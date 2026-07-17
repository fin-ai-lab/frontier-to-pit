"""Live degeneration guard: detect decoding collapse DURING generation and rewind.

The failure mode this targets: under a strong DD push the fused decode can fall into
a repeating loop / symbol spam / non-words and burn the whole generation budget on
garbage (measured at alpha=1.5 NOTHINK: 45% of M&A and 35% of pharma generations
destroyed). Suppress-only fusion fixed the destruction but gave up most of the
unlearning — so instead of weakening the intervention, this guard keeps the fusion
and repairs the rare collapse when it happens.

How it works (three cooperating pieces):

  ENGINE SIDE — :class:`ftp.vllm.GuardLogitsProcessor` (lives in vLLM's engine-core
  process, judge model on the aux GPU). Every ``interval`` ENGINE steps it sweeps
  the whole batch: for each active request it decodes the last ``backtrack`` tokens
  (crossing into the prompt tail if the output is shorter) and asks a small judge LM
  "has this degenerated? yes/no" in ONE batched forward — so the steady-state cost
  is one small forward per ``interval`` steps regardless of batch width. When
  p(yes) >= threshold for a row, it forces the reserved MARKER token on THAT row
  only, ending that request exactly there; the rest of the batch is untouched.

  WIRE PROTOCOL — the client registers the marker in ``stop_token_ids``, so a tripped
  request comes back as an ordinary stop with ``stop_reason == marker_id``. No IPC
  with the engine-core process is needed.

  CLIENT SIDE — :func:`rollback_generate`. Tripped requests are truncated by
  ``backtrack`` tokens and resubmitted (prompt + kept output) with the remaining
  budget; sampling is stochastic so the retry takes a different path. Escalation
  policy when a request keeps breaking AT THE SAME POINT: after ``tries``
  no-progress attempts the walk-back deepens by another ``backtrack`` (50 -> 100 ->
  150 ...), on the theory that the collapse was seeded earlier than the window
  shows. A request that escalates all the way back to its FIRST tokens and still
  breaks returns a visible failure string instead of garbage; a request that runs
  out of budget or rounds mid-rescue returns the clean accepted prefix it has.

vLLM cannot rewind a request's KV mid-flight, so the rewind is this stop+resubmit;
the clean path (no degeneration) is never interrupted, and the only steady-state cost
is one small batched judge forward per ``interval`` tokens on the aux GPU.

This module is import-safe without vLLM (config, judge, pure helpers, and the
client loop — which only duck-types vLLM's outputs). The logits processor itself
lives in :mod:`ftp.vllm`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, fields

# field name -> env var name. Same round-trip contract as DDConfig: env is the only
# config channel that survives vLLM's engine-core spawn.
_ENV_NAMES: dict[str, str] = {
    "model": "DD_GUARD_MODEL",
    "device": "DD_GUARD_DEVICE",
    "interval": "DD_GUARD_INTERVAL",
    "backtrack": "DD_GUARD_BACKTRACK",
    "threshold": "DD_GUARD_THRESHOLD",
    "marker": "DD_GUARD_MARKER",
    "dtype": "DD_GUARD_DTYPE",
    "tries": "DD_GUARD_TRIES",
    "max_rounds": "DD_GUARD_MAX_ROUNDS",
    "min_tokens": "DD_GUARD_MIN_TOKENS",
    "gate_ratio": "DD_GUARD_GATE_RATIO",
    "judge_rows": "DD_GUARD_JUDGE_ROWS",
}

#: reserved-token candidates for the stop marker, tried in order. Must be a real
#: single token in P's vocabulary that sampling can never produce naturally.
_MARKER_CANDIDATES = ("<|fim_pad|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|repo_name|>")


@dataclass(frozen=True)
class GuardConfig:
    """Configuration for the live degeneration guard.

    Args:
        model: HF id / local path of the judge LM (a small instruct model). The
            judge reads the window as TEXT (decoded engine-side with P's
            tokenizer), so it need not share P's family. Default
            ``Qwen/Qwen3-1.7B`` — chosen so the DOCUMENTED QUICKSTART (2x80GB,
            judge co-hosted with the 8B aux pair on cuda:1) always fits:
            ~3.4 GB resident, plain attention, small transients, ungated,
            and end-to-end validated (2026-07-16 judge x threshold sweep,
            ma+pharma at alpha=1.5: destroyed 31.4% unguarded -> 10.8%).
            WITH >80 GB OF JUDGE-GPU HEADROOM (h200/141GB co-hosted, or a
            spare 3rd GPU) PREFER ``unsloth/gemma-3-4b-it`` WITH ``threshold
            0.5`` — the sweep winner: destroyed -> 7.7%, zero visible
            failures, ~42ms per gated call — but its ~8.6 GB weights tipped
            an 80 GB card over next to the 8B pair (the benchmark eval arms
            pin it explicitly; they run on h200s). Also measured: the old
            ``Qwen/Qwen3.5-2B`` judge repairs to 8.6% but its GDN layers cost
            a ~113ms fixed HF-eager scan per fire; gemma-3-1b is unusable (no
            class separation: gated FPR 22-46% at every threshold).
        device: Device for the judge (e.g. ``"cuda:1"``). ``None`` = the default
            layout (:func:`ftp.config.default_device_layout`): the first free GPU
            after P's tensor-parallel ranks and the DD aux devices; failing that,
            under TP the LAST TP rank (sharded P leaves spare memory there — the
            judge loads post-profiling, so keep ``gpu_memory_utilization`` low
            enough to leave it ~6 GB), else the aux GPU. On 2xGPU the judge sits
            next to the aux pair on cuda:1; on 4xGPU TP=2 (aux pair split over
            cuda:2/cuda:3) it rides cuda:1; single-GPU boxes co-host it with P.
        interval: Judge every N ENGINE steps — one batched forward over all
            active rows per sweep, so the cost is independent of batch width
            (per-row token counters would desynchronize in large batches and
            fire nearly every step). Each row still accrues ~N new tokens
            between sweeps (one token per step).
        backtrack: Tokens shown to the judge per check AND tokens discarded on a
            rewind. The window is the last ``backtrack`` tokens of prompt+output.
        threshold: Trip when p(degenerated) >= threshold. Default 0.8 = the
            Qwen3-1.7B judge's end-to-end winner (vs 0.5: less destruction,
            35% fewer rollbacks, 1 vs 5 visible failures) — and both finalist
            judges are nearly threshold-FLAT (saturated yes/no logits;
            offline gated operating points move only a few points from 0.3 to
            0.99), so 0.8 is also fine for gemma-3-4b (which the eval arms pin
            at 0.5, within noise of 0.9 end-to-end).
            tools/calibrate_guard_threshold.py re-derives the operating table
            for any candidate judge from the labeled windows. A persistent
            loop is re-checked every sweep, so the effective catch rate
            compounds toward 1 while a false trip costs only one recoverable
            rewind. (History: the original Qwen3.5-2B judge measured TPR 0.80
            / FPR 0.02 per check at gate 1.6 + threshold 0.5.)
        gate_ratio: zlib pre-gate: a window is only shown to the judge when
            ``len(bytes)/len(zlib(bytes)) >= gate_ratio`` (loops compress;
            normal prose sits ~0.6-1.2 at window size). Kills most judge
            false-positives AND ~10x-es down judge load. 0 disables the gate.
            Default 1.3 (2026-07-16 gate x rounds sweep): NONWORDS garble
            escapes a 1.6 gate because a half-prose/half-garble window
            compresses ~1.3-1.4 — 1.3 cut ma text-destruction 15.5 -> 5.4% and
            pharma 26 -> 11% at unchanged leak, while 1.0 (gate ~open) traded
            it for 7-15% visible FAIL_TEXT and halved generation lengths.
        marker: Token string forced on a trip and registered as a stop token.
        dtype: Judge dtype.
        tries: No-progress rescue attempts at one stuck point before the
            walk-back ESCALATES by another ``backtrack`` (50 -> 100 -> 150 ...).
            A request that escalates back to its first tokens and still breaks
            fails visibly (see :func:`rollback_generate`).
        max_rounds: GLOBAL cap on generation rounds per request — since each
            round can trip the judge at most once, this is the "you can only
            fail the judge N times" bound that makes near-infinite rescue loops
            impossible. Past it, the clean accepted prefix is returned as-is.
            Default 10, paired with gate 1.3 (same sweep): rounds only matter
            once the gate actually generates trips, and 10 ends the rescue
            marathons that used to run to the token budget and leave loop
            debris; alone (gate 1.6) neither 10 nor 5 moved anything.
            THINKING MODE runs 40 (run.py --think, the eval thinking twin, the
            chat deployment): outline/list-style CoT is compressible enough to
            pass the gate and trip the judge routinely, and at 10 rounds a
            reasoning reply dies mid-CoT (~3k of a 16k budget, observed) with
            no answer ever reached. 10 stays the NOTHINK/instruct default.
        min_tokens: A row is only judged once it has generated at least this
            many tokens. Guards the early steps: a 2-token output like ``**``
            inside a sweep window is indistinguishable from symbol spam even
            though nothing has gone wrong yet — give the generation room to
            look like text before the judge sees it.
        judge_rows: Max windows per judge FORWARD — a sweep with more due rows
            runs several forwards back to back. This bounds the judge's
            activation transient, which for the gemma-3-4b judge reached ~18 GB
            on a 64-row call (HF eager) and OOM'd cuda:1 next to the 8B aux
            pair on 80 GB cards (clean-run jobs 168309/11/12, 2026-07-16);
            the h200s only survived by having 141 GB. At 16 rows the transient
            is a few GB, and the common gated case (~a couple rows per fired
            sweep) is one forward either way. 0 disables the cap.
    """

    model: str = "Qwen/Qwen3-1.7B"
    device: str | None = None
    interval: int = 25
    backtrack: int = 50
    threshold: float = 0.8
    marker: str = "<|fim_pad|>"
    dtype: str = "bfloat16"
    tries: int = 2
    max_rounds: int = 10
    min_tokens: int = 25
    gate_ratio: float = 1.3
    judge_rows: int = 16

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError(f"interval must be >= 1, got {self.interval}")
        if self.min_tokens < 1:
            raise ValueError(f"min_tokens must be >= 1, got {self.min_tokens}")
        if self.tries < 1:
            raise ValueError(f"tries must be >= 1, got {self.tries}")
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {self.max_rounds}")
        if self.backtrack < self.interval:
            raise ValueError(
                f"backtrack ({self.backtrack}) must be >= interval ({self.interval}); "
                "a rewind must discard at least everything generated since the last "
                "clean check, or the surviving prefix may end in judged-bad tokens")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {self.threshold}")
        if self.gate_ratio < 0:
            raise ValueError(f"gate_ratio must be >= 0 (0 disables), got {self.gate_ratio}")
        if self.judge_rows < 0:
            raise ValueError(f"judge_rows must be >= 0 (0 disables), got {self.judge_rows}")

    # ── Environment round-trip (mirrors DDConfig) ────────────────────────────

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> GuardConfig:
        kwargs: dict = {}
        defaults = {f.name: f.default for f in fields(cls)}
        for name, env_name in _ENV_NAMES.items():
            if env_name not in env:
                continue
            raw = env[env_name]
            if name == "device":  # nullable string
                kwargs[name] = raw or None
            elif isinstance(defaults[name], bool):
                kwargs[name] = raw == "1"
            elif isinstance(defaults[name], int):
                kwargs[name] = int(raw)
            elif isinstance(defaults[name], float):
                kwargs[name] = float(raw)
            else:
                kwargs[name] = raw
        return cls(**kwargs)

    def to_env(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, env_name in _ENV_NAMES.items():
            value = getattr(self, name)
            if name == "device":
                out[env_name] = value or ""
            elif isinstance(value, bool):
                out[env_name] = "1" if value else "0"
            else:
                out[env_name] = str(value)
        return out

    def apply_env(self, env: MutableMapping[str, str] = os.environ) -> None:
        env.update(self.to_env())


def resolve_marker_id(tokenizer, marker: str | None = None) -> int:
    """Resolve the marker token string to a single id in ``tokenizer``.

    Engine (forcing) and client (stop detection) MUST call this with the same
    tokenizer so both sides agree on the id. Falls through the reserved-token
    candidates when the configured marker is absent from the vocabulary.
    """
    unk = getattr(tokenizer, "unk_token_id", None)
    for cand in ((marker,) if marker else ()) + _MARKER_CANDIDATES:
        if cand is None:
            continue
        tid = tokenizer.convert_tokens_to_ids(cand)
        if tid is not None and tid != unk and tid >= 0:
            return int(tid)
    raise ValueError(
        f"no guard marker token resolves in {type(tokenizer).__name__} "
        f"(tried {marker!r} then {_MARKER_CANDIDATES}); set DD_GUARD_MARKER to a "
        "reserved token that exists in P's vocabulary")


def window_ids(prompt_ids, output_ids, backtrack: int) -> list[int]:
    """The judge window: last ``backtrack`` tokens of prompt+output.

    Crossing into the prompt tail matters twice: early in a generation the output
    alone is too short to judge, and after a rewind-resubmit the loop's earlier
    tokens live in the (new) prompt — a window that ignored them would let a loop
    straddle the boundary undetected.
    """
    prompt_ids = list(prompt_ids or [])
    out = list(output_ids or [])
    need = backtrack - len(out)
    if need > 0:
        return prompt_ids[max(0, len(prompt_ids) - need):] + out
    return out[-backtrack:]


def window_suspicious(text: str, gate_ratio: float) -> bool:
    """zlib pre-gate: is this window compressible enough to be a loop?

    Measured on gemma-labeled 50-token windows from the alpha=1.5 results:
    ratio >= 1.6 passes 89% of degenerate windows and 3% of clean ones (normal
    prose sits ~0.6-1.2 at this size; loops 2-15). Windows failing the gate are
    treated as clean without consulting the judge.
    """
    if gate_ratio <= 0:
        return True
    import zlib

    b = text.encode("utf-8", "replace")
    return len(b) / max(len(zlib.compress(b, 6)), 1) >= gate_ratio


# The judge prompt. Kept deliberately narrow: the offline destruction judge's failure
# modes (tools/README_degradation.md) showed that "is this text bad?" flags rambling
# and hallucination — which are EXPECTED under unlearning. Only mechanical collapse
# counts. Calibrated on gemma-labeled windows from the alpha=1.5 arm
# (tools/calibrate_guard_threshold.py): with the default gemma-3-4b judge this
# few-shot form measures gated TPR ~86% / FPR ~6-7%, flat across thresholds
# 0.3-0.99 (the judge's yes/no logits saturate). History: on the original
# Qwen3.5-2B judge it measured TPR 0.80 / FPR 0.02 per check at threshold 0.5;
# the zero-shot fenced form scored mushier on both (7% FPR at its best point);
# gemma-3-1b cannot use this prompt at all (clean-window p50 = 0.73 — no
# separation at any threshold).
_JUDGE_PROMPT = """\
Judge whether TEXT (a snippet from the middle of a longer output; it starts and ends \
mid-sentence, which is normal) has DEGENERATED into mechanical repetition.

Examples:
TEXT: "the deal; the deal; the deal; the deal; the deal; the deal; the deal" -> yes
TEXT: "regulators approved the merger in March, and the stock rose 4% on the news" -> no

Degenerated = the same word, phrase, or symbol pattern repeated over and over, or \
keyboard-mash non-words. Normal prose, lists, tables, code, quotes, or wrong facts \
are NOT degenerated. If unsure, answer no.

TEXT: "{window}"

Answer yes or no."""


class DegenJudge:
    """Small-LM yes/no degeneration judge: one batched forward per check.

    Loads ``cfg.model`` on ``device`` (HF transformers, eager) and reads the
    first-token logits after the chat template's generation prompt, comparing the
    "yes" group against the "no" group — a single constrained step, no decoding
    loop, so a check costs one short prefill.
    """

    def __init__(self, cfg: GuardConfig, device) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._device = device
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[cfg.dtype]
        self._tok = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        self._tok.padding_side = "left"  # last position = readout position for every row
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model, dtype=dtype, trust_remote_code=True)
        except Exception:  # noqa: BLE001 — fall through to the declared architecture
            # Multimodal-wrapped checkpoints (gemma-3-4b-it:
            # Gemma3ForConditionalGeneration; Qwen3.5-*: config nests the LM
            # under text_config) crash the plain-CausalLM auto route
            # ('...Config' object has no attribute 'vocab_size'). Load the
            # class the checkpoint DECLARES — text-only forwards work fine;
            # the vision tower just sits unused.
            import transformers as _tf
            from transformers import AutoConfig

            arch = (AutoConfig.from_pretrained(cfg.model, trust_remote_code=True)
                    .architectures or [None])[0]
            cls = getattr(_tf, arch or "", None)
            if cls is None:
                raise
            model = cls.from_pretrained(cfg.model, dtype=dtype, trust_remote_code=True)
        self._model = model.to(device).eval()
        self._threshold = cfg.threshold
        self._rows_cap = cfg.judge_rows

        def first_ids(variants):
            ids = []
            for v in variants:
                enc = self._tok.encode(v, add_special_tokens=False)
                if enc:
                    ids.append(enc[0])
            return sorted(set(ids))

        # "yes"/"Yes" and "no"/"No" (no leading-space variants: both the gemma and
        # Qwen chat templates end the generation prompt with a newline, so the
        # answer token starts a line).
        self._yes_ids = first_ids(["yes", "Yes", "YES"])
        self._no_ids = first_ids(["no", "No", "NO"])
        if not self._yes_ids or not self._no_ids:
            raise ValueError(f"judge tokenizer for {cfg.model} has no yes/no tokens")

    def _render(self, window_text: str) -> str:
        msgs = [{"role": "user", "content": _JUDGE_PROMPT.format(window=window_text)}]
        try:
            return self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:  # template without the enable_thinking kwarg
            return self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def p_degen(self, window_texts: list[str]) -> list[float]:
        """p(degenerated) per window; forwards capped at ``judge_rows`` windows.

        The cap bounds the activation transient (gemma-3-4b HF-eager measured
        ~18 GB on one 64-row forward — enough to OOM an 80 GB card co-hosting
        the aux pair); the common gated case is a couple of rows = one forward.
        """
        cap = self._rows_cap
        if cap and len(window_texts) > cap:
            out: list[float] = []
            for i in range(0, len(window_texts), cap):
                out += self.p_degen(window_texts[i:i + cap])
            return out
        torch = self._torch
        prompts = [self._render(t) for t in window_texts]
        enc = self._tok(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(self._device)
        with torch.inference_mode():
            # Only the LAST position is read, so don't materialize the full
            # [rows, seq, vocab] logits tensor: at Qwen's 248k vocab that is
            # ~0.5 MB/position, and a 26-row sweep asked for 3.25 GiB — OOMing
            # a judge co-hosted on a P TP rank (measured, 4xH100 job 168230).
            # logits_to_keep=1 computes [rows, 1, vocab] (~13 MB) instead.
            try:
                out = self._model(**enc, logits_to_keep=1)
            except TypeError:  # a forward without the kwarg (exotic judge arch)
                out = self._model(**enc)
            logits = out.logits[:, -1, :].float()
        yes = torch.logsumexp(logits[:, self._yes_ids], dim=-1)
        no = torch.logsumexp(logits[:, self._no_ids], dim=-1)
        return torch.sigmoid(yes - no).tolist()  # p(yes) within the {yes, no} pair

    def tripped(self, window_texts: list[str]) -> list[bool]:
        return [p >= self._threshold for p in self.p_degen(window_texts)]


#: returned as the whole output when a request degenerates all the way back at its
#: very first tokens: better a visible failure than plausible-looking garbage.
FAIL_TEXT = "[Could not generate without degeneration]"


def rollback_generate(
    llm,
    token_prompts: list[list[int]],
    sampling_params,
    *,
    marker_id: int,
    backtrack: int,
    tries: int = 2,
    max_rounds: int = 10,
    decode,
    use_tqdm: bool = False,
):
    """``llm.generate`` with guard rewinds: rollback-and-resample on marker stops.

    Escalation policy: a tripped round keeps its clean prefix (everything but the
    last ``backtrack`` tokens) and resamples. ``tries`` consecutive NO-PROGRESS
    trips (the rewind ate the whole round) mean the collapse is seeded earlier
    than the window shows — the walk-back escalates by another ``backtrack`` into
    previously accepted text (cumulative 50 -> 100 -> 150 ...) and the counter
    resets. A request that escalates back to EMPTY and still trips returns
    :data:`FAIL_TEXT`; a request that exhausts its token budget or ``max_rounds``
    mid-rescue returns the clean accepted prefix it has.

    Args:
        llm: a vLLM ``LLM`` (duck-typed: anything with ``.generate(prompts, sps)``).
        token_prompts: prompts as token-id lists (the lm-eval calling convention).
        sampling_params: one ``SamplingParams`` or one per prompt. Cloned per
            request — the caller's objects are never mutated.
        marker_id: the guard marker (from :func:`resolve_marker_id` on P's tokenizer).
        backtrack: tokens discarded per rewind (must equal the engine's).
        tries: no-progress trips at one point before the walk-back escalates.
        max_rounds: total generation rounds per request (backstop).
        decode: ``ids -> text`` for re-detokenizing spliced outputs.

    Returns:
        (outputs, stats) — outputs in input order, each a vLLM ``RequestOutput``
        whose first completion has the SPLICED token_ids/text (accepted prefix +
        final continuation); stats = {"rollbacks", "escalations", "failed"}.
    """
    import copy

    n = len(token_prompts)
    base = sampling_params if isinstance(sampling_params, list) else [sampling_params] * n
    if len(base) != n:
        raise ValueError(f"{len(base)} sampling_params for {n} prompts")

    sps, budgets = [], []
    for sp in base:
        sp = copy.deepcopy(sp)
        # Register the marker as a stop token: a trip surfaces as a normal stop
        # with stop_reason == marker_id (vLLM reports the stopping token id there).
        stops = set(getattr(sp, "stop_token_ids", None) or [])
        stops.add(marker_id)
        sp.stop_token_ids = sorted(stops)
        sps.append(sp)
        budgets.append(sp.max_tokens)

    accepted: list[list[int]] = [[] for _ in range(n)]
    consec = [0] * n   # consecutive no-progress trips at the current stuck point
    rounds = [0] * n
    results: list = [None] * n
    stats = {"rollbacks": 0, "escalations": 0, "failed": 0}
    pending = list(range(n))

    def finalize(i, out, tail_ids, *, failed: bool = False) -> None:
        comp = out.outputs[0]
        if failed:
            stats["failed"] += 1
            comp.token_ids = []
            comp.text = FAIL_TEXT
        else:
            merged = accepted[i] + list(tail_ids)
            comp.token_ids = merged
            comp.text = decode(merged)
        results[i] = out

    while pending:
        outs = llm.generate(
            [{"prompt_token_ids": token_prompts[i] + accepted[i]} for i in pending],
            [sps[i] for i in pending],
            use_tqdm=use_tqdm,
        )
        nxt = []
        for i, out in zip(pending, outs, strict=True):
            rounds[i] += 1
            comp = out.outputs[0]
            ids = list(comp.token_ids)
            tripped = getattr(comp, "stop_reason", None) == marker_id or (
                ids and ids[-1] == marker_id)
            if ids and ids[-1] == marker_id:
                ids = ids[:-1]  # the marker is bookkeeping, never output
            if not tripped:
                finalize(i, out, ids)
                continue
            stats["rollbacks"] += 1
            kept = ids[:max(0, len(ids) - backtrack)]
            if kept:
                accepted[i] += kept
                consec[i] = 1  # progress — this trip starts the count at the NEW point
            else:
                consec[i] += 1
                if consec[i] >= tries:
                    if not accepted[i]:
                        # Broke `tries` times at the very first tokens with nothing
                        # to walk back into: give up visibly.
                        finalize(i, out, [], failed=True)
                        continue
                    # Escalate: the collapse is seeded before the window — walk
                    # back another `backtrack` into accepted text and start over.
                    stats["escalations"] += 1
                    accepted[i] = accepted[i][:max(0, len(accepted[i]) - backtrack)]
                    consec[i] = 0
            if rounds[i] >= max_rounds or budgets[i] - len(accepted[i]) <= 0:
                # Out of rounds/budget mid-rescue: return the clean prefix we have
                # (or the visible failure if there is none).
                finalize(i, out, [], failed=not accepted[i])
                continue
            sp = copy.deepcopy(sps[i])
            sp.max_tokens = budgets[i] - len(accepted[i])
            sps[i] = sp
            nxt.append(i)
        pending = nxt
    return results, stats
