"""Typed configuration for Divergence Decoding.

:class:`DDConfig` is the single configuration object for the vLLM integration
(and a convenient bundle of defaults for direct
:class:`~ftp.engine.AuxBatchedEngine` use). It round-trips
to/from ``DD_*`` environment variables because vLLM
instantiates logits-processor *classes* inside its engine-core subprocess —
environment variables are the only configuration channel that survives every
spawn method and the ``vllm serve`` CLI. Library users should prefer
:func:`ftp.vllm.make_processor`, which carries a ``DDConfig``
directly and falls back to the environment transparently.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field, fields

# field name -> env var name. Kept stable: these are the public serve-CLI surface.
_ENV_NAMES: dict[str, str] = {
    "aux_p": "DD_AUX_P",
    "aux_q": "DD_AUX_Q",
    "window": "DD_WINDOW",
    "alpha_default": "DD_ALPHA_DEFAULT",
    "suppress_tokens": "DD_SUPPRESS_TOKENS",
    "tokenizer": "DD_TOK_PATH",
    "dtype": "DD_DTYPE",
    "compile_aux": "DD_COMPILE_AUX",
    "prewarm": "DD_PREWARM",
    "mode": "DD_MODE",
    "max_feeds_per_step": "DD_MAX_FEEDS",
    "aux_device": "DD_AUX_DEVICE",
    "fuse_aux": "DD_FUSE_AUX",
    "fuse_pin": "DD_FUSE_PIN",
    "retemplate": "DD_RETEMPLATE",
    "aux_kv_gb": "DD_AUX_KV_GB",
}

_MODES = ("auto", "shared", "universal")
_FUSE = ("auto", "on", "off")


@dataclass(frozen=True)
class DDConfig:
    """Configuration for Divergence Decoding.

    Args:
        aux_p: Path or HF Hub id of the "forget" aux model (trained WITH the
            knowledge to suppress).
        aux_q: Path or HF Hub id of the "retain" aux model (trained WITHOUT it).
        window: Aux-model context window in tokens. Aux KV caches slide past it.
        alpha_default: DD strength used when a request's ``SamplingParams``
            omits ``extra_args={"dd_alpha": ...}``. ``0.0`` bypasses DD.
        suppress_tokens: Tokens banned during DD generation, given as token ids
            and/or token strings (e.g. ``("<think>",)`` for reasoning models
            whose thinking blocks would blow the generation budget).
        tokenizer: In shared mode: tokenizer used to build the special-token
            whitelist (ids whose probability is pinned to P's); defaults to
            ``aux_p``, with which P must agree. In universal mode: overrides
            the P-tokenizer path otherwise taken from vLLM's model config
            (the whitelist always indexes P's vocabulary).
        dtype: Aux model dtype (``"bfloat16"`` or ``"float16"``/``"float32"``).
        compile_aux: ``torch.compile`` the aux decode step into CUDA graphs
            (~10x faster aux step; slower startup; CUDA only).
        prewarm: If > 0, allocate aux caches and capture the compiled decode
            graphs for this batch size at processor init — before vLLM's memory
            profiling, so the footprint is accounted automatically. Set this to
            your expected peak DD concurrency.
        mode: ``"shared"`` = P and the aux pair share one tokenizer (token ids
            pass straight through); ``"universal"`` = P's tokenizer differs
            from the aux pair's (P's stream is retokenized for the aux models
            and their distributions are mapped onto P's vocabulary);
            ``"auto"`` (default) detects which applies by comparing the
            tokenizers at processor init.
        max_feeds_per_step: Universal mode only — cap on aux tokens fed per
            request per P step (one P token can retokenize into several aux
            tokens; excess queues to the next step).
        aux_device: Device for the aux engines (e.g. ``"cuda:1"``). Default
            ``None`` places them on P's device. Putting the aux pair on its
            own GPU frees P's card for weights and KV; the per-step logits
            transfer is ~16 MB per model (negligible over NVLink).
        fuse_aux: ``"auto"`` (default) stacks the two aux models into ONE
            fused forward when their architectures are identical (~2x faster
            aux step: one batched kernel per layer computes both models),
            falling back loudly to two engines otherwise (e.g. different
            sizes, tied embeddings). ``"on"`` requires fusion (raises if the
            pair is incompatible); ``"off"`` always runs two engines.
        fuse_pin: Whitelist pinning — keep special/control tokens (EOS,
            ``<|im_end|>``, ``<think>``, …) at P's probability EXACTLY so DD
            only steers content tokens and never suppresses the stop tokens.
            ON by default; effectively mandatory (without it α inflates content
            logits, EOS loses relatively, and generations ramble and leak
            more). Set ``False`` only for the deliberate no-pin ablation arms.
        aux_kv_gb: Cap on the aux engines' TOTAL page-pool footprint in GB
            (weights excluded). ``0`` (default) sizes the full-attention pool
            to the ``prewarm x window`` worst case — fine for small windows,
            impossible for 32K-context aux models, so long-context deployments
            must set it to the aux GPU's spare memory. Sliding-layer pools are
            always exact-sized rings and come out of this budget first.
        retemplate: Universal mode only — name of a chat-template adapter preset
            (e.g. ``"gemma"``) that re-renders P's prompt under the *aux*
            tokenizer's own chat template, so an aux pair instruction-tuned on a
            different template (e.g. Qwen ChatML) sees its native wrapper instead
            of P's markup as foreign sub-words. ``None`` (default) retokenizes
            P's stream verbatim. See
            :class:`ftp.translate.ChatTemplateAdapter`.
    """

    aux_p: str
    aux_q: str
    window: int = 2048
    alpha_default: float = 1.375
    suppress_tokens: tuple[int | str, ...] = field(default=())
    tokenizer: str | None = None
    dtype: str = "bfloat16"
    compile_aux: bool = False
    prewarm: int = 0
    mode: str = "auto"
    max_feeds_per_step: int = 3
    aux_device: str | None = None
    fuse_aux: str = "auto"
    fuse_pin: bool = True
    retemplate: str | None = None
    aux_kv_gb: float = 0.0

    def __post_init__(self) -> None:
        if not self.aux_p or not self.aux_q:
            raise ValueError("DDConfig requires both aux_p and aux_q model paths")
        if self.window < 2:
            raise ValueError(f"window must be >= 2, got {self.window}")
        if self.alpha_default < 0:
            raise ValueError(f"alpha_default must be >= 0, got {self.alpha_default}")
        if self.prewarm < 0:
            raise ValueError(f"prewarm must be >= 0, got {self.prewarm}")
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {self.mode!r}")
        if self.fuse_aux not in _FUSE:
            raise ValueError(f"fuse_aux must be one of {_FUSE}, got {self.fuse_aux!r}")
        if self.aux_kv_gb < 0:
            raise ValueError(f"aux_kv_gb must be >= 0, got {self.aux_kv_gb}")
        if self.max_feeds_per_step < 1:
            raise ValueError(f"max_feeds_per_step must be >= 1, got {self.max_feeds_per_step}")
        if isinstance(self.suppress_tokens, list):
            object.__setattr__(self, "suppress_tokens", tuple(self.suppress_tokens))

    # ── Environment round-trip ────────────────────────────────────────────────

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> DDConfig:
        """Build a config from ``DD_*`` environment variables.

        Raises ``KeyError`` if ``DD_AUX_P`` / ``DD_AUX_Q`` are unset.
        """
        kwargs: dict = {"aux_p": env["DD_AUX_P"], "aux_q": env["DD_AUX_Q"]}
        defaults = {f.name: f.default for f in fields(cls)}
        for name, env_name in _ENV_NAMES.items():
            if name in ("aux_p", "aux_q") or env_name not in env:
                continue
            raw = env[env_name]
            if name == "suppress_tokens":
                kwargs[name] = _parse_suppress(raw)
            elif name in ("tokenizer", "aux_device", "retemplate"):  # nullable strings
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
        """Render this config as ``DD_*`` environment variables."""
        out: dict[str, str] = {}
        for name, env_name in _ENV_NAMES.items():
            value = getattr(self, name)
            if name == "suppress_tokens":
                out[env_name] = ",".join(str(t) for t in value)
            elif name in ("tokenizer", "aux_device", "retemplate"):  # nullable strings
                out[env_name] = value or ""
            elif isinstance(value, bool):
                out[env_name] = "1" if value else "0"
            else:
                out[env_name] = str(value)
        return out

    def apply_env(self, env: MutableMapping[str, str] = os.environ) -> None:
        """Write this config into ``env`` (e.g. before constructing ``vllm.LLM``)."""
        env.update(self.to_env())

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def tokenizer_path(self) -> str:
        return self.tokenizer or self.aux_p

    def resolve_suppress_ids(self, tokenizer) -> list[int]:
        """Resolve ``suppress_tokens`` (ids and/or strings) to token ids."""
        ids: list[int] = []
        for tok in self.suppress_tokens:
            if isinstance(tok, int) or (isinstance(tok, str) and tok.lstrip("-").isdigit()):
                ids.append(int(tok))
                continue
            tid = tokenizer.convert_tokens_to_ids(tok)
            unk = getattr(tokenizer, "unk_token_id", None)
            if tid is None or tid == unk:
                raise ValueError(
                    f"suppress token {tok!r} does not resolve to a known token id "
                    f"in {type(tokenizer).__name__}"
                )
            ids.append(int(tid))
        return ids


def _cuda_index(device: str | None) -> int | None:
    """``"cuda:2"`` -> ``2``; anything else (None, "cpu", bare "cuda") -> None."""
    if device and device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return None
    return None


def default_device_layout(
    n_gpu: int,
    tensor_parallel_size: int = 1,
    aux_device: str | None = None,
    guard_device: str | None = None,
) -> tuple[str | None, str | None]:
    """Default ``(aux_device, guard_device)`` for a box with ``n_gpu`` visible GPUs.

    P occupies the tensor-parallel ranks ``cuda:0 .. cuda:{tp-1}``; the aux pair
    takes the first GPU after P; the guard judge takes the next free one. When
    there aren't enough GPUs the placements collapse inward: the guard shares
    the aux GPU, and when P spans every GPU the aux pair co-hosts with the LAST
    TP rank (rank 0 also hosts the logits gather + sampler, so the last rank's
    card has the most headroom). Explicit ``aux_device``/``guard_device`` values
    pass through untouched; the guard default routes around an explicit aux GPU.

    Layouts this produces (P = TP ranks, A = aux pair, G = guard judge)::

        1 GPU            -> (None, None)          everything co-hosts with P
        2 GPUs, TP=1     -> P | A+G               the classic 2xH100 split
        4 GPUs, TP=2     -> P P | A | G           the 4xH100 thinking-mode split
        4 GPUs, TP=1     -> P | A | G  (cuda:3 idle)
        n GPUs, TP=n     -> P .. P+A+G            aux+guard on the last rank
    """
    tp = tensor_parallel_size
    if aux_device is None and n_gpu > 1:
        aux_device = f"cuda:{tp}" if tp < n_gpu else f"cuda:{n_gpu - 1}"
    if guard_device is None:
        aux_idx = _cuda_index(aux_device)
        for i in range(min(tp, n_gpu), n_gpu):
            if i != aux_idx:
                guard_device = f"cuda:{i}"
                break
        else:
            guard_device = aux_device
    return aux_device, guard_device


def _parse_suppress(raw: str) -> tuple[int | str, ...]:
    """Parse ``DD_SUPPRESS_TOKENS``: comma-separated ids and/or token strings."""
    out: list[int | str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part) if part.lstrip("-").isdigit() else part)
    return tuple(out)
