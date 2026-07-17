#!/usr/bin/env python
"""Central vLLM runner for Divergence Decoding + SAE steering.

One entry point that builds P with both interventions (via
``ftp.serve``) and either chats interactively or runs a single
prompt — replies stream token-by-token.

    python run.py chat                         # interactive, one fresh turn per message
    python run.py chat --no-dd                 # steering only
    python run.py chat --think                 # reasoning on for the whole session
    python run.py generate "It is 2019. ..."   # one prompt, stream, exit
    python run.py generate --think "It is ..." # one prompt, with a real <think> block
    python run.py chat --steer "" --no-steer   # plain P

This runner keeps NO chat history: every message is answered as its own fresh
conversation. That is a simplification of this simple runner, not a limit of the
methods — DD + steering work multi-turn. It is off by default here because the
benchmarks (and the alpha/clamp calibrated against them) are single-turn, so keeping
history would quietly move you off the measured config. Press ENTER during a reply to
stop just that reply (Ctrl-C can't — vLLM's engine-core subprocess catches SIGINT and
shuts down, ending the session). Quit with 'exit'/'quit'/'q' or Ctrl-D.

Defaults reproduce the benchmarked config: bf16 P (Qwen/Qwen3.5-27B), DD alpha=1.5,
steer L48:28961@10, thinking OFF, and the benchmarks' forecasting system prompt on
every turn (the point-in-time-expert preamble baked into the temporal eval datasets;
--no-system-prompt drops it, --system-prompt replaces it) — the same P precision,
sampling and <think> ban as evals/lmeval, so run.py matches the benchmarks. --think flips the session to reasoning
mode (real <think> block, no ban, bigger budgets — mirroring the evals'
qwen3_5_27b_think arm); DD + steering are calibrated NOTHINK, so --think is
exploratory, not a benchmarked config. Pass --model Qwen/Qwen3.5-27B-FP8 for FP8
(smaller/faster, but untested with DD+steering — it can destabilize the fused decode).
The 27B P needs an ≥80GB GPU (or --tensor-parallel-size 2 on 40GB cards). Aux
models download from the Hugging Face Hub by default.

Multi-GPU placement is automatic from the visible GPU count and
--tensor-parallel-size: with two or more GPUs free after P's TP ranks the aux
pair SPLITS, one model per card (two-engine path — fusion needs one card); with
one free GPU both share it fused; the guard judge takes a free card if any
remain, else rides the last TP rank's spare memory. 2xGPU TP=1:
P | aux+aux+guard (unchanged); 4xGPU TP=2: P P+guard | aux-retain | aux-forget.
--think at the full 20480 context does not fit next to the aux pair + guard on
2xH100 — on a 4xGPU box run ``python run.py chat --think --tensor-parallel-size 2``
and the layout above is picked up automatically (--aux-device / --guard-device
override it).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import os
import signal
import sys
import uuid

from transformers import AutoTokenizer
from vllm import SamplingParams

from ftp.guard import GuardConfig, resolve_marker_id
from ftp.prompts import FORECAST_SYSTEM_PROMPT
from ftp.serve import (
    SteerArgs,
    build_async_llm,
    install_steering_async,
    parse_steer,
    stream,
)

DEFAULT_MODEL = "Qwen/Qwen3.5-27B"  # bf16 — the benchmarked P precision (evals/lmeval)
DEFAULT_MODEL_FP8 = "Qwen/Qwen3.5-27B-FP8"  # optional FP8 via --model (unbenchmarked w/ DD+steer)
DEFAULT_AUX_P = "fin-ai-lab/aux-2024"  # forget (le2025, HAS post-cutoff knowledge)
DEFAULT_AUX_Q = "fin-ai-lab/aux-2015"  # retain (le2015, LACKS it)
DEFAULT_SAE_REPO = "Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50"
DEFAULT_SAE_CACHE = os.environ.get(
    "FTP_SAE_CACHE", os.path.expanduser("~/.cache/qwen-scope-saes")
)  # SAEs are downloaded here from --sae-repo on first steered run
DEFAULT_STEER = "48:28961:10"  # L:feature:clamp (single production feature; L27 unsteered)
# Budgets: NOTHINK replies are short; a real CoT needs room for the <think> block plus
# the answer. The --think pair mirrors the evals' qwen3_5_27b_think arm (max_gen_toks
# 16384, max_model_len 20480) so reasoning here matches how we measured reasoning there.
DEFAULT_MAX_NEW, THINK_MAX_NEW = 2048, 16384
DEFAULT_MAX_MODEL_LEN, THINK_MAX_MODEL_LEN = 4096, 20480


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"P checkpoint (default {DEFAULT_MODEL}, bf16 — matches the "
                         f"benchmarks). Pass {DEFAULT_MODEL_FP8} for FP8 (smaller/faster, but "
                         f"untested with DD+steering — it can destabilize the fused decode; "
                         f"also needs nvcc>=12.9 for DeepGEMM).")
    ap.add_argument("--aux-p", default=DEFAULT_AUX_P, help="DD forget model (HF)")
    ap.add_argument("--aux-q", default=DEFAULT_AUX_Q, help="DD retain model (HF)")
    ap.add_argument("--alpha", type=float, default=1.5, help="DD strength")
    ap.add_argument("--steer", default=DEFAULT_STEER, help="L:feature:clamp triples (comma-sep)")
    ap.add_argument("--sae-repo", default=DEFAULT_SAE_REPO)
    ap.add_argument("--sae-cache", default=DEFAULT_SAE_CACHE)
    ap.add_argument("--sae-dir", default=None, help="local dir of layer{L}.sae.pt (vs --sae-repo)")
    ap.add_argument("--no-dd", action="store_true")
    ap.add_argument("--no-steer", action="store_true")
    # The forecasting system prompt served with every ma/pharma/covid benchmark
    # generation (baked into the eval parquets' system_prompt column; canonical text in
    # ftp.prompts). ON by default so demo replies are prompted like the benchmarks.
    ap.add_argument("--system-prompt", default=FORECAST_SYSTEM_PROMPT,
                    help="system message for every turn (default: the benchmarks' "
                         "forecasting point-in-time prompt)")
    ap.add_argument("--no-system-prompt", action="store_true",
                    help="send no system message at all")
    ap.add_argument("--think", action="store_true",
                    help="reasoning mode for the whole run: the chat template opens a real "
                         "<think> block instead of an empty one, the sampler-level <think> ban "
                         "is dropped, and the budgets default to the evals' thinking arm "
                         f"(--max-new {THINK_MAX_NEW}, --max-model-len {THINK_MAX_MODEL_LEN}). "
                         "EXPLORATORY: alpha and the steering clamp are calibrated NOTHINK, and "
                         "the benchmarks are NOTHINK, so this is off the measured config.")
    ap.add_argument("--max-new", type=int, default=None,
                    help=f"generation budget (default {DEFAULT_MAX_NEW}, or {THINK_MAX_NEW} "
                         f"with --think)")
    # Qwen 3.5 27B native sampling preset — the card's "thinking mode, general tasks"
    # values, identical to evals/lmeval QWEN_SAMPLING so run.py matches the benchmarks.
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--presence-penalty", type=float, default=1.5)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=None,
                    help=f"P context window (default {DEFAULT_MAX_MODEL_LEN}, or "
                         f"{THINK_MAX_MODEL_LEN} with --think, which must fit prompt + CoT)")
    ap.add_argument("--aux-device", default=None,
                    help="device(s) for the aux models: one GPU for both (e.g. cuda:1) or a "
                         "forget,retain pair (e.g. cuda:3,cuda:2) giving each model its own "
                         "card — the pair form disables fusion. Default: split across the "
                         "free GPUs after P's TP ranks when there are two (4xGPU TP=2: "
                         "retain cuda:2, forget cuda:3), one shared card when there is one "
                         "(2xGPU TP=1: cuda:1)")
    ap.add_argument("--gdn-prefill-backend", default=None, choices=["flashinfer", "triton"],
                    help="Qwen3.5 GDN kernel. Default (flashinfer) JIT-compiles via nvcc on the "
                         "first run (~15-20 min, then cached); 'triton' compiles in seconds — use "
                         "it for a quick try, at a small per-token speed cost.")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="skip torch.compile + CUDA-graph capture (~1-3 min faster startup, "
                         "~2x slower per token). Steering falls back to eager post-build hooks.")
    ap.add_argument("--fast", action="store_true",
                    help="fastest startup: implies --enforce-eager and --gdn-prefill-backend "
                         "triton (skips BOTH the ~1-3 min compile and the ~15-20 min first-run "
                         "nvcc GDN build). Small per-token cost; ideal for a quick try / timing "
                         "startup. Explicit --gdn-prefill-backend still wins.")
    # Live degeneration guard (ftp.guard): a small judge LM on the aux GPU checks the
    # last --guard-backtrack tokens every --guard-interval engine steps; on a trip the
    # reply is rewound that many tokens and resampled. ON by default whenever DD is on
    # (the guard exists to repair the DD push's rare decoding collapse). Streaming
    # holds back the newest --guard-backtrack tokens so a rewind never has to un-print
    # anything — the visible stream trails generation by ~1s and arrives in blocks.
    ap.add_argument("--no-guard", action="store_true",
                    help="disable the live degeneration guard (guard is on by default with DD)")
    ap.add_argument("--guard-model", default=GuardConfig.model,
                    help=f"judge LM (HF; default {GuardConfig.model} fits the 2x80GB "
                         "quickstart next to the aux pair; with >80GB judge-GPU headroom "
                         "— h200, or a spare 3rd GPU — prefer unsloth/gemma-3-4b-it with "
                         "--guard-threshold 0.5: best measured collapse repair)")
    ap.add_argument("--guard-device", default=None,
                    help="GPU for the guard judge (default: first free GPU after P's TP ranks "
                         "and the aux models; else under TP the last TP rank's spare memory — "
                         "cuda:1 on 4xGPU TP=2, where the judge loads post-profiling, so leave "
                         "it ~6GB under --gpu-memory-utilization — else the aux GPU, as on "
                         "2xGPU)")
    ap.add_argument("--guard-interval", type=int, default=25,
                    help="judge every N engine steps (one batched check)")
    ap.add_argument("--guard-backtrack", type=int, default=50,
                    help="tokens judged per check AND discarded per rewind")
    ap.add_argument("--guard-threshold", type=float, default=GuardConfig.threshold,
                    help="trip when p(degenerated) >= this (judges are near threshold-flat; "
                         "pair 0.5 with the gemma-3-4b judge)")
    ap.add_argument("--guard-tries", type=int, default=2,
                    help="stuck-point resamples before the walk-back deepens by "
                         "another --guard-backtrack (50 -> 100 -> 150 ...)")
    ap.add_argument("--guard-max-rounds", type=int, default=None,
                    help="rewind rounds per reply before the guard gives up where "
                         "the text stands (default: 10 — the benchmarked sweep "
                         "winner — or 40 with --think: compressible CoT trips the "
                         "judge routinely and an exhausted reasoning reply dies "
                         "mid-CoT with no answer)")


def _resolve_devices(args, use_dd: bool) -> tuple[str | None, str | None]:
    """Place the aux models and guard judge off P's card(s) on a multi-GPU box.

    P fills most of its TP ranks via gpu_memory_utilization, so co-locating the
    small models there OOMs. ftp.config.default_device_layout does the math:
    2xGPU TP=1 -> aux pair + guard on cuda:1; 4xGPU TP=2 -> aux pair SPLIT
    (retain on cuda:2, forget on cuda:3), guard riding cuda:1's TP spare
    memory. Explicit --aux-device / --guard-device always win."""
    if not use_dd:
        return args.aux_device, args.guard_device
    import torch

    from ftp.config import default_device_layout
    return default_device_layout(
        torch.cuda.device_count(), args.tensor_parallel_size,
        aux_device=args.aux_device, guard_device=args.guard_device,
    )


async def _run(args, prompt: str | None) -> None:
    use_dd = not args.no_dd
    aux_device, guard_device = _resolve_devices(args, use_dd)
    steer = (
        SteerArgs(parse_steer(args.steer), args.sae_repo, args.sae_cache, args.sae_dir)
        if not args.no_steer and args.steer
        else None
    )
    # --fast is the ergonomic bundle: enforce_eager + triton GDN. Individual flags
    # still work on their own; an explicit --gdn-prefill-backend overrides the bundle.
    enforce_eager = args.enforce_eager or args.fast
    gdn_backend = args.gdn_prefill_backend or ("triton" if args.fast else None)
    # --think raises both budgets together (a CoT overruns the 2048/4096 NOTHINK pair);
    # an explicit --max-new / --max-model-len still wins.
    max_new = args.max_new if args.max_new is not None else (
        THINK_MAX_NEW if args.think else DEFAULT_MAX_NEW)
    max_model_len = args.max_model_len if args.max_model_len is not None else (
        THINK_MAX_MODEL_LEN if args.think else DEFAULT_MAX_MODEL_LEN)
    # Live degeneration guard: ON by default with DD (it exists to repair the DD
    # push's rare decoding collapse). With --no-dd there is nothing it needs to
    # repair, and on a single-GPU box the judge would squeeze in next to P.
    guard_cfg = None
    if use_dd and not args.no_guard:
        guard_cfg = GuardConfig(
            model=args.guard_model, device=guard_device,
            interval=args.guard_interval, backtrack=args.guard_backtrack,
            threshold=args.guard_threshold, tries=args.guard_tries,
            # 10 (the benchmarked NOTHINK sweep winner) unless reasoning: CoT
            # trips the judge often enough that 10 rounds strands a think reply
            # mid-CoT — 40 gives it room to finish (see GuardConfig docs).
            max_rounds=args.guard_max_rounds or (40 if args.think else 10),
        )
    engine, _dd_cfg, pairs = build_async_llm(
        args.model,
        aux_p=args.aux_p if use_dd else None,
        aux_q=args.aux_q if use_dd else None,
        dd_kwargs={"aux_device": aux_device} if aux_device else None,
        steer=steer,
        guard=guard_cfg if guard_cfg is not None else False,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        gdn_prefill_backend=gdn_backend,
    )
    print(f"[run] P model: {args.model}", flush=True)
    if enforce_eager:
        print(f"[run] fast start: enforce_eager on (no CUDA-graph capture), "
              f"gdn_prefill_backend={gdn_backend or 'flashinfer'} — slower per token", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if use_dd:
        from ftp.config import split_aux_device
        p_dev, q_dev = split_aux_device(aux_device)
        aux_desc = (f"forget@{p_dev} retain@{q_dev} (split, unfused)" if p_dev != q_dev
                    else f"aux_device={p_dev or 'cuda:0'}")
        print(f"[run] DD on: forget={args.aux_p} retain={args.aux_q} a={args.alpha} "
              f"{aux_desc}", flush=True)
    if pairs:  # eager route: install after build (pre-capture route bakes it in already)
        await install_steering_async(engine, pairs)
    if steer is not None:
        desc = ", ".join(f"L{layer}:{f}@{v:g}" for layer, f, v in parse_steer(args.steer))
        print(f"[run] steering on: {desc}", flush=True)
    marker_id = resolve_marker_id(tok, guard_cfg.marker) if guard_cfg else None
    if guard_cfg:
        print(f"[run] 🛡 guard on: judge={guard_cfg.model} on "
              f"{guard_cfg.device or 'cuda:0'} every {guard_cfg.interval} engine "
              f"steps; a tripped reply rewinds {guard_cfg.backtrack} tokens and resamples. "
              f"Streaming holds back the newest {guard_cfg.backtrack} tokens (so a rewind "
              f"never has to un-print) — the visible stream trails generation by ~1s. "
              f"--no-guard to disable.", flush=True)
    if args.think:
        print(f"[run] 🧠 thinking ON: real <think> block, max_new={max_new} "
              f"max_model_len={max_model_len} — EXPLORATORY: alpha/clamp are calibrated "
              f"NOTHINK, so this is off the benchmarked config", flush=True)
    system_prompt = "" if args.no_system_prompt else args.system_prompt
    if system_prompt:
        tag = ("the benchmarks' forecasting point-in-time prompt"
               if system_prompt == FORECAST_SYSTEM_PROMPT else "custom")
        print(f"[run] system prompt on ({tag}) — --no-system-prompt to disable, "
              f"--system-prompt to replace", flush=True)

    sp = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        min_p=args.min_p, presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        max_tokens=max_new,
        # NOTHINK (default): the template already closes an empty <think></think> block, and
        # this sampler-level ban stops a stray re-open — matching evals QWEN_GEN. Under
        # --think the model must emit <think> ITSELF, so the ban has to go with it (banning
        # it while the template invites reasoning would just gag the model mid-format).
        bad_words=None if args.think else ["<think>"],
        extra_args={"dd_alpha": args.alpha} if use_dd else None,
    )

    # The rid actually in flight (the guarded path issues one request per rewind
    # round under derived ids) — ENTER-abort targets this, not the turn's base rid.
    live_rid: dict[str, str] = {}

    async def reply(messages, request_id: str) -> str:
        # enable_thinking drives the Qwen3.5 template: False appends an empty
        # <think></think> to the generation prompt (reasoning pre-closed), True leaves it
        # open for the model. DD's fuse_pin keeps <think>/</think> at P's probability, so
        # the delimiters survive DD either way.
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, *messages]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.think
        )
        print("asst> ", end="", flush=True)
        live_rid["rid"] = request_id
        if guard_cfg is None:
            full = ""
            async for delta in stream(engine, text, sp, request_id=request_id):
                print(delta, end="", flush=True)
                full += delta
            print()
            return full
        return await guarded_reply(text, request_id)

    async def guarded_reply(text: str, request_id: str) -> str:
        """Stream in APPROVED blocks: only text older than the walk-back window is
        printed, so a rewind never has to un-print anything. The newest
        ``backtrack`` tokens (= the rewindable region, spanning the current and
        previous judge blocks) stay held back until they survive; the stream
        arrives in blocks trailing generation by ~backtrack tokens.

        Same escalation policy as ftp.guard.rollback_generate: ``tries``
        no-progress rewinds at one point deepen the walk-back by another
        ``backtrack``; breaking all the way back at the first tokens prints the
        visible failure string instead of garbage."""
        from ftp.guard import FAIL_TEXT

        prompt_ids = tok(text, add_special_tokens=False)["input_ids"]
        backtrack = guard_cfg.backtrack
        accepted: list[int] = []
        printed = ""
        rewinds = consec = rounds = 0
        failed = False
        while max_new - len(accepted) > 0 and rounds < guard_cfg.max_rounds:
            sp_r = copy.deepcopy(sp)
            sp_r.max_tokens = max_new - len(accepted)
            sp_r.stop_token_ids = sorted(set(sp_r.stop_token_ids or []) | {marker_id})
            rrid = f"{request_id}-r{rounds}"
            live_rid["rid"] = rrid
            rounds += 1
            final = None
            async for out in engine.generate(
                {"prompt_token_ids": prompt_ids + accepted}, sp_r, request_id=rrid
            ):
                final = out
                merged = accepted + list(out.outputs[0].token_ids)
                safe = tok.decode(merged[:max(0, len(merged) - backtrack)],
                                  skip_special_tokens=True)
                if len(safe) > len(printed):
                    print(safe[len(printed):], end="", flush=True)
                    printed = safe
            if final is None:  # aborted before any output
                break
            comp = final.outputs[0]
            ids = list(comp.token_ids)
            tripped = getattr(comp, "stop_reason", None) == marker_id or (
                ids and ids[-1] == marker_id)
            if ids and ids[-1] == marker_id:
                ids = ids[:-1]
            if not tripped:
                accepted += ids
                break
            rewinds += 1
            kept = ids[:max(0, len(ids) - backtrack)]
            if kept:
                accepted += kept
                consec = 1
            else:
                consec += 1
                if consec >= guard_cfg.tries:
                    if not accepted:  # breaking at the very first tokens: fail visibly
                        failed = True
                        break
                    accepted = accepted[:max(0, len(accepted) - backtrack)]
                    consec = 0
                    # A deepened walk-back can retract text that was already printed
                    # (it was outside the original hold-back window). A terminal
                    # can't un-print, so say so and re-stream the passage.
                    safe_now = tok.decode(accepted, skip_special_tokens=True)
                    if len(printed) > len(safe_now):
                        print("\n[guard] ⟲ retracting the passage above and rewriting:",
                              flush=True)
                        printed = safe_now
        if failed and not accepted:
            print(FAIL_TEXT, flush=True)
            return FAIL_TEXT
        full = tok.decode(accepted, skip_special_tokens=True)
        print(full[len(printed):], flush=True)  # flush the held-back tail
        if rewinds:
            print(f"[guard] ⟲ {rewinds} rewind(s); walk-back {backtrack} tokens, "
                  f"deepened after {guard_cfg.tries} stuck tries", flush=True)
        return full

    try:
        if prompt is not None:
            await reply([{"role": "user", "content": prompt}], uuid.uuid4().hex)
            return
        print("[run] ready — press ENTER during a reply to stop it; "
              "🛑 quit with 'exit' / 'quit' / 'q' or Ctrl-D 🛑\n", flush=True)
        loop = asyncio.get_running_loop()

        def _exit_on_sigint() -> None:
            # Ctrl-C ends the session (ENTER stops just a reply). vLLM's engine-core
            # subprocess catches SIGINT and shuts down anyway, so hard-exit now — a
            # normal shutdown would hang joining the stdin reader blocked in the
            # executor thread.
            print("\n[run] 🛑 exiting", flush=True)
            os._exit(0)

        # no signal support (non-main thread / unsupported platform)
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(signal.SIGINT, _exit_on_sigint)

        # One blocking stdin reader, reused across turns. During a reply it doubles
        # as the STOP key (any line — just ENTER — aborts the reply, which is the
        # engine-safe way to stop a request: no SIGINT reaches vLLM's engine-core
        # subprocess, unlike Ctrl-C, so the engine stays alive for the next turn).
        def read_line():
            return loop.run_in_executor(None, sys.stdin.readline)

        pending = None  # a reader started during a reply that wasn't used to stop it
        while True:
            # NO HISTORY: every message is its OWN fresh conversation. Multi-turn works
            # with DD + steering — this simple runner just doesn't keep history, because
            # the benchmarks (and the alpha/clamp tuned against them) are single-turn.
            # Easy to change here: accumulate messages instead of rebuilding the list.
            # Reprint the state in red each turn so it's never a surprise.
            print("🔴🔴 FRESH conversation — previous messages are NOT in context "
                  "(this runner keeps no history; multi-turn works, it's just off "
                  "by default here) 🔴🔴", flush=True)
            print("you> ", end="", flush=True)
            line = await (pending or read_line())
            pending = None
            if line == "":  # EOF (Ctrl-D)
                print("\n[run] 🛑 exiting", flush=True)
                break
            user = line.strip()
            if user in ("exit", "quit", "q"):
                print("[run] 🛑 exiting", flush=True)
                break
            if not user:
                continue
            rid = uuid.uuid4().hex
            reply_task = asyncio.ensure_future(
                reply([{"role": "user", "content": user}], rid))  # fresh: no prior turns
            stop_key = read_line()  # a line here (ENTER) stops the reply
            done, _ = await asyncio.wait(
                {reply_task, stop_key}, return_when=asyncio.FIRST_COMPLETED)
            if stop_key in done:
                # ENTER pressed mid-reply: abort the request (engine stays up), drain.
                if stop_key.result() == "":  # it was actually EOF (Ctrl-D) mid-reply
                    with contextlib.suppress(Exception):
                        await engine.abort(live_rid.get("rid", rid))
                    print("\n[run] 🛑 exiting", flush=True)
                    break
                with contextlib.suppress(Exception):
                    await engine.abort(live_rid.get("rid", rid))
                with contextlib.suppress(Exception):
                    await reply_task
                print("[run] ⏹  reply stopped (press ENTER to stop a reply; "
                      "'exit' or Ctrl-D to quit)", flush=True)
            else:
                # Reply finished on its own; the stop-key reader is still blocked —
                # carry it over as the next turn's message reader (don't orphan it,
                # or it would swallow the next message).
                pending = stop_key
                reply_task.result()  # surface a real failure (no-op on success)
    finally:
        engine.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    chat = sub.add_parser("chat", help="interactive streaming chat")
    add_common_args(chat)
    gen = sub.add_parser("generate", help="single prompt, stream, exit")
    gen.add_argument("prompt", help="the user message")
    add_common_args(gen)
    args = ap.parse_args()
    try:
        asyncio.run(_run(args, getattr(args, "prompt", None)))
    except KeyboardInterrupt:
        # Ctrl-C: vLLM's engine-core subprocess catches SIGINT and shuts down, so
        # Ctrl-C ends the whole session (use ENTER to stop just a reply). Exit clean
        # — hard-exit past the stdin reader thread blocked in the executor.
        print("\n[run] 🛑 exiting", flush=True)
        os._exit(0)


if __name__ == "__main__":
    main()
