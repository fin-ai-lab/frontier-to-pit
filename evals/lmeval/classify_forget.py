#!/usr/bin/env python
"""INDEPENDENT forget-set cleaner (multi-model).

For every candidate forget question, a strong reasoning model decides whether a
2015-cutoff model could still probably answer it — by guessing/reasoning, or because the
answer is a stable pre-2016 fact. INDEPENDENT of the DD aux models, so the eval's
forget-set selection isn't circular. Run it with >1 model and correlate the labels.

  YES = answerable by a 2015 model  -> DISCARD (too easy / not post-2015)
  NO  = genuinely needs post-2015 knowledge -> KEEP

Supported: gemma-* (enable_thinking) and openai/gpt-oss-* (harmony: reasoning_effort +
`final` channel). The model reasons, then emits "FINAL: YES/NO"; we parse that.
Sharded via DD_NUM_CHUNKS/DD_CHUNK_ID (strided); writes a labels parquet per shard.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import time

import pandas as pd

MODEL = os.environ.get("CLASSIFY_MODEL", "google/gemma-4-26B-A4B-it")
GPTOSS = "gpt-oss" in MODEL.lower()

# gemma path: avoid flashinfer (its offline JIT hangs) -> native sampler + flash-attn +
# eager. gpt-oss path: leave vLLM's defaults (its gpt-oss kernels work on these nodes,
# per the teacher harness) and just use spawn for tensor-parallel workers.
if not GPTOSS:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("FLASHINFER_NO_DOWNLOAD", "1")
else:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

PROMPT = """A frontier language model was trained ONLY on data up to the end of 2015 \
(31 December 2015) — equivalently, it has been unlearned back to that cutoff, so it knows \
nothing about any event after 2015.

You are given a question, the date it was asked, and ALL of its accepted reference \
answers. Decide whether this question is EASY ENOUGH that such a 2015-cutoff model could \
still probably answer it correctly, either:
  (a) by guessing or reasoning from general knowledge it already had by the end of 2015, or
  (b) because the correct answer is a stable fact that was already true and known at the \
end of 2015 (it would have held if the same question were asked before the cutoff).

Answer YES if a smart 2015-cutoff model could probably still get it right.
Answer NO only if the correct answer genuinely depends on knowledge of an event, result, \
appointment, election, release, or record from AFTER 31 December 2015.

Be strict and aggressive: if there is a reasonable chance the model could answer it \
without any post-2015 knowledge, answer YES (we would rather discard a borderline \
question than keep one that isn't truly post-2015).

Date asked: {date}
Question: {question}
Accepted answer(s): {answers}

Reason briefly, then end your reply with exactly "FINAL: YES" or "FINAL: NO"."""

# gpt-oss decodes in the harmony format: an `analysis` (CoT) channel then the user-facing
# `final` channel. vLLM usually strips channels already; this is a safety net.
_FINAL = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|<\|channel\|>|\Z)", re.S)


def harmony_final(text):
    m = _FINAL.search(text)
    if m:
        return m.group(1).strip()
    if "assistantfinal" in text:
        return text.split("assistantfinal")[-1].strip()
    return text


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def date_str(ts):
    try:
        return dt.datetime.fromtimestamp(int(ts), dt.UTC).strftime("%B %-d, %Y")
    except Exception:
        return "(unknown date)"


def answers_str(a):
    xs = [str(x).strip() for x in (list(a) if a is not None else []) if str(x).strip()]
    return "; ".join(dict.fromkeys(xs)) or "(none)"


def parse_label(text):
    m = re.search(r"FINAL:\s*\**\s*(YES|NO)", text, re.I)
    if m:
        return m.group(1).upper()
    toks = re.findall(r"\b(YES|NO)\b", text, re.I)   # fallback: last standalone yes/no
    return toks[-1].upper() if toks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="candidate forget pool parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tensor-parallel-size", type=int,
                    default=int(os.environ.get("CLASSIFY_TP", "1")))
    ap.add_argument("--reasoning-effort", default=os.environ.get("CLASSIFY_EFFORT", "low"),
                    help="gpt-oss reasoning effort (low/medium/high)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--limit", type=int, default=int(os.environ.get("CLASSIFY_LIMIT", "0")))
    args = ap.parse_args()
    gptoss = "gpt-oss" in args.model.lower()

    df = pd.read_parquet(args.parquet)
    nc = int(os.environ.get("DD_NUM_CHUNKS", "1"))
    ci = int(os.environ.get("DD_CHUNK_ID", "0"))
    if nc > 1:
        df = df.iloc[ci::nc].copy()      # strided shard; merged downstream
    if args.limit:
        df = df.head(args.limit).copy()
    df = df.reset_index(drop=True)
    log(f"classify {len(df)} questions (chunk {ci}/{nc}) model={args.model} "
        f"tp={args.tensor_parallel_size}")

    msgs = [[{"role": "user", "content": PROMPT.format(
                date=date_str(r["question_ts"]), question=str(r["question"]).strip(),
                answers=answers_str(r["answers"]))}]
            for _, r in df.iterrows()]

    from vllm import LLM, SamplingParams
    llm_kw = dict(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  max_num_seqs=args.max_num_seqs,
                  tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True,
                  limit_mm_per_prompt={"image": 0, "video": 0})
    if gptoss:
        llm_kw["enable_prefix_caching"] = False
        # enforce_eager: skip CUDA-graph capture. Capturing ~118 graphs for the 120B MoE
        # across 4 TP workers OOM'd a worker at init; eager is plenty for a batch classify.
        llm_kw["enforce_eager"] = True
        sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens)
        ct_kwargs = {"reasoning_effort": args.reasoning_effort}
    else:
        llm_kw["enforce_eager"] = True
        sp = SamplingParams(temperature=args.temperature, top_p=0.95, seed=0,
                            max_tokens=args.max_tokens)
        ct_kwargs = {"enable_thinking": True}
    llm = LLM(**llm_kw)

    t0 = time.time()
    try:
        outs = llm.chat(msgs, sp, chat_template_kwargs=ct_kwargs)
    except (TypeError, ValueError):
        outs = llm.chat(msgs, sp)
    log(f"generated {len(outs)} in {time.time()-t0:.1f}s ({len(outs)/max(time.time()-t0,1):.1f}/s)")

    raw = [o.outputs[0].text for o in outs]
    texts = [harmony_final(t) for t in raw] if gptoss else raw
    labels = [parse_label(t) for t in texts]
    df["answerable_2015"] = [(lab == "YES") if lab else None for lab in labels]  # YES => discard
    df["classify_label"] = labels
    df["classify_model"] = args.model
    df["classify_text"] = [t[-600:] for t in texts]                        # tail, for audit
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out)
    ny = labels.count("YES")
    nn = labels.count("NO")
    nb = labels.count(None)
    log(f"YES/discard={ny} NO/keep={nn} unparsed={nb}  "
        f"({100 * nn / max(len(labels), 1):.1f}% kept) -> {args.out}")


if __name__ == "__main__":
    main()
