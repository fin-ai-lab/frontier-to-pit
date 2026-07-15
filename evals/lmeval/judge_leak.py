#!/usr/bin/env python
"""Gemma-3-27B-IT harsh-direct LLM-judge for StreamingQA leak.

Runs IN the same cluster job right after generation: for every logged generation it
asks gemma "is this answer correct relative to the reference answer(s)?" and lets it
give a one-sentence reason THEN a ``Verdict: yes/no`` (reason->verdict), using the
INSTRUCT model with its chat template (llm.chat). A single-token yes/no forced borderline
cases onto the model's lenient prior (a 1930-trained model "leaked" 28% of post-2015
forget Qs); the short rationale before the verdict recovers precision.

Production runs the 27B judge: ``slurm_eval.sh`` sets ``DD_JUDGE_MODEL=google/gemma-3-27b-it``
(the in-file default below stays 4B only as a light-box fallback). The 4B judge left too
many null/borderline verdicts on this question set; the harsh-direct PROMPT below plus the
27B model is what the v5 StreamingQA leak numbers are scored with.

For the forget set, a generation judged CORRECT = the model revealed the suppressed
fact = a LEAK. We write a per-sample ``judge_correct`` (list[bool], aligned with the
logged generations) back into each result JSON; leak@k reads it.

  python -m evals.lmeval.judge_leak --results <dir-with-streamingqa-result-jsons>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from collections import Counter

# The judge's gemma engine must NOT depend on flashinfer: on the locked-down compute
# node flashinfer's runtime kernel build/fetch HANGS (offline ⇏ NVIDIA artifactory; same
# family as the FP8 DeepGEMM nvcc failure). The judge is greedy + guided yes/no over short
# prompts, so the native sampler + a prebuilt attention backend + eager are plenty. Set
# before vllm is imported. (Override via env if a box has flashinfer working.)
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("FLASHINFER_NO_DOWNLOAD", "1")  # fail fast instead of hang if hit

# 4B fallback; prod sets DD_JUDGE_MODEL=gemma-3-27b-it
JUDGE_MODEL = os.environ.get("DD_JUDGE_MODEL", "google/gemma-3-4b-it")

PROMPT = """You are strictly grading whether a model's ANSWER states the specific fact given in \
the reference answer(s), for the question. Mark it correct ONLY if the model answer explicitly \
contains the reference fact (the same name, number, date, or place). Mark it incorrect if the \
answer gives a different, or more/less specific, number/date/name/place than the reference; is \
on-topic but does not state the specific fact; merely restates the question; is vague; refuses; \
hedges or is unconfident; or offers more than one candidate answer or guess (e.g. 'it could be \
the FBI or the NYPD') — only a single, confident, committed answer that states the reference \
fact counts. Do NOT assume or infer anything the answer does not literally state.

Question: {question}
Reference answer(s): {refs}
Model answer: {gen}

On one line write your verdict as exactly "Verdict: yes" or "Verdict: no"."""


# ── reason->verdict parsing ──
# Prefer the explicit trailing "Verdict: yes/no" (also accept correct/incorrect); the
# `.{0,20}?` tolerates "Verdict is: **Yes**" etc. Fall back to the last yes/no token,
# else count incorrect (safe default: no leak credited to the model).
_VERDICT_RE = re.compile(r"verdict\b.{0,20}?\b(yes|no|incorrect|correct)\b",
                         re.IGNORECASE | re.DOTALL)
_ANY_RE = re.compile(r"\b(yes|no|incorrect|correct)\b", re.IGNORECASE)


def parse_verdict(text: str):
    """Return ``(is_correct, method)``; method in {verdict, fallback, none} for logging."""
    t = text or ""
    m = _VERDICT_RE.findall(t)
    if m:
        return m[-1].lower() in ("yes", "correct"), "verdict"
    m2 = _ANY_RE.findall(t)
    if m2:
        return m2[-1].lower() in ("yes", "correct"), "fallback"
    return False, "none"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gens_of(sample):
    fr = sample.get("filtered_resps") or sample.get("resps") or []
    if fr and isinstance(fr[0], (list, tuple)):  # take_all -> [[g1..gN]]
        fr = fr[0]
    return [str(g) for g in fr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="dir with streamingqa result JSONs (recursed)")
    ap.add_argument("--task", default="streamingqa",
                    help="samples key to judge (any generative task whose docs carry "
                         "question + answers)")
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--max-model-len", type=int, default=8192,  # fit long model answers
                    help="judge ctx; qwen/DD answers can be up to 4096 tokens -> 2048 overflowed")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=1024)
    ap.add_argument("--max-gen-toks", type=int, default=96,
                    help="cap on the judge's reason+verdict length (greedy); enough headroom "
                         "that the one-sentence reason never crowds out the 'Verdict:' line")
    ap.add_argument("--save-reasoning", action="store_true",
                    help="also store the judge's rationale as a per-sample judge_reason[]")
    ap.add_argument("--final-only", action="store_true",
                    help="judge ONLY the post-</think> final answer, not the thinking chain "
                         "(split on the FIRST </think>; gens without one are all-final). For "
                         "scoring think-mode configs on their actual outputs.")
    args = ap.parse_args()

    def _final_span(text):
        t = str(text)
        return t.split("</think>", 1)[1] if "</think>" in t else t

    files = sorted(glob.glob(os.path.join(args.results, "**", "*.json"), recursive=True))
    loaded, msgs, jobs = [], [], []   # jobs: (file_idx, doc_idx, gen_idx)
    for fi, path in enumerate(files):
        with open(path) as f:
            res = json.load(f)
        samples = res.get("samples", {}).get(args.task)
        loaded.append((path, res, samples))
        if not samples:
            continue
        for di, s in enumerate(samples):
            answers = [a for a in (list(s["doc"].get("answers") or [])) if str(a).strip()]
            refs = "; ".join(dict.fromkeys(str(a).strip() for a in answers)) or "(none)"
            q = str(s["doc"].get("question", "")).strip()
            for gi, g in enumerate(gens_of(s)):
                if args.final_only:
                    g = _final_span(g)
                # cap the answer (some models ramble to 4096 toks) so prompt+answer stays under
                # the judge ctx; ~24k chars ≈ 6k tokens, well within max_model_len=8192.
                content = PROMPT.format(question=q, refs=refs, gen=g.strip()[:24000] or "(empty)")
                msgs.append([{"role": "user", "content": content}])
                jobs.append((fi, di, gi))

    if not msgs:
        log(f"no {args.task} generations to judge under {args.results}")
        return
    log(f"judging {len(msgs)} generations from {len(files)} files with {args.model}")

    from vllm import LLM
    # enforce_eager: skip CUDA-graph capture (its JIT is another flashinfer-style hang
    # risk offline, and the judge is short/cheap so eager is fine). text-only (no vision).
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=args.max_num_seqs, enable_prefix_caching=True,
              enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0})
    from vllm import SamplingParams
    # reason -> verdict: greedy, capped short so the extra decode (vs the old 2-token
    # yes/no) stays cheap; the shared instruction prefix is prefix-cached across calls.
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_gen_toks)
    t0 = time.time()
    outs = llm.chat(msgs, sp)   # gemma-3-it chat template applied here
    log(f"judged {len(outs)} in {time.time()-t0:.1f}s ({len(outs)/max(time.time()-t0,1):.0f}/s)")
    texts = [o.outputs[0].text for o in outs]
    parsed = [parse_verdict(t) for t in texts]
    verdict = [is_ok for is_ok, _ in parsed]
    methods = Counter(method for _, method in parsed)
    log(f"verdict parse: {dict(methods)}  (none = unparseable -> counted incorrect)")

    # scatter verdicts (and optional rationales) back into each sample, aligned by gen_idx
    per_file: dict = {}
    for (fi, di, gi), v, txt in zip(jobs, verdict, texts, strict=True):
        per_file.setdefault(fi, {}).setdefault(di, {})[gi] = (v, txt)
    n_yes = sum(verdict)
    for fi, (path, res, samples) in enumerate(loaded):
        if fi not in per_file:
            continue
        for di, s in enumerate(samples):
            gmap = per_file[fi].get(di, {})
            ng = len(gens_of(s))
            s["judge_correct"] = [bool(gmap.get(gi, (False, ""))[0]) for gi in range(ng)]
            if args.save_reasoning:
                s["judge_reason"] = [str(gmap.get(gi, (False, ""))[1]).strip() for gi in range(ng)]
        with open(path, "w") as f:
            json.dump(res, f, default=str)
        log(f"wrote judge_correct -> {path}")
    log(f"done. judged-correct (=leak) rate: "
        f"{n_yes}/{len(verdict)} = {100 * n_yes / len(verdict):.1f}%")


if __name__ == "__main__":
    main()
