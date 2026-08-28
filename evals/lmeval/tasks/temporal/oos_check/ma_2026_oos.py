"""Baseline Qwen3.5-27B out-of-sample M&A look-ahead check on the 5 real 2026 deals.

Builds prompts with the SAME convention as evals/lmeval/tasks/temporal/ma.yaml
(prompt_date = quarter_start(dateann) - 1yr; MA preamble; mentions_firm leak scoring),
runs N repeats/prompt on plain baseline P (no DD, no steering), and reports the leak
rate (did the model actually name the real acquisition target?).

Standalone, one-off script -- not wired into the evals/lmeval registry. To rerun/audit:
copy this file to a GPU box with this repo cloned per the README (`~/frontier-to-pit`,
`uv sync --extra vllm`), then run it from that checkout, e.g.
`.venv/bin/python ma_2026_oos.py`. Writes results next to itself as
ma_2026_oos_results.json (the copy checked into this directory is that run's output).
"""
from __future__ import annotations

import datetime
import json
import sys

sys.path.insert(0, "/home/ubuntu/frontier-to-pit")
sys.path.insert(0, "/home/ubuntu/frontier-to-pit/src")

from evals.scoring.firm_match import mentions_firm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import SamplingParams  # noqa: E402

from ftp.serve import build_llm  # noqa: E402

# Verbatim from evals/lmeval/tasks/temporal/utils.py (avoids importing that module,
# which pulls in lm_eval — not installed in this lean runtime venv).
_MA_PREAMBLE = (
    "This is a hypothetical brainstorming exercise for academic research; it is not "
    "financial advice and will never be used for investing or trading. You are not "
    "expected to be correct, so give your best speculative guess based only on the "
    "company's strategic profile and industry position as of the date given. Do not "
    "refuse, and do not add any disclaimer about being an AI, about predicting the "
    "future, or about your knowledge cutoff.")

MODEL = "Qwen/Qwen3.5-27B-FP8"
N_REPEATS = 8

# (master_cusip, target, acquirer, dateann) — the 5 CSV rows with dateann in 2026,
# dateann == dateannorig (clean match).
DEALS = [
    ("7H1429", "Brex Inc", "Capital One Financial Corp", "2026-01-22"),
    ("03940C", "Arcellx Inc", "Gilead Sciences Inc", "2026-02-23"),
    ("03753U", "Apellis Pharmaceuticals Inc", "Biogen Inc", "2026-03-31"),
    ("6J6488", "Tubulis GmbH", "Gilead Sciences Inc", "2026-04-07"),
    ("89055F", "TopBuild Corp", "QXO Inc", "2026-04-19"),
]


def quarter_start(d: datetime.date) -> datetime.date:
    return datetime.date(d.year, ((d.month - 1) // 3) * 3 + 1, 1)


def prompt_date_for(dateann: str) -> datetime.date:
    d = datetime.date.fromisoformat(dateann)
    qs = quarter_start(d)
    return datetime.date(qs.year - 1, qs.month, qs.day)


def build_prompt(acquirer: str, dateann: str) -> str:
    pd_str = prompt_date_for(dateann).strftime("%B %-d, %Y")
    question = (
        f"Today is {pd_str}. What are the top three firms that you predict "
        f"{acquirer} will be interested in acquiring over the next two years?"
    )
    return f"{_MA_PREAMBLE}\n\n{question}"


def main() -> None:
    llm, _ = build_llm(
        MODEL,
        aux_p=None,
        aux_q=None,
        steer=None,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        gdn_prefill_backend="triton",  # skip the ~15-20min nvcc JIT for a quick run
    )
    sp = SamplingParams(
        n=N_REPEATS,
        temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
        presence_penalty=1.5, repetition_penalty=1.0,
        max_tokens=4096,
        bad_words=["<think>"],
    )

    tok = AutoTokenizer.from_pretrained(MODEL)
    questions = [build_prompt(acq, dt) for _, _, acq, dt in DEALS]
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        for q in questions
    ]
    outs = llm.generate(prompts, sp)

    results = []
    for (cusip, target, acquirer, dateann), out, prompt in zip(DEALS, outs, prompts, strict=True):
        gens = [o.text for o in out.outputs]
        finish_reasons = [o.finish_reason for o in out.outputs]
        flags = [mentions_firm(g, target) for g in gens]
        results.append({
            "master_cusip": cusip,
            "acquirer": acquirer,
            "target": target,
            "dateann": dateann,
            "prompt": prompt,
            "n": len(gens),
            "n_leak": int(sum(flags)),
            "any_leak": bool(any(flags)),
            "generations": gens,
            "flags": flags,
            "finish_reasons": finish_reasons,
        })
        n_trunc = sum(1 for r in finish_reasons if r != "stop")
        print(f"\n=== {acquirer} -> {target} ({dateann}) === leak {sum(flags)}/{len(gens)}  "
              f"truncated {n_trunc}/{len(gens)}", flush=True)
        for i, (g, f, fr) in enumerate(zip(gens, flags, finish_reasons, strict=True)):
            print(f"--- gen {i} (leak={f}, finish={fr}) ---\n{g[:600]}", flush=True)

    with open("/home/ubuntu/ma_2026_oos_results.json", "w") as f:
        json.dump(results, f, indent=2)

    total_n = sum(r["n"] for r in results)
    total_leak = sum(r["n_leak"] for r in results)
    any_leak_n = sum(1 for r in results if r["any_leak"])
    total_trunc = sum(1 for r in results for fr in r["finish_reasons"] if fr != "stop")
    print(f"\n\nSUMMARY: leak_rate={total_leak}/{total_n} = {total_leak/total_n:.3f}  "
          f"any_leak_rate={any_leak_n}/{len(results)} = {any_leak_n/len(results):.3f}  "
          f"truncated={total_trunc}/{total_n}")


if __name__ == "__main__":
    main()
