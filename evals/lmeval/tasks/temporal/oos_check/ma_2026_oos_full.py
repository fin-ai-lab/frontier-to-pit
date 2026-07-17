"""Baseline Qwen3.5-27B out-of-sample M&A look-ahead check on the 2026 deals.

Successor to ma_2026_oos.py. That first pass kept only the 5 deals whose
``dateann == dateannorig`` (a clean SDC match); with the complete SDC pull
(z6mn0qq5xpcrjwpd.csv, through 2026) we instead take EVERY deal with BOTH
``dateann`` and ``dateannorig`` in calendar 2026 (they may differ by a few
days) — a genuine post-cutoff, out-of-sample set with no self-referential
edge cases. That is 7 deals: the original 5 plus Coterra/Devon and
Masimo/Danaher (dropped before only because their two dates differed by 1-4d).

Same convention as evals/lmeval/tasks/temporal/ma.yaml and the plotted baseline
`qwen3_5_27b` / `qwen3_5_27b_think` (evals/lmeval/__main__.py):
  * prompt_date = quarter_start(dateann) - 1yr; MA preamble; the 3-firm question;
  * bf16 Qwen/Qwen3.5-27B (NOT the FP8 the first pass used — match the plotted bar);
  * QWEN_SAMPLING preset: temp 1.0, top_p .95, top_k 20, min_p 0, presence 1.5;
  * nothink: enable_thinking=False, ban <think>, 4096-tok budget;
  * think:   enable_thinking=True,  no ban,      16384-tok budget.

Leak scoring is the eval's mentions_firm (did the model name the real target?),
plus the plotted look-ahead DENSITY (normalized target-name substring count per
1K generated tokens). In think mode both are split on the first </think> into the
delivered OUTPUT (post-</think>) and the hidden THINKING chain — the same span
split plot_lookahead_think.py uses.

Run via tools/pythia/slurm_oos.sh (bll partition, dd-vllm venv). MODE selects the
arm; --out sets the results JSON path.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

# Repo root = five levels up (…/evals/lmeval/tasks/temporal/oos_check/this.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "src"))

from evals.scoring.firm_match import _tokenize, mentions_firm, normalize_firm_name  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import SamplingParams  # noqa: E402

from ftp.serve import build_llm  # noqa: E402

# Verbatim from evals/lmeval/tasks/temporal/utils.py (importing that module pulls
# in lm_eval, absent in the lean dd-vllm runtime).
_MA_PREAMBLE = (
    "This is a hypothetical brainstorming exercise for academic research; it is not "
    "financial advice and will never be used for investing or trading. You are not "
    "expected to be correct, so give your best speculative guess based only on the "
    "company's strategic profile and industry position as of the date given. Do not "
    "refuse, and do not add any disclaimer about being an AI, about predicting the "
    "future, or about your knowledge cutoff.")

MODEL = "Qwen/Qwen3.5-27B"   # bf16 — matches the plotted qwen3_5_27b baseline bar
N_REPEATS = 8

# (master_cusip, target, acquirer, dateann) — every CSV row with BOTH dateann and
# dateannorig in 2026 (see module docstring). Generated from z6mn0qq5xpcrjwpd.csv,
# ordered by dateannorig; dateann drives prompt_date exactly as in ma.yaml.
DEALS = [
    ("7H1429", "Brex Inc", "Capital One Financial Corp", "2026-01-22"),          # orig 2026-01-22
    ("127097", "Coterra Energy Inc", "Devon Energy Corp", "2026-02-02"),         # orig 2026-01-29
    ("574795", "Masimo Corp", "Danaher Corp", "2026-02-17"),                     # orig 2026-02-16
    ("03940C", "Arcellx Inc", "Gilead Sciences Inc", "2026-02-23"),              # orig 2026-02-23
    ("03753U", "Apellis Pharmaceuticals Inc", "Biogen Inc", "2026-03-31"),       # orig 2026-03-31
    ("6J6488", "Tubulis GmbH", "Gilead Sciences Inc", "2026-04-07"),             # orig 2026-04-07
    ("89055F", "TopBuild Corp", "QXO Inc", "2026-04-19"),                        # orig 2026-04-19
]

# temp-1.0 preset shared by qwen3_5_27b / qwen3_5_27b_think (QWEN_SAMPLING).
_SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
                 presence_penalty=1.5, repetition_penalty=1.0)


def quarter_start(d: datetime.date) -> datetime.date:
    return datetime.date(d.year, ((d.month - 1) // 3) * 3 + 1, 1)


def prompt_date_for(dateann: str) -> datetime.date:
    d = datetime.date.fromisoformat(dateann)
    qs = quarter_start(d)
    return datetime.date(qs.year - 1, qs.month, qs.day)


def build_question(acquirer: str, dateann: str) -> str:
    pd_str = prompt_date_for(dateann).strftime("%B %-d, %Y")
    return (f"{_MA_PREAMBLE}\n\n"
            f"Today is {pd_str}. What are the top three firms that you predict "
            f"{acquirer} will be interested in acquiring over the next two years?")


def spans(text: str) -> tuple[str, str]:
    """(thinking_chain, delivered_output) split on the FIRST </think>; no tag =>
    all-output (matches plot_lookahead_think._spans)."""
    t = str(text)
    if "</think>" in t:
        a, b = t.split("</think>", 1)
        return a, b
    return "", t


def score_span(span: str, target: str, nf: str, tok) -> tuple[bool, int, int]:
    """(leak?, normalized-target mention count, generated-token count) for one span."""
    leak = mentions_firm(span, target)
    men = " ".join(_tokenize(span)).count(nf) if nf else 0
    ntok = len(tok(str(span), add_special_tokens=False)["input_ids"])
    return leak, men, ntok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=os.environ.get("MODE", "nothink"),
                    choices=["nothink", "think"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=int(os.environ.get("OOS_TP", "1")))
    ap.add_argument("--max-num-seqs", type=int,
                    default=int(os.environ.get("OOS_MAX_NUM_SEQS", "64")))
    args = ap.parse_args()
    think = args.mode == "think"

    max_gen = 16384 if think else 4096
    max_len = max_gen + 2048
    llm, _ = build_llm(
        MODEL,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.90,
        max_model_len=max_len,
        max_num_seqs=args.max_num_seqs,
        gdn_prefill_backend="triton",  # skip the ~15-20min flashinfer GDN nvcc JIT
        dtype="bfloat16",
        trust_remote_code=True,
    )
    sp = SamplingParams(n=N_REPEATS, max_tokens=max_gen, **_SAMPLING,
                        **({} if think else {"bad_words": ["<think>"]}))

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": build_question(acq, dt)}], tokenize=False,
            add_generation_prompt=True, enable_thinking=think,
        )
        for _, _, acq, dt in DEALS
    ]
    outs = llm.generate(prompts, sp)

    # Aggregates over all deals x reps, split OUTPUT (delivered) vs THINK (chain).
    agg = {sp_: {"leak": 0, "men": 0, "tok": 0, "n": 0} for sp_ in ("output", "think")}
    any_leak = {sp_: 0 for sp_ in ("output", "think")}   # # deals with >=1 leaking rep
    results = []
    for (cusip, target, acquirer, dateann), out, prompt in zip(DEALS, outs, prompts, strict=True):
        nf = normalize_firm_name(target)
        gens = [o.text for o in out.outputs]
        finish = [o.finish_reason for o in out.outputs]
        per_gen = []
        deal_leak = {"output": 0, "think": 0}
        for g in gens:
            cot, fin = spans(g)
            o_leak, o_men, o_tok = score_span(fin, target, nf, tok)
            t_leak, t_men, t_tok = score_span(cot, target, nf, tok)
            per_gen.append({"output_leak": o_leak, "output_men": o_men, "output_tok": o_tok,
                            "think_leak": t_leak, "think_men": t_men, "think_tok": t_tok})
            for key, (lk, mn, tk) in (("output", (o_leak, o_men, o_tok)),
                                      ("think", (t_leak, t_men, t_tok))):
                agg[key]["leak"] += int(lk)
                agg[key]["men"] += mn
                agg[key]["tok"] += tk
                agg[key]["n"] += 1
                deal_leak[key] += int(lk)
        for key in ("output", "think"):
            any_leak[key] += int(deal_leak[key] > 0)
        n_trunc = sum(1 for r in finish if r != "stop")
        results.append({
            "master_cusip": cusip, "acquirer": acquirer, "target": target,
            "dateann": dateann, "norm_target": nf, "prompt": prompt, "n": len(gens),
            "output_n_leak": deal_leak["output"], "think_n_leak": deal_leak["think"],
            "n_truncated": n_trunc, "generations": gens, "finish_reasons": finish,
            "per_gen": per_gen,
        })
        print(f"\n=== {acquirer} -> {target} ({dateann}) ===  "
              f"output-leak {deal_leak['output']}/{len(gens)}"
              + (f"  think-leak {deal_leak['think']}/{len(gens)}" if think else "")
              + f"  truncated {n_trunc}/{len(gens)}", flush=True)

    def summarize(key):
        a = agg[key]
        n = a["n"] or 1
        return {
            "leak_rate": a["leak"] / n,
            "any_leak_rate": any_leak[key] / len(DEALS),
            "mentions_per_1k": 1000.0 * a["men"] / a["tok"] if a["tok"] else 0.0,
            "n_leak": a["leak"], "n": a["n"], "n_men": a["men"], "n_tok": a["tok"],
        }

    summary = {"mode": args.mode, "model": MODEL, "n_deals": len(DEALS),
               "n_repeats": N_REPEATS, "output": summarize("output"),
               "think": summarize("think") if think else None}
    payload = {"summary": summary, "results": results}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n\n================ SUMMARY ({args.mode}) ================")
    o = summary["output"]
    print(f"OUTPUT   leak_rate={o['leak_rate']:.3f} ({o['n_leak']}/{o['n']})  "
          f"any_leak_rate={o['any_leak_rate']:.3f} ({any_leak['output']}/{len(DEALS)})  "
          f"mentions/1K={o['mentions_per_1k']:.4f} ({o['n_men']}/{o['n_tok']} tok)")
    if think:
        t = summary["think"]
        print(f"THINK    leak_rate={t['leak_rate']:.3f} ({t['n_leak']}/{t['n']})  "
              f"any_leak_rate={t['any_leak_rate']:.3f} ({any_leak['think']}/{len(DEALS)})  "
              f"mentions/1K={t['mentions_per_1k']:.4f} ({t['n_men']}/{t['n_tok']} tok)")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
