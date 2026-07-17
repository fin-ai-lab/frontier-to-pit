#!/usr/bin/env python3
"""Strong/weak look-ahead-bias (LAB) judge for the M&A task, per generation.

Ported from forecasting-thoughts/scripts/label_ma_lab_local.py (the validated
Gemma-3-4B-IT judge; SYSTEM_PROMPT/USER_TEMPLATE kept verbatim), adapted to read
this repo's lm-eval results JSONs instead of that repo's freeform-sweep parquets.

The mention matcher (evals/scoring/firm_match.mentions_firm) flags a generation
whenever the post-cutoff deal target is NAMED, but that conflates two very
different behaviors plus an innocent one. The judge splits them:

  - strong_LAB: the target appears AS ONE OF THE THREE PREDICTED ACQUISITION
    TARGETS in the model's delivered forecast — the prediction itself names the
    real answer. THE metric (2026-08-03, user): strong LAB is what we report;
    weak LAB is kept in the output as a diagnostic only.
  - weak_LAB:   the model acknowledges (aside/hindsight/"in reality") that the
    acquisition actually occurred, outside the forecast list.
  - neither:    the target shows up only as context (competitor list, industry
    landscape, a rationale for a different pick).

Only mention-positive generations are judged (strong/weak is meaningless
otherwise); mention-negative generations are strong_LAB=False by construction,
so the summary's strong-LAB %gens is over ALL generations and is directly
comparable to the %gens leak rate in tools/report_lab_rates.py and the website
export.

The judge sees ONLY the delivered output, never the thinking chain (2026-08-03,
user). For think-mode samples (the logged prompt opens the assistant turn with
<think>) the output is what follows the first </think>; a generation whose chain
NEVER closes delivered no output at all — it is not mention-filtered, not
judged, and counts strong_LAB=False in the denominator. Its text belongs to the
chain, whose leak is measured separately as mentions/1K think tokens. Instruct
samples have no chain; the whole generation is the output. Use --span whole to
override and judge entire generations (chain included).

Usage (any single >=32GB GPU; ~70 mention-positive rows judge in minutes):
  python3 evals/scoring/label_ma_lab.py                        # all ma/*.json in the v6 store
  python3 evals/scoring/label_ma_lab.py --results /data/lab/frontier-to-pit/results/ma/qwen3_5_27b_think__base.json
  python3 evals/scoring/label_ma_lab.py --report               # re-print summaries from saved labels

Output: one JSON per input results file under --out-dir
(<results-basename>.labels.json) with per-generation labels + a summary block.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

# repo root = two levels up (this file lives in evals/scoring/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("HF_HOME", "/data/lab/hf-cache")

from evals.scoring.firm_match import mentions_firm, normalize_firm_name  # noqa: E402

FTP_RESULTS = "/data/lab/frontier-to-pit/results"
DEFAULT_OUT = "/data/lab/frontier-to-pit/ma_lab_labels"

# 12B, not 4B: the 4B judge marked backdated "already announced/completed" deal
# assertions strong even while its own notes said otherwise (26/27 strong on the
# 27B think arm vs 20 by manual read); 12B + the IMPORTANT rule below gets 26/27
# rows right. ~24GB bf16 — fits a single A40/L40S/H100. The google/ id is gated;
# the unsloth mirror is byte-identical and needs no token — override with --model.
DEFAULT_JUDGE = "unsloth/gemma-3-12b-it"

# Matcher-FP targets excluded from every LAB rate (same set as
# tools/report_lab_rates.py and tools/aggregate_score.py — keep in sync).
MA_DROP_TARGETS = {"Targa Resources Partners LP", "ONEOK Partners LP"}

# --- Judge prompt: from forecasting-thoughts (validated there), with ONE added
# rule ("IMPORTANT: ...") — on this repo's completions the frontier model often
# BACKDATES the real deal ("the pending Worldpay transaction", "completed in
# November 2015") without picking it, and the 4B judge marked those strong even
# while its own notes said "mentioned as having been acquired". The added rule
# pins strong_LAB to the target actually being one of the three picks; spot-
# checked 2026-08-03 against a manual read of all 27B/ours think-mode flags. ----
SYSTEM_PROMPT = """You label LLM completions for two types of look-ahead bias (LAB) on an M&A acquirer-prediction task.

Background: a model was asked at a past date T to predict the top three firms a given acquirer might acquire over the next two years. We later know one specific firm — target — was actually acquired by that acquirer. Each completion has been pre-filtered to mention the target. Your job: classify HOW the target is mentioned.

Two non-mutually-exclusive labels:

- strong_LAB (true/false): the target firm appears AS ONE OF THE THREE PREDICTED ACQUISITION TARGETS in the model's main forecast (positions 1, 2, or 3, or labeled "Most likely target", "Top target", etc., within the forecast itself). The prediction itself names the correct answer.

- weak_LAB (true/false): OUTSIDE the top-three forecast — typically at the very beginning or very end of the response, in an aside, footnote, "Important Note", "Disclaimer", "In reality", "What actually happened", "Looking back", "Update", "Post-script", "Editor's Note", or similar — the model acknowledges that the acquisition actually occurred. The model broke character to reference the real outcome.

Both can be true (model predicts AND acknowledges). Either alone. Or both false (target appears only as context, e.g. as a competitor or partner with no prediction or acknowledgment).

IMPORTANT: strong_LAB requires the target to actually BE one of the three forecast picks. To decide it, first identify the three firms the completion offers as its picks (its numbered or headed list); strong_LAB is true only if the target is one of those three firms. A completion that merely states the acquisition already happened, was announced, or is pending or integrating — without the target being one of its picks — is weak_LAB, not strong_LAB, even when that statement appears inside the forecast section or serves as the rationale for a different pick. A target named only as context (a competitor, a rival suitor, or part of the rationale for a DIFFERENT pick) is neither.

Match the target name generously — ignore Inc / Corp / PLC / Ltd / Holdings / LLC suffixes.

Respond with a single JSON object on one line, nothing else:
{"strong_LAB": <true|false>, "weak_LAB": <true|false>, "notes": "<≤12 word evidence phrase>"}"""

USER_TEMPLATE = """TARGET COMPANY TO LOOK FOR: "{target}"

Your only job is to determine how this specific target — "{target}" — is referenced in the model's completion below. Ignore other firms; you are looking for "{target}" (match generously, ignoring Inc / Corp / PLC / Ltd / Holdings / LLC suffixes).

Acquirer (for context only): {acquirer}
Prompt date in the model's view: {prompt_date}

Model's completion:
---
{completion}
---

Reminder — the target you must look for is: "{target}".

- strong_LAB = "{target}" appears as one of the top-three predicted acquisition targets in the model's forecast — an actual pick, not a statement that the deal already happened and not a context/competitor mention.
- weak_LAB = "{target}" is acknowledged as a real outcome without being a pick ("in reality", "actually", "already announced/completed/pending", "looking back", "what happened", etc.).

Output the JSON object only, no other text."""
# ------------------------------------------------------------------------------


def _gens(sample):
    """The N repeat-generations logged for one doc (take_all keeps them nested).
    Same access logic as the website's export_plot_data.py."""
    fr = sample.get("filtered_resps") or sample.get("resps") or []
    g0 = fr[0] if fr else None
    return [str(g) for g in g0] if isinstance(g0, (list, tuple)) else [str(g) for g in fr]


def _prompt_text(sample):
    a = sample.get("arguments") or []
    first = a[0] if a else None
    if isinstance(first, (list, tuple)) and first:
        return str(first[0])
    return str(first or "")


def _output_span(text, in_think):
    """The DELIVERED output only. Think-mode (prompt opened a <think> block):
    text after the first </think>; empty if the chain never closed — the model
    never stopped thinking, so nothing was delivered. Instruct: the whole text."""
    if not in_think:
        return text
    return text.split("</think>", 1)[1] if "</think>" in text else ""


def _elide(text, max_chars):
    """Tail-weighted middle elision: the forecast list normally sits at the END
    of a completion, and blind right-truncation (tokenizer truncation) would cut
    exactly the part strong_LAB is about. Head kept for early asides (weak LAB)."""
    if len(text) <= max_chars:
        return text
    head, tail = max_chars // 3, max_chars - max_chars // 3
    return text[:head] + "\n[... middle elided for length ...]\n" + text[-tail:]


def parse_label_text(text):
    txt = text.strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```\s*$", "", txt)
    m = re.search(r"\{.*?\}", txt, re.DOTALL)
    if not m:
        return {"strong_LAB": None, "weak_LAB": None, "notes": f"PARSE_FAIL: {txt[:80]}"}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return {"strong_LAB": None, "weak_LAB": None,
                "notes": f"JSON_FAIL: {str(e)[:30]}: {txt[:60]}"}
    strong, weak = obj.get("strong_LAB"), obj.get("weak_LAB")
    return {
        "strong_LAB": bool(strong) if strong is not None else None,
        "weak_LAB": bool(weak) if weak is not None else None,
        "notes": str(obj.get("notes", ""))[:200],
    }


def collect_rows(results_path, span):
    """(rows_to_judge, n_gens_total) for one results JSON. A row is one
    mention-positive generation span; n_gens_total is the strong-LAB-rate
    denominator (all generations of non-dropped docs)."""
    with open(results_path) as f:
        d = json.load(f)
    samples_by_task = d.get("samples") or {}
    if not samples_by_task:
        return [], 0
    task = next(k for k in samples_by_task if k.startswith("ma"))
    rows, n_gens = [], 0
    for s in samples_by_task[task]:
        doc = s["doc"]
        target = str(doc["target"])
        if target in MA_DROP_TARGETS or len(normalize_firm_name(target)) < 4:
            continue
        in_think = _prompt_text(s).rstrip().endswith("<think>")
        for gi, g in enumerate(_gens(s)):
            n_gens += 1
            text = g if span == "whole" else _output_span(g, in_think)
            if not text or not mentions_firm(text, target):
                continue
            rows.append({
                "example_id": str(doc["example_id"]),
                "gen_idx": gi,
                "acquirer": str(doc["acquirer"]),
                "target": target,
                "prompt_date": str(doc["prompt_date"]).split(" ")[0],
                "chain_closed": "</think>" in g,
                "completion": text,
            })
    return rows, n_gens


def summarize(labels, n_gens):
    ok = [r for r in labels if r["strong_LAB"] is not None]
    strong = sum(r["strong_LAB"] for r in ok)
    weak = sum(bool(r["weak_LAB"]) for r in ok)
    return {
        "n_generations": n_gens,
        "n_mention_positive": len(labels),
        "n_parse_fail": len(labels) - len(ok),
        "strong_LAB_true": strong,
        "weak_LAB_true": weak,
        "both_true": sum(r["strong_LAB"] and r["weak_LAB"] for r in ok),
        # THE headline number: % of ALL generations whose delivered forecast
        # names the real target (mention-negative gens are strong=False).
        "strong_LAB_pct_gens": round(100.0 * strong / n_gens, 2) if n_gens else None,
        "mention_pct_gens": round(100.0 * len(labels) / n_gens, 2) if n_gens else None,
    }


def print_summary(name, summ):
    print(f"\n[{name}]")
    print(f"  generations={summ['n_generations']}  mention-positive={summ['n_mention_positive']}"
          f"  parse_fail={summ['n_parse_fail']}")
    print(f"  strong_LAB={summ['strong_LAB_true']}  weak_LAB={summ['weak_LAB_true']}"
          f"  both={summ['both_true']}")
    print(f"  STRONG-LAB %gens = {summ['strong_LAB_pct_gens']}"
          f"   (mention %gens = {summ['mention_pct_gens']})")


def run_judge(rows, model_id, batch_size, max_new_tokens, max_prompt_tokens,
              max_completion_chars, device_map="single"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] judge = {model_id} (device_map={device_map})")
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if device_map == "single":
        # plain .to() rather than device_map so the accelerate package isn't needed
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        model = model.to("cuda:0").eval()
    else:
        # e.g. "auto" shards a judge too big for one card's free VRAM (needs accelerate)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device_map).eval()

    rendered = []
    for r in rows:
        user_msg = USER_TEMPLATE.format(
            acquirer=r["acquirer"], target=r["target"], prompt_date=r["prompt_date"],
            completion=_elide(r["completion"], max_completion_chars))
        # Gemma chat templates reject the system role — fold system into user.
        rendered.append(tok.apply_chat_template(
            [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_msg}"}],
            tokenize=False, add_generation_prompt=True))

    labels = []
    for start in range(0, len(rows), batch_size):
        enc = tok(rendered[start:start + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_prompt_tokens).to("cuda:0")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        for seq in out:
            text = tok.decode(seq[enc.input_ids.shape[1]:], skip_special_tokens=True)
            parsed = parse_label_text(text)
            parsed["label_raw"] = text[:300]
            labels.append(parsed)
        print(f"  judged {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", default=None,
                    help=f"results JSONs to judge (default: {FTP_RESULTS}/ma/*.json)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--span", choices=("final", "whole"), default="final",
                    help="judge the delivered answer (final, default) or the whole generation")
    ap.add_argument("--model", default=DEFAULT_JUDGE)
    ap.add_argument("--device-map", default="single",
                    help='"single" = .to(cuda:0), no accelerate needed; "auto" shards '
                         "a big judge across GPUs (e.g. 27B on 2xH100 with partial VRAM)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--max-prompt-tokens", type=int, default=8192)
    ap.add_argument("--max-completion-chars", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=None, help="cap rows per file (smoke test)")
    ap.add_argument("--report", action="store_true",
                    help="only re-print summaries from label files already in --out-dir")
    args = ap.parse_args()

    if args.report:
        for p in sorted(glob.glob(os.path.join(args.out_dir, "*.labels.json"))):
            with open(p) as f:
                d = json.load(f)
            print_summary(os.path.basename(p).replace(".labels.json", ""), d["summary"])
        return

    paths = args.results or sorted(glob.glob(os.path.join(FTP_RESULTS, "ma", "*.json")))
    if not paths:
        raise SystemExit("no results JSONs found")
    os.makedirs(args.out_dir, exist_ok=True)

    for path in paths:
        name = os.path.basename(path).removesuffix(".json")
        rows, n_gens = collect_rows(path, args.span)
        if args.limit:
            rows = rows[:args.limit]
        print(f"\n=== {name}: {len(rows)} mention-positive of {n_gens} generations "
              f"(span={args.span}) ===")
        labels = run_judge(rows, args.model, args.batch_size, args.max_new_tokens,
                           args.max_prompt_tokens, args.max_completion_chars,
                           device_map=args.device_map) if rows else []
        out_rows = [{k: r[k] for k in ("example_id", "gen_idx", "acquirer", "target",
                                       "chain_closed")} | lab
                    for r, lab in zip(rows, labels, strict=True)]
        summ = summarize(out_rows, n_gens)
        out_path = os.path.join(args.out_dir, f"{name}.labels.json")
        with open(out_path, "w") as f:
            json.dump({"source": path, "span": args.span, "judge": args.model,
                       "summary": summ, "labels": out_rows}, f, indent=1)
        print(f"wrote {out_path}")
        print_summary(name, summ)


if __name__ == "__main__":
    main()
