"""Data loading + scoring for the temporal-leakage tasks (ma / pharma / covid).

Each runs generatively at temperature 0.7 with ``repeats: N`` (the launcher overrides
N via ``--temporal-n``). A custom ``take_all`` filter keeps every repeat so
``process_results`` sees all N generations per doc; a generation "leaks" if a
post-cutoff term surfaces in the free-form prediction (lower is better). Grouped
metrics (e.g. the covid look-ahead window) are emitted as separate ``(num, den)``
scalars and reduced response-weighted by ``ratio_agg``.

The parquets ARE the shipped artifact: prompts are served verbatim (no rewriting
here), and each doc carries its system prompt in the ``system_prompt`` column,
which the task YAMLs reference via ``description: system_prompt`` (the harness
resolves a doc column named there into the chat template's system message).

Datasets load from parquet next to this file (path from ``__file__``, not CWD) so the
task is portable across machines.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from lm_eval.api.filter import Filter
from lm_eval.api.registry import FILTER_REGISTRY, register_filter

# firm-name matcher for M&A leak detection (kept from the old harness).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evals.scoring.answer_match import answer_leak  # noqa: E402
from evals.scoring.firm_match import _tokenize, mentions_firm, normalize_firm_name  # noqa: E402

_DIR = Path(__file__).resolve().parent


# ── keep-all-repeats filter (default take_first would drop samples 2..N) ──
if "take_all" not in FILTER_REGISTRY:

    @register_filter("take_all")
    class TakeAllFilter(Filter):
        def apply(self, resps, docs):
            # each entry of resps is one doc's list of `repeats` generations — keep all.
            return [list(r) for r in resps]


def ratio_agg(items) -> float:
    """Response-weighted rate from a list of ``(num, den)`` pairs."""
    num = sum(n for n, _ in items)
    den = sum(d for _, d in items)
    return num / den if den else 0.0


def _gens(results) -> list[str]:
    """The list of `repeats` generations for one doc (take_all keeps them all)."""
    g = results[0]
    return list(g) if isinstance(g, (list, tuple)) else [g]


# ── LAB mentions per 1K tokens (whole / thinking chain / final response) ──
# Mirrors the lab-2026 plot_lab_combined convention: per-completion
# mention_count / token_count (real Qwen tokenizer), MEAN over completions, x1000
# at aggregation. Temporal tasks keep the RAW generation — run_task disables the
# harness's model-level </think> strip for them — so thinking runs arrive as
# "trace</think>final" and both spans can be scored separately.

_LAB_TOK: object = None


def _lab_tok():
    """Qwen tokenizer for token counts (byte-identical across the Qwen3.5 family AND
    the v4 aux checkpoints). Resolution favors node-local copies (offline nodes):
    DD_QWEN27B_DIR -> staged v4 aux dirs -> the hub name. None if nothing loads
    (mentions metrics then emit 0 and the offline re-scorer recomputes from samples)."""
    global _LAB_TOK
    if _LAB_TOK is None:
        import os

        from transformers import AutoTokenizer

        for c in (os.environ.get("DD_QWEN27B_DIR"),
                  os.path.join(os.environ.get("DD_V4_BASES", ""), "le2015_3b_v4"),
                  os.path.join(os.environ.get("DD_V4_ROOT", ""),
                               "le2015_3b_v4", "lr5e-6_dual", "final"),
                  "Qwen/Qwen3.5-27B"):
            if not c or not str(c).strip("/"):
                continue
            try:
                _LAB_TOK = AutoTokenizer.from_pretrained(c, trust_remote_code=True)
                break
            except Exception:  # noqa: BLE001 — try the next candidate
                continue
        if _LAB_TOK is None:
            _LAB_TOK = False
    return _LAB_TOK or None


def _think_spans(text) -> tuple[str, str]:
    """(thinking_chain, final_response), split on the FIRST '</think>'. A generation
    with no '</think>' is ALL final response (the nothink case; NB a thinking trace
    truncated before it closed also lands there)."""
    t = str(text)
    if "</think>" in t:
        cot, fin = t.split("</think>", 1)
        return cot, fin
    return "", t


def _term_count(text_lower, term) -> int:
    """Whole-word occurrences of a (possibly multi-word) lowercase term, so short
    leak terms can't fire inside unrelated words ('lease' in 'released')."""
    return len(re.findall(rf"\b{re.escape(term)}\b", text_lower))


def _mention_count(span, key, task) -> int:
    """Occurrences of the doc's leak terms in a span. ma: normalized-firm-name
    substring count over the normalized span (>=4 chars, as in the lab-2026 plots);
    others: summed lowercase whole-word counts of the word list."""
    s = str(span)
    if task == "ma":
        nf = normalize_firm_name(str(key))
        return " ".join(_tokenize(s)).count(nf) if len(nf) >= 4 else 0
    lo = s.lower()
    return sum(_term_count(lo, str(w).lower()) for w in (key or []) if str(w))


def _mentions_per_1k(gens, key, task) -> dict:
    """{'': (sum_ratios, n), '_think': ..., '_final': ...} — per-completion
    mentions/token ratios summed per span (zero-token spans skipped), for per1k_agg."""
    tok = _lab_tok()
    out = {sfx: (0.0, 0) for sfx in ("", "_think", "_final")}
    if tok is None:
        return out
    for g in gens:
        cot, fin = _think_spans(g)
        for sfx, span in (("", str(g)), ("_think", cot), ("_final", fin)):
            nt = len(tok(span, add_special_tokens=False)["input_ids"]) if span else 0
            if nt:
                num, den = out[sfx]
                out[sfx] = (num + _mention_count(span, key, task) / nt, den + 1)
    return out


def per1k_agg(items) -> float:
    """1000 x mean-over-completions of (mentions / tokens); items = per-doc
    (sum_of_ratios, n_completions) pairs."""
    num = sum(n for n, _ in items)
    den = sum(d for _, d in items)
    return 1000.0 * num / den if den else 0.0


# ── dataset loaders (portable parquet paths) ──
def _load(name):
    import datasets
    import pandas as pd

    # Read the parquet via pandas and build the Dataset in-memory — do NOT use
    # datasets.load_dataset("parquet", ...). Its packaged-builder cache (node-local
    # /hpc_temp HF cache) returned STALE arrows even with download_mode="force_redownload":
    # after the forget+retain rebuild, warm nodes served the old forget-only set (and one
    # node a retain-only set), so shards silently disagreed on the data. pandas reads the
    # file fresh every call; from_pandas builds from that DataFrame -> always current.
    df = pd.read_parquet(str(_DIR / f"{name}.parquet"))
    return datasets.DatasetDict({"test": datasets.Dataset.from_pandas(df, preserve_index=False)})


def load_ma(**kwargs):
    return _load("ma")


def load_pharma(**kwargs):
    return _load("pharma")


def load_covid(**kwargs):
    return _load("covid")


def load_streamingqa(**kwargs):
    # Data-parallel sharding: DD_NUM_CHUNKS jobs each take a strided shard of the
    # forget+retain set (one GPU per shard); results are merged downstream and split by
    # `bucket` for leak@k (forget) / retained@k (retain).
    import os
    import random

    ds = _load("streamingqa")
    # DD_STREAMINGQA_PER_BUCKET=N: deterministic (seed-0) N-question subsample from EACH
    # bucket (forget/retain) — the SAME questions in every arm/job, so sweep cells stay
    # comparable (the v4 LR x alpha heatmap runs 1000+1000). The v5 parquet is
    # block-ordered (all forget, then retain), so a plain --limit can't do this.
    # Applied BEFORE chunk sharding: shards partition the subset.
    npb = int(os.environ.get("DD_STREAMINGQA_PER_BUCKET", "0"))
    if npb:
        by_bucket = {}
        for i, b in enumerate(ds["test"]["bucket"]):
            by_bucket.setdefault(str(b), []).append(i)
        keep = []
        for b in sorted(by_bucket):
            idx = by_bucket[b]
            random.Random(0).shuffle(idx)
            keep.extend(idx[:npb])
        ds["test"] = ds["test"].select(sorted(keep))
    nc = int(os.environ.get("DD_NUM_CHUNKS", "1"))
    ci = int(os.environ.get("DD_CHUNK_ID", "0"))
    if nc > 1:
        # The parquet interleaves forget/retain with period 2, so a stride-nc shard (nc
        # even) aliases onto one parity -> each shard ends up single-bucket. Shuffle first
        # with a FIXED seed (identical order on every shard) so the shards still partition
        # the full set disjointly, but each shard is bucket-mixed.
        ds["test"] = ds["test"].shuffle(seed=0).shard(num_shards=nc, index=ci, contiguous=False)
    return ds


# ── StreamingQA forget-set leak@k (leak = the generation states a gold answer) ──
# The forget set is post-2015 knowledge we steer away from; a generation "leaks" if it
# contains a gold answer. Runs with `repeats: N` at temperature 0.7 so leak@k can be
# computed from the N samples downstream (tools/.../streamingqa_leak_at_k.py re-scores the
# logged generations, so the matcher can change without re-generating).
def streamingqa_doc_to_text(doc) -> str:
    # Brevity instruction lives in the doc's `system_prompt` column (yaml `description`).
    import datetime as _dt

    date = _dt.datetime.utcfromtimestamp(int(doc["question_ts"])).strftime("%B %-d, %Y")
    return f"Today is {date}.\n\n{str(doc['question']).strip()}"


def streamingqa_process_results(doc, results) -> dict:
    gens = _gens(results)
    answers = list(doc["answers"]) if doc["answers"] is not None else []
    flags = [answer_leak(g, answers) for g in gens]
    n, nl = len(flags), int(sum(flags))
    yr = int(doc["evidence_year"])
    # headline leak rate (response-weighted) + any-leak (>=1 of N). leak@k for k=1..N is
    # recomputed downstream from the logged generations. Split by 2020 vs 2016-2019.
    is2020 = yr >= 2020
    # bucket split: the set is 2000 forget (post-2015, leak = bad, want LOW) + 2000 retain
    # (pre-2015, "leak" = the model correctly states the answer = retention, want HIGH).
    is_forget = str(doc.get("bucket", "forget")) == "forget"
    return {
        "leak_rate": (nl, n),
        "any_leak_rate": 1.0 if nl else 0.0,
        "leak_rate_2020": (nl, n) if is2020 else (0, 0),
        "leak_rate_2016_2019": (0, 0) if is2020 else (nl, n),
        "leak_rate_forget": (nl, n) if is_forget else (0, 0),
        "leak_rate_retain": (0, 0) if is_forget else (nl, n),
    }


# ── M&A look-ahead leakage (leak = naming the firm that was in fact acquired) ──
# Every deal in ma.parquet was announced AFTER 2015-12-31 (`dateann`; the pre-cutoff
# deals and the two same-name acquirer/target false positives were dropped from the
# artifact), so naming the target is always look-ahead leakage. Prompts are served
# verbatim (yaml `doc_to_text: prompt_text`).
def ma_process_results(doc, results) -> dict:
    gens = _gens(results)
    flags = [mentions_firm(g, doc["target"]) for g in gens]
    n, nl = len(flags), int(sum(flags))
    out = {
        "leak_rate": (nl, n),
        "any_leak_rate": 1.0 if nl else 0.0,
    }
    # LAB mentions/1K tokens, per span (whole/think/final).
    for sfx, pair in _mentions_per_1k(gens, doc["target"], "ma").items():
        out[f"mentions_per_1k{sfx}"] = pair
    return out


# ── pharma look-ahead leakage (leak = naming a post-cutoff drug/approval) ──
def pharma_process_results(doc, results) -> dict:
    gens = _gens(results)
    words = [str(x).lower() for x in doc["lab_words"]]
    flags = [any(_term_count(g.lower(), w) for w in words) for g in gens]
    n, nl = len(flags), int(sum(flags))
    strong = doc["verdict"] == "Strong LAB"
    out = {
        "leak_rate": (nl, n),
        "any_leak_rate": 1.0 if nl else 0.0,
        "leak_rate_strong": (nl, n) if strong else (0, 0),
        "leak_rate_weak": (0, 0) if strong else (nl, n),
    }
    for sfx, pair in _mentions_per_1k(gens, words, "pharma").items():
        out[f"mentions_per_1k{sfx}"] = pair
    return out


# ── covid look-ahead leakage (replicates/adjusts Sarkar & Vafa 2024) ──
_COVID_PROMPT = (
    "The following is a section of an MD&A for {firm}:\n{mda}\n"
    "The call took place on {date}.\n"
    "Consider only information up to and including the earnings call.\n"
    "Predict the potential risks for this company in {risk_year}"
)
# metric-key -> phrase to match (lowercased substring).
_COVID_WORDS = {
    "covid": "covid",
    "pandemic": "pandemic",
    "disease_outbreak": "disease outbreak",
    "supply_chain": "supply chain",
}


def covid_doc_to_text(doc) -> str:
    import pandas as pd

    date = pd.Timestamp(doc["date"]).strftime("%B %-d, %Y")
    return _COVID_PROMPT.format(
        firm=doc["conml"],
        mda=str(doc["text_mda"])[:2000],
        date=date,
        risk_year=int(doc["year"]) + 1,
    )


def covid_process_results(doc, results) -> dict:
    raw_gens = _gens(results)
    gens = [g.lower() for g in raw_gens]
    n = len(gens)
    year = int(doc["year"])
    counts = {k: int(sum(w in g for g in gens)) for k, w in _COVID_WORDS.items()}
    # year==2019 -> predicting 2020 (the look-ahead window); year==2018 -> predicting 2019
    # (pre-pandemic control). Emit each phrase for BOTH windows so the table can show the
    # 2020/2019 pair per phrase.
    win = "2020" if year == 2019 else "2019"
    out = {}
    for k, c in counts.items():
        out[f"{k}_2020_rate"] = (c, n) if win == "2020" else (0, 0)
        out[f"{k}_2019_rate"] = (c, n) if win == "2019" else (0, 0)
    # LAB mentions/1K over ALL covid leak phrases combined, per span x window.
    m1k = _mentions_per_1k(raw_gens, list(_COVID_WORDS.values()), "covid")
    for sfx, pair in m1k.items():
        out[f"mentions_per_1k_2020{sfx}"] = pair if win == "2020" else (0.0, 0)
        out[f"mentions_per_1k_2019{sfx}"] = (0.0, 0) if win == "2020" else pair
    return out
