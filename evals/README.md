# evals — Divergence Decoding evals on lm-evaluation-harness

We run everything through the vendored EleutherAI **lm-evaluation-harness**
(`../lm-evaluation-harness`, committed into this repo). The launcher
`python -m evals.lmeval` loads one model once and sweeps it over the requested
tasks via `lm_eval.simple_evaluate`. Two families:

- **utility** (temp 0.0, generative) — `mmlu_pro` (5-shot CoT), `mmlu_redux_generative`
  (0-shot, stock upstream), `ifeval`, `gpqa_diamond_generative_n_shot`. Capability
  preservation. All run **generatively** (DD only affects generation, not loglikelihood).
- **temporal** (temp 0.7, `repeats`) — `ma`, `pharma`, `covid`, `streamingqa` (custom tasks
  in `lmeval/tasks/temporal`). Post-cutoff knowledge leakage; **lower is better**.

## Install

```bash
uv pip install -e ".[evals]" -e "./lm-evaluation-harness[ifeval]"   # from repo root
# local-only: flashinfer's sampler JIT needs the `ninja` binary on PATH, or:
#   export VLLM_USE_FLASHINFER_SAMPLER=0
```

## Run

```bash
python -m evals.lmeval --model le2015_3b_v1 --tasks ifeval --limit 4      # smoke
python -m evals.lmeval --model pit_4b_2015  --tasks all --temporal-n 6
python -m evals.lmeval --model chrono_gpt_2015 --tasks ma,pharma,covid
```

`--tasks` takes `utility` / `temporal` / `all` or a comma list. Each `(model, arm, task)`
writes `results/<task>/<model>__<arm>.json` (lm-eval's results + `--log_samples`),
skipped if it exists (resumable; `--overwrite` to redo). `--temporal-n` overrides the
temporal tasks' `repeats`.

## Models

The active sweep is **6 little models + a reference** (DD's `ftp_qwen_v1` is defined but
deferred — it needs an 80 GB GPU):

| model | backend | notes |
|-------|---------|-------|
| `le2015_3b_v1`, `le2025_3b_v1` | stock `vllm` | aux pair run standalone (leakage floor / upper bound). Qwen3.5 template, `enable_thinking=False`, `<think>` banned |
| `qwen3_5_2b` | stock `vllm` | capability-ceiling reference (vision pinned out) |
| `pit_4b_2015`, `pit_4b_2024` | `pit` (HF) | custom `PITForCausalLM`; **batch-safe attention patch** lets it run at a large hardcoded `batch_size` (per-GPU) despite ignoring the padding mask + having no KV cache. Alpaca `### Instruction` prompt |
| `chrono_gpt_2015`, `chrono_gpt_2024` | `chronogpt` | bespoke modded-nanogpt, tiktoken — not vLLM/HF-loadable, so a thin custom `LM` wraps its loader + `generate` |
| `ftp_qwen_v1` *(deferred)* | `dd` | Divergence Decoding (+ optional SAE steering); two arms (α=1.5 / 0.0) from one engine load |

All per-model knobs (paths, templates, fewshot caps, generation budget) are hardcoded in
`lmeval/__main__.py`. The few-shot count is **capped to 0** for these small-context models
(PIT 1024, ChronoGPT 1792, aux 2048) — they can't fit MMLU-Pro's 5-shot CoT.

## Custom code (everything else is stock lm-eval)

- `lmeval/backends.py` — `@register_model` for `pit` (HF + attention/batch patch + Alpaca
  template), `chronogpt` (custom `LM`), `dd` (thin `VLLM` subclass: `DDLogitsProcessor`
  + per-request `dd_alpha` + optional SAE steering composed with DD). Steering defaults
  to the PRE-CAPTURE route (hooks recorded inside P's CUDA graphs — full-speed, fixed
  clamps). A spec passing `steer_sweep=True` in its `args` instead takes the eager
  post-build route, where `set_steer_clamp`/`set_steer_feature` remove and re-install the
  hook in-process to sweep the `clamp_value` range from one engine load (clamp 0 = pure-DD
  baseline); the sweep-driver loop and its specs were removed with the pre-v4 registry, so
  no current model uses it. `BATCH_BY_GPU` sizes the HF backends.
- `lmeval/__main__.py` — launcher + hardcoded `MODELS` config; drives `simple_evaluate`.
- `lmeval/tasks/temporal/` — `ma`/`pharma`/`covid`/`streamingqa` YAMLs + `utils.py`: portable parquet
  loaders, a `take_all` keep-repeats filter (so all `repeats` samples are scored), leak
  `process_results` (reusing `scoring/firm_match.mentions_firm` for M&A), and grouped
  response-weighted aggregation (`ratio_agg`; pre/post cutoff, covid look-ahead window).

## Cluster

Cluster sweeps go through the lab's pythia glue (`tools/pythia/submit_eval.sh`,
gitignored), one SLURM job per `(model × task)` with node-local staging. It installs the
vendored harness and calls `python -m evals.lmeval` (same `--model/--tasks/...` interface
as before).
