# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Live degeneration guard** (`ftp.guard` + `ftp.vllm.GuardLogitsProcessor`;
  `run.py` flags `--no-guard`, `--guard-{model,interval,backtrack,threshold,tries}`;
  `build_llm(..., guard=GuardConfig(...))`; env `DD_GUARD_*`). Under a strong DD
  push a generation can collapse into a repeating loop / symbol spam and burn its
  whole budget (measured at α=1.5 NOTHINK: 45% of M&A and 35% of pharma
  generations destroyed). Weakening the fusion to fix this costs the unlearning
  (a suppress-only variant saturated ~2× above the two-sided leak floor and was
  dropped), so the guard keeps the fusion and repairs the collapse instead:
  a small judge LM (default `Qwen/Qwen3.5-2B`, on the aux GPU) sweeps the batch
  every 25 engine steps — ONE batched yes/no forward regardless of batch width —
  reading each request's last 50 tokens through a zlib pre-gate (only
  loop-compressible windows reach the judge — measured on gemma-labeled windows:
  the gate passes 89% of degenerate and 3% of clean ones); on p(degenerated) ≥
  0.5 that request (only) is force-stopped via a reserved marker token, rewound
  50 tokens, and resampled. The gate+threshold pair is CALIBRATED on real labeled
  windows from the α=1.5 results: TPR 0.80 / FPR 0.02 per check, and a persistent
  loop is re-checked every sweep so the effective catch rate compounds toward 1.
  vLLM cannot rewind KV mid-request, so the rewind is a stop+resubmit
  (`ftp.guard.rollback_generate`); the clean path is never interrupted. Stuck
  requests escalate: after 2 no-progress resamples the walk-back deepens by
  another 50 tokens (50 → 100 → 150 …); a request that still breaks at its very
  first tokens returns a visible `[Could not generate without degeneration]`
  instead of garbage, and one that runs out of budget/rounds returns its clean
  accepted prefix (`max_rounds`, default 20, globally caps judge failures per
  request). The judge loads multimodal-wrapped checkpoints (Qwen3.5-\*) via the
  architecture the checkpoint declares when the plain-CausalLM auto route
  rejects the nested config. `run.py` streaming now emits APPROVED blocks: the newest 50
  tokens are held back until they survive the walk-back window, so a rewind
  never has to un-print (the visible stream trails generation by ~1 s).

- **4×GPU layout support** (`ftp.config.default_device_layout` /
  `split_aux_device`; per-model `DDConfig.aux_device` pair form; `run.py
  --guard-device`; `ftp.probe --tp`). Device placement is now derived from the
  visible GPU count and P's `--tensor-parallel-size` instead of assuming the
  2×GPU split. With two or more GPUs free after P's TP ranks the aux pair
  SPLITS, one model per card — `DDConfig.aux_device` accepts an
  `"aux_p_dev,aux_q_dev"` pair (e.g. `"cuda:3,cuda:2"`), which runs the
  two-engine path (fusion needs one card; shared tokenizer mode only, and each
  logits plane crosses to P's GPU independently). The guard judge takes a free
  card if any remain, else rides the LAST TP rank's spare memory (sharded P
  leaves headroom); like the DD aux engines, under TP the judge now loads at
  its first sweep — post-profiling, where an external allocation on a TP
  rank's card would crash the worker — so leave it ~6 GB of
  `gpu_memory_utilization` headroom. Layouts: 2×GPU TP=1 unchanged
  (P | aux pair+guard on `cuda:1`); 4×GPU TP=2: P over `cuda:0`–`1` + guard on
  `cuda:1`, retain aux on `cuda:2`, forget aux on `cuda:3`. Motivation:
  `--think` at its 20480 default context does not fit next to the aux pair +
  guard on 2×H100 — on a 4×GPU box `run.py chat --think --tensor-parallel-size 2`
  now lays itself out correctly with no device flags. `ftp.probe --tp N` models
  P's weights+KV sharded over N ranks and prints the per-GPU fit, per-model aux
  placement, and `gpu_memory_utilization`.

### Fixed

- **README stated the DD formula with the operands swapped** — `l_P + α·(l_forget
  − l_retain)`, which is the sign that would ADD the post-cutoff knowledge. The
  implementation is and always was `l_P + α·(l_q − l_p)` with `aux_q` = retain and
  `aux_p` = forget, i.e. `l_P + α·(l_retain − l_forget)`. Docs only; no behaviour
  change.

### Changed

- **The degeneration guard is ON BY DEFAULT whenever DD is configured, and its
  judge is now `unsloth/gemma-3-4b-it` (threshold 0.5).**
  `build_llm`/`build_async_llm` install the guard automatically with DD
  (`guard=False` opts out; a `GuardConfig` customizes) — a strong DD push is
  exactly what produces the collapse the guard repairs, and the steady state
  costs ~0.1–0.2% of step time. The judge swap follows the 2026-07-16
  judge × threshold sweep (ma+pharma at α=1.5, gemma-27B-judged): the previous
  `Qwen/Qwen3.5-2B` judge's GDN linear-attention layers run HF-eager as a
  ~113 ms fixed-cost sequential scan per fired check, so a plain-attention
  judge is ~3× cheaper (~42 ms vs ~121 ms per gated call) — and gemma-3-4b
  also *detects better*: pooled destroyed 31.4% (unguarded) → 7.7% vs 8.6%
  for the old judge, leak unchanged, zero visible failures. Its operating
  point is threshold-insensitive (offline gated TPR ~86% / FPR ~6–7%, flat
  from 0.3 to 0.99), so 0.5 stays the default with no tuning cliff. The
  unsloth mirror is ungated — fresh boxes pull it without a HF login.
  (gemma-3-1b was evaluated and rejected: no class separation — gated FPR
  22–46% at every threshold, 2882 rollbacks/325 requests end-to-end;
  `Qwen3-1.7B` is a viable budget pick at threshold 0.8 if judge latency
  matters more than the last ~2 pp of repair.
  `tools/calibrate_guard_threshold.py` re-derives the operating table for any
  candidate judge from the labeled windows.)

- **The aux engine's sliding window is GONE: the aux models now see P's full
  stream.** `window` was an engine-level truncation — a relic of the 2K-context
  v1 aux models: rows crossing it re-primed on a truncated tail
  (`window × (1 − DD_REPRIME_MARGIN)` tokens, re-based positions), and prompts
  longer than it primed on the tail only. EVERY DD result generated before
  2026-07-16 ran with `dd_window=2048` (verified across the registry and git
  history: think and nothink alike) — so the aux pair scored divergence from at
  most the trailing 2048 tokens, mmlu_pro's ~2.5K prompts were never fully seen
  by the aux, and think-mode CoTs lost the prompt from aux view entirely —
  while P and every baseline saw full context. Now `window` is a hard CAPACITY
  (page tables, ring sizing): exceeding it raises loudly instead of truncating
  silently, and `DDConfig.window=0` (the new default) auto-sizes it to P's
  `max_model_len` (×2 in universal mode for retokenization inflation), where
  the error is unreachable. The upfront `batch × window` pool sizing is now
  capped at ~80% of free device memory (the full-context worst case is huge
  while live usage tracks tokens in flight; demand growth + graph recapture
  covers overflow). The 512-token sliding-attention layers in the aux
  ARCHITECTURE (23 of 28 layers) are model semantics and are unchanged — they
  are what keeps full-context KV cheap (~5 full-attention layers pay context).
  `DD_REPRIME_MARGIN` is gone; `AuxBatchedEngine.step`/`step_pairs` raise past
  capacity. NOTE: alpha/clamp calibrations predate this change — DD numbers
  produced after it are not directly comparable to the 2048-window results.

- **Temporal eval datasets are now the final shipped artifacts** — the parquets
  under `evals/lmeval/tasks/temporal/` are served verbatim; `utils.py` no longer
  rewrites prompts or drops rows at load time, so running the parquets outside
  this harness reproduces our prompts exactly.
  - Every doc now carries a `system_prompt` column, rendered as the chat
    system message (task yaml `description: system_prompt`). ma/pharma/covid
    share a forecasting system prompt (point-in-time expert as of 2015-12-31,
    predict-don't-refuse); streamingqa's is "Answer in a short and concise
    sentence." (the brevity instruction moved out of the user turn).
  - `ma.parquet` 174 → 129 docs: dropped the two same-name acquirer/target
    false positives (Targa Resources Partners LP, ONEOK Partners LP — formerly
    excluded at load time) and all pre-cutoff deals (`dateann` ≤ 2015-12-31),
    so naming the target is always look-ahead leakage. The old in-prompt
    preamble and weak anti-refusal suffix are gone (replaced by the system
    prompt); the misleading `regime` column (a different project's boundary)
    was removed; the pre/post metric split is gone (`leak_rate`,
    `any_leak_rate`, `mentions_per_1k{,_think,_final}` remain).
  - `pharma.parquet` 50 → 49 docs: dropped example_id 17 (Nurix Therapeutics
    2023), whose only lab word "pfizer" is too guessable to signal look-ahead;
    removed the generic "study" lab word (formerly a scoring-time stop word).
  - PIT's Alpaca/card chat templates now render a system message as a leading
    paragraph (they silently dropped it before); prompts without a system
    message are byte-identical to before.
  - Stale temporal results were deleted from the result stores; all temporal
    numbers must be regenerated on the new datasets.
  - The demo tooling now serves the same forecasting system prompt by default:
    `run.py` (both subcommands; `--no-system-prompt` / `--system-prompt` to
    drop/replace) and the website-examples client (`ask.sh` /
    `tools/websrv/client.py`, same flags). Canonical text lives in
    `ftp.prompts.FORECAST_SYSTEM_PROMPT`; `tests/test_prompts.py` pins it
    byte-identical to the datasets' `system_prompt` column.

- **Aux engine replaced with a paged-KV design** (same class name and public
  API: `AuxBatchedEngine`; `PagedAuxEngine` kept as an alias). KV lives in
  16-token pages allocated on demand; decode attention reads each request's
  ACTUAL context (flashinfer `BatchDecodeWithPagedKVCacheWrapper` on CUDA, an
  exact gather+SDPA fallback elsewhere); a step costs only the rows active
  that step. The previous window-wide `StaticCache` + CUDA-graph engine —
  whose memory AND per-step cost scaled with `window x prewarmed_batch`
  regardless of occupancy (measured 0.47 ms/row/step at window 2048 whether
  rows held 256 or 2048 tokens) and which could not host 32K-context aux
  models (3.8 GB per slot per plane) — was removed; it lives in git history.
  Measured on 2xH100 (v3.3 fused 3B pair, bf16): decode 31.1 ms vs 105.9 ms
  at B=112 full window (3.4x), 25.0 ms at fill 256; DD-in-vLLM engine build
  117 s vs 426 s. Past the window, rows RE-PRIME (pages freed, tail
  `window*(1-DD_REPRIME_MARGIN)` tokens re-prefilled, re-based positions)
  instead of the old approximate eviction shift. `compile_aux` is accepted
  and ignored (the engine runs eager; compiling/graphing the paged decode is
  the open follow-up). New env knobs: `DD_PAGED_FLASHINFER=0` (force the
  fallback attention), `DD_REPRIME_MARGIN` (default 0.25),
  `DD_PREFILL_TOKEN_BUDGET` (default 32768).

### Added

- **Fused aux pair: both aux models in ONE forward** (`DDConfig.fuse_aux` /
  `DD_FUSE_AUX`, default `auto`). When the two aux models are architecturally
  identical, their weights are stacked at engine init (`[2, ...]` leading
  model dim) and every layer runs ONE batched kernel for both models: linears
  via `torch.bmm` over pre-transposed `[2, in, out]` weights, attention with
  the model dim folded into the batch, one KV cache with a doubled batch dim,
  ONE CUDA graph replay per decode step returning `[2, N, V]`. At decode
  batch sizes each small model underutilizes the GPU, so the fused step costs
  ~1.05–1.3× a single model instead of 2× — removing the serial p-then-q aux
  block that was the entire residual DD overhead in 2-GPU overlap mode.
  `auto` falls back loudly to the two-engine path when the pair is not
  fusable (different sizes, tied embeddings); `on` makes that an error;
  `off` opts out. New `ftp.paired` module: `PairedLinear`,
  `PairedEmbedding`, `PairedRMSNorm`, `fuse_pair`, `check_fusable`.
- `AuxBatchedEngine(model, ..., model2=...)`: fused-pair construction;
  `step`/`step_pairs` return stacked `[2, N, V]` logits (plane 0 = `model`,
  plane 1 = `model2`). `UniversalBridge(..., aux_q=None)` accepts a fused
  engine. `DD_AUX_STREAMS` (experimental, default off) is ignored when the
  pair is fused — a fused engine is one forward; there is nothing to overlap.
- **GQA-fold attention** (`DD_GQA_FOLD=0` opts out): the engine's per-row
  validity mask disabled HF's native-GQA sdpa path, so every layer of every
  decode step materialized the kv→q-head expansion of the full KV cache
  (~10 ms of copy traffic per 3B forward at batch 32). The engine now folds
  the GQA group dim into the query-length dim and lets SDPA read the cache
  in place: single 3B replay 18.2 → 11.3 ms, fused pair 34.0 → 20.9 ms
  (H100, batch 32, window 1024). Combined result on 2× H100 with
  `aux_device` overlap (Qwen3.5-27B P + 2× 3B aux, batch 32): exposed aux
  wait 13.2 → **0.1 ms/step**; DD throughput 1061–1082 tok/s vs 1128
  same-run baseline (**94–96%**, vs 76% in v0.3) — DD is effectively
  latency-invisible given a second GPU.
- **Tensor-parallel support**: under `tensor_parallel_size > 1` the
  processor is constructed in every TP rank's worker, but vLLM gathers
  logits to rank 0 only — non-zero ranks are now inert (no aux load; was: a
  full aux copy per rank), and rank 0 defers aux construction to the first
  batch, after vLLM's per-card memory profiling. TP=1 behavior unchanged.
  Known issue: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is
  incompatible with TP > 1 (workers die natively during memory profiling;
  reproduced without any DD code) — use the flag only for single-GPU
  placements.
- **Universal decoding: P may use a different tokenizer than the aux pair**
  (the aux pair always shares one). Auto-detected at processor init by
  comparing vocabularies (`DDConfig.mode`: `auto`/`shared`/`universal`); the
  shared-tokenizer path is unchanged. New `ftp.translate`
  module: `TokenTextTable` (id→bytes for P's tokenizer), `StreamRetokenizer`
  (incremental aux retokenization of P's stream with exact cache rewinds),
  `VocabMapper` (precomputed first-token map, fp32 log-prob gather onto P's
  vocab, disk-cached), and `UniversalBridge` (drain loop over the engines'
  single-token batched step, `max_feeds_per_step` cap). In universal mode the
  special-token whitelist and `suppress_tokens` resolve from P's tokenizer.
- `AuxBatchedEngine.rewind(req_id, k)`: drop the last k cached tokens for one
  request (pure bookkeeping under the per-row cursors; exact below the window
  cap).
- `AuxBatchedEngine.step_pairs()`: 1–2 new tokens per row in one batched
  forward (per-row two-column cursors, per-query validity masks, grouped 1/2
  column evictions at the window cap), compiled and captured as a second CUDA
  graph that shares the single-token graph's memory pool. The universal
  bridge packs retokenization bursts through it: measured on GH200
  (DeepSeek-R1-Distill-Llama-8B P + 2× 3B aux, 32 requests, 1024 new tokens)
  universal DD went 195–216 → 265–269 tok/s (baseline 467–470); mean aux
  rounds/step 2.04 → 1.09. gemma-4-31B-it (SentencePiece) + 2× 0.8B aux:
  439–447 tok/s (baseline 795–889), map coverage 100%.
- GPU smoke test for universal mode (`tests/gpu/test_vllm_universal_smoke.py`):
  with `aux_p == aux_q` the mapped shift is exactly zero, so greedy universal
  DD must reproduce the greedy `dd_alpha=0` baseline token-for-token.

### Fixed

- The engine's decode index cache was keyed by output-index tuples only; the
  same index tuple recurring with different requests (vLLM recompute pauses,
  universal drain rounds) replayed the wrong slots, writing K/V into another
  request's cache rows. Now keyed by (index, request) pairs.

### Removed

- The plausibility gate (`gate_mode` none/P/q/union/intersect and its
  parameters). It compensated for tail noise from non-instruction-tuned aux
  models; with instruction-tuned aux pairs it is unnecessary, and ordinary
  top-k/top-p sampling filters apply to the fused distribution instead. The
  implementation remains available in git history (<= v0.2.0).

## [0.2.0] - 2026-06-12

### Changed

- **Decode is now a single batched forward for any mix of sequence positions.**
  Every cache layer shares a per-row write cursor (`_DDStaticLayer`), per-row
  validity is enforced by a prebuilt 4D additive attention mask, and rows at
  the window cap evict their oldest token via a subset shift. This replaces
  the uniform/grouped/per-request decode paths: position-staggered batches
  (vLLM's normal scheduling) cost one forward and one CUDA-graph replay
  instead of one per position group.
- Measured on GH200 (Qwen3.5-27B, 32 requests, 1024 new tokens): 3B-aux DD
  throughput 95-177 -> 213-242 tok/s (+23% to +155% per arm); aux step
  p50 54 -> 17.8 ms.
- Engine registrations are reconciled against vLLM's persistent batch after
  every update (a missed unregister leaked KV slots and grew the cache), and
  invalidated CUDA graphs release their memory pools via reset().

### Removed

- The shared-cursor resync, grouped column backup/restore, and the
  per-request decode fallback (all subsumed by the single path).

## [0.1.0] - 2026-06-XX

### Added

- Initial public release.
- `ftp.core`: the fusion math — `dd_fuse` (linear/rank DD with
  plausibility gating and special-token probability pinning), `make_gate`,
  `build_special_ids`, `log1mexp`.
- `ftp.engine.AuxBatchedEngine`: batched auxiliary-model
  inference on HF `StaticCache` with slot recycling, uniform / grouped-staggered
  / per-request decode paths, a sliding context window, and optional
  `torch.compile` CUDA-graph decode.
- `ftp.vllm.DDLogitsProcessor` + `make_processor(cfg)`:
  Divergence Decoding inside vLLM v1's engine core, configured by a typed
  `DDConfig` (with a `DD_*` environment-variable fallback for `vllm serve`),
  per-request strength via `SamplingParams.extra_args["dd_alpha"]`.
- CPU test suite incl. tiny-model ground-truth tests of every engine decode
  path; GPU test suite (opt-in, `pytest -m gpu`).
- `examples/forget_probe.py`: leak-rate probe demonstrating end-to-end usage.

### Notes

- Versions 0.1.0 and 0.2.0 predate this public repository; they document the
  method's evolution in the internal research codebase (then named
  `divergence-decoding`, import package `divergence_decoding` — identifiers
  above are shown under their current names).
