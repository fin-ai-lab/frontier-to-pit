# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- **README stated the DD formula with the operands swapped** — `l_P + α·(l_forget
  − l_retain)`, which is the sign that would ADD the post-cutoff knowledge. The
  implementation is and always was `l_P + α·(l_q − l_p)` with `aux_q` = retain and
  `aux_p` = forget, i.e. `l_P + α·(l_retain − l_forget)`. Docs only; no behaviour
  change.

### Changed

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
