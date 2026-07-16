"""Run our evals through the vendored lm-evaluation-harness.

    python -m evals.lmeval --model qwen3_5_27b --tasks utility --limit 8   # smoke
    python -m evals.lmeval --model pit_4b_2015 --tasks all --temporal-n 6
    python -m evals.lmeval --model ftp_qwen_v4_lr0 --tasks ma,pharma

One model is loaded once and swept over the requested tasks; each (model, arm,
task) writes ``results/<task>/<model>__<arm>.json`` (resumable; ``--overwrite`` to
redo). All model/task knobs are hardcoded in ``MODELS`` below — the CLI only picks
the model + tasks and a few engine overrides, mirroring the old ``run_evals`` so
the pythia glue is unchanged.

Backends: the Qwen baselines and talkie use lm-eval's stock ``vllm``; PIT uses
``pit`` (HF + batch patch), ChronoGPT uses ``chronogpt`` — all from
``evals.lmeval.backends``. The ``ftp_qwen_v4_*`` specs (``dd`` backend) are the DD
method and need 2 GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from lm_eval import simple_evaluate
from lm_eval.api.registry import get_model
from lm_eval.tasks import TaskManager, get_task_dict

import evals.lmeval.backends  # noqa: F401  -- registers pit/chronogpt/dd model types

# humaneval is scored by the `code_eval` metric, which executes model-generated code and
# refuses to run unless this is set; it must be set before lm-eval loads the task module.
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

TEMPORAL_DIR = str(Path(__file__).resolve().parent / "tasks" / "temporal")

# Main DD model P (Qwen3.5-27B). Override with a local path on boxes w/o the HF cache.
QWEN27B = os.environ.get("DD_QWEN27B_DIR", "Qwen/Qwen3.5-27B")

# Belt-and-suspenders <think> ban for Qwen3.5 models (sampler-level, merged into
# every task's generation_kwargs). enable_thinking=False already emits the empty
# <think></think> block via the template; this stops any stray <think> emission.
QWEN_GEN = {"bad_words": ["<think>"]}

# Qwen3.5 native sampling for the plain (non-DD) Qwen3.5 baselines — the card's "thinking
# mode, general tasks" preset, applied to EVERY task (utility + temporal leak@k) for both the
# reasoning-ON and reasoning-OFF Qwen3.5 27B/2B runs. We set it explicitly (not via HF's
# generation_config auto-load) because lm-eval normalizes temperature back to 0/greedy unless
# do_sample=True is present. The point-in-time baselines (talkie/pit/chrono) ship no sampling
# defaults, so they stay at their HF default (greedy on utility) — see run_task. The v4 DD
# specs use this same preset so DD vs qwen3_5_27b differ ONLY in the intervention.
QWEN_SAMPLING = {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
                 "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}

# Generative utility tasks (DD only affects generation, so MC tasks run generatively).
# humaneval_instruct is execution-scored (pass@1) and runs model-generated code.
UTILITY_TASKS = ["mmlu_pro", "mmlu_redux_generative", "ifeval", "gpqa_diamond_cot_zeroshot",
                 "humaneval_instruct"]
# streamingqa = forget-set leak@k; in TEMPORAL_TASKS so --temporal-n sets its `repeats`
# and it runs offline from the local parquet. Not in the default "temporal" 3-task sweep
# UI sugar below — run it explicitly with --tasks streamingqa.
# streamingqa_bradfordsystemprompt = the advisor's prompt-format variant (date-bearing
# system prompt, bare-question user turn); same data/judge, own results dir.
TEMPORAL_TASKS = ["ma", "pharma", "covid", "streamingqa", "streamingqa_bradfordsystemprompt"]
# Default leak@k repeats per temporal task (the canonical sweep config; --temporal-n
# overrides ALL temporal tasks uniformly if given). ma/covid single-shot (leak@1),
# pharma leak@4. streamingqa is ALWAYS single-shot now (repeats: 1) — it's judged (gemma
# in-job), not leak@k, and 8 reps made the full-set think runs take ~day-scale for no gain.
TEMPORAL_N = {"ma": 1, "covid": 1, "pharma": 4, "streamingqa": 1,
              "streamingqa_bradfordsystemprompt": 1}
# Per-CATEGORY doc cap for the heavy utility tasks. mmlu_pro (14 categories) and mmlu_redux
# (57 subjects) are multi-subtask GROUPS, so lm-eval's int `limit` caps EACH subtask (the
# first N of every category): 30 -> mmlu_pro ~420 docs, mmlu_redux ~1710. Keeps the long-CoT
# mmlu_pro fast under DD; identical first-30 across arms so the dose-response stays comparable.
# Other tasks (gpqa/ifeval/humaneval/ma/pharma/covid) run full (fall back to args.limit).
TASK_LIMITS = {"mmlu_pro": 30, "mmlu_redux_generative": 30}

# Per-model config. Fewshot is left at each task's lm-eval default (mmlu_pro = 5-shot
# CoT; redux/ifeval/gpqa = 0). `util_max_gen_toks` is the only context cap: mmlu_pro
# defaults to 2048 generation tokens, which can't coexist with a 5-shot prompt in a
# 2048-ctx model, so we shrink the generation budget for the small-context models.
# alphas: DD arms from one engine load; (None,) = a plain model (single run).

# Shared dd-backend engine args for the Qwen3.5-27B DD configs (aux pair set per-spec).
# TWO-GPU deployment: P (+KV) on cuda:0, the aux pair on cuda:1 so it never competes with
# P for memory — submit with 2 GPUs (tools/pythia/submit_eval.sh --gpus-per-job 2). P is
# bf16 (not FP8: flashinfer's fp8_blockscale cubin isn't fetchable on offline nodes).
# cudagraph capture sizes trimmed to 5 (vs ~51 defaults) because each pre-capture warmup
# pass runs the 27B + the cross-device DD aux — cuts engine init ~8x.
_DD_QWEN_ARGS = {
    "pretrained": QWEN27B,
    "aux_device": "cuda:1", "compile_aux": True,
    "prewarm": int(os.environ.get("DD_PREWARM", "256")),  # lower on 80GB GPUs (aux OOMs cuda:1)
    # max_model_len matches qwen3_5_27b: full prompt + the 16384-tok utility CoT budget
    # untruncated. The aux stays cheap via its dd_window sliding window (engine.py), so DD
    # does full-context utility while the 3B aux never pays for the long context.
    "max_model_len": 20480, "gpu_memory_utilization": 0.93,
    "max_num_seqs": int(os.environ.get("DD_MAX_NUM_SEQS", "256")),
    "dd_window": 2048, "dtype": "bfloat16",
    "compilation_config": {"cudagraph_capture_sizes": [16, 32, 64, 128, 256]},
    "trust_remote_code": True, "enable_thinking": False,
    "limit_mm_per_prompt": {"image": 0, "video": 0},
}

MODELS: dict[str, dict] = {
    "qwen3_5_2b": {
        "backend": "vllm",
        # max_model_len 8192: at 4096 the 5-shot mmlu_pro prompt (~2512) + 2048 gen overflowed
        # and the prompt was left-truncated (lost few-shot examples). 8192 fits both untruncated.
        "args": {"pretrained": "Qwen/Qwen3.5-2B", "max_model_len": 20480,
                 "gpu_memory_utilization": 0.85, "dtype": "bfloat16", "trust_remote_code": True,
                 "enable_thinking": False, "limit_mm_per_prompt": {"image": 0, "video": 0}},
        "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
        "util_max_gen_toks": 16384,  # all utility at the think-level budget so long CoT (gpqa,
        # mmlu_pro) isn't truncated; temporal stays 4096 via gen_kwargs (natural len ~1k).
    },
    # Reasoning-ON Qwen3.5-2B: enable_thinking emits a real <think> block, so we must NOT
    # ban <think> (drop QWEN_GEN). 8192-token budget on EVERY task (util via util_max_gen_toks,
    # temporal via gen_kwargs max_gen_toks override) so CoT isn't truncated; bigger ctx for it.
    "qwen3_5_2b_think": {
        "backend": "vllm",
        "args": {"pretrained": "Qwen/Qwen3.5-2B", "max_model_len": 20480,
                 "gpu_memory_utilization": 0.85, "dtype": "bfloat16", "trust_remote_code": True,
                 "enable_thinking": True, "think_end_token": "</think>",
                 "limit_mm_per_prompt": {"image": 0, "video": 0}},
        "gen_kwargs": {**QWEN_SAMPLING, "max_gen_toks": 16384}, "util_max_gen_toks": 16384,
    },
    # Plain bf16 Qwen3.5-27B base (no DD) -- the unsteered baseline that matches the DD
    # P-model (same 27B, dtype, gen). 1 GPU. FP8 is broken on pythia's offline nodes
    # (flashinfer fp8_blockscale !cubin), so the base runs bf16 like the DD P. The 27B
    # is a GDN hybrid -> same gdn warmup; cudagraph sizes trimmed to 5 (see gdn doc).
    "qwen3_5_27b": {
        "backend": "vllm",
        # max_model_len 8192 so utility fits the full prompt + 4096-tok CoT untruncated
        # (mmlu_pro 5-shot <=2512 tok, gpqa 0-shot <=526). At the old 2048 the prompt was
        # truncated to empty -> "decoder prompt cannot be empty" and every utility task died.
        # Temporal gen is short (<=768) so its numbers are identical to the 2048-ctx runs.
        "args": {"pretrained": QWEN27B, "max_model_len": 20480,
                 "gpu_memory_utilization": 0.9, "max_num_seqs": 256,
                 "dtype": "bfloat16", "trust_remote_code": True, "enable_thinking": False,
                 "compilation_config": {"cudagraph_capture_sizes": [16, 32, 64, 128, 256]},
                 "limit_mm_per_prompt": {"image": 0, "video": 0}},
        "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
        "util_max_gen_toks": 16384,  # all utility at the think-level budget so long CoT (gpqa,
        # mmlu_pro) isn't truncated; temporal stays 4096 via gen_kwargs (natural len ~1k).
    },
    # Reasoning-ON Qwen3.5-27B: enable_thinking emits a real <think> block, so do NOT ban
    # <think> (drop QWEN_GEN). 8192-token budget on every task (util + temporal) for full CoT;
    # max_model_len 12288 fits prompt + 8192 gen. Single GPU (no DD); same gdn cudagraph trim.
    "qwen3_5_27b_think": {
        "backend": "vllm",
        "args": {"pretrained": QWEN27B, "max_model_len": 20480,
                 "gpu_memory_utilization": 0.9, "max_num_seqs": 256,
                 "dtype": "bfloat16", "trust_remote_code": True, "enable_thinking": True,
                 "think_end_token": "</think>",
                 "compilation_config": {"cudagraph_capture_sizes": [16, 32, 64, 128, 256]},
                 "limit_mm_per_prompt": {"image": 0, "video": 0}},
        "gen_kwargs": {**QWEN_SAMPLING, "max_gen_toks": 16384}, "util_max_gen_toks": 16384,
    },
    # PIT model card claims a 2048 context window (matches the aux models -> fits 5-shot).
    "pit_4b_2015": {
        "backend": "pit", "args": {"pretrained": "Diamegs/PIT-4B-FT-201511", "max_length": 2048},
        "util_max_gen_toks": 1536,  # ~ctx window (2048) minus prompt reserve
    },
    "chrono_gpt_2015": {
        "backend": "chronogpt",
        "args": {"pretrained": "manelalab/chrono-gpt-instruct-v1-20151231", "max_length": 1792},
        "util_max_gen_toks": 1280,  # ~ctx window (1792) minus prompt reserve
    },
    # talkie-1930: a bespoke 13B GPT trained on pre-1931 text (1930 knowledge cutoff).
    # awilliamson/talkie-1930-13b-it-vllm repackages it as a custom-arch HF model
    # (TalkieForCausalLM + its own chat template), so we run it on the stock vllm backend
    # via the transformers fallback (model_impl="transformers" + trust_remote_code). bf16
    # only (fp8 is broken on this arch); native 2048 ctx fits 5-shot. The temporal tasks
    # already sample at T=0.7 (the card warns greedy loops, so greedy utility runs may
    # want do_sample too — left at the task defaults for now).
    "talkie_1930_13b": {
        "backend": "vllm",
        # custom-arch (TalkieForCausalLM) needs trust_remote_code; the XET HF-cache doesn't
        # materialize the remote .py as a plain blob, so point at a node-local FLAT dir
        # (DD_TALKIE_DIR, staged from bll01:/data/lab/talkie-flat) where the .py sits beside config.
        "args": {"pretrained": os.environ.get("DD_TALKIE_DIR",
                                              "awilliamson/talkie-1930-13b-it-vllm"),
                 "model_impl": "transformers", "max_model_len": 2048,
                 "gpu_memory_utilization": 0.85, "dtype": "bfloat16",
                 "trust_remote_code": True},
        "util_max_gen_toks": 1536,  # ~ctx window (2048) minus prompt reserve
    },
}

# Layer-48 Qwen-Scope SAE (W80K-L0_50) for feature steering on the main model P. Points at a
# flat dir holding layer48.sae.pt (node-staged via DD_QWEN_SAE_DIR; defaults to the bll01 copy).
_SAE_DIR = os.environ.get("DD_QWEN_SAE_DIR", "/data/lab/dd-steer-saes/L48")
# SAE feature steering on the main model P (used by the v4 serving-parity spec).
from ftp.serve import SteerArgs  # noqa: E402

# v4 DD: 32K-context FlexLlama flexdoc DUAL-MODE aux pairs (ONE model SFT'd on dolly+tulu
# x think+nothink; per-row prompt mode) — the v4 LR x alpha streamingqa heatmap. NOTHINK
# eval only (streamingqa is always a nothink evaluation). "lr0" = the UN-SFT'd v4 bases
# (config-patched sliding_window 512->32768), testing whether NO SFT beats every LR.
# Run with DD_STREAMINGQA_PER_BUCKET=1000 (submit_eval.sh --sqa-per-bucket 1000) so all
# cells score the same 1000 forget + 1000 retain questions. No steering.
# Sampling/budgets are IDENTICAL to the plain `qwen3_5_27b` baseline (temp-1.0 QWEN_SAMPLING
# preset on every task, 4096-tok temporal gen, 16384 utility budget, 20480 ctx) so DD vs
# base differ only in the intervention.
_V4_ROOT = os.environ.get("DD_V4_ROOT", "/data/lab/dd-aux-hf/v4")
_V4_BASES = os.environ.get("DD_V4_BASES", "/data/lab/dd-aux-hf/v4_bases")
_v4_pairs = {"lr0": (f"{_V4_BASES}/le2025_3b_v4", f"{_V4_BASES}/le2015_3b_v4")}
for _lr in ("5e-6", "1e-5", "2e-5", "5e-5"):
    _v4_pairs[f"lr{_lr}"] = (f"{_V4_ROOT}/le2025_3b_v4/lr{_lr}_dual/final",
                             f"{_V4_ROOT}/le2015_3b_v4/lr{_lr}_dual/final")
for _tag, (_p4, _q4) in _v4_pairs.items():
    MODELS[f"ftp_qwen_v4_{_tag}"] = {
        "backend": "dd",
        "args": {**_DD_QWEN_ARGS, "aux_p": _p4, "aux_q": _q4, "fuse_pin": True},
        "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
        "util_max_gen_toks": 16384,
        "alphas": [1.25, 1.5, 1.75],
    }
    # Steered variant for the utility+temporal LR-selection sweep: DD plus the
    # PRODUCTION dual-feature steering (the website "Ours" config) — L48:28961 @
    # clamp 22.5 + L27:24365 @ clamp 15 (both .sae.pt files live in the one L48
    # flat dir), at the "Ours" alpha 1.375. streamingqa runs UNSTEERED via the
    # entries above. THINKING OFF; the _steer_think twin below is the reasoning-ON
    # arm (qwen3_5_27b_think budgets, no <think> ban) probing whether DD+steering
    # works under thinking at all and whether aux LR moves it.
    MODELS[f"ftp_qwen_v4_{_tag}_steer"] = {
        "backend": "dd",
        "args": {**_DD_QWEN_ARGS, "aux_p": _p4, "aux_q": _q4, "fuse_pin": True,
                 "steer": SteerArgs(triples=[(48, 28961, 22.5), (27, 24365, 15.0)],
                                    family="topk", sae_dir=_SAE_DIR)},
        "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
        "util_max_gen_toks": 16384,
        "alphas": [1.375],
    }
    MODELS[f"ftp_qwen_v4_{_tag}_steer_think"] = {
        "backend": "dd",
        "args": {**_DD_QWEN_ARGS, "aux_p": _p4, "aux_q": _q4, "fuse_pin": True,
                 "enable_thinking": True, "think_end_token": "</think>",
                 "steer": SteerArgs(triples=[(48, 28961, 22.5), (27, 24365, 15.0)],
                                    family="topk", sae_dir=_SAE_DIR)},
        "gen_kwargs": {**QWEN_SAMPLING, "max_gen_toks": 16384},
        "util_max_gen_toks": 16384,
        "alphas": [1.375],
    }

# v4 aux "little models" evaluated STANDALONE (no DD): each era x LR (lr0 = the raw
# patched base) on the stock vllm backend via the transformers fallback (FlexLlama is
# a custom arch; with sliding_window==32768 every hybrid layer acts full-attention at
# these context lengths, so vLLM's attention is semantically exact). Sampling/budgets
# mirror qwen3_5_2b (nothink, temp-1.0 preset + <think> ban) / qwen3_5_2b_think
# (reasoning ON, 16384 budget, no ban). Dual-mode SFT means ONE model serves both.
_V4_LITTLE_ARGS = {"max_model_len": 20480, "gpu_memory_utilization": 0.85,
                   "dtype": "bfloat16", "trust_remote_code": True,
                   "model_impl": "transformers",
                   # custom-arch remote code breaks vLLM's CUDA-graph capture
                   # ("operation not permitted when stream is capturing") — run eager,
                   # exactly like the v3.3 little-model entries did.
                   "enforce_eager": True,
                   "limit_mm_per_prompt": {"image": 0, "video": 0}}
for _tag, (_p4, _q4) in _v4_pairs.items():  # _p4 = le2025 (forget), _q4 = le2015 (retain)
    for _year, _path in (("2025", _p4), ("2015", _q4)):
        MODELS[f"v4_{_tag}_{_year}"] = {
            "backend": "vllm",
            "args": {**_V4_LITTLE_ARGS, "pretrained": _path, "enable_thinking": False},
            "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
            "util_max_gen_toks": 16384,
        }
        MODELS[f"v4_{_tag}_{_year}_think"] = {
            "backend": "vllm",
            "args": {**_V4_LITTLE_ARGS, "pretrained": _path,
                     "enable_thinking": True, "think_end_token": "</think>"},
            "gen_kwargs": {**QWEN_SAMPLING, "max_gen_toks": 16384},
            "util_max_gen_toks": 16384,
        }

# v5 aux "little models" (the retrained v4 replacements): 32K flexdoc bases on the custom
# MinistralDualRope arch (trust_remote_code; Gemma3-style dual rope — llama3-scaled global
# rope on the 5 full-attention layers, UNSCALED local rope on the sliding layers). Unlike
# v4 the bases are NOT config-patched: sliding_window=512 is the trained configuration, and
# the vLLM transformers fallback honors per-layer layer_types + sliding_window, so serving
# matches training. Standalone vllm entries per era x LR (lr0 = raw base). SFT is
# NOTHINK-ONLY (the v3.3 dolly+tulu nothink teacher file — thinking-mode SFT deferred
# to v6), so the _think entries are probe-only for the SFT'd arms; sampling/budgets
# mirror qwen3_5_2b / qwen3_5_2b_think.
_V5_ROOT = os.environ.get("DD_V5_ROOT", "/data/lab/dd-aux-hf/v5")
_V5_BASES = os.environ.get("DD_V5_BASES", "/data/lab/dd-aux-hf/v5_bases")
_v5_pairs = {"lr0": (f"{_V5_BASES}/le2025_3b_v5", f"{_V5_BASES}/le2015_3b_v5")}
for _lr in ("5e-6", "1e-5", "2e-5", "5e-5"):
    _v5_pairs[f"lr{_lr}"] = (f"{_V5_ROOT}/le2025_3b_v5/lr{_lr}_nothink/final",
                             f"{_V5_ROOT}/le2015_3b_v5/lr{_lr}_nothink/final")
for _tag, (_p5, _q5) in _v5_pairs.items():  # _p5 = le2025 (forget), _q5 = le2015 (retain)
    for _year, _path in (("2025", _p5), ("2015", _q5)):
        MODELS[f"v5_{_tag}_{_year}"] = {
            "backend": "vllm",
            "args": {**_V4_LITTLE_ARGS, "pretrained": _path, "enable_thinking": False},
            "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
            "util_max_gen_toks": 16384,
        }
        MODELS[f"v5_{_tag}_{_year}_think"] = {
            "backend": "vllm",
            "args": {**_V4_LITTLE_ARGS, "pretrained": _path,
                     "enable_thinking": True, "think_end_token": "</think>"},
            "gen_kwargs": {**QWEN_SAMPLING, "max_gen_toks": 16384},
            "util_max_gen_toks": 16384,
        }

# v6 aux models, named by SFT stage: `nosft` = raw cooldown bases (32K ctx; le2015 =
# v2_corrected, le2025 = v1_corrected upstream), `partialsft` = single128k-8b SFT at
# step 1250 (config already claims 128K max_position_embeddings), `fullsft` = pending.
# Same MinistralDualRope arch as v5 (trust_remote_code, dual rope, sliding_window=512
# as trained). Chat templates carry enable_thinking, but reasoning support is
# UNCONFIRMED — the standalone _think entries are the probe.
# `ftp_qwen_v6_<tag>` = DD streamingqa arm: NO steering, NOTHINK, alpha swept
# 1.25..1.75 in-process from one engine load.
_V6_ROOT = os.environ.get("DD_V6_ROOT", "/data/lab/dd-aux-hf/v6")
_v6_pairs = {
    "nosft": (f"{_V6_ROOT}/nosft/le2025_3b_v6", f"{_V6_ROOT}/nosft/le2015_3b_v6"),
    "partialsft": (f"{_V6_ROOT}/partialsft/le2025_3b_v6",
                   f"{_V6_ROOT}/partialsft/le2015_3b_v6"),
    # fullsft pair complete 2026-07-13 (both eras single128k-8b SFT run to completion).
    "fullsft": (f"{_V6_ROOT}/fullsft/le2025_3b_v6",
                f"{_V6_ROOT}/fullsft/le2015_3b_v6"),
}
for _tag, (_p6, _q6) in _v6_pairs.items():  # _p6 = le2025 (forget), _q6 = le2015 (retain)
    MODELS[f"ftp_qwen_v6_{_tag}"] = {
        "backend": "dd",
        "args": {**_DD_QWEN_ARGS, "aux_p": _p6, "aux_q": _q6, "fuse_pin": True},
        "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
        "util_max_gen_toks": 16384,
        "alphas": [1.25, 1.375, 1.5, 1.625, 1.75],
    }
    # Standalone "little" aux models, per era, nothink + think. MUST live inside the
    # _tag loop: a module-level version below only ever saw the LAST _tag ("fullsft"),
    # silently dropping the nosft/partialsft littles (2026-07-14 bug).
    for _year, _lpath in (("2025", _p6), ("2015", _q6)):
        MODELS[f"v6_{_tag}_{_year}"] = {
            "backend": "vllm",
            "args": {**_V4_LITTLE_ARGS, "pretrained": _lpath, "enable_thinking": False},
            "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
            "util_max_gen_toks": 16384,
        }
        MODELS[f"v6_{_tag}_{_year}_think"] = {
            "backend": "vllm",
            "args": {**_V4_LITTLE_ARGS, "pretrained": _lpath,
                     "enable_thinking": True, "think_end_token": "</think>"},
            "gen_kwargs": {**QWEN_SAMPLING, "max_gen_toks": 16384},
            "util_max_gen_toks": 16384,
        }

# GREEDY (temp-0) twin of the v6 partialsft aux LITTLE model, 2015 (le2015=retain) ONLY,
# NOTHINK — probe whether greedy decoding beats the temp-1.0 QWEN_SAMPLING preset on the
# standalone-aux benchmarks (2026-07-14 curiosity run). DISTINCT `_temp0` name so the
# temp-1.0 result (v6_partialsft_2015__base.json) is never overwritten. Greedy =
# do_sample off + temperature 0; the sampling knobs (top_p/top_k/min_p) and the
# presence/repetition penalties are DROPPED (meaningless or actively harmful at greedy — a
# large presence_penalty on argmax decoding blocks needed token reuse). <think> ban kept;
# same budgets (4096 temporal / 16384 utility) as the temp-1.0 twin.
MODELS["v6_partialsft_2015_temp0"] = {
    "backend": "vllm",
    "args": {**_V4_LITTLE_ARGS, "pretrained": _v6_pairs["partialsft"][1],
             "enable_thinking": False},
    "gen_kwargs": {**QWEN_GEN, "do_sample": False, "temperature": 0.0,
                   "max_gen_toks": 4096},
    "util_max_gen_toks": 16384,
}

# PIT-4B-FT-201511 prompted the way its own model card documents (<|user|>/<|assistant|>
# role markers, stop on <|end|>) rather than the Alpaca format the PIT repo's IFEval script
# uses. `pit_4b_2015` (Alpaca + greedy) produced every published PIT number and stays exactly
# as-is; these are DISTINCT model keys, so `results/<task>/pit_4b_2015__base.json` is never
# touched and the two can be compared side by side.
#
# Two arms, because format and decoding are separate variables and the existing results are
# greedy:
#   _chat        = card format + the card's recommended sampling (T=0.7, top_p=0.9) — this
#                  is "we ran it exactly as documented".
#   _chat_greedy = card format + greedy — decoding-matched to `pit_4b_2015`, so the diff
#                  isolates the prompt format alone. Greedy is HF's default, hence no
#                  gen_kwargs (mirrors how `pit_4b_2015` gets greedy).
# Same weights/context/budget as `pit_4b_2015` so nothing else moves.
#
# OUTCOME (2026-07-16, full suite; /data/lab/frontier-to-pit/results-pit-cardformat):
# the documented format is WORSE than the Alpaca format we ship on every task that carries
# any signal — MMLU-Redux 0.2205 -> 0.0012, GPQA 0.1616 -> 0.0000, IFEval 0.2421 -> 0.1257
# (card sampling; greedy lands the same) — and ties at the floor on MMLU-Pro and HumanEval,
# which Alpaca already bottoms out on at 0.0000.
# Under the role markers it emits bare <|assistant|>/<|user|> loops, so the answer
# extractors find nothing to score. _chat vs _chat_greedy agree within noise, so the format
# is doing this, not the decoding. Keeping the Alpaca numbers on the site is therefore the
# charitable choice, not a thumb on the scale. Full reasoning + the KV-cache equivalence
# check: see the PIT_CHAT_TEMPLATE comment in backends.py and documents/pit_issues.md.
# These specs are kept so the claim stays reproducible — do not delete them to "clean up".
_PIT_2015_ARGS = {"pretrained": "Diamegs/PIT-4B-FT-201511", "max_length": 2048}
MODELS["pit_4b_2015_chat"] = {
    "backend": "pit_chat", "args": dict(_PIT_2015_ARGS),
    "gen_kwargs": {"do_sample": True, "temperature": 0.7, "top_p": 0.9},
    "util_max_gen_toks": 1536,
}
MODELS["pit_4b_2015_chat_greedy"] = {
    "backend": "pit_chat", "args": dict(_PIT_2015_ARGS),
    "util_max_gen_toks": 1536,
}

# Production "Ours" eval config (kept standalone after the v6 steering grid was pruned):
# alpha=1.5, single feature L48:28961@10 (L27 off), rank OFF, NOTHINK, partialsft aux —
# the recipe run.py serves. Lets the full eval suite still be reproduced for the shipped
# config. Full-config scoring: streamingqa runs with the steering on, same as the sweep.
MODELS["ftp_v6lin_a1.5_L48c10_L27c0"] = {
    "backend": "dd",
    "args": {**_DD_QWEN_ARGS, "aux_p": _v6_pairs["partialsft"][0],
             "aux_q": _v6_pairs["partialsft"][1], "fuse_pin": True,
             "steer": SteerArgs(triples=[(48, 28961, 10.0)], family="topk",
                                sae_dir=_SAE_DIR)},
    "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
    "util_max_gen_toks": 16384,
    "alphas": [1.5],
}

# Think-mode DD dose-response sweep (NO steering, thinking ON, v6 partialsft aux):
# one entry PER alpha so results land as ftp_v6lin_nosteer_think_a<a>__alpha=<a>.json —
# the exact filenames plot_unlearning_think.py (website) reads for its alpha curve.
# Re-added 2026-07-16 after the think grid was pruned (the sqa results were deleted in
# the temporal final-artifact cleanup and need regenerating). The _v6guard_ twins are
# the same sweep with the live degeneration guard on (guard-vs-not on the dose-response,
# internal record — the website reads the unguarded stems).
for _a in (1.125, 1.25, 1.375, 1.5):
    for _guard in (False, True):
        _stem = "ftp_v6guard_nosteer_think_a" if _guard else "ftp_v6lin_nosteer_think_a"
        MODELS[f"{_stem}{_a:g}"] = {
            "backend": "dd",
            "args": {**_DD_QWEN_ARGS, "aux_p": _v6_pairs["partialsft"][0],
                     "aux_q": _v6_pairs["partialsft"][1], "fuse_pin": True,
                     **({"dd_guard": True} if _guard else {}),
                     "enable_thinking": True, "think_end_token": "</think>"},
            "gen_kwargs": {**QWEN_SAMPLING, "max_gen_toks": 16384},
            "util_max_gen_toks": 16384,
            "alphas": [_a],
        }

# Guarded twin of the INSTRUCT sqa dose-response sweep (ftp_qwen_v6_partialsft: pure DD,
# no steering, NOTHINK, alphas swept in-process from one engine load). Same grid, only
# `dd_guard` differs — guard-vs-not on the instruct dose-response (internal record).
MODELS["ftp_v6guard_nosteer"] = {
    "backend": "dd",
    "args": {**_DD_QWEN_ARGS, "aux_p": _v6_pairs["partialsft"][0],
             "aux_q": _v6_pairs["partialsft"][1], "fuse_pin": True, "dd_guard": True},
    "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
    "util_max_gen_toks": 16384,
    "alphas": [1.25, 1.375, 1.5, 1.625, 1.75],
}

# Thinking twin of the production config (production think candidate: same single
# feature, alpha=1.125, reasoning budgets, no <think> ban). Matches the pruned
# ftp_v6think_* grid's conventions: enable_thinking template + </think> strip on
# utility.
MODELS["ftp_v6lin_think_L48c10_L27c0"] = {
    "backend": "dd",
    "args": {**_DD_QWEN_ARGS, "aux_p": _v6_pairs["partialsft"][0],
             "aux_q": _v6_pairs["partialsft"][1], "fuse_pin": True,
             "enable_thinking": True, "think_end_token": "</think>",
             "steer": SteerArgs(triples=[(48, 28961, 10.0)], family="topk",
                                sae_dir=_SAE_DIR)},
    "gen_kwargs": {**QWEN_SAMPLING, "max_gen_toks": 16384},
    "util_max_gen_toks": 16384,
    "alphas": [1.125],
}

# GUARDED twins of the production config: the live degeneration guard (ftp.guard —
# Qwen3.5-2B judge on the aux GPU sweeping the batch every 25 engine steps; a tripped
# generation rewinds 50 tokens and resamples). Byte-identical to the production arm
# except `dd_guard`, so guard-vs-prod deltas are the guard alone. Motivation
# (2026-07-16): at alpha=1.5 NOTHINK, 45%/35% of ma/pharma generations are judged
# destroyed; suppress-only fusion fixed that but saturated ~2x above the two-sided
# leak floor (streamingqa alpha sweep to 4.0) — so keep the fusion, repair the
# collapse. The judge model must be reachable on the node (submit_eval stages
# Qwen/Qwen3.5-2B for ftp_v6guard_* arms; DD_GUARD_MODEL overrides).
MODELS["ftp_v6guard_a1.5_L48c10_L27c0"] = {
    "backend": "dd",
    "args": {**_DD_QWEN_ARGS, "aux_p": _v6_pairs["partialsft"][0],
             "aux_q": _v6_pairs["partialsft"][1], "fuse_pin": True,
             "dd_guard": True,
             "steer": SteerArgs(triples=[(48, 28961, 10.0)], family="topk",
                                sae_dir=_SAE_DIR)},
    "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
    "util_max_gen_toks": 16384,
    "alphas": [1.5],
}
# Guarded thinking twin (same single feature, alpha=1.125, reasoning budgets,
# no <think> ban).
MODELS["ftp_v6guard_think_L48c10_L27c0"] = {
    "backend": "dd",
    "args": {**_DD_QWEN_ARGS, "aux_p": _v6_pairs["partialsft"][0],
             "aux_q": _v6_pairs["partialsft"][1], "fuse_pin": True,
             "dd_guard": True,
             "enable_thinking": True, "think_end_token": "</think>",
             "steer": SteerArgs(triples=[(48, 28961, 10.0)], family="topk",
                                sae_dir=_SAE_DIR)},
    "gen_kwargs": {**QWEN_SAMPLING, "max_gen_toks": 16384},
    "util_max_gen_toks": 16384,
    "alphas": [1.125],
}

# Guard hyperparameter sweep (2026-07-16, ma+pharma): the two residual failure
# modes map onto two knobs. NONWORDS garble slips the zlib pre-gate — a half-
# prose/half-garble window compresses ~1.3-1.4, under the 1.6 gate — so lower
# gate_ratio shows the judge more windows (1.0 ~= gate off for prose-like text).
# Long rescue sessions accumulate loop debris and run to the token budget, so
# lower max_rounds gives up earlier and returns the clean accepted prefix.
# One factor at a time plus both-lever combos; all else = the production guard.
# OUTCOME: g13r10 won (ma text-destroyed 15.5->5.4%, pharma 26->11.2%, leak
# unchanged) and its values are now the GuardConfig DEFAULTS (gate 1.3, rounds
# 10) — so the plain ftp_v6guard_ production arms already run the winner, and
# re-running these sweep cells would no longer reproduce the original grid
# (partial kwargs inherit the new defaults, not the old 1.6/20 baseline).
for _tag, _gk in {
    "g13":    {"gate_ratio": 1.3},
    "g10":    {"gate_ratio": 1.0},
    "r10":    {"max_rounds": 10},
    "r5":     {"max_rounds": 5},
    "g13r10": {"gate_ratio": 1.3, "max_rounds": 10},
    "g10r5":  {"gate_ratio": 1.0, "max_rounds": 5},
}.items():
    MODELS[f"ftp_v6guard_{_tag}_a1.5_L48c10_L27c0"] = {
        "backend": "dd",
        "args": {**_DD_QWEN_ARGS, "aux_p": _v6_pairs["partialsft"][0],
                 "aux_q": _v6_pairs["partialsft"][1], "fuse_pin": True,
                 "dd_guard": True, "dd_guard_kwargs": dict(_gk),
                 "steer": SteerArgs(triples=[(48, 28961, 10.0)], family="topk",
                                    sae_dir=_SAE_DIR)},
        "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
        "util_max_gen_toks": 16384,
        "alphas": [1.5],
    }

# v3.3.1 = the chosen v3.3 "Ours" config RE-RUN under the v4 sampling policy (temp-1.0
# preset, matched budgets) so the v3.3-vs-v4 aux comparison is decoding-matched — the
# original v3.3 diamonds ran T=0.7/rep-pen 1.1/2048-tok utility gen. Aux pair = the
# v3.3 lr2e-5_nomix dense Llamas. `ftp_qwen_v331` (pure DD) = the streamingqa arm
# (alpha 1.5); `_ours` adds the production dual-feature steering for the 8-task sweep
# (alpha 1.375).
_V33_ROOT = os.environ.get("DD_V33_ROOT", "/data/lab/dd-aux-hf/v3.3")
_V331_P = f"{_V33_ROOT}/le2025_3b_v3/lr2e-5_nomix/final"
_V331_Q = f"{_V33_ROOT}/le2015_3b_v3/lr2e-5_nomix/final"
MODELS["ftp_qwen_v331"] = {
    "backend": "dd",
    "args": {**_DD_QWEN_ARGS, "aux_p": _V331_P, "aux_q": _V331_Q, "fuse_pin": True},
    "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
    "util_max_gen_toks": 16384,
    "alphas": [1.5],
}
MODELS["ftp_qwen_v331_ours"] = {
    "backend": "dd",
    # PRODUCTION (2026-07-05, steering rescue grid): L48:28961 @ 11.25 ONLY — L27
    # dropped, both clamps halved from the original 22.5/15+15 recipe that wrecked
    # GPQA (0.859 -> 0.48; this config: 0.788, util-agg 0.678 -> 0.814, LAB%
    # 2.1 -> 3.5). Canonical results live under ftp_v331_a1.375_L48c11.25_L27c0
    # (the grid cell that IS this config); this entry is the serving recipe.
    "args": {**_DD_QWEN_ARGS, "aux_p": _V331_P, "aux_q": _V331_Q, "fuse_pin": True,
             "steer": SteerArgs(triples=[(48, 28961, 11.25)],
                                family="topk", sae_dir=_SAE_DIR)},
    "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
    "util_max_gen_toks": 16384,
    "alphas": [1.375],
}
# The v3.3.1 production recipe with the aux pair swapped for the RAW (no-SFT) v5
# dual-rope bases: probes whether the un-instruction-tuned v5 bases can already serve
# as DD aux at the production operating point. `ftp_qwen_v5_lr0` (pure DD, NO steering,
# alpha swept 1.25/1.375/1.5 in-process) = the streamingqa arm, mirroring ftp_qwen_v331;
# `_ours` = the steered 8-task arm mirroring ftp_qwen_v331_ours (alpha 1.375, fuse_pin,
# L48:28961 @ 11.25).
MODELS["ftp_qwen_v5_lr0"] = {
    "backend": "dd",
    "args": {**_DD_QWEN_ARGS, "aux_p": _v5_pairs["lr0"][0], "aux_q": _v5_pairs["lr0"][1],
             "fuse_pin": True},
    "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
    "util_max_gen_toks": 16384,
    "alphas": [1.25, 1.375, 1.5],
}
MODELS["ftp_qwen_v5_lr0_ours"] = {
    "backend": "dd",
    "args": {**_DD_QWEN_ARGS, "aux_p": _v5_pairs["lr0"][0], "aux_q": _v5_pairs["lr0"][1],
             "fuse_pin": True,
             "steer": SteerArgs(triples=[(48, 28961, 11.25)],
                                family="topk", sae_dir=_SAE_DIR)},
    "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
    "util_max_gen_toks": 16384,
    "alphas": [1.375],
}

# v5 SFT'd aux pair at the CHOSEN LR (2e-5 nothink, picked from the le2015 utility
# sweep) as DD aux. `ftp_qwen_v5_lr2e5` (pure DD, NO steering, alphas 1.25/1.375/1.5
# in-process) = the streamingqa arm, mirroring ftp_qwen_v5_lr0; the steered 8-task
# arms are the production-neighborhood grid below.
MODELS["ftp_qwen_v5_lr2e5"] = {
    "backend": "dd",
    "args": {**_DD_QWEN_ARGS, "aux_p": _v5_pairs["lr2e-5"][0],
             "aux_q": _v5_pairs["lr2e-5"][1], "fuse_pin": True},
    "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
    "util_max_gen_toks": 16384,
    "alphas": [1.25, 1.375, 1.5],
}
# Production-neighborhood grid on the v5 lr2e-5 pair: the production cell (alpha
# 1.375, L48:28961 @ 11.25, L27 off) + one-knob perturbations — alpha +-0.125,
# L48 clamp +-5, and L27:24365 re-enabled at half its old production strength
# (7.5). GRAPHED single-cell entries (clamps baked into the triples, no
# steer_sweep) exactly like the _RESCUE_CELLS fan-out; one 2-GPU job per
# (cell x task).
_V5_PROD_CELLS = [
    (1.375, 11.25, 0.0),   # production config on the v5 pair
    (1.25,  11.25, 0.0),   # alpha -0.125
    (1.5,   11.25, 0.0),   # alpha +0.125
    (1.375,  6.25, 0.0),   # L48 -5
    (1.375, 16.25, 0.0),   # L48 +5
    (1.375, 11.25, 7.5),   # + L27 @ 7.5
]
for _a, _l48, _l27 in _V5_PROD_CELLS:
    _triples = [t for t in [(27, 24365, _l27), (48, 28961, _l48)] if t[2] != 0.0]
    MODELS[f"ftp_v5lr2e5_a{_a:g}_L48c{_l48:g}_L27c{_l27:g}"] = {
        "backend": "dd",
        "args": {**_DD_QWEN_ARGS, "aux_p": _v5_pairs["lr2e-5"][0],
                 "aux_q": _v5_pairs["lr2e-5"][1], "fuse_pin": True,
                 "steer": SteerArgs(triples=_triples, family="topk",
                                    sae_dir=_SAE_DIR)},
        "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
        "util_max_gen_toks": 16384,
        "alphas": [_a],
    }

# v3.3 aux little model evaluated STANDALONE under the temp-1.0 policy (the website's
# "Aux 2015" bars; keeps the LEGACY key so results land as v3.3_lr2e-5_nomix_2015__base.json
# and frontier-to-pit-website/common.py resolves unchanged). Dense 2048-ctx LlamaForCausalLM:
# vLLM-native, prompt-truncating on the long 5-shot utility prompts exactly like the
# original v3.3 little runs; 512-tok utility budget (~ctx minus prompt reserve).
MODELS["v3.3_lr2e-5_nomix_2015"] = {
    "backend": "vllm",
    "args": {"pretrained": _V331_Q, "max_model_len": 2048,
             "gpu_memory_utilization": 0.85, "dtype": "bfloat16",
             "trust_remote_code": True, "enable_thinking": False, "enforce_eager": True,
             "limit_mm_per_prompt": {"image": 0, "video": 0}},
    "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING},
    "util_max_gen_toks": 512,
}

# Steering-strength rescue grid (2026-07-05): the chosen "Ours" steering wrecks GPQA
# (86 -> 48), so sweep BOTH feature clamps from 0 up to production at alpha {1.25, 1.375}.
# Each entry sweeps ONE feature in-process (steer_clamps re-clamps the LAST triple via
# set_steer_clamp; steer_sweep=True forces the eager hook route) while the other feature
# is HELD at a level that varies across entries: 5 job-shapes x 2 alphas = 10 jobs.
# L48 sweep grid {0,7.5,15,22.5} x L27 held {0,7.5,15}; L27 sweep grid {0,5,10,15} x
# L48 held {0,22.5}. clamp 0 = that feature's hook dropped. GPQA is the sentinel metric.
for _a in (1.25, 1.375):
    for _l27fix in (0.0, 7.5, 15.0):
        MODELS[f"ftp_v331_a{_a:g}_L27c{_l27fix:g}_L48sweep"] = {
            "backend": "dd",
            "args": {**_DD_QWEN_ARGS, "aux_p": _V331_P, "aux_q": _V331_Q,
                     "fuse_pin": True, "steer_sweep": True,
                     "steer": SteerArgs(triples=[(27, 24365, _l27fix), (48, 28961, 22.5)],
                                        family="topk", sae_dir=_SAE_DIR)},
            "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
            "util_max_gen_toks": 16384,
            "alphas": [_a],
            "steer_clamps": [0, 7.5, 15, 22.5],   # sweeps L48 (the last triple)
        }
    for _l48fix in (0.0, 22.5):
        MODELS[f"ftp_v331_a{_a:g}_L48c{_l48fix:g}_L27sweep"] = {
            "backend": "dd",
            "args": {**_DD_QWEN_ARGS, "aux_p": _V331_P, "aux_q": _V331_Q,
                     "fuse_pin": True, "steer_sweep": True,
                     "steer": SteerArgs(triples=[(48, 28961, _l48fix), (27, 24365, 15.0)],
                                        family="topk", sae_dir=_SAE_DIR)},
            "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
            "util_max_gen_toks": 16384,
            "alphas": [_a],
            "steer_clamps": [0, 5, 10, 15],       # sweeps L27 (the last triple)
        }

# GRAPHED single-cell fan-out of the same rescue grid: steer_sweep's in-process clamp
# swaps force enforce_eager on ALL of P (measured 180-500 tok/s vs the pre-capture
# route's graphed speed; a graphed steered GPQA job = ~31 min total incl. warm-cache
# build, an eager sweep = 1-2.5 h PER CELL). One entry per (alpha, L48, L27) cell so
# jobs backfill nodes independently. Zero-clamp triples are DROPPED to match
# set_steer_clamp's sweep semantics (clamp 0 = hook removed; both 0 = pure DD, no
# steering worker at all). Results land as <name>__alpha=<a>.json (no _c suffix).
_RESCUE_CELLS = sorted(
    (l48, l27)
    for l48 in (0.0, 5.625, 11.25, 16.875, 22.5)  # {0, .25, .5, .75, 1} x production 22.5
    for l27 in (0.0, 3.75, 7.5, 11.25, 15.0))     # {0, .25, .5, .75, 1} x production 15
for _a in (1.25, 1.375):
    for _l48, _l27 in _RESCUE_CELLS:
        _triples = [t for t in [(27, 24365, _l27), (48, 28961, _l48)] if t[2] != 0.0]
        MODELS[f"ftp_v331_a{_a:g}_L48c{_l48:g}_L27c{_l27:g}"] = {
            "backend": "dd",
            "args": {**_DD_QWEN_ARGS, "aux_p": _V331_P, "aux_q": _V331_Q,
                     "fuse_pin": True,
                     **({"steer": SteerArgs(triples=_triples, family="topk",
                                            sae_dir=_SAE_DIR)} if _triples else {})},
            "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
            "util_max_gen_toks": 16384,
            "alphas": [_a],
        }

# Serving-parity check: the FULL production path — DD (graphed aux) + PRE-CAPTURE
# steering — at NEAR-ZERO strength (alpha 0.01, clamps 0.01) must reproduce the plain
# `qwen3_5_27b` baseline within sampling noise; any real gap is a serving-stack bug,
# not an intervention. Mirrors qwen3_5_27b's sampling and budgets exactly (temp-1.0
# preset, 4096-tok gen, 16384 utility budget, 20480 ctx); v4 BASE pair; L48+L27
# steering (the production features) at clamp 0.01.
MODELS["ftp_qwen_v4_micro"] = {
    "backend": "dd",
    "args": {**_DD_QWEN_ARGS, "aux_p": _v4_pairs["lr0"][0], "aux_q": _v4_pairs["lr0"][1],
             "fuse_pin": True,
             "steer": SteerArgs(triples=[(48, 28961, 0.01), (27, 24365, 0.01)],
                                family="topk", sae_dir=_SAE_DIR)},
    "gen_kwargs": {**QWEN_GEN, **QWEN_SAMPLING, "max_gen_toks": 4096},
    "util_max_gen_toks": 16384,
    "alphas": [0.01],
}


def resolve_tasks(spec_str: str) -> list[str]:
    # "temporal" = the original 3-task leakage sweep; streamingqa (leak@k) is run
    # explicitly via --tasks streamingqa (it's in TEMPORAL_TASKS only for the
    # --temporal-n / offline handling).
    temporal_sweep = ["ma", "pharma", "covid"]
    if spec_str == "all":
        return UTILITY_TASKS + temporal_sweep
    if spec_str == "utility":
        return list(UTILITY_TASKS)
    if spec_str == "temporal":
        return list(temporal_sweep)
    return [t.strip() for t in spec_str.split(",") if t.strip()]


def build_model(spec: dict, overrides: dict):
    cls = get_model(spec["backend"])
    args = {**spec["args"], **{k: v for k, v in overrides.items() if v is not None}}
    return cls(**args)


def run_task(inst, spec, task_name, *, tm, args, alpha) -> None:
    # alpha: None = plain model ("base"); float = linear DD arm; ("rank", k) =
    # rank-DD arm; ("linrank", a, k) = composed linear+rank arm.
    if alpha is None:
        arm = "base"
    elif isinstance(alpha, tuple) and alpha[0] == "rank":
        arm = f"rank_k={alpha[1]}"
    elif isinstance(alpha, tuple):
        arm = f"alpha={alpha[1]:g}_rank_k={alpha[2]}"
    else:
        arm = f"alpha={alpha}"
    # Organize by model -> task -> arm (alpha) so multiple models/strengths don't collide
    # (we'll sweep other base models after Qwen3.5-27B). Data-parallel shards (DD_NUM_CHUNKS
    # jobs, one GPU each) get a per-chunk filename and are merged downstream for leak@k.
    nc = int(os.environ.get("DD_NUM_CHUNKS", "1"))
    ci = int(os.environ.get("DD_CHUNK_ID", "0"))
    chunk = f"__chunk{ci}of{nc}" if nc > 1 else ""
    out_path = Path(args.out) / args.model / task_name / f"{arm}{chunk}.json"
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {out_path} exists", flush=True)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    is_temporal = task_name in TEMPORAL_TASKS
    # Leave fewshot at each task's lm-eval default (None): mmlu_pro = 5-shot CoT;
    # redux/ifeval/gpqa = 0; the temporal YAMLs pin num_fewshot: 0.
    num_fewshot = None
    gen_kwargs = dict(spec.get("gen_kwargs", {}))
    if not is_temporal and spec.get("util_max_gen_toks"):
        gen_kwargs["max_gen_toks"] = spec["util_max_gen_toks"]
    # GPQA cot_zeroshot elicits long step-by-step reasoning that the general util cap truncates
    # (no-think Qwen @4096 loses ~1/3 of legit answers); a spec can set a bigger GPQA-only budget.
    if task_name == "gpqa_diamond_cot_zeroshot" and spec.get("gpqa_max_gen_toks"):
        gen_kwargs["max_gen_toks"] = spec["gpqa_max_gen_toks"]

    # Set the temporal task's `repeats`: explicit --temporal-n wins; otherwise fall back to
    # the per-task TEMPORAL_N default (ma/covid/streamingqa=1, pharma=4). Load the Task,
    # set it, pass the object.
    tasks_arg: list = [task_name]
    temporal_n = args.temporal_n if args.temporal_n is not None else TEMPORAL_N.get(task_name)
    if is_temporal and temporal_n is not None:
        td = get_task_dict([task_name], tm)
        for t in td.values():
            t.set_config(key="repeats", value=temporal_n)
        tasks_arg = list(td.values())

    # LAB tasks keep the RAW generation (thinking chain + final response): disable the
    # harness's model-level </think> strip for temporal tasks, so utils._think_spans can
    # score mentions_per_1k on the think and final spans separately. Utility tasks keep
    # the strip — their answer extraction wants the final span only.
    prev_think_tok = getattr(inst, "think_end_token", None)
    if is_temporal and prev_think_tok:
        inst.think_end_token = None

    t0 = time.time()
    print(f"[run] {task_name} / {args.model} / {arm} (nfs={num_fewshot}) ...", flush=True)
    try:
        res = simple_evaluate(
            model=inst,
            tasks=tasks_arg,
            num_fewshot=num_fewshot,
            gen_kwargs=(gen_kwargs or None),
            limit=TASK_LIMITS.get(task_name, args.limit),
            apply_chat_template=True,
            fewshot_as_multiturn=False,
            task_manager=tm,
            log_samples=True,
            write_out=False,
            confirm_run_unsafe_code=True,  # humaneval executes model-generated code (pass@1)
            random_seed=args.seed,
            numpy_random_seed=args.seed,
            torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )
    finally:
        if is_temporal and prev_think_tok:
            inst.think_end_token = prev_think_tok
    dt = time.time() - t0
    if isinstance(alpha, tuple):
        _arm_alpha = alpha[1] if alpha[0] == "linrank" else None
        _arm_k = alpha[-1]
    else:
        _arm_alpha, _arm_k = alpha, 0
    res["dd_arm"] = {
        "model": args.model, "arm": arm,
        "alpha": _arm_alpha, "rank_k": _arm_k,
        "elapsed_s": round(dt, 1),
    }
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"[done] {out_path}  ({dt:.0f}s)  {res.get('results', {})}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--tasks", required=True, help="utility|temporal|all or a comma list")
    ap.add_argument("--limit", type=int, default=None, help="examples per task (testing)")
    ap.add_argument("--temporal-n", type=int, default=None, help="override temporal `repeats`")
    ap.add_argument("--alphas", type=str, default=None,
                    help="comma list of DD alphas to run; overrides the model spec's alphas "
                         "(e.g. '1.5' to run a single arm). Ignored for non-DD models.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="results", help="results root dir")
    ap.add_argument("--overwrite", action="store_true")
    # Engine overrides (forwarded to the backend constructor; None = use spec default).
    ap.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    ap.add_argument("--tensor-parallel-size", type=int, default=None, dest="tensor_parallel_size")
    ap.add_argument("--gpu-memory-utilization", type=float, default=None,
                    dest="gpu_memory_utilization")
    ap.add_argument("--max-model-len", type=int, default=None, dest="max_model_len")
    ap.add_argument("--aux-device", default=None, dest="aux_device",
                    help="DD backend only: device(s) for the aux models — one GPU for both "
                         "(e.g. cuda:1) or an 'aux_p_dev,aux_q_dev' pair (e.g. cuda:3,cuda:2) "
                         "for one engine per card (4xGPU TP=2 layout). Overrides the spec.")
    args = ap.parse_args()

    spec = MODELS[args.model]
    task_names = resolve_tasks(args.tasks)
    # Arms: linear alphas + rank-DD k values ("rank_ks") + composed linear+rank
    # (alpha, k) pairs ("alpha_rank_arms"), all from ONE engine load.
    arms = list(spec.get("alphas") or []) + [("rank", int(k)) for k in spec.get("rank_ks") or []]
    arms += [("linrank", float(a), int(k)) for a, k in spec.get("alpha_rank_arms") or []]
    if not arms:
        arms = [None]
    if args.alphas is not None and spec.get("alphas"):  # only override linear DD models
        arms = [float(a) for a in args.alphas.split(",") if a.strip()]
    print(f"[plan] model={args.model} backend={spec['backend']} arms={arms} tasks={task_names}",
          flush=True)

    overrides = {k: getattr(args, k) for k in
                 ("batch_size", "tensor_parallel_size", "gpu_memory_utilization",
                  "max_model_len", "aux_device")}
    inst = build_model(spec, overrides)
    tm = TaskManager(include_path=TEMPORAL_DIR)

    # All DD arms run from ONE engine load; alpha=None = a plain (non-DD) model.
    # The DD backend reads (dd_alpha, dd_rank_k) per request; a rank arm zeroes the
    # linear shift and a linear arm zeroes the mask, so the arms stay pure.
    for alpha in arms:
        if isinstance(alpha, tuple) and alpha[0] == "rank":  # ("rank", k)
            inst.dd_alpha, inst.dd_rank_k = 0.0, alpha[1]
        elif isinstance(alpha, tuple):  # ("linrank", a, k): both adjustments on
            inst.dd_alpha, inst.dd_rank_k = alpha[1], alpha[2]
        elif alpha is not None:
            inst.dd_alpha, inst.dd_rank_k = alpha, 0
        for task_name in task_names:
            try:
                run_task(inst, spec, task_name, tm=tm, args=args, alpha=alpha)
            except Exception as e:  # noqa: BLE001 -- one task failing shouldn't kill the sweep
                print(f"[error] {task_name} / {args.model} alpha={alpha}: "
                      f"{type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
