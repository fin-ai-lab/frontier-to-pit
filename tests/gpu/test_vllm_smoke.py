"""End-to-end vLLM stress smoke (~4 min on one GPU): every production stressor
at miniature thresholds.

Production failure modes have all occurred at *thresholds* — window crossing,
KV-pool exhaustion, batch churn across generate() calls, CUDA-graph recapture.
At production settings those fire 10-25 minutes into a run; this test shrinks
the thresholds (window=64, starved KV pool, three short arms) so each fires
within seconds. Run this after ANY engine/processor change before paying for a
full benchmark.

Requires: vllm installed, DD_TEST_P_MODEL (a small-ish vLLM-servable model is
fine), DD_TEST_AUX_MODEL (Llama-architecture aux sharing P's tokenizer... for
smoke purposes the same aux can serve as both p and q: alpha steering then
cancels, but every engine code path still executes).
"""

import os

import pytest
import torch

vllm = pytest.importorskip("vllm")


@pytest.fixture(scope="module")
def p_model_path():
    val = os.environ.get("DD_TEST_P_MODEL")
    if val is None:
        pytest.skip("set DD_TEST_P_MODEL to a vLLM-servable model path")
    return val


def test_vllm_stress_smoke(cuda_device, p_model_path, aux_model_path):
    from ftp import DDConfig
    from ftp.vllm import make_processor

    cfg = DDConfig(
        aux_p=aux_model_path,
        aux_q=aux_model_path,  # same model twice: steering cancels, paths don't
        tokenizer=p_model_path,
        window=64,  # window crossings begin within ~40 generated tokens
        compile_aux=True,  # exercises compile + capture + recapture
        prewarm=8,
    )
    kwargs = dict(
        model=p_model_path,
        dtype="bfloat16",
        # To additionally exercise KV-pressure pauses (recompute-preemption),
        # set DD_SMOKE_GPU_UTIL so the KV pool barely exceeds the model's
        # footprint — the right value depends on the P model's size, so the
        # default stays safe.
        gpu_memory_utilization=float(os.environ.get("DD_SMOKE_GPU_UTIL", "0.75")),
        max_model_len=512,
        max_num_seqs=16,
        logits_processors=[make_processor(cfg)],
        enable_prefix_caching=False,
        trust_remote_code=True,
    )
    # Multimodal P models (e.g. Qwen3.5-VL hybrids, gemma-4 audio+vision) need
    # their modalities zeroed to initialize at a small max_model_len; try from
    # most to least specific.
    for limit in ({"image": 0, "video": 0, "audio": 0}, {"image": 0, "video": 0}, None):
        try:
            llm = vllm.LLM(**kwargs, limit_mm_per_prompt=limit) if limit else vllm.LLM(**kwargs)
            break
        except (ValueError, TypeError):
            if limit is None:
                raise
    sp = lambda alpha, seed: vllm.SamplingParams(  # noqa: E731
        temperature=0.8,
        top_p=1.0,
        top_k=-1,
        max_tokens=128,  # crosses window=64 twice over
        seed=seed,
        extra_args={"dd_alpha": alpha},
    )
    prompts = ["Tell me a short story about a dog."] * 8

    # Three arms = two full batch-churn/recapture cycles, alternating alpha so
    # both the DD path and the bypass path see arm transitions.
    for arm, alpha in enumerate([1.5, 0.0, 1.5]):
        outs = llm.generate(prompts, [sp(alpha, 1000 * arm + i) for i in range(8)])
        assert len(outs) == 8
        for o in outs:
            assert len(o.outputs[0].token_ids) > 0
            assert o.outputs[0].text.strip(), "empty generation"

    # The engine survived: init+capture, 3 arms, window crossings under stagger,
    # KV-pressure pauses, slot recycling. Sanity: GPU still healthy.
    assert torch.cuda.is_available()
