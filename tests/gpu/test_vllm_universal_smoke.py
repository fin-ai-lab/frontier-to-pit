"""End-to-end vLLM stress smoke for UNIVERSAL (cross-tokenizer) decoding.

Same miniature-threshold philosophy as test_vllm_smoke.py (window=64, short
arms, churn), plus the universal-mode correctness lever: with aux_p == aux_q
the mapped shift l_q − l_p is exactly zero, so GREEDY universal-DD output must
equal greedy α=0 output token-for-token — any retokenization/rewind/mapping
corruption breaks the equality.

Requires: vllm installed, DD_TEST_P_UNIVERSAL_MODEL (a vLLM-servable model
whose tokenizer DIFFERS from the aux model's, e.g.
deepseek-ai/DeepSeek-R1-Distill-Llama-8B vs Qwen-tokenizer aux),
DD_TEST_AUX_MODEL (the aux checkpoint).
"""

import os

import pytest
import torch

vllm = pytest.importorskip("vllm")


@pytest.fixture(scope="module")
def p_model_path():
    val = os.environ.get("DD_TEST_P_UNIVERSAL_MODEL")
    if val is None:
        pytest.skip("set DD_TEST_P_UNIVERSAL_MODEL to a tokenizer-mismatched P model path")
    return val


def test_vllm_universal_stress_smoke(cuda_device, p_model_path, aux_model_path):
    from ftp import DDConfig
    from ftp.vllm import make_processor

    cfg = DDConfig(
        aux_p=aux_model_path,
        aux_q=aux_model_path,  # same model twice: shift is exactly zero
        window=64,  # window crossings (and at-cap rewinds) within ~40 tokens
        compile_aux=True,
        prewarm=8,
        mode="auto",  # must auto-resolve to universal for this pairing
    )
    kwargs = dict(
        model=p_model_path,
        dtype="bfloat16",
        gpu_memory_utilization=float(os.environ.get("DD_SMOKE_GPU_UTIL", "0.75")),
        max_model_len=512,
        max_num_seqs=16,
        logits_processors=[make_processor(cfg)],
        enable_prefix_caching=False,
        trust_remote_code=True,
    )
    # Multimodal P models need their modalities zeroed to initialize at a
    # small max_model_len; try from most to least specific (audio for
    # gemma-4-style audio+vision models).
    for limit in ({"image": 0, "video": 0, "audio": 0}, {"image": 0, "video": 0}, None):
        try:
            llm = vllm.LLM(**kwargs, limit_mm_per_prompt=limit) if limit else vllm.LLM(**kwargs)
            break
        except (ValueError, TypeError):
            if limit is None:
                raise

    prompts = [
        "Tell me a short story about a dog.",
        "List the first ten prime numbers and explain primality.",
        "Write a Python function that reverses a linked list.",
        "Describe a rainy day in Tokyo using vivid imagery. 你好",
    ] * 2

    # Greedy equivalence: universal DD with a zero shift must reproduce the
    # α=0 baseline token-for-token, except at genuine numerical ties (see
    # assert_greedy_equiv in test_aux_device.py).
    from test_aux_device import assert_greedy_equiv

    greedy = lambda alpha: vllm.SamplingParams(  # noqa: E731
        temperature=0.0, max_tokens=128, logprobs=4, extra_args={"dd_alpha": alpha}
    )
    base = llm.generate(prompts, [greedy(0.0)] * len(prompts))
    dd = llm.generate(prompts, [greedy(1.5)] * len(prompts))
    for b, d in zip(base, dd, strict=True):
        assert_greedy_equiv(b.outputs[0], d.outputs[0])

    # Stress arms: sampling, window crossings under stagger, batch churn,
    # alternating alpha (DD path vs bypass path transitions).
    sp = lambda alpha, seed: vllm.SamplingParams(  # noqa: E731
        temperature=0.8,
        top_p=1.0,
        top_k=-1,
        max_tokens=128,
        seed=seed,
        extra_args={"dd_alpha": alpha},
    )
    for arm, alpha in enumerate([1.5, 0.0, 1.5]):
        outs = llm.generate(prompts, [sp(alpha, 1000 * arm + i) for i in range(len(prompts))])
        assert len(outs) == len(prompts)
        for o in outs:
            assert len(o.outputs[0].token_ids) > 0
        # Instruct P models may EOS immediately on raw (non-chat) prompts;
        # require only that the batch as a whole isn't degenerate.
        nonempty = sum(1 for o in outs if o.outputs[0].text.strip())
        assert nonempty >= len(prompts) // 2, f"degenerate batch: {nonempty}/{len(prompts)}"

    assert torch.cuda.is_available()
