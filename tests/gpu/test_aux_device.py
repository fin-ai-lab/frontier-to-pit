"""Two-GPU placement tests: aux engines on a different GPU than P.

Validates the aux_device path — compiled CUDA graphs captured and replayed on
a non-default device (the engine's device guard), and cross-device logits
transfer back to P's GPU. Requires >= 2 CUDA devices.
"""

import os

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ftp import AuxBatchedEngine


@pytest.fixture(scope="module")
def second_gpu():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs >= 2 CUDA devices")
    return torch.device("cuda:1")


def test_engine_compiled_on_second_gpu(second_gpu, aux_model_path):
    """Engine (compiled, CUDA graphs) on cuda:1 while cuda:0 is the process
    default: ground truth vs full-context reference forwards, including a
    cross-device copy of every output (the vLLM apply() pattern)."""
    torch.cuda.set_device(0)  # P's device is the default, as under vLLM
    window, steps = 64, 24
    tok = AutoTokenizer.from_pretrained(aux_model_path)
    prompts = [
        tok(t)["input_ids"]
        for t in (
            "The capital of France is Paris, and the capital of Germany is",
            "In 2014, the fastest production car made by Bugatti was the",
        )
    ]

    ref_model = (
        AutoModelForCausalLM.from_pretrained(aux_model_path, dtype=torch.bfloat16)
        .to(second_gpu)
        .eval()
    )
    seqs = [list(p) for p in prompts]
    ref, conts = {}, [[] for _ in prompts]
    with torch.no_grad():
        for s in range(steps + 1):
            for i in range(len(prompts)):
                ids = torch.tensor([seqs[i][-window:]], device=second_gpu)
                lg = ref_model(input_ids=ids).logits[0, -1].float().cpu()
                ref[(s, i)] = lg
                if s < steps:
                    nxt = int(lg.argmax())
                    seqs[i].append(nxt)
                    conts[i].append(nxt)
    del ref_model
    torch.cuda.empty_cache()

    eng = AuxBatchedEngine(aux_model_path, second_gpu, torch.bfloat16, window, compile_model=True)
    for i in range(len(prompts)):
        eng.register(i)

    def check(out, s):
        assert out.device == second_gpu
        moved = out.to("cuda:0").float().cpu()  # the apply() cross-device hop
        for i in range(len(prompts)):
            a, b = ref[(s, i)], moved[i]
            assert int(a.argmax()) == int(b.argmax()), f"step {s} row {i} argmax"
            assert (a.softmax(-1) - b.softmax(-1)).abs().max() < 0.05

    check(eng.step([(i, prompts[i], []) for i in range(len(prompts))]), 0)
    for s in range(steps):
        out = eng.step([(i, prompts[i], conts[i][: s + 1]) for i in range(len(prompts))])
        check(out, s + 1)


@pytest.mark.parametrize("fuse", ["off", "on"])
def test_vllm_split_smoke(second_gpu, aux_model_path, fuse):
    """End-to-end split placement: P on cuda:0 (vLLM), aux models on cuda:1,
    covering BOTH the two-engine and the fused aux path.
    aux_p == aux_q ⇒ greedy DD must equal greedy α=0 token-for-token."""
    vllm = pytest.importorskip("vllm")
    from ftp import DDConfig
    from ftp.vllm import make_processor

    p_model = os.environ.get("DD_TEST_P_MODEL") or os.environ.get("DD_TEST_P_UNIVERSAL_MODEL")
    if p_model is None:
        pytest.skip("set DD_TEST_P_MODEL or DD_TEST_P_UNIVERSAL_MODEL")

    cfg = DDConfig(
        aux_p=aux_model_path,
        aux_q=aux_model_path,
        window=64,
        compile_aux=True,
        prewarm=8,
        aux_device="cuda:1",
        fuse_aux=fuse,
    )
    kwargs = dict(
        model=p_model,
        dtype="bfloat16",
        gpu_memory_utilization=float(os.environ.get("DD_SMOKE_GPU_UTIL", "0.75")),
        max_model_len=512,
        max_num_seqs=16,
        logits_processors=[make_processor(cfg)],
        enable_prefix_caching=False,
        trust_remote_code=True,
    )
    for limit in ({"image": 0, "video": 0, "audio": 0}, {"image": 0, "video": 0}, None):
        try:
            llm = vllm.LLM(**kwargs, limit_mm_per_prompt=limit) if limit else vllm.LLM(**kwargs)
            break
        except (ValueError, TypeError):
            if limit is None:
                raise

    prompts = ["Tell me a short story about a dog.", "List the first ten primes."] * 4
    greedy = lambda alpha: vllm.SamplingParams(  # noqa: E731
        temperature=0.0, max_tokens=96, logprobs=4, extra_args={"dd_alpha": alpha}
    )
    base = llm.generate(prompts, [greedy(0.0)] * len(prompts))
    dd = llm.generate(prompts, [greedy(1.5)] * len(prompts))
    for b, d in zip(base, dd, strict=True):
        assert_greedy_equiv(b.outputs[0], d.outputs[0])


def assert_greedy_equiv(base_out, dd_out, tie_tol: float = 2e-3) -> None:
    """Greedy DD with a zero shift must match the greedy baseline, except at a
    genuine numerical tie: the fused log-probs round-trip through the sampler
    dtype, so two tokens within rounding distance can legitimately flip (and
    the suffix then diverges). Anything beyond tie tolerance is a real bug."""
    b_ids, d_ids = list(base_out.token_ids), list(dd_out.token_ids)
    for i in range(min(len(b_ids), len(d_ids))):
        if b_ids[i] == d_ids[i]:
            continue
        lps = base_out.logprobs[i]
        lb = lps.get(b_ids[i])
        ld = lps.get(d_ids[i])
        assert lb is not None and ld is not None, (
            f"divergence at {i}: {b_ids[i]} vs {d_ids[i]} — not a near-tie "
            f"(DD pick absent from baseline top logprobs)"
        )
        gap = abs(lb.logprob - ld.logprob)
        assert gap < tie_tol, f"divergence at {i}: logprob gap {gap:.4f} exceeds tie tolerance"
        return  # legitimate tie — the suffix is expected to differ
    assert b_ids == d_ids or len(b_ids) != len(d_ids)
