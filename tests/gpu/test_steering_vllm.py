"""End-to-end vLLM steering test (run on a CUDA-13 box — the repo's vLLM range
is CUDA-13). Validates the apply_model install path, that the forward hook fires
under enforce_eager, and that the vLLM (hidden, residual) reconstruction is an
exact identity at clamp_value=0.

Uses a SYNTHETIC SAE (random encoder/decoder of the model's d_model) — this
tests the *mechanism*, not real features, so any small Qwen3/Llama works. Set
DD_TEST_STEER_MODEL (falls back to DD_TEST_P_MODEL) to a small model path.

    DD_TEST_STEER_MODEL=Qwen/Qwen3-0.6B pytest tests/gpu/test_steering_vllm.py
"""

import os

import pytest
import torch

# Steering installs forward hooks on the in-process model, so the engine must run
# in-process — same contract as production (slurm_steer.sh exports this).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

vllm = pytest.importorskip("vllm")


@pytest.fixture(scope="module")
def steer_model_path():
    val = os.environ.get("DD_TEST_STEER_MODEL") or os.environ.get("DD_TEST_P_MODEL")
    if val is None:
        pytest.skip("set DD_TEST_STEER_MODEL (or DD_TEST_P_MODEL) to a small model")
    return val


def _write_synthetic_sae(path, d_model, d_sae=4096, seed=0):
    """On-disk Qwen-Scope layout: W_enc (d_sae,d_model), W_dec (d_model,d_sae)."""
    g = torch.Generator().manual_seed(seed)
    torch.save(
        {
            "W_enc": torch.randn(d_sae, d_model, generator=g),
            "b_enc": torch.randn(d_sae, generator=g),
            "W_dec": torch.randn(d_model, d_sae, generator=g) * 0.1,
            "b_dec": torch.zeros(d_model),
        },
        path,
    )


def test_vllm_steering_install_fire_remove(cuda_device, steer_model_path, tmp_path):
    from transformers import AutoConfig

    from ftp.steering import SaeSource, SteerSpec, steered_vllm

    # get_text_config(): Qwen3.5-family configs nest hidden_size/num_hidden_layers
    # under text_config; plain configs return themselves.
    cfg = AutoConfig.from_pretrained(steer_model_path).get_text_config()
    d_model, n_layers = cfg.hidden_size, cfg.num_hidden_layers
    layer = n_layers // 2
    sae_path = tmp_path / f"layer{layer}.sae.pt"
    _write_synthetic_sae(str(sae_path), d_model)
    src = SaeSource(path=str(sae_path))
    ids = [10, 200, 1000]

    llm = vllm.LLM(
        model=steer_model_path,
        enforce_eager=True,  # hooks do not fire under CUDA-graph replay
        gpu_memory_utilization=float(os.environ.get("DD_STEER_GPU_UTIL", "0.5")),
        max_model_len=2048,
    )
    sp = vllm.SamplingParams(temperature=0.0, max_tokens=40)  # greedy → deterministic
    prompt = "The capital of France is"

    def gen():
        return llm.generate([prompt], sp, use_tqdm=False)[0].outputs[0].text

    base = gen()

    # clamp_value=0 ⇒ delta=0 ⇒ exact identity, even though the hook runs. This
    # also proves resid_post = hidden + residual is reconstructed losslessly.
    with steered_vllm(llm, [(src, SteerSpec(layer=layer, feature_id=ids, clamp_value=0.0))]):
        noop = gen()
    assert noop == base, "clamp_value=0 must reproduce the baseline exactly"

    # A large clamp must perturb the residual stream enough to change the text —
    # if it doesn't, the hook is not firing.
    with steered_vllm(llm, [(src, SteerSpec(layer=layer, feature_id=ids, clamp_value=200.0))]):
        steered = gen()
    assert steered != base, "large clamp_value did not change output (hook not firing?)"

    # Hooks removed on context exit → back to baseline.
    assert gen() == base, "steering not removed after the context closed"


def test_vllm_steering_runs_with_dd(cuda_device, steer_model_path, aux_model_path, tmp_path):
    """Steering and Divergence Decoding active in the SAME engine / generate().

    They are orthogonal — steering hooks shift P's residual stream, DD fuses the
    aux pair into P's logits — so both effects must be present at once. We assert
    that steer+DD differs from DD-alone (steering still bites with DD running)
    and from steer-alone (DD still bites with steering running). Self-fused aux
    (same model as p and q) exercises every DD path; alpha steering cancels, so
    its logit shift is ~0 — hence DD's *presence* is checked via the engine
    running end-to-end, and steering's effect via the cross-comparison.
    """
    from transformers import AutoConfig

    from ftp import DDConfig
    from ftp.steering import SaeSource, SteerSpec, steered_vllm
    from ftp.vllm import make_processor

    cfg = AutoConfig.from_pretrained(steer_model_path).get_text_config()
    layer = cfg.num_hidden_layers // 2
    sae_path = tmp_path / f"layer{layer}.sae.pt"
    _write_synthetic_sae(str(sae_path), cfg.hidden_size)
    src = SaeSource(path=str(sae_path))
    spec = SteerSpec(layer=layer, feature_id=[10, 200, 1000], clamp_value=200.0)

    dd = DDConfig(aux_p=aux_model_path, aux_q=aux_model_path, tokenizer=steer_model_path)
    llm = vllm.LLM(
        model=steer_model_path,
        enforce_eager=True,
        gpu_memory_utilization=float(os.environ.get("DD_STEER_GPU_UTIL", "0.6")),
        max_model_len=2048,
        logits_processors=[make_processor(dd)],
        enable_prefix_caching=False,
    )
    sp = vllm.SamplingParams(temperature=0.0, max_tokens=40, extra_args={"dd_alpha": 1.5})
    prompt = "The capital of France is"

    def gen():
        return llm.generate([prompt], sp, use_tqdm=False)[0].outputs[0].text

    dd_only = gen()  # DD active, no steering
    with steered_vllm(llm, [(src, spec)]):
        dd_and_steer = gen()  # DD + steering active together
    assert dd_and_steer != dd_only, "steering had no effect while DD was running"

