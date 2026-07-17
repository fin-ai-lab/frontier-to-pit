"""CPU unit tests for SAE-feature steering (ftp.steering).

Covers the model-agnostic hook math (fast relu gate + clamp_floor), both
residual conventions (plain HF tensor output vs vLLM's (hidden, residual)
deferred-residual tuple), the clamp_value=0 no-op, multi-feature additivity, and
the on-disk SAE slice/transpose in _load_weights. The vLLM install path
(apply_model) needs a GPU engine and lives in tests/gpu/.
"""

import torch

from ftp.steering import (
    SaeSource,
    SteerSpec,
    _load_weights,
    _SteerWeights,
    make_steer_hook,
)

D_MODEL, D_SAE = 16, 64


def _rand_sae(nf_ids, seed=0):
    """A synthetic SAE slice for features `nf_ids` (feature-major rows)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor(nf_ids)
    W_enc = torch.randn(D_SAE, D_MODEL, generator=g)
    b_enc = torch.randn(D_SAE, generator=g)
    W_dec = torch.randn(D_SAE, D_MODEL, generator=g)
    return _SteerWeights(W_enc_f=W_enc[idx], b_enc_f=b_enc[idx], W_dec_f=W_dec[idx])


def _ref_clamp_floor(x, w, cv):
    """Reference fast clamp_floor: relu gate, raise feat up to cv, inject."""
    pre = x @ w.W_enc_f.T + w.b_enc_f
    feat = pre.clamp(min=0)
    delta = (cv - feat).clamp(min=0)
    return x + delta @ w.W_dec_f


def _ref_clamp_floor_jumprelu(x, w, cv):
    """Reference exact JumpReLU clamp_floor (the HF Gemma gate): a feature is
    active iff pre > its threshold; feat is the pre there, else 0."""
    pre = x @ w.W_enc_f.T + w.b_enc_f
    feat = pre * (pre > w.thr_f)
    delta = (cv - feat).clamp(min=0)
    return x + delta @ w.W_dec_f


def test_hf_convention_matches_reference():
    w = _rand_sae([2, 7, 30])
    spec = SteerSpec(layer=0, feature_id=[2, 7, 30], clamp_value=5.0)
    hook = make_steer_hook(w, spec, set_residual=False)
    x = torch.randn(4, D_MODEL)
    out = hook(None, None, x)
    torch.testing.assert_close(out, _ref_clamp_floor(x, w, 5.0), rtol=1e-5, atol=1e-5)


def test_hf_tuple_output_steers_first_element():
    """HF blocks often return (resid_post, attn_weights, ...); only [0] is steered."""
    w = _rand_sae([1, 4, 9])
    spec = SteerSpec(layer=0, feature_id=[1, 4, 9], clamp_value=3.0)
    hook = make_steer_hook(w, spec, set_residual=False)
    x = torch.randn(4, D_MODEL)
    extra = torch.randn(4, 4)
    out = hook(None, None, (x, extra))
    assert isinstance(out, tuple) and out[1] is extra
    torch.testing.assert_close(out[0], _ref_clamp_floor(x, w, 3.0), rtol=1e-5, atol=1e-5)


def test_vllm_residual_tuple_injects_into_stream():
    """vLLM convention: resid_post = hidden + residual; inject into hidden so the
    next block sees resid_post + delta, with the residual tensor left untouched."""
    w = _rand_sae([3, 5, 11])
    spec = SteerSpec(layer=0, feature_id=[3, 5, 11], clamp_value=4.0)
    hook = make_steer_hook(w, spec, set_residual=True)
    hidden = torch.randn(4, D_MODEL)
    residual = torch.randn(4, D_MODEL)
    new_hidden, new_residual = hook(None, None, (hidden, residual))
    assert new_residual is residual  # residual plane untouched
    # The next block's input norm consumes hidden + residual:
    got_resid_post = new_hidden + new_residual
    want = _ref_clamp_floor(hidden + residual, w, 4.0)
    torch.testing.assert_close(got_resid_post, want, rtol=1e-5, atol=1e-5)


def test_clamp_zero_is_noop():
    w = _rand_sae([0, 8, 20])
    spec = SteerSpec(layer=0, feature_id=[0, 8, 20], clamp_value=0.0)
    x = torch.randn(4, D_MODEL)
    out_hf = make_steer_hook(w, spec, set_residual=False)(None, None, x)
    torch.testing.assert_close(out_hf, x, rtol=0, atol=0)
    h, r = torch.randn(4, D_MODEL), torch.randn(4, D_MODEL)
    nh, nr = make_steer_hook(w, spec, set_residual=True)(None, None, (h, r))
    torch.testing.assert_close(nh, h, rtol=0, atol=0)


def test_multi_feature_is_additive():
    """Steering {a, b} together == steering a alone + steering b alone (the relu
    gate is per-feature and the decoder injection is a linear sum)."""
    ids = [6, 19]
    w_both = _rand_sae(ids, seed=1)
    cv = 2.5
    x = torch.randn(5, D_MODEL)
    both = make_steer_hook(w_both, SteerSpec(0, ids, cv), set_residual=False)(None, None, x) - x
    sep = torch.zeros_like(x)
    for j, fid in enumerate(ids):
        s = slice(j, j + 1)
        wj = _SteerWeights(w_both.W_enc_f[s], w_both.b_enc_f[s], w_both.W_dec_f[s])
        sep += make_steer_hook(wj, SteerSpec(0, [fid], cv), set_residual=False)(None, None, x) - x
    torch.testing.assert_close(both, sep, rtol=1e-5, atol=1e-5)


def test_jumprelu_gate_matches_exact_threshold_reference():
    """JumpReLU hook (thr_f set) == the exact per-feature threshold gate used by
    the HF Gemma steering, for both residual conventions."""
    base = _rand_sae([2, 7, 30])
    g = torch.Generator().manual_seed(3)
    thr = torch.rand(3, generator=g) * 2.0 + 0.5  # positive thresholds
    w = _SteerWeights(base.W_enc_f, base.b_enc_f, base.W_dec_f, thr_f=thr)
    spec = SteerSpec(layer=0, feature_id=[2, 7, 30], clamp_value=5.0)
    x = torch.randn(8, D_MODEL, generator=torch.Generator().manual_seed(8))
    out = make_steer_hook(w, spec, set_residual=False)(None, None, x)
    torch.testing.assert_close(out, _ref_clamp_floor_jumprelu(x, w, 5.0), rtol=1e-5, atol=1e-5)
    # vLLM (hidden, residual) convention: gate uses resid_post = hidden + residual.
    h = torch.randn(8, D_MODEL, generator=torch.Generator().manual_seed(9))
    r = torch.randn(8, D_MODEL, generator=torch.Generator().manual_seed(10))
    nh, nr = make_steer_hook(w, spec, set_residual=True)(None, None, (h, r))
    assert nr is r
    torch.testing.assert_close(
        nh + nr, _ref_clamp_floor_jumprelu(h + r, w, 5.0), rtol=1e-5, atol=1e-5
    )


def test_jumprelu_threshold_changes_gate_vs_relu():
    """A feature with 0 < pre < threshold is OFF under JumpReLU (inject full
    clamp_value) but ON under relu (inject clamp_value - pre): the two gates give
    different deltas, so the threshold is not a no-op."""
    base = _rand_sae([5, 12])
    cv = 9000.0
    x = torch.randn(6, D_MODEL, generator=torch.Generator().manual_seed(11))
    pre = x @ base.W_enc_f.T + base.b_enc_f
    assert (pre > 0).any()  # some positions are relu-active
    big_thr = pre.amax(0) + 1.0  # above every position's pre → JumpReLU-inactive everywhere
    wj = _SteerWeights(base.W_enc_f, base.b_enc_f, base.W_dec_f, thr_f=big_thr)
    out_j = make_steer_hook(wj, SteerSpec(0, [5, 12], cv), set_residual=False)(None, None, x)
    out_relu = make_steer_hook(base, SteerSpec(0, [5, 12], cv), set_residual=False)(None, None, x)
    # All inactive → inject exactly cv along each decoder row.
    torch.testing.assert_close(
        out_j, x + torch.full_like(pre, cv) @ base.W_dec_f, rtol=1e-5, atol=1e-5
    )
    assert not torch.allclose(out_j, out_relu)


def test_load_weights_jumprelu_reads_gemma_scope(tmp_path):
    """family='jumprelu' reads Gemma-Scope params.safetensors: w_enc
    (d_model, d_sae) → (nf, d_model), w_dec (d_sae, d_model) feature-major as-is,
    and the per-feature threshold."""
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(4)
    w_enc = torch.randn(D_MODEL, D_SAE, generator=g)  # Gemma-Scope on-disk layout
    b_enc = torch.randn(D_SAE, generator=g)
    threshold = torch.rand(D_SAE, generator=g)
    w_dec = torch.randn(D_SAE, D_MODEL, generator=g)
    b_dec = torch.randn(D_MODEL, generator=g)
    base = tmp_path / "resid_post" / "layer_0_width_65k_l0_medium"
    base.mkdir(parents=True)
    save_file(
        {"w_enc": w_enc, "b_enc": b_enc, "threshold": threshold, "w_dec": w_dec, "b_dec": b_dec},
        str(base / "params.safetensors"),
    )
    ids = [4, 40, 2]
    src = SaeSource(path=str(base / "params.safetensors"), family="jumprelu")
    w = _load_weights(src, SteerSpec(0, ids), torch.device("cpu"))
    idx = torch.tensor(ids)
    torch.testing.assert_close(w.W_enc_f, w_enc[:, idx].T)
    torch.testing.assert_close(w.b_enc_f, b_enc[idx])
    torch.testing.assert_close(w.thr_f, threshold[idx])
    torch.testing.assert_close(w.W_dec_f, w_dec[idx])
    assert w.W_dec_f.shape == (len(ids), D_MODEL)


def test_load_weights_slices_and_transposes(tmp_path):
    """_load_weights reads the on-disk (d_model, d_sae) W_dec and the (d_sae,
    d_model) W_enc, slicing the target rows and transposing the decoder."""
    g = torch.Generator().manual_seed(2)
    W_enc = torch.randn(D_SAE, D_MODEL, generator=g)
    b_enc = torch.randn(D_SAE, generator=g)
    W_dec_disk = torch.randn(D_MODEL, D_SAE, generator=g)  # on-disk layout
    b_dec = torch.randn(D_MODEL, generator=g)
    path = tmp_path / "layer0.sae.pt"
    torch.save({"W_enc": W_enc, "b_enc": b_enc, "W_dec": W_dec_disk, "b_dec": b_dec}, path)

    ids = [4, 40, 2]
    w = _load_weights(SaeSource(path=str(path)), SteerSpec(0, ids), torch.device("cpu"))
    idx = torch.tensor(ids)
    torch.testing.assert_close(w.W_enc_f, W_enc[idx])
    torch.testing.assert_close(w.b_enc_f, b_enc[idx])
    torch.testing.assert_close(w.W_dec_f, W_dec_disk[:, idx].T)
    assert w.W_dec_f.shape == (len(ids), D_MODEL)
