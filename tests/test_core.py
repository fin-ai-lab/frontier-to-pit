"""CPU unit tests for the fusion math (ftp.core)."""

import math

import pytest
import torch

from ftp import build_special_ids, dd_fuse, log1mexp

V, N = 512, 8


@pytest.fixture(scope="module")
def logits():
    g = torch.Generator().manual_seed(0)
    lP = torch.randn(N, V, generator=g) * 4
    lp = torch.randn(N, V, generator=g) * 4
    lq = torch.randn(N, V, generator=g) * 4
    return lP, lp, lq


@pytest.fixture(scope="module")
def whitelist():
    return torch.tensor([0, 1, 5, V - 1], dtype=torch.long)


@pytest.mark.parametrize("alpha", [0.0, 1.5])
def test_pin_false_is_plain_linear_fusion(logits, whitelist, alpha):
    """pin=False: raw l̂ = l_P + α·(l_q − l_p); whitelist ignored."""
    lP, lp, lq = logits
    expected = lP.float() if alpha == 0.0 else (lP + alpha * (lq - lp)).float()
    out = dd_fuse(lP, lp, lq, alpha=alpha, pin=False)
    torch.testing.assert_close(out, expected.to(lP.dtype), rtol=1e-5, atol=1e-5)
    # whitelist/wl_mask are no-ops when pinning is off
    out_wl = dd_fuse(lP, lp, lq, alpha=alpha, whitelist=whitelist, pin=False)
    torch.testing.assert_close(out, out_wl, rtol=0, atol=0)


def test_default_pins(logits, whitelist):
    """The default (no pin arg) is the pinned/renormalized path (pin=True)."""
    lP, lp, lq = logits
    default = dd_fuse(lP, lp, lq, alpha=1.5, whitelist=whitelist)
    pinned = dd_fuse(lP, lp, lq, alpha=1.5, whitelist=whitelist, pin=True)
    torch.testing.assert_close(default, pinned, rtol=0, atol=0)


def test_batched_equals_looped(logits, whitelist):
    lP, lp, lq = logits
    batched = dd_fuse(lP, lp, lq, alpha=1.5, whitelist=whitelist, pin=True)
    for i in range(N):
        row = dd_fuse(
            lP[i : i + 1], lp[i : i + 1], lq[i : i + 1], alpha=1.5, whitelist=whitelist, pin=True
        )
        torch.testing.assert_close(row[0], batched[i], rtol=1e-5, atol=1e-5, equal_nan=True)


def test_wl_mask_equals_whitelist(logits, whitelist):
    lP, lp, lq = logits
    wl_mask = torch.zeros(V, dtype=torch.bool)
    wl_mask[whitelist] = True
    a = dd_fuse(lP, lp, lq, alpha=1.5, whitelist=whitelist, pin=True)
    b = dd_fuse(lP, lp, lq, alpha=1.5, wl_mask=wl_mask, pin=True)
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6, equal_nan=True)


@pytest.mark.parametrize("alpha", [0.0, 1.5])
def test_normalized(logits, whitelist, alpha):
    lP, lp, lq = logits
    out = dd_fuse(lP, lp, lq, alpha=alpha, whitelist=whitelist, pin=True)
    torch.testing.assert_close(out.exp().sum(-1), torch.ones(N), rtol=1e-4, atol=1e-4)


def test_pinning_exact(logits, whitelist):
    """Whitelist tokens keep P's probability EXACTLY (pin=True)."""
    lP, lp, lq = logits
    out = dd_fuse(lP, lp, lq, alpha=1.5, whitelist=whitelist, pin=True)
    P = lP.float().softmax(-1)
    for i in range(N):
        for t in whitelist.tolist():
            assert math.isclose(out[i, t].exp().item(), P[i, t].item(), rel_tol=1e-4, abs_tol=1e-7)


def test_alpha_zero_is_pinned_P(logits, whitelist):
    """alpha=0 must reduce to the whitelist-pinned P distribution — and with no
    whitelist, to plain log_softmax(P). Aux logits must not matter at all."""
    lP, lp, lq = logits
    out = dd_fuse(lP, lp, lq, alpha=0.0, pin=True)
    torch.testing.assert_close(out, lP.float().log_softmax(-1).to(lP.dtype), rtol=1e-5, atol=1e-5)
    out_a = dd_fuse(lP, lp, lq, alpha=0.0, whitelist=whitelist, pin=True)
    out_b = dd_fuse(lP, lq, lp, alpha=0.0, whitelist=whitelist, pin=True)  # p/q swapped
    torch.testing.assert_close(out_a, out_b, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_rank_masks_topk_leaves_rest(logits):
    """Pure rank DD (alpha=0, pin=False): exactly the top-k of (l_p − l_q) go to
    −inf; every other token keeps P's logit untouched."""
    lP, lp, lq = logits
    out = dd_fuse(lP, lp, lq, alpha=0.0, rank_k=10, pin=False)
    suppressed = (lp - lq).topk(10, dim=-1).indices
    assert torch.isneginf(out.gather(-1, suppressed)).all()
    keep = torch.ones_like(out, dtype=torch.bool).scatter(-1, suppressed, False)
    torch.testing.assert_close(out[keep], lP[keep], rtol=1e-6, atol=1e-6)


def test_rank_does_not_mutate_inputs(logits):
    """The rank scatter must be out-of-place: at alpha=0 the fused tensor can alias
    the caller's fp32 logits."""
    lP, lp, lq = logits
    orig = lP.clone()
    dd_fuse(lP, lp, lq, alpha=0.0, rank_k=10, pin=False)
    torch.testing.assert_close(lP, orig, rtol=0, atol=0)


def test_rank_composes_with_linear(logits):
    """alpha and rank_k together: linear shift everywhere + −inf on the top-k."""
    lP, lp, lq = logits
    out = dd_fuse(lP, lp, lq, alpha=1.5, rank_k=10, pin=False)
    suppressed = (lp - lq).topk(10, dim=-1).indices
    expected = (lP + 1.5 * (lq - lp)).float().scatter(-1, suppressed, float("-inf"))
    torch.testing.assert_close(out, expected.to(lP.dtype), rtol=1e-5, atol=1e-5)


def test_rank_pin_protects_whitelist(logits):
    """pin=True: a whitelisted token keeps P's probability exactly even when it is
    inside the top-k divergent set; the result stays normalized."""
    lP, lp, lq = logits
    suppressed = (lp - lq).topk(10, dim=-1).indices
    wl = suppressed[0, :2]  # two tokens the rank mask would otherwise kill (row 0)
    out = dd_fuse(lP, lp, lq, alpha=0.0, rank_k=10, whitelist=wl, pin=True)
    P = lP.float().softmax(-1)
    for t in wl.tolist():
        assert math.isclose(out[0, t].exp().item(), P[0, t].item(), rel_tol=1e-4, abs_tol=1e-7)
    torch.testing.assert_close(out.exp().sum(-1), torch.ones(N), rtol=1e-4, atol=1e-4)


def test_log1mexp():
    x = torch.linspace(-20.0, -1e-4, steps=2001, dtype=torch.float64)
    ref = torch.log1p(-torch.exp(x))
    torch.testing.assert_close(log1mexp(x), ref, rtol=1e-9, atol=1e-12)


def test_build_special_ids_includes_unflagged_added_tokens(tmp_path):
    """Added control tokens NOT flagged `special` must still be whitelisted —
    this is the documented sharp edge build_special_ids exists for."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {f"w{i}": i for i in range(20)}
    tok = Tokenizer(WordLevel(vocab, unk_token="w0"))
    tok.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="w1")
    fast.add_tokens(["<ctrl>"], special_tokens=False)  # added but NOT special
    fast.add_special_tokens({"additional_special_tokens": ["<extra>"]})

    ids = build_special_ids(fast)
    assert fast.convert_tokens_to_ids("<ctrl>") in ids
    assert fast.convert_tokens_to_ids("<extra>") in ids
    assert fast.eos_token_id in ids
