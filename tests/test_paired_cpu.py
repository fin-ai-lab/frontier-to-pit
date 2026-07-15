"""CPU ground truth for the paired (stacked-weight) modules.

The fused model must reproduce BOTH source models exactly: a plane-major
folded batch (rows [0, B/2) = model p, [B/2, B) = model q) through the fused
forward equals each source model run separately on its half. fp32 on CPU; the
1e-4 tolerance absorbs bmm-vs-mm reassociation.
"""

from copy import deepcopy

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from ftp.paired import (
    PairedEmbedding,
    PairedLinear,
    PairedRMSNorm,
    check_fusable,
    fuse_pair,
)

V = 512


def make_llama(seed: int, **cfg_kwargs) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    kwargs = dict(
        vocab_size=V,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    kwargs.update(cfg_kwargs)
    return LlamaForCausalLM(LlamaConfig(**kwargs)).eval()


@pytest.fixture()
def fused(tiny_llama, tiny_llama_q):
    return fuse_pair(deepcopy(tiny_llama), deepcopy(tiny_llama_q))


def assert_fused_matches(fused_model, model_p, model_q, ids: torch.Tensor) -> None:
    """ids: [N, T]; runs the fused model on the plane-major doubled batch."""
    with torch.no_grad():
        got = fused_model(input_ids=torch.cat([ids, ids])).logits
        ref_p = model_p(input_ids=ids).logits
        ref_q = model_q(input_ids=ids).logits
    n = ids.shape[0]
    torch.testing.assert_close(got[:n], ref_p, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(got[n:], ref_q, rtol=1e-4, atol=1e-4)


def test_fused_forward_matches_both_sources(tiny_llama, tiny_llama_q, fused):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(2, V, (3, 12), generator=g)
    assert_fused_matches(fused, tiny_llama, tiny_llama_q, ids)
    # Single-token rows (decode shape).
    assert_fused_matches(fused, tiny_llama, tiny_llama_q, ids[:, :1])


def test_planes_are_independent(fused):
    """The two planes must produce genuinely different distributions — a
    plane-marriage bug (one model's weights serving both halves) is invisible
    to per-plane reference checks alone when the sources are near-identical,
    so assert the planes disagree for different-seed sources."""
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(2, V, (2, 8), generator=g)
    with torch.no_grad():
        lg = fused(input_ids=torch.cat([ids, ids])).logits
    assert not torch.allclose(lg[:2], lg[2:], rtol=1e-3, atol=1e-3)


def test_fused_with_biases():
    p = make_llama(0, attention_bias=True, mlp_bias=True)
    q = make_llama(1, attention_bias=True, mlp_bias=True)
    fused_model = fuse_pair(deepcopy(p), deepcopy(q))
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(2, V, (2, 6), generator=g)
    assert_fused_matches(fused_model, p, q, ids)


def test_free_source_false_keeps_sources_usable(tiny_llama, tiny_llama_q):
    p, q = deepcopy(tiny_llama), deepcopy(tiny_llama_q)
    fused_model = fuse_pair(deepcopy(tiny_llama), q, free_source=False)
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(2, V, (1, 5), generator=g)
    assert_fused_matches(fused_model, p, q, ids)  # q itself must still work
    with torch.no_grad():
        q(input_ids=ids)


def test_weights_are_bmm_oriented_contiguous(fused):
    for m in fused.modules():
        if isinstance(m, PairedLinear):
            assert m.weight.shape == (2, m.in_features, m.out_features)
            assert m.weight.is_contiguous()
        elif isinstance(m, PairedEmbedding):
            assert m.weight.shape == (2, m.num_embeddings, m.embedding_dim)
        elif isinstance(m, PairedRMSNorm):
            assert m.weight.shape[0] == 2 and m.weight.is_contiguous()


def test_coverage_no_unfused_modules_and_param_count(tiny_llama, tiny_llama_q, fused):
    for _, m in fused.named_modules():
        assert not isinstance(m, (nn.Linear, nn.Embedding))
        if any(p is not None for p in m._parameters.values()):
            assert isinstance(m, (PairedLinear, PairedEmbedding, PairedRMSNorm))
    expected = sum(p.numel() for p in tiny_llama.parameters()) + sum(
        p.numel() for p in tiny_llama_q.parameters()
    )
    assert sum(p.numel() for p in fused.parameters()) == expected


def test_unknown_parameterized_module_trips_assert(tiny_llama, tiny_llama_q):
    p, q = deepcopy(tiny_llama), deepcopy(tiny_llama_q)
    p.model.mystery = nn.Conv1d(4, 4, 1)
    q.model.mystery = nn.Conv1d(4, 4, 1)
    with pytest.raises(AssertionError, match="does not know how to stack"):
        fuse_pair(p, q)


def test_param_shape_mismatch_raises(tiny_llama, tiny_llama_q):
    p, q = deepcopy(tiny_llama), deepcopy(tiny_llama_q)
    p.model.extra = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="name/shape mismatch"):
        fuse_pair(p, q)


def test_tied_embeddings_rejected():
    p = make_llama(0, tie_word_embeddings=True)
    q = make_llama(1, tie_word_embeddings=True)
    ok, why = check_fusable(p.config, q.config)
    assert not ok and "tie_word_embeddings" in why
    with pytest.raises(NotImplementedError, match="tie_word_embeddings"):
        fuse_pair(p, q)


@pytest.mark.parametrize(
    "field,value",
    [("num_hidden_layers", 3), ("vocab_size", 256), ("rope_theta", 50000.0)],
)
def test_check_fusable_flags_differing_field(field, value):
    p = make_llama(0)
    q = make_llama(1, **{field: value})
    ok, why = check_fusable(p.config, q.config)
    assert not ok and field in why
    with pytest.raises(ValueError, match=field):
        fuse_pair(p, q)


def test_check_fusable_identical_ok(tiny_llama, tiny_llama_q):
    ok, why = check_fusable(tiny_llama.config, tiny_llama_q.config)
    assert ok, why
