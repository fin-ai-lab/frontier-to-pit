"""CPU tests for the vectorized decode staging (page-table mirrors, packed
flashinfer plan inputs, bucket padding).

The ground-truth engine tests already validate staged VALUES end-to-end (the
fallback attention consumes the same staged write targets, and a desynced
mirror would corrupt KV writes and fail their exactness asserts). These tests
pin what CPU runs can't reach through logits alone: mirror/list consistency
across the whole row lifecycle, the packed plan arrays against a naive
per-row reference (the algorithm the deleted ``_plan`` loop used), and the
plane-major pad layout used by the CUDA-graph buckets.
"""

import math

import torch

from ftp import AuxBatchedEngine

DEV = torch.device("cpu")
V = 512
_PAGE = 16


def rand_tokens(g: torch.Generator, n: int) -> list[int]:
    return torch.randint(2, V, (n,), generator=g).tolist()


def ref_plan(eng, items, *, sliding):
    """The pre-vectorization per-row plan builder, as ground truth."""
    indptr, indices, lpl = [0], [], []
    for p, j in eng._plane_rows(items):
        st = eng._states[items[j][1]]
        pages, length = eng._row_table(st, p, sliding)
        npg = math.ceil(length / _PAGE)
        indices.extend(pages[:npg])
        indptr.append(len(indices))
        lpl.append(length - (npg - 1) * _PAGE)
    return indptr, indices, lpl


def drive(eng, reqs, outs, steps, g):
    """Decode `steps` tokens for every request, validating mirrors per step."""
    for _ in range(steps):
        eng.step([(rid, p, o) for (rid, p), o in zip(reqs, outs, strict=True)])
        eng._validate_tables()
        for o in outs:
            o.append(int(torch.randint(2, V, (1,), generator=g).item()))


def test_tables_stay_in_sync_through_lifecycle(tiny_flexllama_factory):
    """Hybrid fused pair: prefill short/long, decode through ring evictions
    and page appends, rewind, re-prime, unregister/register — the table
    mirrors must equal the python page lists after every step."""
    eng = AuxBatchedEngine(
        tiny_flexllama_factory(0), DEV, torch.float32, 64,
        model2=tiny_flexllama_factory(1),
    )
    g = torch.Generator().manual_seed(0)
    reqs = [(1, rand_tokens(g, 9)), (2, rand_tokens(g, 40)), (3, rand_tokens(g, 21))]
    for rid, _ in reqs:
        eng.register(rid)
    outs = [[int(torch.randint(2, V, (1,), generator=g).item())] for _ in reqs]
    eng.step([(rid, p, []) for rid, p in reqs])
    eng._validate_tables()

    drive(eng, reqs, outs, 3 * _PAGE, g)  # crosses evictions + page appends

    eng.rewind(1, 2)  # logical drop; lists/tables unchanged but consistent
    eng._validate_tables()
    drive(eng, reqs, outs, 30, g)  # long row crosses window 64 -> re-prime

    eng.unregister(2)
    eng.register(9)
    eng.step([(9, rand_tokens(g, 33), [])])
    eng._validate_tables()
    eng.unregister(9)
    eng._validate_tables()


def test_plan_inputs_match_reference(tiny_flexllama_factory):
    """Packed indptr/indices/lpl from the vectorized builder == the naive
    per-row reference, for both layer groups, mid-lifecycle."""
    eng = AuxBatchedEngine(
        tiny_flexllama_factory(0), DEV, torch.float32, 128,
        model2=tiny_flexllama_factory(1),
    )
    g = torch.Generator().manual_seed(1)
    reqs = [(1, rand_tokens(g, 9)), (2, rand_tokens(g, 40)), (3, rand_tokens(g, 17))]
    for rid, _ in reqs:
        eng.register(rid)
    outs = [[int(torch.randint(2, V, (1,), generator=g).item())] for _ in reqs]
    eng.step([(rid, p, []) for rid, p in reqs])
    drive(eng, reqs, outs, 2 * _PAGE + 3, g)  # past evictions, mid-page slots

    items = [(j, rid, o[-1]) for j, ((rid, _), o) in enumerate(zip(reqs, outs, strict=True))]
    eng._fi_full = object()  # force plan building without flashinfer
    try:
        staged = eng._stage_decode(items, len(items) * eng._replicas)
    finally:
        eng._fi_full = None
    for sliding, ip, ind, lp in (
        (False, staged.indptr_full, staged.indices_full, staged.lpl_full),
        (True, staged.indptr_sw, staged.indices_sw, staged.lpl_sw),
    ):
        r_ip, r_ind, r_lp = ref_plan(eng, items, sliding=sliding)
        assert ip.tolist() == r_ip
        assert ind.tolist() == r_ind
        assert lp.tolist() == r_lp


def test_bucket_padding_is_plane_major(tiny_flexllama_factory):
    """Staging at a bucket larger than R: each plane's real rows sit at the
    START of its half (the fused reshape depends on it); pad rows point at the
    scratch pages with plan length 1."""
    from ftp.engine import (
        _ST_POS,
        _ST_TOK,
        _ST_WRPF,
        _ST_WRPS,
    )

    eng = AuxBatchedEngine(
        tiny_flexllama_factory(0), DEV, torch.float32, 64,
        model2=tiny_flexllama_factory(1),
    )
    g = torch.Generator().manual_seed(2)
    reqs = [(1, rand_tokens(g, 9)), (2, rand_tokens(g, 20)), (3, rand_tokens(g, 33))]
    for rid, _ in reqs:
        eng.register(rid)
    eng.step([(rid, p, []) for rid, p in reqs])
    eng._scratch_full = eng._alloc_full(1)[0]
    eng._scratch_sw = eng._alloc_sw(1)[0]

    n, Rb = len(reqs), 16
    half = Rb // 2
    items = [(j, rid, 7) for j, (rid, _) in enumerate(reqs)]
    eng._fi_full = object()
    try:
        staged = eng._stage_decode(items, Rb)
    finally:
        eng._fi_full = None

    pk = staged.pack
    assert staged.half == half
    # real rows: this step's token everywhere, positions == seq_len
    for p in range(2):
        base = p * half
        assert pk[_ST_TOK, base: base + n].tolist() == [7] * n
        assert pk[_ST_POS, base: base + n].tolist() == [len(pr) for _, pr in reqs]
        # pads: scratch write targets at position 0
        pad = slice(base + n, base + half)
        assert pk[_ST_TOK, pad].eq(0).all() and pk[_ST_POS, pad].eq(0).all()
        assert pk[_ST_WRPF, pad].eq(eng._scratch_full).all()
        assert pk[_ST_WRPS, pad].eq(eng._scratch_sw).all()
    # plan arrays: pads are length-1 scratch reads; reals match the reference
    npg = staged.indptr_full.diff().tolist()
    lpl = staged.lpl_full.tolist()
    for p in range(2):
        for r in range(p * half + n, (p + 1) * half):
            assert npg[r] == 1 and lpl[r] == 1
    r_ip, r_ind, _ = ref_plan(eng, items, sliding=False)
    real_rows = [r for p in range(2) for r in range(p * half, p * half + n)]
    got_real = [staged.indices_full[staged.indptr_full[r]: staged.indptr_full[r + 1]].tolist()
                for r in real_rows]
    ref_real = [r_ind[r_ip[k]: r_ip[k + 1]] for k in range(len(real_rows))]
    assert got_real == ref_real


def test_window_rope_bound_and_pairs_guard(tiny_flexllama_factory):
    """window == rope cache_len must BUILD (the v4 32K pair runs exactly
    there: step() keeps positions <= window-1), window > cache_len must not.
    step_pairs may cross the WINDOW (step() re-primes after) but never the
    ROPE CACHE — the graphed decode skips the per-step bounds check."""
    import pytest

    # fixture rope caches hold 256 positions
    with pytest.raises(ValueError, match="rope cache_len"):
        AuxBatchedEngine(
            tiny_flexllama_factory(0), DEV, torch.float32, 272,
            model2=tiny_flexllama_factory(1),
        )
    eng = AuxBatchedEngine(
        tiny_flexllama_factory(0), DEV, torch.float32, 256,
        model2=tiny_flexllama_factory(1),
    )
    assert eng._window == 256 and eng._rope_cache_len == 256
    g = torch.Generator().manual_seed(3)
    eng.register(1)
    eng.step([(1, rand_tokens(g, 255), [])])  # primed at seq_len 255
    with pytest.raises(ValueError, match="rope cache"):
        eng.step_pairs([(1, [5, 6])])  # positions 255, 256 — 256 is OOB
    eng.step_pairs([(1, [5])])  # position 255 == cache_len - 1: exact
    with pytest.raises(ValueError, match="rope cache"):
        eng.step_pairs([(1, [7])])  # position 256 would be OOB
    # a regular step() re-primes the row instead (window crossed)
    out = eng.step([(1, rand_tokens(g, 200), [5, 7])])
    assert torch.isfinite(out).all()
