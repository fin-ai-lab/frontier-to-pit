"""Full-vocab GPU equivalence test: batched dd_fuse == per-row calls.

(The same property is covered at V=512 on CPU in tests/test_core.py; this
verifies it at the real vocab size on CUDA.)
"""

import pytest
import torch
from transformers import AutoTokenizer

from ftp import build_special_ids, dd_fuse

N = 32


@pytest.fixture(scope="module")
def setup(cuda_device, tokenizer_path):
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    v = max(len(tok), max(build_special_ids(tok)) + 1)
    wl = torch.tensor(build_special_ids(tok), device=cuda_device, dtype=torch.long)
    wl_mask = torch.zeros(v, dtype=torch.bool, device=cuda_device)
    wl_mask[wl[wl < v]] = True
    g = torch.Generator(device=cuda_device).manual_seed(0)
    logits = [torch.randn(N, v, generator=g, device=cuda_device) * 4 for _ in range(3)]
    return logits, wl, wl_mask


def test_batched_equals_looped(setup):
    (lP, lp, lq), wl, wl_mask = setup
    kw = dict(alpha=1.5, pin=True)
    batched = dd_fuse(lP, lp, lq, wl_mask=wl_mask, **kw)
    for i in range(N):
        row = dd_fuse(lP[i : i + 1], lp[i : i + 1], lq[i : i + 1], whitelist=wl, **kw)
        torch.testing.assert_close(row[0], batched[i], rtol=1e-5, atol=1e-5, equal_nan=True)
