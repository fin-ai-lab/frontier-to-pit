import os

import pytest
import torch


def _need(var: str) -> str:
    val = os.environ.get(var) or os.environ.get("DD_TEST_AUX_MODEL")
    if val is None:
        pytest.skip(f"set {var} (or DD_TEST_AUX_MODEL) to a local checkpoint path")
    return val


@pytest.fixture(scope="session")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def aux_model_path():
    """A real aux checkpoint (LlamaForCausalLM), e.g. the le2015 retain model."""
    return _need("DD_TEST_AUX_MODEL")


@pytest.fixture(scope="session")
def aux_model_path_q():
    """A second aux checkpoint with the SAME architecture (e.g. le2025) for
    fused-pair tests. Falls back to the primary aux model — self-fusion still
    exercises every fused code path, but genuinely different weights also
    catch plane-swap/marriage bugs, so set DD_TEST_AUX_MODEL_Q when you can."""
    return os.environ.get("DD_TEST_AUX_MODEL_Q") or _need("DD_TEST_AUX_MODEL")


@pytest.fixture(scope="session")
def tokenizer_path():
    return _need("DD_TEST_TOKENIZER")
