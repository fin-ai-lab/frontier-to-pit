import importlib.util

import pytest


def test_package_imports_without_vllm():
    """The core package must import with vLLM absent; the vllm submodule must
    fail with an error that names the missing dependency."""
    import ftp  # noqa: F401

    if importlib.util.find_spec("vllm") is not None:
        pytest.skip("vllm installed in this environment")
    with pytest.raises(ImportError, match="vllm"):
        import ftp.vllm  # noqa: F401
