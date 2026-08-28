"""Divergence Decoding: inference-time knowledge suppression.

A large model P is steered at decode time by two small auxiliary models — p (the
"forget" model, trained WITH the knowledge to suppress) and q (the "retain"
model, trained WITHOUT it) — via the fused distribution

    l̂ = l_P + α · (l_q − l_p)

with special/control tokens pinned to P's probabilities so structure (EOS,
chat markup) is never distorted.

The vLLM integration lives in :mod:`ftp.vllm` and is not
imported here so the core package works without vLLM installed.
"""

from importlib.metadata import PackageNotFoundError, version

from ftp.config import DDConfig
from ftp.core import build_special_ids, dd_fuse, log1mexp
from ftp.engine import AuxBatchedEngine, PagedAuxEngine
from ftp.paired import (
    PairedEmbedding,
    PairedLinear,
    PairedRMSNorm,
    check_fusable,
    fuse_pair,
)
from ftp.steering import (
    SaeSource,
    SteerSpec,
    install_steering,
    remove_steering,
    steered_vllm,
)
from ftp.translate import (
    ChatTemplateAdapter,
    StreamRetokenizer,
    TokenTextTable,
    UniversalBridge,
    VocabMapper,
    make_chat_adapter,
)

try:
    __version__ = version("frontier-to-pit")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"

__all__ = [
    "AuxBatchedEngine",
    "PagedAuxEngine",
    "ChatTemplateAdapter",
    "DDConfig",
    "PairedEmbedding",
    "PairedLinear",
    "PairedRMSNorm",
    "SaeSource",
    "SteerSpec",
    "StreamRetokenizer",
    "TokenTextTable",
    "UniversalBridge",
    "VocabMapper",
    "__version__",
    "build_special_ids",
    "check_fusable",
    "dd_fuse",
    "fuse_pair",
    "install_steering",
    "log1mexp",
    "make_chat_adapter",
    "remove_steering",
    "steered_vllm",
]
