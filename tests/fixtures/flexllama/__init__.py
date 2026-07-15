"""Vendored copy of the FlexLlama trust_remote_code files (v4 aux arch) so the
CPU ground-truth tests can build tiny hybrid sliding/full models without
network or trust_remote_code machinery. Source of truth: the checkpoint dirs
under bll01:/data/gpt-backtest-models-hf/llama-qwen3.5-3b-flexdoc_32768_*."""

from .configuration_flexllama import FlexLlamaConfig
from .modeling_flexllama import FlexLlamaForCausalLM, FlexLlamaModel

__all__ = ["FlexLlamaConfig", "FlexLlamaForCausalLM", "FlexLlamaModel"]
