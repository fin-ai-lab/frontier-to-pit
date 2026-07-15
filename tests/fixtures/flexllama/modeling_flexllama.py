"""FlexLlama — LLaMA with a per-layer sliding-window / full attention pattern.

Implementation note: this deliberately reuses the *stock* transformers LLaMA
building blocks (attention, MLP, RMSNorm, rotary embedding, decoder layer,
ForCausalLM head) unchanged. The math is therefore identical to
``LlamaForCausalLM`` to the bit. The single behavioural addition is in
``FlexLlamaModel.forward``: instead of building one causal mask and handing it to
every layer, it builds two masks — a full causal mask and a sliding-window causal
mask — and routes each decoder layer to the one named by
``config.layer_types[layer_idx]`` (the exact pattern transformers' own Gemma-3
text model uses).

Sliding-window semantics match our pretraining (``flexattn_patch.py``) exactly:
transformers' ``sliding_window_overlay`` keeps ``kv_idx > q_idx - sliding_window``
which, combined with causality, is ``[q - window + 1, q]`` — identical to the
flex ``mask_mod`` ``(q >= kv) & (q - kv < window)``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

# Pull the building blocks + helpers straight from the installed llama module so we
# stay aligned with whatever transformers version is loading this checkpoint.
from transformers.models.llama import modeling_llama as _L
from transformers.masking_utils import create_sliding_window_causal_mask

from .configuration_flexllama import FlexLlamaConfig

LlamaModel = _L.LlamaModel
LlamaForCausalLM = _L.LlamaForCausalLM
LlamaDecoderLayer = _L.LlamaDecoderLayer
BaseModelOutputWithPast = _L.BaseModelOutputWithPast
DynamicCache = _L.DynamicCache


def _resolve(name, *modules):
    """Fetch a symbol from the first module that has it (transformers moves these
    helpers between minor/major versions; e.g. `check_model_inputs` is re-exported by
    `modeling_llama` in 4.5x but lives in `transformers.utils.generic` in 5.x)."""
    import importlib

    for modname in modules:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        if hasattr(mod, name):
            return getattr(mod, name)
    raise ImportError(f"could not resolve `{name}` from any of {modules}")


def _rope_theta_of(config) -> float:
    """The RoPE base of the GLOBAL rotary, robust across transformers versions
    (5.x stores it in ``config.rope_parameters['rope_theta']``; 4.x in ``rope_theta``)."""
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict) and "rope_theta" in rp:
        return float(rp["rope_theta"])
    return float(config.rope_theta)


create_causal_mask = _resolve(
    "create_causal_mask",
    "transformers.models.llama.modeling_llama",
    "transformers.masking_utils",
)
check_model_inputs = _resolve(
    "check_model_inputs",
    "transformers.models.llama.modeling_llama",
    "transformers.utils.generic",
    "transformers.utils",
)
auto_docstring = _resolve(
    "auto_docstring",
    "transformers.models.llama.modeling_llama",
    "transformers.utils",
)


class PrecomputedRotaryEmbedding(nn.Module):
    """Rotary embedding that REPLAYS an explicit, checkpoint-provided cos/sin cache
    instead of recomputing it analytically.

    Used to ship the EXACT training RoPE buffers of the flex_doc cooldown. Its LOCAL
    (sliding-layer) cache was trained ≈0 (``cos_local[0]`` is noise near 0, not the
    ``[1,1,...]`` a valid rotary starts at) — a real training bug the 23 sliding
    layers adapted to. Injecting a VALID analytic local rope here is catastrophic
    (real-data CE 8.65 vs 2.21 with the ≈0 buffer). So we do NOT compute rope; we
    replay the buffers the run trained with, verbatim.

    The caches are PERSISTENT buffers of shape ``(cache_len, head_dim)`` so they save
    into ``model.safetensors`` and reload via ``from_pretrained``. They are created at
    the correct size in ``__init__`` (BEFORE the weight load) and stored fp32; the
    forward casts to the activation dtype. The buffer layout is ``[half, half]`` over
    ``head_dim`` — identical to both litgpt's ``.repeat(1,2)`` rope cache and HF's
    ``cat((freqs, freqs))`` — so no reordering is needed.
    """

    def __init__(self, cache_len: int, head_dim: int):
        super().__init__()
        self.cache_len = int(cache_len)
        self.register_buffer(
            "cos_cached", torch.zeros(self.cache_len, head_dim, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "sin_cached", torch.zeros(self.cache_len, head_dim, dtype=torch.float32), persistent=True
        )

    @torch.no_grad()
    def forward(self, x, position_ids):
        # position_ids: (bs, seq) long -> (cos, sin) each (bs, seq, head_dim), matching
        # the shape a stock LlamaRotaryEmbedding returns (apply_rotary_pos_emb then
        # unsqueezes dim 1). Guard against reading past the cached context length.
        if position_ids.numel() and int(position_ids.max()) >= self.cache_len:
            raise ValueError(
                f"position id {int(position_ids.max())} >= rope_cache_len {self.cache_len}; "
                f"this precomputed-RoPE FlexLlama only covers positions [0, {self.cache_len})."
            )
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class FlexLlamaDecoderLayer(LlamaDecoderLayer):
    """Stock Llama decoder layer that remembers its attention type."""

    def __init__(self, config: FlexLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention_type = config.layer_types[layer_idx]
        # Record the per-layer sliding window on the attention module. For the
        # forward/no-cache path correctness comes entirely from the mask routed in
        # below; this attribute makes the layer's intent explicit and lets cached
        # generation treat global layers as full attention.
        self.self_attn.sliding_window = (
            config.sliding_window if self.attention_type == "sliding_attention" else None
        )


class FlexLlamaModel(LlamaModel):
    config_class = FlexLlamaConfig

    def __init__(self, config: FlexLlamaConfig):
        super().__init__(config)
        # Replace the stock all-full layer stack with attention-type-aware layers.
        self.layers = nn.ModuleList(
            [FlexLlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.has_sliding_layers = "sliding_attention" in config.layer_types

        if getattr(config, "rope_precomputed", False):
            # Buffer-injection RoPE (the faithful path for the flex_doc cooldown). Replace
            # BOTH the parent's analytic global rotary and the local rotary with caches
            # that REPLAY the exact training buffers (global cos/sin AND the ≈0 local
            # cos_local/sin_local). Populated from the checkpoint at load time. See
            # PrecomputedRotaryEmbedding for why an analytic local rope is WRONG here.
            head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
            cache_len = int(getattr(config, "rope_cache_len", None) or config.max_position_embeddings)
            self.rotary_emb = PrecomputedRotaryEmbedding(cache_len, head_dim)
            self.rotary_emb_local = PrecomputedRotaryEmbedding(cache_len, head_dim)
        else:
            # Analytic dual RoPE (fallback for factor-1 / non-degenerate checkpoints).
            # `self.rotary_emb` (built by the parent from the config's rope params) is the
            # GLOBAL rotary used by full-attention layers -- it carries the llama3 scaling
            # for context-extended checkpoints. Build a second, UNSCALED rotary at
            # `rope_local_base_freq` for the sliding (local) layers, exactly as
            # pretraining/flexattn_patch.py does (cos_local/sin_local are the unscaled
            # cache). When there is no scaling and the bases are equal, the two are
            # numerically identical, so factor-1 checkpoints are byte-for-byte unchanged.
            import copy as _copy

            local_base = float(getattr(config, "rope_local_base_freq", None) or _rope_theta_of(config))
            local_cfg = _copy.deepcopy(config)
            if hasattr(local_cfg, "rope_parameters"):
                # transformers >= 5: rope config is a unified dict; unscaled == rope_type "default".
                local_cfg.rope_parameters = {"rope_type": "default", "rope_theta": local_base}
            else:
                # transformers 4.x: separate attributes.
                local_cfg.rope_scaling = None
                local_cfg.rope_theta = local_base
            self.rotary_emb_local = _L.LlamaRotaryEmbedding(config=local_cfg)

        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Build BOTH masks once and route each layer to the right one. `generate`
        # may already pass a prepared {type: mask} dict, in which case reuse it.
        # Mask-builder kwargs adapt to the installed transformers: 5.6 took
        # `input_embeds` + `cache_position`; 5.12 renamed to `inputs_embeds`
        # and dropped `cache_position`.
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            import inspect as _inspect

            _mask_params = set(_inspect.signature(create_causal_mask).parameters)
            mask_kwargs = {
                "config": self.config,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
            }
            mask_kwargs[
                "inputs_embeds" if "inputs_embeds" in _mask_params else "input_embeds"
            ] = inputs_embeds
            if "cache_position" in _mask_params:
                mask_kwargs["cache_position"] = cache_position
            if "position_ids" in _mask_params:
                mask_kwargs["position_ids"] = position_ids
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        # Dual RoPE: full-attention layers get the (possibly scaled) global rotary,
        # sliding layers get the unscaled local rotary. Routed by layer type exactly
        # like the attention mask above.
        position_embeddings = {
            "full_attention": self.rotary_emb(hidden_states, position_ids),
            "sliding_attention": self.rotary_emb_local(hidden_states, position_ids),
        }

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings[decoder_layer.attention_type],
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class FlexLlamaForCausalLM(LlamaForCausalLM):
    config_class = FlexLlamaConfig

    def __init__(self, config: FlexLlamaConfig):
        super().__init__(config)
        self.model = FlexLlamaModel(config)
        self.post_init()


__all__ = ["FlexLlamaForCausalLM", "FlexLlamaModel"]
