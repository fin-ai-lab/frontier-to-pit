"""Configuration for FlexLlama — a Llama-architecture model with a Gemma-3-style
per-layer attention pattern (alternating sliding-window / full attention).

The base model is byte-for-byte a LLaMA (SiLU gated MLP, RMSNorm without unit
offset, GQA, rotary embeddings, no biases, no QK-norm, no logit softcapping). The
*only* architectural addition over stock Llama is that a fixed set of layers
(``global_layer_indices``) use full causal attention while every other layer uses
sliding-window causal attention with window ``sliding_window``.

This mirrors how our pretraining stack (``pretraining/flexattn_patch.py``) ran the
3B cooldown: ``attention_impl="flex_doc"`` with ``local_window_size=512`` and
``global_layer_indices=[5, 11, 17, 23, 27]``. Document-boundary masking is a
training-time-only construct (it only changes attention when a packed sequence
contains the ``<|endoftext|>`` separator); for single-document inference it reduces
to the causal/sliding masks defined here.

Dual RoPE: the global (full-attention) layers use ``rope_theta`` + ``rope_scaling``
(the llama3-scaled cache); the local (sliding) layers use an UNSCALED rotary at
``rope_local_base_freq``. For a factor-1 checkpoint with equal bases and no scaling
the two collapse to one and a single ``rope_theta`` is exact -- but for a
context-extended checkpoint (``rope_scaling`` set, e.g. llama3 factor 16) the global
and local caches DIVERGE beyond ``original_max_position_embeddings`` and BOTH are
required. A single unscaled rotary is only faithful up to that length.
"""

from __future__ import annotations

from transformers.models.llama.configuration_llama import LlamaConfig


class FlexLlamaConfig(LlamaConfig):
    r"""LlamaConfig + per-layer sliding-window/full attention pattern.

    Extra args beyond :class:`~transformers.LlamaConfig`:
        global_layer_indices (`list[int]`):
            0-based indices of layers that use **full** causal attention. Every
            other layer uses sliding-window causal attention. Defaults to ``[]``
            (i.e. all-full, identical to stock Llama).
        sliding_window (`int`, defaults to 1024):
            Window size for the sliding-window (local) layers. A query at position
            ``q`` attends to keys in ``[q - sliding_window + 1, q]``.
        layer_types (`list[str]`, optional):
            Per-layer attention type, one of ``"full_attention"`` /
            ``"sliding_attention"``. If omitted it is derived from
            ``global_layer_indices`` and ``num_hidden_layers``.
    """

    model_type = "flexllama"

    def __init__(
        self,
        global_layer_indices=None,
        sliding_window: int = 1024,
        layer_types=None,
        rope_local_base_freq=None,
        rope_precomputed: bool = False,
        rope_cache_len=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.global_layer_indices = list(global_layer_indices) if global_layer_indices else []
        self.sliding_window = sliding_window
        # Dual RoPE: the GLOBAL (full-attention) layers use the standard rotary built
        # from `rope_theta` + `rope_scaling` (so a context-extended checkpoint gets its
        # llama3-scaled RoPE); the LOCAL (sliding) layers use an UNSCALED rotary at
        # `rope_local_base_freq`. Mirrors pretraining/flexattn_patch.py's dual cache.
        # Defaults to `rope_theta` (with rope_scaling=None on both -> single-RoPE, i.e.
        # byte-identical to the pre-dual-RoPE behaviour for factor-1 checkpoints).
        self.rope_local_base_freq = rope_local_base_freq

        # Precomputed (buffer-injection) RoPE. When True, FlexLlamaModel does NOT build
        # analytic rotary embeddings; instead it REPLAYS explicit cos/sin caches that
        # ship inside the checkpoint (model.rotary_emb{,_local}.cos_cached/sin_cached).
        # This is how we ship the EXACT training rope buffers of the flex_doc cooldown,
        # whose LOCAL cache (cos_local/sin_local) was trained ≈0 (a real training bug the
        # model adapted to). Recomputing an analytic local rope here is WRONG for that
        # model (real-data CE 8.65 vs 2.21). Persistent buffers of shape
        # (rope_cache_len, head_dim) are created at load time and populated from the
        # checkpoint. When False, the analytic dual-rope path is used (factor-1 models).
        self.rope_precomputed = bool(rope_precomputed)
        self.rope_cache_len = (
            int(rope_cache_len) if rope_cache_len is not None else int(self.max_position_embeddings)
        )

        if layer_types is None:
            global_set = set(self.global_layer_indices)
            layer_types = [
                "full_attention" if i in global_set else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        self.layer_types = layer_types

        # Sanity: layer_types must agree with depth.
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but "
                f"num_hidden_layers={self.num_hidden_layers}"
            )


__all__ = ["FlexLlamaConfig"]
