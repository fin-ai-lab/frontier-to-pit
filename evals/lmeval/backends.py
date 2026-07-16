"""Custom lm-eval model backends — only what the stock backends can't provide.

Three of our models need glue on top of lm-eval's built-ins:

* **PIT** (``pit``) — a bespoke ``PITForCausalLM`` (HF, ``trust_remote_code``) whose
  attention ignores the ``attention_mask`` (hardcoded ``is_causal=True``) and has no
  KV cache. Stock ``hf`` would corrupt any padded batch, forcing ``batch_size=1``. We
  monkeypatch its attention to honor a causal+key-padding mask so it batches at large
  sizes. RoPE is relative, so a uniform left-pad offset cancels in every real·real dot
  product — masking alone is correct, no position-id surgery needed.
* **ChronoGPT** (``chronogpt``) — a modded-nanogpt arch loaded from the repo's own
  ``ChronoGPT_instruct.py`` with a tiktoken tokenizer and a custom ``generate``; not
  loadable by vLLM or HF. A thin ``LM`` wraps its loader + generate for ``generate_until``.
* **DD** (``dd``, deferred) — Divergence Decoding (+ optional SAE steering) as a thin
  ``VLLM`` subclass. NOT used by the small-model sweep; only ``ftp_qwen``.

``le2015``/``le2025``/``qwen3_5_2b`` use the **stock** ``vllm`` backend (see
``__main__.py``); they need no code here.
"""

from __future__ import annotations

import sys

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.vllm_causallms import VLLM
from tqdm import tqdm

# --- Per-GPU cap on PIT's HF auto-batch (vLLM models self-batch via continuous batching).
# PIT now has a KV cache, so its memory scales with batch x seq-length: lm-eval auto-batch
# probes the largest batch that fits at max_length, and these cap that probe per GPU so it
# starts sensibly (and stays "far greater than 8" where memory allows). ---
BATCH_BY_GPU: dict[str, int] = {"H200": 256, "H100": 128, "A100": 96, "L40S": 64, "A40": 32}
_DEFAULT_HF_BATCH = 16


def gpu_batch_size(default: int = _DEFAULT_HF_BATCH) -> int:
    """Hardcoded batch size for the running GPU (substring match on the device name)."""
    import torch

    if not torch.cuda.is_available():
        return default
    name = torch.cuda.get_device_name(0).upper()
    for key, bs in BATCH_BY_GPU.items():
        if key in name:
            return bs
    return default


# PIT-4B-FT-201511 prompt format — why we send Alpaca, in full, because it is contested.
#
# PIT ships two mutually inconsistent formats, and neither of us picked this fight:
#   * its SFT tokenizer (post_training/sft_tokens.py) trains on <|user|>/<|assistant|>/<|end|>
#     role markers, which its HF model card also documents;
#   * its own IFEval script (eval/ifeval_test.py) evaluates HF candidates with Alpaca
#     "### Instruction:/### Response:".
# We took the format PIT's own evaluator uses. A third-party audit (documents/pit_issues.md)
# argues that was the wrong call. We then went and measured it rather than argue.
#
# WHAT WE TRIED, in good faith, to make this checkpoint work:
#   1. Both templates. The card's role-marker format runs as `pit_4b_2015_chat*` (see the
#      specs in __main__.py); results in /data/lab/frontier-to-pit/results-pit-cardformat.
#   2. Both decodings, since the card recommends sampling and our suite runs greedy: the
#      card format at its own T=0.7/top_p=0.9 AND at greedy, so format and decoding are
#      separated rather than confounded.
#   3. <|end|> stopping, as the card specifies (see PITCardHFLM.generate_until).
#
# WHAT WE FOUND (PIT-4B-FT-201511, full utility suite, metrics per tools/aggregate_score.py):
#
#                                 MMLU-Pro  MMLU-Redux   GPQA-D   IFEval  HumanEval
#     Alpaca + greedy  (shipped)     0.0000      0.2205   0.1616   0.2421     0.0000
#     card chat + T=0.7/p=0.9        0.0024      0.0012   0.0000   0.1257     0.0000
#     card chat + greedy             0.0000      0.0000   0.0000   0.1275     0.0000
#
# The Alpaca format we ship is this checkpoint's BEST case. The documented format scores
# lower on every task that has any signal and collapses MMLU-Redux/GPQA to zero — it emits
# bare role-marker loops, so nothing is extractable. Card-greedy ~= card-sampling, so this
# is the format, not the decoding. Our formatter choice therefore flatters PIT; it does not
# penalise it, and no number on the site depends on which of the two we use.
#
# THE FAST RUNNER IS NOT THE CAUSE. _patch_pit_kv_cache below gives PIT a KV cache it does
# not ship. The same audit checked it against stock full-prefix recomputation on an A10:
# 64/64 greedy tokens identical on all three site prompts, mean |logit| delta 0.037–0.067
# against max logit magnitude ~29–33 (bf16 kernel/op-order noise), and not one greedy
# decision changed. Behaviourally equivalent for greedy; do not extend that claim to
# sampling, where small logit deltas can eventually diverge even at a fixed seed.
#
# PIT-4B-FT-202412 IS FINE — AND IS NOT WHAT WE BENCHMARK. The audit's 2024 control does
# respond under the role-marker format, so PIT's architecture, HF wrapper and export stack
# are not broken; the defect is specific to the 2015 checkpoint. But this benchmark is a
# 2015 point-in-time comparison: a 2024-cutoff model has no look-ahead constraint to hold
# and cannot stand in for 201511. 201511 is the only PIT checkpoint at our cutoff, so it is
# the one that gets compared, degenerate or not. That it degenerates under BOTH formats is
# the audit's own finding, not ours.
PIT_CHAT_TEMPLATE = (
    # Alpaca has no system slot: a system message (the temporal tasks' per-doc
    # system_prompt) renders as a preamble paragraph before the instruction block.
    # No-system prompts (all utility tasks) are byte-identical to the pre-system template.
    "{% for m in messages %}{% if m['role'] == 'system' %}"
    "{{ m['content'] }}\n\n"
    "{% elif m['role'] == 'user' %}"
    "### Instruction:\n{{ m['content'] }}\n\n### Response:\n"
    "{% elif m['role'] == 'assistant' %}{{ m['content'] }}"  # gen_prefix (e.g. humaneval)
    "{% endif %}{% endfor %}"
)

# TiMaGPT2 is a bare completion GPT-2 (never trained on chat/instruct markers or a system
# prompt), so its "chat template" is pass-through: emit each message's content verbatim and
# let the model continue the raw text. No role scaffolding, no generation-prompt suffix.
TIMAGPT2_CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"

# ChronoGPT's training-time system preamble (Alpaca instruct). Mirrors evals/harness.py.
CHRONO_SYSTEM = (
    "You are ChronoGPT, a large language model trained by ManelaLab at WashU.\n"
    "Below is an instruction that describes a task.\n"
    "Write a response that appropriately completes the request."
)


# ──────────────────── PIT: KV-cache + batch-safe modeling patch ────────────────────


def _patch_pit_kv_cache(model) -> None:
    """Give a loaded PIT model a KV cache + batch/pad-safe attention.

    Stock PIT has no KV cache (``_supports_cache_class=False``; ``forward`` recomputes
    the full sequence every step) and its attention ignores ``attention_mask``. We
    rewrite the forward chain to:

    * apply RoPE at **absolute cache positions** (PIT's ``Rotary`` keys off the raw seq
      index, which is wrong once decoding feeds one token at a time) — reproducing the
      full-recompute result exactly;
    * use a transformers ``DynamicCache`` so decoding is O(T) instead of O(T^2);
    * honor a causal + key-padding mask so left-padded batches are correct (no padding =>
      ``is_causal=True`` fast path, valid for both prefill and cached decode);
    * project ``lm_head`` on the last token only during generation (a ``[B,T,vocab]``
      float32 tensor would otherwise dominate memory at a large hardcoded batch size).

    Generation then runs through the generic (cache-aware) ``prepare_inputs_for_generation``.
    """
    import torch
    import torch.nn.functional as F
    from transformers import DynamicCache
    from transformers.modeling_outputs import CausalLMOutputWithPast

    model_cls = type(model)
    if getattr(model_cls, "_dd_kv_patched", False):
        return
    mod = sys.modules[model_cls.__module__]
    apply_rotary = mod._apply_rotary_emb
    blocks = model.transformer["h"]
    block_cls, attn_cls = type(blocks[0]), type(blocks[0].attn)
    for i, blk in enumerate(blocks):
        blk.attn.layer_idx = i  # DynamicCache keys k/v by layer index

    def rope_cos_sin(position_ids, head_dim, device, dtype):
        # Matches PIT's Rotary (base 10000) exactly, but at arbitrary absolute positions.
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
        )
        freqs = position_ids[..., None].float() * inv_freq  # [B, T, hd/2]
        return freqs.cos().to(dtype)[:, :, None, :], freqs.sin().to(dtype)[:, :, None, :]

    def attn_forward(self, x, attn_bias=None, position_ids=None, past_key_values=None,
                     cache_position=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = rope_cos_sin(position_ids, self.head_dim, x.device, q.dtype)
        q = apply_rotary(F.rms_norm(q, (q.size(-1),)), cos, sin).transpose(1, 2)
        k = apply_rotary(F.rms_norm(k, (k.size(-1),)), cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)  # [B, nh, T, hd]
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias,
                                            is_causal=attn_bias is None)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

    def block_forward(self, x, attn_bias=None, position_ids=None, past_key_values=None,
                      cache_position=None):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)), attn_bias=attn_bias,
                          position_ids=position_ids, past_key_values=past_key_values,
                          cache_position=cache_position)
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

    def model_forward(self, input_ids=None, attention_mask=None, labels=None,
                      past_key_values=None, use_cache=None, cache_position=None,
                      position_ids=None, **kwargs):
        x = self.transformer["wte"](input_ids)
        B, T = input_ids.shape[0], input_ids.shape[1]
        use_cache = use_cache if use_cache is not None else (labels is None)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        total_len = past_seen + T
        if cache_position is None:
            cache_position = torch.arange(past_seen, total_len, device=x.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(B, -1)  # absolute positions

        # is_causal=True (fast flash path) is only valid for a SQUARE prefill
        # (q_len == kv_len). With a cache (decode: q_len=1, kv_len=total) SDPA's is_causal
        # uses top-left alignment, which would make the new token attend to only the first
        # cached key — so we build an explicit position-based causal mask whenever a cache
        # is present. Padding also requires the mask.
        attn_bias = None
        has_pad = attention_mask is not None and not bool(attention_mask.all())
        if past_seen > 0 or has_pad:
            key_pos = torch.arange(total_len, device=x.device)
            # causal by absolute position: query at pos p attends to keys 0..p
            keep = key_pos[None, None, None, :] <= position_ids[:, None, :, None]
            if attention_mask is not None:
                keep = keep & attention_mask.bool()[:, None, None, :]
            diag = key_pos[None, None, None, :] == position_ids[:, None, :, None]
            attn_bias = keep | diag  # diagonal avoids NaN on fully-padded query rows

        for blk in self.transformer["h"]:
            x = blk(x, attn_bias=attn_bias, position_ids=position_ids,
                    past_key_values=past_key_values, cache_position=cache_position)
        x = F.rms_norm(x, (x.size(-1),))
        hid = x[:, -1:, :] if labels is None else x  # generation needs only the last token
        logits = self.lm_head(hid).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=past_key_values if use_cache else None
        )

    def prepare_inputs(self, input_ids, past_key_values=None, attention_mask=None,
                       use_cache=True, **kwargs):
        # transformers 5.x drops cache_position for remote-code models and won't slice
        # input_ids itself, so we slice to the unprocessed tokens when a cache is present.
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            input_ids = input_ids[:, past_key_values.get_seq_length():]
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "past_key_values": past_key_values, "use_cache": use_cache}

    attn_cls.forward = attn_forward
    block_cls.forward = block_forward
    model_cls.forward = model_forward
    model_cls.prepare_inputs_for_generation = prepare_inputs
    model_cls._supports_cache_class = True
    model_cls._dd_kv_patched = True
    model_cls._dd_batch_patched = True


@register_model("pit")
class PITHFLM(HFLM):
    """Stock HF backend for PIT-4B-FT + KV-cache/batch-safe patch + Alpaca template."""

    def __init__(self, pretrained, batch_size=None, max_length=2048, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("dtype", "bfloat16")
        if batch_size is None:
            # Auto-fit: the KV cache memory scales with seq length, so a fixed large
            # batch that fits short tasks OOMs on long-context ones (mmlu_pro 5-shot ~ a
            # 2048-token cache). lm-eval probes the largest batch that fits at max_length
            # (which bounds the cache), capped per GPU. Still "far greater than 8" where
            # memory allows; never OOMs.
            kwargs.setdefault("max_batch_size", gpu_batch_size())
            batch_size = "auto"
        super().__init__(
            pretrained=pretrained, batch_size=batch_size, max_length=max_length, **kwargs
        )
        _patch_pit_kv_cache(self.model)
        # PIT's tokenizer ships no usable chat template; install the Alpaca format so
        # lm-eval's apply_chat_template renders the prompt the official PIT eval uses.
        self.tokenizer.chat_template = PIT_CHAT_TEMPLATE


# PIT-4B-FT's own model card documents this format instead of the Alpaca one above:
#
#     def format_prompt(instruction): return f"<|user|>\n{instruction}\n<|assistant|>\n"
#
# with generation stopped at <|end|> and sampled at T=0.7 / top_p=0.9. `pit_4b_2015` keeps
# the Alpaca format (that is what the published numbers were produced with, and what the
# PIT repo's own IFEval script uses); this runs the documented format alongside it so the
# two are directly comparable rather than one replacing the other.
PIT_CARD_CHAT_TEMPLATE = (
    # The card format has no system slot either: render a system message (temporal
    # tasks' per-doc system_prompt) as a leading paragraph before the first role marker.
    "{% for m in messages %}{% if m['role'] == 'system' %}"
    "{{ m['content'] }}\n\n"
    "{% elif m['role'] == 'user' %}"
    "<|user|>\n{{ m['content'] }}\n<|assistant|>\n"
    "{% elif m['role'] == 'assistant' %}{{ m['content'] }}"  # gen_prefix (e.g. humaneval)
    "{% endif %}{% endfor %}"
)
PIT_END = "<|end|>"


@register_model("pit_chat")
class PITCardHFLM(PITHFLM):
    """PIT-4B-FT prompted as its model card documents: role markers + <|end|> stop.

    Identical weights, KV patch and batch handling to :class:`PITHFLM` -- the prompt
    format and the stop string are the only differences, so a diff against the ``pit``
    results isolates the format.
    """

    def __init__(self, pretrained, **kwargs):
        super().__init__(pretrained=pretrained, **kwargs)
        self.tokenizer.chat_template = PIT_CARD_CHAT_TEMPLATE  # replaces PITHFLM's Alpaca

    def generate_until(self, requests, disable_tqdm: bool = False):
        # Append <|end|> to each request's stop strings rather than passing `until` through
        # the spec's gen_kwargs: lm-eval merges gen_kwargs into the task config with a dict
        # update, so an `until` there would REPLACE each task's own stop strings and break
        # its answer extraction. <|end|> is ordinary GPT-2 text under this tokenizer, not an
        # atomic special token, so a stop string is the only way to honour it.
        for req in requests:
            kwargs = req.args[1]
            if not isinstance(kwargs, dict):
                continue
            until = kwargs.get("until") or []
            if isinstance(until, str):
                until = [until]
            if PIT_END not in until:
                kwargs["until"] = [*until, PIT_END]
        return super().generate_until(requests, disable_tqdm=disable_tqdm)


@register_model("timagpt2")
class TiMaGPT2LM(HFLM):
    """Ti-Ma/TiMaGPT2-* temporal GPT-2 baselines (point-in-time knowledge cutoff).

    A bare GPT-2 (1024-token context, model_type ``gpt2`` under a Graphcore
    ``PoptorchPipelinedGPT2LMHeadModel`` arch label). Loaded on the stock HF backend:
    ``AutoModelForCausalLM`` maps model_type=gpt2 -> ``GPT2LMHeadModel`` (the Poptorch
    ``architectures`` label and any auto_map are ignored with trust_remote_code off, so we
    never need optimum-graphcore). The tokenizer ships no chat template, so we install a
    pass-through one (:data:`TIMAGPT2_CHAT_TEMPLATE`) — the model is a completion LM, so the
    fair rendering is the raw prompt text, not chat/instruct scaffolding.
    """

    def __init__(self, pretrained, batch_size=None, max_length=1024, **kwargs):
        kwargs.setdefault("dtype", "bfloat16")
        if batch_size is None:
            kwargs.setdefault("max_batch_size", gpu_batch_size())
            batch_size = "auto"
        super().__init__(
            pretrained=pretrained, batch_size=batch_size, max_length=max_length, **kwargs
        )
        self.tokenizer.chat_template = TIMAGPT2_CHAT_TEMPLATE


# ──────────────── ChronoGPT: KV-cache + batch-safe modeling patch ───────────────

_CHRONO_EOS = 50256  # tiktoken gpt2 <|endoftext|>
_CHRONO_STOP_WIN = 24  # tokens of generated tail decoded each step to test `until` stops


class _ChronoKVCache:
    """Per-attention-layer KV cache for ChronoGPT's custom incremental decode.

    Holds one ``[B, n_head, T, head_dim]`` key/value pair per attention-layer index.
    Beyond ``update`` (append the step's k/v), it supports the two batch ops the custom
    generation loop needs and a transformers ``DynamicCache`` doesn't expose cleanly:
    ``select_rows`` (drop finished sequences from the batch) and ``evict_left`` (slide
    the context window by discarding the oldest cached positions).
    """

    def __init__(self, torch):
        self._t = torch
        self._k: dict = {}
        self._v: dict = {}

    def seq_len(self) -> int:
        return next(iter(self._k.values())).shape[2] if self._k else 0

    def update(self, k, v, layer_idx):
        if layer_idx in self._k:
            k = self._t.cat([self._k[layer_idx], k], dim=2)
            v = self._t.cat([self._v[layer_idx], v], dim=2)
        self._k[layer_idx], self._v[layer_idx] = k, v
        return k, v

    def select_rows(self, sel) -> None:
        for layer in self._k:
            self._k[layer] = self._k[layer].index_select(0, sel)
            self._v[layer] = self._v[layer].index_select(0, sel)

    def evict_left(self, n: int) -> None:
        for layer in self._k:
            self._k[layer] = self._k[layer][:, :, n:, :]
            self._v[layer] = self._v[layer][:, :, n:, :]


def _patch_chrono_kv_cache(model) -> None:
    """Give ChronoGPT a KV cache + batch/pad-safe attention for fast generation.

    Stock ChronoGPT (``ChronoGPT_instruct.py``) recomputes the **whole** window every
    decode step (O(T^2) over a generation), hardcodes ``is_causal=True`` (so a
    left-padded batch lets real tokens attend to pad) and projects ``lm_head`` over the
    full sequence. Its bundled ``kv_cache`` buffer is unusable: RoPE is applied *before*
    the cache concat (a cached single-token step would rotate the new token at position
    0) and the assembled ``present`` is never returned. We rewrite the forward chain to:

    * apply RoPE at **absolute positions** (``rope_pos``) so a cached step rotates the new
      token at its true position. RoPE here is relative (q·k depends only on the offset;
      half the modded-nanogpt freqs are zero, the rest rotate by absolute position), so
      cached keys keep their original bake and the offset to the newest query stays
      correct even after sliding-window eviction — and a uniform left-pad shift cancels in
      every real·real product, so masking alone handles padding (same as the PIT patch);
    * keep a per-layer ``_ChronoKVCache`` so decoding is O(T), not O(T^2);
    * honor a causal + key-padding mask. No padding on a square prefill => ``is_causal``
      flash fast-path; a cache present => explicit position mask, since SDPA's
      ``is_causal`` top-left alignment is wrong once q_len=1 < kv_len (the same trap the
      bundled cache fell into). Mask indices are cache-relative (the causal/padding
      structure is window-local); ``rope_pos`` carries the absolute positions;
    * project ``lm_head`` on the **last token only**, killing the ``[B,T,vocab]`` alloc.

    The U-Net skip connections are within a single forward (decoder layer i adds encoder
    layer i's output for the *same* tokens), so incremental decode reproduces the full
    computation exactly. Value embeddings are per-token and already folded into cached
    ``v``, so they need no separate caching.
    """
    import torch
    import torch.nn.functional as F

    model_cls = type(model)
    if getattr(model_cls, "_dd_kv_patched", False):
        return
    blocks = model.blocks
    block_cls = type(blocks[0])
    attn = next(b.attn for b in blocks if b.attn is not None)
    attn_cls, head_dim = type(attn), attn.head_dim
    li = 0
    for blk in blocks:  # cache keys k/v by attention-layer index, in execution order
        if blk.attn is not None:
            blk.attn.layer_idx = li
            li += 1

    # ChronoGPT's `norm` is just weightless RMSNorm over the last dim — define it here
    # rather than importing from the importlib-loaded (not in sys.modules) model module.
    def norm(t):
        return F.rms_norm(t, (t.size(-1),))

    def rope_cos_sin(position_ids, device):
        # Matches ChronoGPT's Rotary (base 1/1024, half the freqs zeroed) exactly, but at
        # arbitrary absolute positions instead of the fixed 0..T-1.
        af = (1.0 / 1024.0) ** torch.linspace(
            0, 1, steps=head_dim // 4, device=device, dtype=torch.float32
        )
        af = torch.cat([af, af.new_zeros(head_dim // 4)])        # [head_dim/2]
        theta = position_ids.float()[:, None] * af[None, :]      # [T, head_dim/2]
        return theta.cos()[None, :, None, :], theta.sin()[None, :, None, :]  # [1,T,1,hd/2]

    def apply_rotary(x, cos, sin):  # x: [B, T, n_head, head_dim]
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1).type_as(x)

    def attn_forward(self, x, ve, attn_bias=None, cos=None, sin=None, past_key_values=None):
        B, T = x.size(0), x.size(1)
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.num_heads, self.head_dim)
        v = self.lambdas[0] * v + (self.lambdas[1] * ve.view_as(v) if ve is not None else 0)
        q, k = norm(q), norm(k)
        q = apply_rotary(q, cos, sin).transpose(1, 2)  # [B, nh, T, hd]
        k = apply_rotary(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias,
                                            is_causal=attn_bias is None)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, -1))

    def block_forward(self, x, ve, x0, attn_bias=None, cos=None, sin=None,
                      past_key_values=None):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, attn_bias, cos, sin, past_key_values)
        return x + self.mlp(norm(x))

    def model_forward(self, inputs, attention_mask=None, past_key_values=None, rope_pos=None):
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        B, T = inputs.shape
        dev = inputs.device
        past_len = past_key_values.seq_len() if past_key_values is not None else 0
        total = past_len + T
        if rope_pos is None:  # no cache: positions are just 0..T-1 (full recompute)
            rope_pos = torch.arange(past_len, total, device=dev)
        cos, sin = rope_cos_sin(rope_pos, dev)

        # Cache-relative causal + key-padding mask (rope_pos carries absolute positions;
        # the mask only needs the window-local structure). A square no-pad prefill skips
        # it and takes SDPA's is_causal fast path.
        attn_bias = None
        has_pad = attention_mask is not None and not bool(attention_mask.all())
        if past_len > 0 or has_pad:
            q_pos = torch.arange(past_len, total, device=dev)
            k_pos = torch.arange(total, device=dev)
            keep = k_pos[None, None, None, :] <= q_pos[None, None, :, None]  # causal
            if attention_mask is not None:
                keep = keep & attention_mask.bool()[:, None, None, :]
            diag = k_pos[None, None, None, :] == q_pos[None, None, :, None]
            attn_bias = keep | diag  # diagonal avoids NaN on fully-padded query rows

        x0 = norm(self.embed(inputs).bfloat16())
        x = x0
        ve = [self.value_embeds(inputs[i].view(-1)) for i in range(B)]
        ve = [torch.stack([ve[b][i] for b in range(B)]) if ve[0][i] is not None else None
              for i in range(len(ve[0]))]
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]
        skip = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0, attn_bias, cos, sin, past_key_values)
            skip.append(x)
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip.pop()
            x = self.blocks[self.num_encoder_layers + i](
                x, ve_dec[i], x0, attn_bias, cos, sin, past_key_values)
        logits = self.lm_head(norm(x)[:, -1:, :])  # last token only
        return (15 * torch.tanh(logits / 15)).float()

    attn_cls.forward = attn_forward
    block_cls.forward = block_forward
    model_cls.forward = model_forward
    model_cls._dd_kv_patched = True
    model_cls._dd_batch_patched = True


# ─────────────────────────── ChronoGPT: thin custom LM ────────────────────────────


@register_model("chronogpt")
class ChronoGPTLM(LM):
    """ChronoGPT-Instruct (bespoke modded-nanogpt) as a generate-only lm-eval model.

    Loads the repo's own ``ChronoGPT_instruct.py`` + tiktoken gpt2 and applies a
    KV-cache + batch/pad-safe modeling patch (see ``_patch_chrono_kv_cache``).
    ``generate_until`` prefills each left-padded batch once, then decodes one token at a
    time against the cache (O(T) instead of upstream's O(T^2) full-window recompute),
    stops each sequence as soon as it emits ``eos`` or an ``until`` string (upstream
    ``generate`` only broke on ``eos`` and ran to the token cap), and compacts finished
    rows out of the batch + cache so one long generation doesn't hold up the rest. Only
    ``generate_until`` is supported (we run generative tasks only).
    """

    def __init__(self, pretrained, max_length=1792, batch_size=None, device="cuda", **kwargs):
        super().__init__()
        import importlib.util

        import tiktoken
        import torch
        from huggingface_hub import hf_hub_download

        self._pretrained = pretrained
        self._max_length = int(max_length)
        self._device = device
        self._torch = torch
        mod_path = hf_hub_download(pretrained, "ChronoGPT_instruct.py")
        ispec = importlib.util.spec_from_file_location("chronogpt_mod", mod_path)
        mod = importlib.util.module_from_spec(ispec)
        ispec.loader.exec_module(mod)
        self._mod = mod
        self.model = mod.ChronoGPT.from_pretrained(pretrained).to(device).eval()
        _patch_chrono_kv_cache(self.model)
        self.enc = tiktoken.get_encoding("gpt2")
        # KV-cache memory scales with batch x window; cap below the GPU's batch default.
        self._batch_size = int(batch_size) if batch_size else min(gpu_batch_size(), 64)

    @property
    def tokenizer_name(self) -> str:
        return self._pretrained.replace("/", "__")

    def apply_chat_template(self, chat_history, add_generation_prompt=True) -> str:
        instr = "\n".join(m["content"] for m in chat_history)
        return f"\n\n### Instruction:\n{CHRONO_SYSTEM}\n{instr}\n\n### Input:\n### Response:\n"

    @staticmethod
    def _gen_sig(gen_kwargs: dict):
        until = gen_kwargs.get("until") or []
        if isinstance(until, str):
            until = [until]
        return (
            int(gen_kwargs.get("max_gen_toks", 512)),
            float(gen_kwargs.get("temperature", 0.0) or 0.0),
            int(gen_kwargs.get("top_k") or 0),
            tuple(until),
        )

    def _generate_batch(self, id_lists, max_new, temperature, top_k, until) -> list[str]:
        """KV-cached left-padded batched greedy/sampled decode with per-sequence early-stop.

        Prefills the (left-padded) prompts once, then decodes one token at a time against a
        growing per-layer KV cache, bounding context to a ``max_length`` window by evicting
        the oldest cache entries — RoPE is relative so the offset to the newest query stays
        within the window (in-distribution positions). Finished rows are compacted out of
        the batch + cache so one long generation doesn't hold up the rest.

        For sequences that fit in ``max_length`` (every task but mmlu_pro's long 5-shot
        prompts) this is bit-identical to upstream's full-recompute ``idx[:, -ctx:]`` decode.
        Past that, it's a *streaming* window (cached k/v retain evicted tokens' deep-layer
        influence) rather than upstream's per-step full recompute — chosen to keep positions
        in-distribution while still O(T) instead of O(T^2).
        """
        torch = self._torch
        dev, eos, ctx = self._device, _CHRONO_EOS, self._max_length
        B = len(id_lists)
        L0 = max(len(x) for x in id_lists)
        idx = torch.full((B, L0), eos, dtype=torch.long)
        attn = torch.zeros((B, L0), dtype=torch.long)
        for b, ids in enumerate(id_lists):
            idx[b, L0 - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attn[b, L0 - len(ids):] = 1
        idx, attn = idx.to(dev), attn.to(dev)
        if idx.size(1) > ctx:  # never prefill beyond the context window
            idx, attn = idx[:, -ctx:], attn[:, -ctx:]

        def sample(logits):
            if top_k:
                kth = torch.topk(logits, top_k).values[:, -1:]
                logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
            if temperature and temperature > 0.0:
                return torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)
            return torch.argmax(logits, dim=-1, keepdim=True)

        cache = _ChronoKVCache(torch)
        n_pos = idx.size(1)  # absolute position of the next (first generated) token
        with torch.no_grad():
            logits = self.model(idx, attention_mask=attn, past_key_values=cache,
                                 rope_pos=torch.arange(n_pos, device=dev))[:, -1, :]

        active = list(range(B))         # tensor row r -> original request index active[r]
        gen = [[] for _ in range(B)]    # generated token ids per original request
        for _ in range(max_new):
            if not active:
                break
            nxt = sample(logits)  # [len(active), 1]
            keep = []
            for r, tok in enumerate(nxt.squeeze(1).tolist()):
                orig = active[r]
                if tok == eos:
                    continue  # done; eos itself is not part of the response
                gen[orig].append(tok)
                tail = self.enc.decode(gen[orig][-_CHRONO_STOP_WIN:])
                if until and any(s in tail for s in until):
                    continue  # `until` hit; stop (final truncation drops it + trailing)
                keep.append(r)
            if len(keep) < len(active):  # compact finished rows out of the batch + cache
                sel = torch.tensor(keep, device=dev, dtype=torch.long)
                cache.select_rows(sel)
                attn, nxt = attn.index_select(0, sel), nxt.index_select(0, sel)
                active = [active[r] for r in keep]
            if not active:
                break
            if attn.size(1) >= ctx:  # slide the window: keep only the last ctx-1 cached keys
                drop = attn.size(1) - ctx + 1
                cache.evict_left(drop)
                attn = attn[:, drop:]
            attn = torch.cat(
                [attn, torch.ones((len(active), 1), dtype=attn.dtype, device=dev)], 1)
            with torch.no_grad():
                logits = self.model(nxt, attention_mask=attn, past_key_values=cache,
                                     rope_pos=torch.tensor([n_pos], device=dev))[:, -1, :]
            n_pos += 1

        out = []
        for b in range(B):
            text = self.enc.decode(gen[b]).replace("### Response:", "")
            for s in until:
                cut = text.find(s)
                if cut != -1:
                    text = text[:cut]
            out.append(text.strip())
        return out

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        from collections import defaultdict

        results: list = [None] * len(requests)
        groups: dict = defaultdict(list)  # homogeneous gen_kwargs -> [(idx, token_ids)]
        for i, req in enumerate(requests):
            ctx_str = req.args[0]
            ids = self.enc.encode(ctx_str, allowed_special={"<|endoftext|>"})
            groups[self._gen_sig(req.args[1])].append((i, ids))

        pbar = tqdm(total=len(requests), disable=disable_tqdm, desc="chronogpt")
        for (max_new, temperature, top_k, until), items in groups.items():
            items.sort(key=lambda t: len(t[1]))  # bucket similar lengths => less padding
            for s in range(0, len(items), self._batch_size):
                chunk = items[s:s + self._batch_size]
                conts = self._generate_batch(
                    [ids for _, ids in chunk], max_new, temperature, top_k or None, list(until)
                )
                for (i, _), cont in zip(chunk, conts, strict=True):
                    results[i] = cont
                    self.cache_hook.add_partial(
                        "generate_until", (requests[i].args[0], requests[i].args[1]), cont
                    )
                    pbar.update(1)
        pbar.close()
        return results

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("ChronoGPT backend is generate-only (no loglikelihood).")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("ChronoGPT backend is generate-only (no loglikelihood).")


# ─────────────── DD + steering vLLM backend (deferred; only for ftp_qwen) ───────────────


@register_model("dd")
class DDVLLM(VLLM):
    """Divergence Decoding (+ optional SAE steering) on the stock vLLM backend.

    Reuses lm-eval's engine construction: ``DDLogitsProcessor`` is installed via the
    standard ``logits_processors`` LLM kwarg (no double build), and ``dd_alpha`` is
    injected per request in ``_model_generate``. Steering (if given) installs forward
    hooks on the built engine. NOT exercised by the small-model sweep.
    """

    def __init__(
        self, pretrained, aux_p=None, aux_q=None, dd_alpha=1.5, dd_window=2048, dd_mode="auto",
        aux_device=None, compile_aux=False, prewarm=0, fuse_pin=True, steer=None,
        steer_sweep=False, dd_retemplate=None, **kwargs,
    ):
        import os

        from ftp.config import DDConfig
        from ftp.vllm import DDLogitsProcessor

        # DD is OPTIONAL: with no aux pair this is a PURE feature-steering engine — steering
        # hooks only, no divergence logits processor, single GPU (no aux on cuda:1). Give both
        # aux_p and aux_q to enable full Divergence Decoding.
        self._dd = bool(aux_p) and bool(aux_q)

        # aux_device (e.g. "cuda:1") runs the aux pair on a SECOND GPU so it doesn't
        # compete with P for memory — the README's recommended 27B + 2x3B deployment
        # (P at util ~0.93 on card 0, aux on card 1; DD effectively latency-free).
        # compile_aux CUDA-graphs the aux decode (~10x faster than eager ~27ms/step);
        # prewarm captures those graphs at init (set to peak concurrency = max_num_seqs)
        # so the costly profiling/warmup run and every generated token use the graphed
        # (fast) aux path instead of the launch-bound eager one.
        # fuse_pin: whitelist pinning in core.dd_fuse keeps special/control tokens
        # (EOS, <|im_end|>, <think>, ...) at P's probability EXACTLY, so DD only steers
        # content tokens and never suppresses the stop tokens (without it, α inflates
        # content logits, EOS loses relatively -> generations ramble). ON by default; the
        # deliberate no-pin A/B arms pass fuse_pin=False. Carried on the frozen, per-instance
        # DDConfig so it can't leak across engine builds in one process.
        # dd_retemplate (universal mode): re-render P's prompt under the aux chat
        # template so the aux see their native wrapper (e.g. Qwen ChatML +
        # <think></think>) instead of P's markup retokenized into foreign sub-words.
        if self._dd:
            DDConfig(
                aux_p=aux_p, aux_q=aux_q, tokenizer=pretrained, window=dd_window, mode=dd_mode,
                aux_device=aux_device, compile_aux=compile_aux, prewarm=prewarm, fuse_pin=fuse_pin,
                retemplate=dd_retemplate,
            ).apply_env()
        self.dd_alpha = float(dd_alpha)
        # Rank-based DD (top-k most forget-divergent tokens masked to -inf); 0 = off.
        # Orthogonal to dd_alpha — the harness sets (dd_alpha, dd_rank_k) per arm, and
        # linear+rank composed is a valid future arm.
        self.dd_rank_k = 0
        self._steer = steer  # base SteerArgs, kept for in-process clamp re-install (clamp sweeps)
        self._steer_pairs = steer.pairs() if steer is not None else []
        # Steering routes: fixed-config steering (the default) installs PRE-CAPTURE —
        # DDSteeringWorker records the hook kernels inside P's CUDA graphs, so the
        # steered engine serves at unsteered throughput (the eager route halves it).
        # Pass steer_sweep=True in a spec's `args` to keep the old post-build eager
        # hooks — required whenever set_steer_clamp/set_steer_feature mutate steering
        # in-process. (No current spec does: the in-process sweep driver was removed
        # with the pre-v4 registry, so these methods are otherwise unreachable.)
        self._steer_sweep = bool(steer_sweep)
        if self._steer_pairs and self._steer_sweep:
            os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
            kwargs["enforce_eager"] = True
        elif self._steer_pairs:
            from ftp.serve import steer_precapture_kwargs

            kwargs.update(steer_precapture_kwargs(steer))
        if self._dd:  # steering-only (no aux) skips the DD logits processor entirely
            kwargs.setdefault("logits_processors", [DDLogitsProcessor])
        kwargs.setdefault("enable_prefix_caching", False)
        super().__init__(pretrained=pretrained, **kwargs)
        if self._steer_pairs and self._steer_sweep:
            from ftp.steering import install_steering

            install_steering(self.model, self._steer_pairs)

    def set_steer_clamp(self, clamp) -> None:
        """Sweep the LAST steer triple's clamp_value in-process; EARLIER triples stay fixed at
        their spec clamp. Lets one engine load sweep one feature while holding others fixed
        (e.g. L48 fixed @ base + L27 swept), or sweep a single feature (the only triple).
        clamp 0 drops the swept feature (so a single feature -> pure-DD baseline). No-op if
        built without steering (steer=None; enforce_eager wasn't set so hooks are skipped)."""
        if self._steer is None:
            return
        if not self._steer_sweep:
            raise RuntimeError(
                "in-process steering mutation on a pre-capture engine (clamps are "
                "capture-time constants there); set steer_sweep=True in the spec's "
                "args to force the eager post-build steering route")
        from dataclasses import replace

        from ftp.steering import install_steering, remove_steering

        remove_steering(self.model)
        triples = list(self._steer.triples)
        L, F, _ = triples[-1]
        triples[-1] = (L, F, float(clamp))           # sweep the last feature
        active = [t for t in triples if t[2] != 0.0]  # drop any feature clamped to 0
        if active:
            install_steering(self.model, replace(self._steer, triples=active).pairs())
            print(f"[steer] {active}", flush=True)
        else:
            print("[steer] removed (all clamps 0)", flush=True)

    def set_steer_feature(self, layer, feat, clamp) -> None:
        """Swap the installed steering to a SINGLE feature (layer, feat) at clamp_value, in one
        engine load — lets ONE job sweep MANY features that live on different layers/SAEs (each
        a fresh remove+install). Reuses the base SteerArgs' SAE source (sae_dir/family), so the
        combined SAE dir must hold every feature's layer{L}.sae.pt. clamp 0 removes steering
        (pure-DD baseline). No-op if built without steering (steer=None)."""
        if self._steer is None:
            return
        if not self._steer_sweep:
            raise RuntimeError(
                "in-process steering mutation on a pre-capture engine (clamps are "
                "capture-time constants there); set steer_sweep=True in the spec's "
                "args to force the eager post-build steering route")
        from dataclasses import replace

        from ftp.steering import install_steering, remove_steering

        remove_steering(self.model)
        if clamp and float(clamp) != 0.0:
            sa = replace(self._steer, triples=[(int(layer), int(feat), float(clamp))])
            install_steering(self.model, sa.pairs())
            print(f"[steer] L{layer}f{feat} @ {clamp}", flush=True)
        else:
            print(f"[steer] removed (L{layer}f{feat} c0 baseline)", flush=True)

    def _model_generate(self, requests, generate: bool = False, sampling_params=None):
        if self._dd and generate and sampling_params is not None:
            sps = sampling_params if isinstance(sampling_params, list) else None
            for sp in sps or [sampling_params]:
                extra = dict(sp.extra_args or {})
                extra.setdefault("dd_alpha", self.dd_alpha)
                extra.setdefault("dd_rank_k", self.dd_rank_k)
                sp.extra_args = extra
        return super()._model_generate(requests, generate=generate, sampling_params=sampling_params)
