"""Cross-tokenizer translation for universal decoding.

Universal mode lets the big model P use a different tokenizer than the aux
pair (p, q). Three pieces bridge the gap:

- :class:`TokenTextTable` — precomputed id→bytes for P's tokenizer, so P's
  token stream can be detokenized incrementally without per-step tokenizer
  calls.
- :class:`StreamRetokenizer` — per-request: maintains the aux-tokenizer
  tokenization of P's decoded text. Retokenization is non-causal (new text can
  merge with the previous trailing aux token), so each step yields a
  ``(rewind_k, new_aux_ids)`` delta that the caller applies to the aux KV
  caches via :meth:`AuxBatchedEngine.rewind`.
- :class:`VocabMapper` — maps the aux models' next-token distributions onto
  P's vocabulary with a precomputed first-token map: one log_softmax + gather
  per model per step. Gathering LOG-PROBS (not raw logits) matters: with
  partial coverage, the per-row log-partition constant of each aux model does
  not cancel in the downstream softmax, whereas log-probs make "uncovered ⇒
  zero shift" the neutral default (q/p ratio = 1).

The same approach family ships in HF transformers' universal assisted
generation (decode→re-encode + cache crop) and the heterogeneous-vocabulary
speculative decoding literature (token-level intersection mapping); DD differs
in needing a neutral zero *shift* for unmapped tokens rather than suppression.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ftp.core import build_special_ids
from ftp.engine import AuxBatchedEngine

_BYTE_TOKEN_RE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def _byte_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 byte→unicode table used by byte-level BPE vocabs."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


def _utf8_complete_len(buf: bytes | bytearray) -> int:
    """Length of the longest prefix of ``buf`` ending on a UTF-8 code-point
    boundary (a trailing incomplete multi-byte sequence is held back)."""
    n = len(buf)
    for back in range(1, min(4, n) + 1):
        c = buf[n - back]
        if c < 0x80:  # ASCII — complete
            break
        if c >= 0xC0:  # lead byte of a multi-byte sequence
            need = 2 if c < 0xE0 else 3 if c < 0xF0 else 4
            if back < need:
                return n - back
            break
        # else: continuation byte — keep scanning backwards
    return n


def _lcp(old: list[int], new: list[int], tail: int) -> int:
    """Longest common prefix length. Fast path: everything but the last
    ``tail`` elements matches (the overwhelmingly common retokenization case),
    verified with one C-level slice compare."""
    m = min(len(old), len(new))
    i = m - tail
    if i <= 0 or old[:i] != new[:i]:
        i = 0
        step = 256
        while i < m and old[i : i + step] == new[i : i + step]:
            i += step
        i = min(i, m)
    while i < m and old[i] == new[i]:
        i += 1
    return i


class TokenTextTable:
    """Precomputed id→bytes for one tokenizer (P's side).

    Handles byte-level BPE vocabs (GPT-2 unicode↔byte table), SentencePiece
    style vocabs (``▁`` word markers + ``<0xNN>`` byte-fallback pieces), and
    falls back to ``convert_tokens_to_string`` per id for anything else.
    Special/added tokens map to their surface string bytes (``b"<|im_end|>"``)
    so the aux models see P's markup as text — consistent with shared mode,
    where the aux models are fed P's markup ids directly.
    """

    def __init__(self, tok) -> None:
        self.special_ids = frozenset(build_special_ids(tok))
        vocab: dict[str, int] = tok.get_vocab()
        n = max(vocab.values()) + 1 if vocab else 0
        self.bytes_of: list[bytes] = [b""] * n
        bdec = _byte_decoder()
        # SentencePiece-style vocabs contain ▁-prefixed pieces; byte-level BPE
        # cannot (U+2581 would itself be byte-encoded into other characters).
        is_sp = any(t.startswith("▁") for t in vocab)
        for t, i in vocab.items():
            if i in self.special_ids:
                self.bytes_of[i] = t.encode("utf-8")
                continue
            m = _BYTE_TOKEN_RE.fullmatch(t)
            if m:
                self.bytes_of[i] = bytes([int(m.group(1), 16)])
                continue
            if is_sp:
                self.bytes_of[i] = t.replace("▁", " ").encode("utf-8")
                continue
            try:
                self.bytes_of[i] = bytes(bdec[ch] for ch in t)
            except KeyError:  # not byte-level after all — ask the tokenizer
                self.bytes_of[i] = tok.convert_tokens_to_string([t]).encode("utf-8")

    def bytes_for(self, token_id: int) -> bytes:
        if 0 <= token_id < len(self.bytes_of):
            return self.bytes_of[token_id]
        return b""


class StreamRetokenizer:
    """Maintains the aux tokenization of one request's P-token byte stream.

    Every :meth:`push` re-encodes the full decoded text with the aux tokenizer
    (Rust-side, tens of µs at typical lengths) and diffs against the committed
    ids, so ``committed == aux_tok(full_text)`` holds *by construction* — no
    anchoring heuristics. The diff cost is one C-level slice compare in the
    common case. Incomplete trailing UTF-8 sequences are held back, so a push
    can legitimately yield no new text (``(0, [])``).
    """

    def __init__(
        self, aux_tok, table: TokenTextTable, *, diff_tail: int = 8, reenc_window: int = 48
    ) -> None:
        self._tok = aux_tok
        self._table = table
        self._diff_tail = diff_tail
        bos = getattr(aux_tok, "bos_token_id", None)
        self._prefix: list[int] = [bos] if bos is not None else []
        self._buf = bytearray()
        self._done = 0  # bytes already decoded into _text
        self._text = ""
        self._aux_ids: list[int] = list(self._prefix)
        # Bounded re-encode: only the last `reenc_window` committed tokens are
        # re-tokenized each step; everything before `_frozen` is frozen (its
        # text won't be re-encoded), making the per-step encode O(window)
        # instead of O(context). Retokenization changes are local to the last
        # few tokens, so a 48-token margin is far beyond any BPE merge range —
        # the per-push `committed == aux_tok(full_text)` invariant
        # (test_translate_cpu) is the guard. Needs char offsets, so it is on
        # only for fast tokenizers; others keep the exact full-text path.
        self._reenc = reenc_window
        self._bounded = bool(getattr(aux_tok, "is_fast", False))
        self._frozen = len(self._prefix)  # committed tokens whose text is frozen
        self._frozen_chars = 0  # char offset in _text where the unstable region starts

    @property
    def aux_ids(self) -> list[int]:
        """The committed aux ids (aux BOS + tokenization of the decoded text)."""
        return self._aux_ids

    def _encode(self, text: str) -> list[int]:
        return self._prefix + self._tok(text, add_special_tokens=False)["input_ids"]

    def _advance_freeze(self, body: list[int], offsets) -> None:
        """Move the frozen boundary forward, keeping the last `_reenc` committed
        tokens unstable. `offsets[i]` is the (start, end) char span of body token
        i within the unstable text that was just encoded."""
        if not self._bounded or offsets is None or not body:
            return
        target = max(self._frozen, len(self._aux_ids) - self._reenc)
        b = target - self._frozen  # candidate body tokens to newly freeze
        if b <= 0 or b >= len(body):
            return
        # Never split a multi-byte char's sub-tokens: byte-level BPE emits >1
        # token per such char, all sharing one char offset, so a cut where
        # body[b] shares body[b-1]'s start char would double-count bytes. Back
        # off to the nearest clean char boundary (freezing slightly less is
        # always safe — it only widens the re-encoded window).
        while b > 0 and offsets[b][0] == offsets[b - 1][0]:
            b -= 1
        if b > 0:
            self._frozen_chars += offsets[b][0]
            self._frozen += b

    def reset(self, p_ids: list[int]) -> list[int]:
        """Rebuild from a full P-id context; returns the full aux ids (for prefill)."""
        self._buf = bytearray()
        for i in p_ids:
            self._buf += self._table.bytes_for(i)
        self._done = _utf8_complete_len(self._buf)
        self._text = bytes(self._buf[: self._done]).decode("utf-8", "replace")
        self._aux_ids = self._encode(self._text)
        # Re-stabilize the frozen boundary from scratch; the next stage()
        # re-encodes the (now short) unstable tail.
        self._frozen = len(self._prefix)
        self._frozen_chars = 0
        return list(self._aux_ids)

    def reset_with_prefix(self, prefix_ids: list[int], output_ids: list[int]) -> list[int]:
        """Like :meth:`reset`, but the aux prefix is supplied externally — the
        aux model's OWN chat-template rendering of the prompt — and frozen in
        place; only ``output_ids`` (P's *generated* tokens) seed the mutable
        retokenized region. Used by :class:`UniversalBridge` when a
        :class:`ChatTemplateAdapter` swaps P's prompt markup for the aux
        template. The prompt/response seam is clean by construction (the aux SFT
        tokenized prompt and response separately), so the whole prefix is
        hard-frozen. Returns the full aux ids (prefill = prefix + retok(gen))."""
        self._prefix = list(prefix_ids)
        self._buf = bytearray()
        for i in output_ids:
            self._buf += self._table.bytes_for(i)
        self._done = _utf8_complete_len(self._buf)
        self._text = bytes(self._buf[: self._done]).decode("utf-8", "replace")
        self._aux_ids = self._encode(self._text)
        self._frozen = len(self._prefix)
        self._frozen_chars = 0
        return list(self._aux_ids)

    def reset_with_prefix_text(self, prefix_ids: list[int], gen_text: str) -> list[int]:
        """Re-base onto ``prefix_ids`` with the mutable region seeded directly
        from ``gen_text`` (not P ids). Used for the one-time reasoning re-base:
        when a reasoning P crosses its "stop thinking" marker, the aux stream is
        rebuilt as prefix + ``gen_text`` — P's generated text with its CoT
        delimiters already rewritten to the aux model's native ``<think>``/
        ``</think>`` (see :meth:`ChatTemplateAdapter.translate_reasoning`). The CoT
        content is KEPT (it carries the forget knowledge DD must see), not dropped.
        Subsequent P tokens append normally (they are all post-marker). Returns the
        full aux ids (for re-prefill)."""
        self._prefix = list(prefix_ids)
        self._buf = bytearray(gen_text.encode("utf-8"))
        self._done = len(self._buf)
        self._text = gen_text
        self._aux_ids = self._encode(self._text)
        self._frozen = len(self._prefix)
        self._frozen_chars = 0
        return list(self._aux_ids)

    @property
    def decoded_text(self) -> str:
        """The decoded text of the mutable region (P's generated stream in
        adapter mode; the full P stream otherwise)."""
        return self._text

    def stage(self, new_p_ids: list[int]) -> str | None:
        """Append new P tokens' bytes and advance the decoded text; return the
        text to (re)encode this step, or ``None`` if no new complete text
        emerged (held-back UTF-8, the d=0 case).

        Pairs with :meth:`commit`, which consumes the aux-tokenization of the
        returned text. Splitting the encode out lets a caller batch the
        encodes of many requests into ONE tokenizer call — the dominant
        universal-mode host cost at scale (32 separate full-text encodes
        ~48 ms vs ~11 ms batched on gpt-oss×Qwen at 1024-token context)."""
        for p_id in new_p_ids:
            b = self._table.bytes_for(p_id)
            if b:
                self._buf += b
        new_done = _utf8_complete_len(self._buf)
        if new_done == self._done:
            return None
        self._text += bytes(self._buf[self._done : new_done]).decode("utf-8", "replace")
        self._done = new_done
        # Bounded mode: re-encode only the unstable tail (text after the frozen
        # boundary); exact mode: the whole text.
        return self._text[self._frozen_chars :] if self._bounded else self._text

    def commit(self, body_ids: list[int], offsets=None) -> tuple[int, list[int]]:
        """Consume ``body_ids`` = the aux tokenization (WITHOUT the BOS prefix)
        of the text returned by the matching :meth:`stage`; diff against the
        committed ids and return ``(rewind_k, new_aux_ids)``: drop the last
        ``rewind_k`` committed aux tokens, then append ``new_aux_ids``.

        In bounded mode ``body_ids`` covers only the unstable tail; the frozen
        prefix (``_aux_ids[:_frozen]``, text-identical by construction) is
        prepended, and ``offsets`` (char spans of the body tokens) advances the
        frozen boundary for next step."""
        new_ids = self._aux_ids[: self._frozen] + body_ids
        old_ids = self._aux_ids
        lcp = _lcp(old_ids, new_ids, self._diff_tail)
        self._aux_ids = new_ids
        self._advance_freeze(body_ids, offsets)
        return (len(old_ids) - lcp, new_ids[lcp:])

    def push(self, p_id: int) -> tuple[int, list[int]]:
        """Append one P token (unbatched convenience over stage/commit)."""
        text = self.stage([p_id])
        if text is None:
            return (0, [])
        if self._bounded:
            enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
            return self.commit(enc["input_ids"], enc["offset_mapping"])
        return self.commit(self._tok(text, add_special_tokens=False)["input_ids"])


class VocabMapper:
    """First-token map from P's vocabulary onto the aux vocabulary.

    For each P token id, its bytes are encoded as a *continuation* with the
    aux tokenizer and the first resulting aux id becomes the map entry — on
    vocabulary-intersection tokens this reduces to the exact-string mapping,
    elsewhere it is the longest-aux-prefix approximation. P tokens with no
    sensible mapping (specials, empty decodes, unrepresentable bytes, vLLM
    vocab padding) are *uncovered*: they receive a zero shift, i.e. an assumed
    aux ratio q/p = 1.

    :meth:`map_logits` is the hot path: one fp32 ``log_softmax`` + ``gather``
    per aux model per step. Build cost is one batched tokenizer call
    (~seconds for a 256k vocab); pass ``cache_dir`` to persist the map keyed
    by a content hash of both vocabularies.
    """

    def __init__(self, aux_tok, table: TokenTextTable, device, *, cache_dir=None) -> None:
        self._device = torch.device(device)
        n = len(table.bytes_of)
        path = None
        if cache_dir is not None:
            path = Path(cache_dir) / f"vocab_map_{self._cache_key(aux_tok, table)}.pt"
        if path is not None and path.exists():
            blob = torch.load(path)
            map_cpu, cov_cpu = blob["map"], blob["cov"]
        else:
            map_cpu, cov_cpu = self._build(aux_tok, table, n)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                torch.save({"map": map_cpu, "cov": cov_cpu}, tmp)
                tmp.replace(path)
        self._map_cpu = map_cpu
        self._cov_cpu = cov_cpu
        self.map_idx: torch.Tensor = map_cpu.to(self._device)
        self.coverage: torch.Tensor = cov_cpu.to(self._device)

    @staticmethod
    def _cache_key(aux_tok, table: TokenTextTable) -> str:
        h = hashlib.sha256(b"divergence-decoding vocab map v1")
        for b in table.bytes_of:
            h.update(b)
            h.update(b"\x00")
        for t, i in sorted(aux_tok.get_vocab().items()):
            h.update(f"{t}\x00{i}\x00".encode())
        return h.hexdigest()[:16]

    @staticmethod
    def _build(aux_tok, table: TokenTextTable, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        map_l = [0] * n
        cov = [False] * n
        aux_vocab: dict[str, int] = aux_tok.get_vocab()
        byte_enc = {b: ch for ch, b in _byte_decoder().items()}

        texts: list[str] = []
        text_pos: list[int] = []
        for i in range(n):
            b = table.bytes_of[i]
            if not b or i in table.special_ids:
                continue
            try:
                texts.append(b.decode("utf-8"))
                text_pos.append(i)
            except UnicodeDecodeError:
                # Partial/invalid UTF-8 (byte-fallback tokens): map to the aux
                # token for the first raw byte, if the aux vocab has one.
                tid = aux_vocab.get(f"<0x{b[0]:02X}>")
                if tid is None:
                    tid = aux_vocab.get(byte_enc[b[0]])
                if tid is not None:
                    map_l[i] = tid
                    cov[i] = True

        if texts:
            # Continuation-faithful encoding: raw if decode(encode(t)) round-trips
            # (byte-level BPE); otherwise prefix a "\n" sentinel and strip it
            # (SentencePiece-style normalization eats leading spaces).
            def enc(batch: list[str]) -> list[list[int]]:
                return aux_tok(batch, add_special_tokens=False)["input_ids"]

            raw_ok = all(aux_tok.decode(enc([t])[0]) == t for t in ("ing", "tion", " the"))
            if raw_ok:
                for i, ids in zip(text_pos, enc(texts), strict=True):
                    if ids:
                        map_l[i] = ids[0]
                        cov[i] = True
            else:
                nl = enc(["\n"])[0]
                for i, ids in zip(text_pos, enc(["\n" + t for t in texts]), strict=True):
                    if len(ids) > len(nl) and ids[: len(nl)] == nl:
                        map_l[i] = ids[len(nl)]
                        cov[i] = True

        return torch.tensor(map_l, dtype=torch.long), torch.tensor(cov, dtype=torch.bool)

    def ensure(self, V: int) -> None:
        """Size the device tensors to vLLM's runtime logits width (which may
        exceed the tokenizer vocab — padded entries are uncovered)."""
        if self.map_idx.shape[0] >= V:
            return
        n = self._map_cpu.shape[0]
        map_idx = torch.zeros(V, dtype=torch.long, device=self._device)
        coverage = torch.zeros(V, dtype=torch.bool, device=self._device)
        map_idx[:n] = self._map_cpu.to(self._device)
        coverage[:n] = self._cov_cpu.to(self._device)
        self.map_idx, self.coverage = map_idx, coverage

    def map_logits(self, l_aux: torch.Tensor, V: int) -> torch.Tensor:
        """Map aux logits ``[N, V_aux]`` to per-P-token log-probs ``[N, V]``
        (fp32), 0.0 where uncovered."""
        self.ensure(V)
        logp = torch.log_softmax(l_aux.float(), dim=-1)
        idx = self.map_idx[:V].unsqueeze(0).expand(l_aux.shape[0], V)
        return logp.gather(1, idx).masked_fill(~self.coverage[:V].unsqueeze(0), 0.0)


# The aux model's (Qwen ChatML) native think delimiters that P's CoT markers are
# rewritten to at the one-time reasoning re-base.
_AUX_THINK_OPEN = "<think>\n"
_AUX_THINK_CLOSE = "\n</think>\n\n"


@dataclass(frozen=True)
class ChatTemplateAdapter:
    """Re-render P's prompt under the *aux* tokenizer's own chat template.

    Universal mode normally feeds the aux models a retokenization of P's full
    decoded stream, so the aux pair sees P's chat markup (e.g. Gemma's
    ``<start_of_turn>user`` …) as foreign sub-word text. When the aux pair was
    instruction-tuned under its OWN template (e.g. Qwen ChatML), that wrapper is
    out of distribution and degrades the aux distributions DD steers with — and
    it is token-inefficient (P's special tokens explode into many aux sub-words,
    eating the aux models' small context window).

    This adapter slices the user turn out of P's rendered prompt and re-wraps it
    with ``aux_tok.apply_chat_template`` so the frozen aux prefix is
    byte-identical to the aux SFT prompt; only P's *generated* tokens are
    retokenized (the response content is identical in either tokenizer). The
    prompt/response seam is clean by construction — the aux SFT tokenized prompt
    and response separately — so the prefix is hard-frozen (see
    :meth:`StreamRetokenizer.reset_with_prefix`).

    ``user_open``/``user_close`` are P's template delimiters bracketing the user
    turn's content; ``template_kwargs`` is forwarded to ``apply_chat_template``
    (e.g. ``{"enable_thinking": False}`` to match a non-thinking SFT). Single
    user turn only — a multi-turn transcript re-renders just the first user
    turn (the DD evals are single-turn ``generate_until``).

    ``reason_open``/``reason_close`` (reasoning P only) are the markers around P's
    chain-of-thought in its *generated* stream (e.g. gpt-oss harmony
    ``<|channel|>analysis<|message|>`` … ``<|start|>assistant<|channel|>final<|message|>``).
    The CoT is NOT dropped — it carries the forget knowledge DD must see. While P
    is still thinking the bridge feeds the stream verbatim (out of distribution,
    but transient); when ``reason_close`` appears it re-bases the aux stream ONCE,
    rewriting P's CoT delimiters to the aux model's native think tokens
    (``<think>`` … ``</think>``) so the aux read a well-formed Qwen thinking trace
    (CoT inside the block, answer after). For a reasoning P use a thinking-mode
    prefix (``enable_thinking=True``) so the prefix opens the assistant turn
    WITHOUT a forced empty think block. ``None`` = P never reasons.
    """

    user_open: str
    user_close: str
    template_kwargs: dict = field(default_factory=dict)
    reason_open: str | None = None
    reason_close: str | None = None

    def user_content(self, p_prompt_text: str) -> str | None:
        """The user turn's content inside P's rendered prompt, or ``None`` if P's
        delimiters are absent (the caller then falls back to raw retokenization)."""
        i = p_prompt_text.find(self.user_open)
        if i < 0:
            return None
        i += len(self.user_open)
        j = p_prompt_text.find(self.user_close, i)
        return (p_prompt_text[i:j] if j >= 0 else p_prompt_text[i:]).strip()

    def aux_prefix_text(self, p_prompt_text: str, aux_tok) -> str | None:
        """P's prompt re-rendered under the aux chat template, or ``None``."""
        content = self.user_content(p_prompt_text)
        if content is None:
            return None
        return aux_tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            **self.template_kwargs,
        )

    def translate_reasoning(self, output_text: str) -> str | None:
        """If P's end-of-think marker (``reason_close``) has appeared in its
        generated text, rewrite P's CoT delimiters to the aux model's native
        ``<think>``/``</think>`` and return the rewritten text; else ``None`` (P
        is still thinking — no re-base yet). The CoT content is kept verbatim."""
        if not self.reason_close or self.reason_close not in output_text:
            return None
        text = output_text
        if self.reason_open:
            text = text.replace(self.reason_open, _AUX_THINK_OPEN, 1)
        return text.replace(self.reason_close, _AUX_THINK_CLOSE, 1)


# Presets keyed by P's (the big model's) chat template — they encode where P's
# rendered prompt brackets the user turn. The aux side always re-renders under
# its OWN template; ``template_kwargs={"enable_thinking": False}`` matches the
# aux models' non-thinking Qwen-ChatML SFT recipe (forwarded to the AUX
# tokenizer, not P's). Add a P family by adding its user-turn delimiters here.
_NO_THINK = {"enable_thinking": False}
_RETEMPLATE_PRESETS: dict[str, ChatTemplateAdapter] = {
    # Gemma-2/3: ``<start_of_turn>user\n{content}<end_of_turn>``.
    "gemma": ChatTemplateAdapter(
        user_open="<start_of_turn>user\n",
        user_close="<end_of_turn>",
        template_kwargs=_NO_THINK,
    ),
    # Llama-3.x: ``<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>``.
    "llama3": ChatTemplateAdapter(
        user_open="<|start_header_id|>user<|end_header_id|>\n\n",
        user_close="<|eot_id|>",
        template_kwargs=_NO_THINK,
    ),
    # GPT-OSS (harmony format): user ``<|start|>user<|message|>{content}<|end|>``;
    # reasoning-only, so the aux render in thinking mode and P's analysis->final
    # channel transition is rewritten to the aux <think>/</think> at re-base.
    # (Harmony marker strings are best-effort — verify against real gpt-oss output.)
    "gptoss": ChatTemplateAdapter(
        user_open="<|start|>user<|message|>",
        user_close="<|end|>",
        template_kwargs={"enable_thinking": True},
        reason_open="<|channel|>analysis<|message|>",
        reason_close="<|end|><|start|>assistant<|channel|>final<|message|>",
    ),
}


def make_chat_adapter(name: str | None) -> ChatTemplateAdapter | None:
    """Resolve a ``retemplate`` preset name to a :class:`ChatTemplateAdapter`
    (``None``/empty → no retemplating)."""
    if not name:
        return None
    try:
        return _RETEMPLATE_PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown retemplate preset {name!r}; known: {sorted(_RETEMPLATE_PRESETS)}"
        ) from None


@dataclass
class _BridgeReqState:
    retok: StreamRetokenizer
    pending: deque[int] = field(default_factory=deque)  # committed aux ids not yet fed
    p_len: int = 0  # P tokens consumed (prompt + outputs)
    fed: int = 0  # aux tokens currently cached by the engines (capped at window)
    prefill: bool = True  # next feed is a full prefill
    needs_reset: bool = False
    post_reason: bool = False  # reasoning P: aux stream already re-based at end-of-think


class UniversalBridge:
    """Owns the cross-tokenizer machinery for one (aux_p, aux_q) engine pair.

    Per P decode step (:meth:`step`):
      1. push each request's new P token through its :class:`StreamRetokenizer`;
         apply rewinds (unfed tokens are dropped from the pending queue first;
         the remainder rewinds both engine caches, or re-primes the row if the
         rewind is deeper than :data:`REWIND_LIMIT`);
      2. drain the pending queues through the engines' existing single-token
         batched step, at most ``max_feeds_per_step`` rounds (rows already
         drained park on the engines' scratch column); each fed row's logits
         land in a persistent per-slot buffer, so rows with no new decodable
         text this step (held-back UTF-8) keep last step's logits — which still
         condition on exactly the current text;
      3. map both buffers onto P's vocabulary via :class:`VocabMapper`.

    The engines are borrowed: registration/unregistration stays with the
    caller (the vLLM processor), mirroring shared mode. ``aux_q=None`` means
    ``aux_p`` is a FUSED pair engine (one forward returns both planes).
    """

    REWIND_LIMIT = 16

    def __init__(
        self,
        aux_tok,
        table: TokenTextTable,
        mapper: VocabMapper,
        aux_p: AuxBatchedEngine,
        aux_q: AuxBatchedEngine | None,
        *,
        max_feeds_per_step: int = 3,
        window: int = 2048,
        aux_streams: tuple | None = None,
        adapter: ChatTemplateAdapter | None = None,
    ) -> None:
        self._aux_tok = aux_tok
        self._table = table
        self._mapper = mapper
        # Optional chat-template swap: re-render P's prompt under the aux
        # template (so the aux models see their native wrapper) instead of
        # retokenizing P's markup. Affects only the prompt prefix at reset; the
        # generated stream is retokenized as usual and slides through the aux
        # KV window unchanged.
        self._adapter = adapter
        self._adapter_warned = False
        self._aux_p = aux_p
        self._aux_q = aux_q
        self._fused = aux_q is None
        self._engines = (aux_p,) if self._fused else (aux_p, aux_q)
        self._max_feeds = max_feeds_per_step
        self._window = window
        self._aux_streams = aux_streams  # run p and q concurrently when set
        bos = getattr(aux_tok, "bos_token_id", None)
        eos = getattr(aux_tok, "eos_token_id", None)
        self._fallback_id = bos if bos is not None else (eos if eos is not None else 0)
        self._states: dict[int, _BridgeReqState] = {}
        self._slot: dict[int, int] = {}
        self._free: set[int] = set()
        self._buf_p: torch.Tensor | None = None  # [cap, V_aux] persistent logits
        self._buf_q: torch.Tensor | None = None
        self._backlog_steps = 0
        # DD_TIMING=1: drain-round histogram (how many engine rounds each P
        # step needed) — the sizing signal for multi-token decode work.
        self._round_hist: list[int] | None = (
            [0] * (max_feeds_per_step + 1) if os.environ.get("DD_TIMING", "0") == "1" else None
        )
        self._stat_steps = 0
        # Slot-index tensors cached per row composition: building them via
        # torch.tensor(..., device=...) is a PAGEABLE H2D copy that blocks the
        # host behind everything already enqueued on the aux stream — fatal in
        # overlap mode, where prefetch must return before P's forward launches.
        self._slots_cache: dict[tuple[int, ...], torch.Tensor] = {}

    def _slots_for(self, rids: list[int], device) -> torch.Tensor:
        key = tuple(rids)
        t = self._slots_cache.get(key)
        if t is None:
            if len(self._slots_cache) > 256:
                self._slots_cache.clear()
            t = torch.tensor([self._slot[rid] for rid in rids], dtype=torch.long, device=device)
            self._slots_cache[key] = t
        return t

    # ── per-request lifecycle (driven by the vLLM processor) ────────────────

    def drop(self, rid: int) -> None:
        self._states.pop(rid, None)
        slot = self._slot.pop(rid, None)
        if slot is not None:
            self._free.add(slot)
            self._slots_cache.clear()  # slot assignments changed

    def mark_reset(self, rid: int) -> None:
        """The caller re-registered this request with the engines (recompute
        gap / id collision) — rebuild its stream from full context next step."""
        st = self._states.get(rid)
        if st is not None:
            st.needs_reset = True

    # ── internals ────────────────────────────────────────────────────────────

    def _aux_prefix_ids(self, p_ids: list[int] | None) -> list[int] | None:
        """Detokenize P's prompt and re-render it under the aux chat template;
        returns the aux prefix ids, or ``None`` to fall back to raw
        retokenization (P's delimiters not found)."""
        buf = bytearray()
        for i in p_ids or []:
            buf += self._table.bytes_for(i)
        p_text = bytes(buf).decode("utf-8", "replace")
        prefix_text = self._adapter.aux_prefix_text(p_text, self._aux_tok)
        if prefix_text is None:
            if not self._adapter_warned:
                print(
                    "[UniversalBridge] WARNING: retemplate adapter found no user "
                    "turn in P's prompt; falling back to raw retokenization.",
                    flush=True,
                )
                self._adapter_warned = True
            return None
        return self._aux_tok(prefix_text, add_special_tokens=False)["input_ids"]

    def _reprime(self, rid: int, st: _BridgeReqState) -> None:
        """Reset the engines' cache for this row and queue a full prefill."""
        for eng in self._engines:
            eng.unregister(rid)
            eng.register(rid)
        st.pending = deque(st.retok.aux_ids)
        st.fed = 0
        st.prefill = True

    def _sync_requests(self, reqs: list[tuple[int, list[int] | None, list[int]]]) -> None:
        """Retokenize all requests' new P tokens for this step, batching the
        aux encodes into ONE tokenizer call.

        Phase 1 stages each request (appends bytes, advances text) and collects
        the texts to encode; phase 2 encodes them all at once; phase 3 commits
        each delta. Cold-start / gap requests (``needs_reset``) re-encode their
        full context inline — rare, off the steady-state path."""
        texts: list[str] = []
        staged: list[tuple[int, _BridgeReqState]] = []
        for rid, p_ids, o_ids in reqs:
            st = self._states.get(rid)
            if st is None:
                st = self._states[rid] = _BridgeReqState(
                    retok=StreamRetokenizer(self._aux_tok, self._table), needs_reset=True
                )
            if st.needs_reset:
                o_list = list(o_ids)
                # With a chat adapter the prompt prefix comes from the aux
                # template (frozen) and only P's generated tokens seed the
                # mutable stream; otherwise retokenize the full P context.
                prefix_ids = (
                    self._aux_prefix_ids(p_ids) if self._adapter is not None else None
                )
                if prefix_ids is not None:
                    ids = st.retok.reset_with_prefix(prefix_ids, o_list) or [self._fallback_id]
                else:
                    ids = st.retok.reset(list(p_ids or []) + o_list) or [self._fallback_id]
                st.pending = deque(ids)
                st.p_len = len(p_ids or []) + len(o_list)
                st.fed = 0
                st.prefill = True
                st.needs_reset = False
                st.post_reason = False  # re-translate reasoning after a verbatim rebuild
                continue
            full_len = len(p_ids or []) + len(o_ids)
            n_new = full_len - st.p_len
            if n_new <= 0:
                continue
            text = st.retok.stage(list(o_ids)[-n_new:])
            st.p_len = full_len
            if text is None:  # d=0: no new complete text this step
                continue
            texts.append(text)
            staged.append((rid, st))
        if texts:
            if getattr(self._aux_tok, "is_fast", False):
                enc = self._aux_tok(texts, add_special_tokens=False, return_offsets_mapping=True)
                offs = enc["offset_mapping"]
            else:
                enc = self._aux_tok(texts, add_special_tokens=False)
                offs = [None] * len(texts)
            for (rid, st), body, off in zip(staged, enc["input_ids"], offs, strict=True):
                rewind, new_ids = st.retok.commit(body, off)
                self._apply_delta(rid, st, rewind, new_ids)

        # Reasoning P: once the stream crosses P's end-of-think, re-base the aux
        # ONCE so they read a Qwen-native think trace (P's CoT delimiters rewritten
        # to <think>/</think>; CoT content kept). Re-primes the aux KV cache.
        adapter = self._adapter
        if adapter is not None and adapter.reason_close:
            for rid, p_ids, _o in reqs:
                st = self._states.get(rid)
                if st is None or st.post_reason or st.needs_reset:
                    continue
                gen = adapter.translate_reasoning(st.retok.decoded_text)
                if gen is None:
                    continue
                prefix_ids = self._aux_prefix_ids(p_ids)
                if prefix_ids is None:
                    continue
                st.retok.reset_with_prefix_text(prefix_ids, gen)
                st.post_reason = True
                self._reprime(rid, st)

    def _apply_delta(self, rid: int, st: _BridgeReqState, rewind: int, new_ids: list[int]) -> None:
        """Apply one request's retokenization delta to its pending queue and
        engine caches: drop unfed pending tokens first, then rewind (or
        re-prime) the cached aux tokens, then enqueue the new ids."""
        if rewind:
            drop = min(rewind, len(st.pending))
            for _ in range(drop):
                st.pending.pop()
            k = rewind - drop
            if k:
                if st.prefill or k > min(st.fed, self.REWIND_LIMIT):
                    self._reprime(rid, st)
                else:
                    for eng in self._engines:
                        eng.rewind(rid, k)
                    st.fed -= k
        st.pending.extend(new_ids)

    def _dual(self, run_p, run_q):
        """Run the two (independent) aux engines, on separate streams if set."""
        if self._aux_streams is None:
            return run_p(), run_q()
        s1, s2 = self._aux_streams
        cur = torch.cuda.current_stream(s1.device)
        s1.wait_stream(cur)
        s2.wait_stream(cur)
        with torch.cuda.stream(s1):
            out_p = run_p()
        with torch.cuda.stream(s2):
            out_q = run_q()
        cur.wait_stream(s1)
        cur.wait_stream(s2)
        return out_p, out_q

    def _run_step(self, feeds, *, sequential: bool = False):
        """One drain round through ``step``: (out_p, out_q) from the fused
        engine's two planes or from the engine pair (``sequential=True`` keeps
        the unfused pair off the streams — eager prefill activation peaks)."""
        if self._fused:
            out = self._aux_p.step(feeds)
            return out[0], out[1]
        if sequential:
            return self._aux_p.step(feeds), self._aux_q.step(feeds)
        return self._dual(lambda: self._aux_p.step(feeds), lambda: self._aux_q.step(feeds))

    def _run_pairs(self, reqs2):
        if self._fused:
            out = self._aux_p.step_pairs(reqs2)
            return out[0], out[1]
        return self._dual(
            lambda: self._aux_p.step_pairs(reqs2), lambda: self._aux_q.step_pairs(reqs2)
        )

    def _ensure_slots(self, rids: list[int]) -> None:
        for rid in rids:
            if rid not in self._slot:
                if not self._free:
                    base = len(self._slot)
                    self._free.update(range(base, base + 8))
                self._slot[rid] = min(self._free)
                self._free.discard(self._slot[rid])

    def _ensure_bufs(self, like: torch.Tensor) -> None:
        cap = max(self._slot.values()) + 1
        if self._buf_p is not None and self._buf_p.shape[0] >= cap:
            return
        new_p = torch.zeros(cap, like.shape[-1], dtype=like.dtype, device=like.device)
        new_q = torch.zeros_like(new_p)
        if self._buf_p is not None:
            new_p[: self._buf_p.shape[0]] = self._buf_p
            new_q[: self._buf_q.shape[0]] = self._buf_q
        self._buf_p, self._buf_q = new_p, new_q

    # ── the per-P-step hot path ──────────────────────────────────────────────

    def step(
        self,
        reqs: list[tuple[int, list[int] | None, list[int]]],
        V: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the aux pair for one P decode step.

        Args:
            reqs: [(req_id, prompt_ids, output_ids), ...] — P-token ids, the
                same tuples shared mode feeds the engines directly.
            V: width of P's logits rows (vLLM's padded vocab size).
        Returns:
            (l_p, l_q): mapped aux log-probs, each [len(reqs), V] fp32.
        """
        self.prefetch(reqs)
        return self.finalize(V)

    def prefetch(self, reqs: list[tuple[int, list[int] | None, list[int]]]) -> None:
        """Phase 1: retokenize and enqueue ALL aux GPU work for this step.

        Touches only the aux device (drain rounds + per-slot buffer writes),
        so under overlap mode it can run BEFORE P's forward without inserting
        anything into P's stream; :meth:`finalize` does the cross-device
        transfer and vocab mapping at fuse time."""
        self._sync_requests(reqs)

        order = [rid for rid, _, _ in reqs]
        self._ensure_slots(order)

        # Drain: per row, feed up to max_feeds_per_step tokens this P step.
        # Prefill rows go through the engines' step(); decode rounds pack up
        # to TWO tokens per row through step_pairs() when any row has a
        # retokenization burst, so a burst costs one two-token forward instead
        # of two rounds.
        budget = dict.fromkeys(order, self._max_feeds)
        rounds = 0
        while True:
            prefills = [rid for rid in order if self._states[rid].prefill]
            pend = [
                rid
                for rid in order
                if not self._states[rid].prefill and self._states[rid].pending and budget[rid] > 0
            ]
            if prefills:
                feeds = [(rid, list(self._states[rid].pending), []) for rid in prefills]
                feeds += [(rid, None, [self._states[rid].pending[0]]) for rid in pend]
                # Prefills run eagerly with batch-sized activation peaks — keep
                # an unfused pair sequential here (concurrent eager prefills
                # double the transient footprint; decode rounds overlap fine
                # because graph replays draw from preallocated pools). A fused
                # engine's single 2N-batch prefill replaces two N-batch peaks.
                out_p, out_q = self._run_step(feeds, sequential=True)
                fed = [f[0] for f in feeds]
                for rid in prefills:
                    st = self._states[rid]
                    st.fed = min(len(st.pending), self._window)
                    st.pending.clear()
                    st.prefill = False
                for rid in pend:
                    st = self._states[rid]
                    st.pending.popleft()
                    st.fed = min(st.fed + 1, self._window)
                    budget[rid] -= 1
            elif pend:
                ks = {rid: min(2, len(self._states[rid].pending), budget[rid]) for rid in pend}
                if max(ks.values()) == 2:
                    reqs2 = []
                    for rid in pend:
                        st = self._states[rid]
                        toks = [st.pending.popleft() for _ in range(ks[rid])]
                        reqs2.append((rid, toks))
                        st.fed = min(st.fed + ks[rid], self._window)
                        budget[rid] -= ks[rid]
                    out_p, out_q = self._run_pairs(reqs2)
                    fed = [r[0] for r in reqs2]
                else:
                    feeds = [(rid, None, [self._states[rid].pending[0]]) for rid in pend]
                    out_p, out_q = self._run_step(feeds)
                    fed = [f[0] for f in feeds]
                    for rid in pend:
                        st = self._states[rid]
                        st.pending.popleft()
                        st.fed = min(st.fed + 1, self._window)
                        budget[rid] -= 1
            else:
                break
            rounds += 1
            self._ensure_bufs(out_p)
            fed_slots = self._slots_for(fed, out_p.device)
            self._buf_p.index_copy_(0, fed_slots, out_p.to(self._buf_p.dtype))
            self._buf_q.index_copy_(0, fed_slots, out_q.to(self._buf_q.dtype))

        if self._round_hist is not None:
            self._round_hist[rounds] += 1
            self._stat_steps += 1
            if self._stat_steps % 500 == 0:
                total = sum(self._round_hist)
                mean = sum(i * c for i, c in enumerate(self._round_hist)) / max(total, 1)
                print(
                    f"[UniversalBridge] drain rounds/step over {total} steps: "
                    f"hist={self._round_hist} mean={mean:.2f}",
                    flush=True,
                )

        if any(self._states[rid].pending for rid in order):
            self._backlog_steps += 1
            if self._backlog_steps & (self._backlog_steps - 1) == 0:  # 1, 2, 4, 8, ...
                print(
                    f"[UniversalBridge] WARNING: aux token backlog persisted past "
                    f"max_feeds_per_step={self._max_feeds} on {self._backlog_steps} "
                    f"step(s); raise max_feeds_per_step if this is sustained.",
                    flush=True,
                )

        self._order = order

    def finalize(self, V: int, rids: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase 2: cross to P's device and map onto its vocabulary.

        Separate from :meth:`prefetch` so that under overlap mode nothing
        lands in P's stream until fuse time. ``rids`` overrides the row order
        (overlap mode merges late-arriving rows after a second prefetch)."""
        order = rids if rids is not None else self._order
        idx = self._slots_for(list(order), self._buf_p.device)
        sel_p = self._buf_p.index_select(0, idx)
        sel_q = self._buf_q.index_select(0, idx)
        target = self._mapper.map_idx.device
        if sel_p.device != target:  # aux_device: raw aux logits cross to P's GPU
            sel_p = sel_p.to(target)
            sel_q = sel_q.to(target)
        return self._mapper.map_logits(sel_p, V), self._mapper.map_logits(sel_q, V)
