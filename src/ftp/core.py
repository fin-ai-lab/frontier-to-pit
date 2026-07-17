"""Shared Divergence-Decoding fusion math (log-space, with whitelist pinning).

Linear DD:  l̂ = l_P + α·(l_q − l_p)                       (α = 0 -> off)
Rank   DD:  suppress the top-k tokens by (l_p − l_q)       (rank_k = 0 -> off)

The two adjustments are ORTHOGONAL and compose: linear first, then the rank mask
on top (the masked set depends only on the aux divergence, so order is moot).
Run either alone or both together.

Whitelist pinning: tokens in the whitelist keep the BIG model's probability
*exactly* (Q̂(i) = P(i)); DD is applied only to the complement, which is then
renormalized to fill the leftover mass (1 − Σ_{i∈S} P(i)). Temperature is applied
to the content side only, so the pinned probabilities (e.g. EOS) are exact under
sampling at any T. Everything is done in log space (logsumexp + log1mexp); we
never materialize the full probability vector. Computed in fp32 for stability.
"""

import contextlib

import torch

_LN2 = 0.6931471805599453


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Stable log(1 - exp(x)) for x <= 0."""
    x = x.clamp(max=0.0)
    return torch.where(x > -_LN2, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def build_special_ids(tok):
    """ALL control/special/added token ids. Do NOT trust only HF's `is_special`
    flag — many added control tokens (<|im_end|>, <|endoftext|>, <think>,
    vision/tool tokens, ...) are not flagged special. The authoritative source is
    the added-tokens table, unioned with the explicit specials."""
    ids = set()
    ids.update(int(i) for i in (tok.all_special_ids or []) if i is not None)
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id", "unk_token_id"):
        v = getattr(tok, attr, None)
        if v is not None:
            ids.add(int(v))
    for v in getattr(tok, "additional_special_tokens_ids", None) or []:
        if v is not None:
            ids.add(int(v))
    with contextlib.suppress(Exception):
        ids.update(int(i) for i in tok.added_tokens_decoder)
    with contextlib.suppress(Exception):
        ids.update(int(i) for i in tok.get_added_vocab().values())
    return sorted(ids)


def dd_fuse(
    l_P,
    l_p,
    l_q,
    *,
    alpha,
    rank_k=0,
    whitelist=None,
    temperature=1.0,
    wl_mask=None,
    pin=True,
):
    """Return the fused logits along the last dim.

    ``alpha`` scales the linear combination (0 = no linear shift); ``rank_k`` > 0
    additionally suppresses the ``rank_k`` tokens most divergent toward the forget
    aux (top-k of ``l_p − l_q``) to −inf, leaving all other tokens unaffected. The
    two compose; run either alone or both.

    By default (``pin=True``) DD is applied across the full vocabulary while
    whitelist token ids are pinned to P's probability exactly (Q̂(i) = P(i)) so
    EOS/control structure is never distorted; the result is normalized log-probs.
    This is the recommended path: without pinning, α inflates content logits and
    EOS loses relatively, so generations ramble and leak more (empirically
    fuse-pin is effectively mandatory). ``temperature`` is content-side (default
    1.0). ``wl_mask`` is an optional precomputed [V_full] bool mask equivalent to
    ``whitelist`` (callers on a hot path build it once instead of scattering per
    call).

    Pass ``pin=False`` (the serving path sets ``DDConfig.fuse_pin=False``) for the plain
    linear DD combination ``l̂ = l_P + α·(l_q − l_p)`` as raw logits (the
    downstream sampler normalizes), with the whitelist-pinning / renormalization
    machinery OFF. ``whitelist``/``wl_mask``/``temperature`` are ignored in that
    case; at temperature 0 the argmax of the plain fusion equals the
    pinned+renormalized result to floating-point.
    """
    V = min(l_P.shape[-1], l_p.shape[-1], l_q.shape[-1])
    a = l_P[..., :V].float()
    lp = l_p[..., :V].float()
    lq = l_q[..., :V].float()

    lhat = a if alpha == 0.0 else a + alpha * (lq - lp)
    if rank_k:
        idx = (lp - lq).topk(rank_k, dim=-1).indices
        # out-of-place: when alpha == 0, `lhat` may alias the caller's fp32 logits
        lhat = lhat.scatter(-1, idx, float("-inf"))

    if not pin:
        # Pinning disabled: return the raw fused logits; the sampler normalizes.
        return lhat.to(l_P.dtype)

    if wl_mask is not None:
        wl_mask = wl_mask[:V]
    else:
        wl_mask = torch.zeros(V, dtype=torch.bool, device=a.device)
        if whitelist is not None and torch.as_tensor(whitelist).numel() > 0:
            wl = whitelist[whitelist < V]
            wl_mask[wl] = True

    pinned = wl_mask  # whitelist tokens -> pin to P(i)
    content = ~wl_mask  # everything else -> DD adjusted

    Z = torch.logsumexp(a, dim=-1, keepdim=True)
    logP = a - Z
    logM = torch.logsumexp(a.masked_fill(~pinned, float("-inf")), dim=-1, keepdim=True) - Z
    logW = log1mexp(logM)  # log(1 - pinned_mass)

    lhat_c = lhat.masked_fill(~content, float("-inf")) / temperature
    logr = torch.log_softmax(lhat_c, dim=-1).masked_fill(~content, float("-inf"))

    out = torch.where(pinned, logP, logW + logr)
    return out.to(l_P.dtype)
