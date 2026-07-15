"""Hardware feasibility probe: will this DD configuration fit, and how?

Analyzes the P model and aux pair against the machine's GPUs and prints a
recommended deployment: single-GPU or split placement, the
``gpu_memory_utilization`` value (vLLM budgets ``util × capacity`` for
*itself* — P weights + activations + KV pool — and is blind to the aux
engines' allocations, so util must leave physical room for them), and the
expected KV headroom at your target concurrency.

Usage::

    python -m ftp.probe \\
        --p-model /path/or/hub-id --aux-p ... --aux-q ... \\
        [--concurrency 32] [--window 1024] [--max-len 2048] [--live-aux]

The aux footprint estimate models the paged engine: its page pool is
prewarmed to the ``concurrency × window`` worst case, so the estimate is the
safe upper bound — live usage is the tokens actually in flight. ``--live-aux``
loads the pair for real (fused when the pair is fusable, like production),
prewarms at the target concurrency, and replaces the estimate with a
measurement plus a per-step latency sample.

All math is static (config.json + safetensors index) except ``--live-aux``;
no P-model load is ever required.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

WORKSPACE_GB = 2.0  # vLLM activation workspace + CUDA context, empirical
BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


def _load_config(model: str) -> dict:
    p = Path(model) / "config.json"
    if p.exists():
        cfg = json.loads(p.read_text())
    else:
        from huggingface_hub import hf_hub_download

        cfg = json.loads(Path(hf_hub_download(model, "config.json")).read_text())
    return cfg.get("text_config", cfg)


def weights_gb(model: str) -> float:
    """Total parameter bytes from the safetensors index (exact, no load)."""
    d = Path(model)
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["metadata"]["total_size"] / 1e9
    single = d / "model.safetensors"
    if single.exists():
        return single.stat().st_size / 1e9
    from huggingface_hub import hf_hub_download

    idx = hf_hub_download(model, "model.safetensors.index.json")
    return json.loads(Path(idx).read_text())["metadata"]["total_size"] / 1e9


def kv_per_seq_gb(cfg: dict, ctx_len: int, dtype_bytes: int = 2) -> float:
    """Per-sequence KV-cache cost at ``ctx_len`` from attention geometry.

    layer_types-aware: full-attention layers cache ctx_len, sliding layers
    cache min(ctx_len, window), linear-attention/mamba layers cache a
    constant-size state (counted as 0 here — small relative to attention KV).
    Parameter count is NOT a proxy for this; a dense 31B can cost 8x a 27B
    hybrid per sequence.
    """
    L = cfg["num_hidden_layers"]
    kv = cfg.get("num_key_value_heads") or cfg["num_attention_heads"]
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    sw = cfg.get("sliding_window") or ctx_len
    layer_types = cfg.get("layer_types") or ["full_attention"] * L
    tokens = 0
    for t in layer_types:
        if "linear" in t or "mamba" in t or "recurrent" in t:
            continue
        tokens += min(ctx_len, sw) if "sliding" in t else ctx_len
    return tokens * kv * hd * 2 * dtype_bytes / 1e9


def aux_pair_footprint_gb(
    aux_p: str, aux_q: str, concurrency: int, window: int, dtype_bytes: int = 2
) -> tuple[float, str]:
    """Estimated aux-pair footprint: weights + the prewarmed page pool.

    The pool is sized to the ``concurrency × window`` worst case, so this is
    an UPPER BOUND — live usage is the tokens actually in flight. The engine
    captures no CUDA graphs; ~0.5 GB/model covers eager activations + the
    flashinfer workspace."""
    total, parts = 0.0, []
    for m in (aux_p, aux_q):
        w = weights_gb(m)
        slots = kv_per_seq_gb(_load_config(m), window, dtype_bytes) * concurrency
        total += w + slots + 0.5
        parts.append(f"{Path(m).name}: weights {w:.1f} + page pool <= {slots:.1f}")
    return total, "; ".join(parts)


def recommend(args) -> None:
    import torch

    n_gpu = torch.cuda.device_count()
    caps = []
    for d in range(n_gpu):
        free, total = torch.cuda.mem_get_info(d)
        caps.append(total / 1e9)
        name = torch.cuda.get_device_properties(d).name
        print(f"GPU {d}: {name}, {total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free")
    if not caps:
        raise SystemExit("no CUDA devices visible")

    p_cfg = _load_config(args.p_model)
    p_w = weights_gb(args.p_model)
    p_kv_seq = kv_per_seq_gb(p_cfg, args.max_len)
    kv_pool = p_kv_seq * args.concurrency * 1.1  # +10% block-rounding headroom
    print(
        f"\nP: {args.p_model}\n  weights {p_w:.1f} GB, KV {p_kv_seq * 1000:.0f} MB/seq "
        f"@{args.max_len} ctx -> pool for {args.concurrency} seqs: {kv_pool:.1f} GB"
    )

    if args.live_aux:
        aux_total, detail = measure_aux_live(args, torch.device(f"cuda:{n_gpu - 1}"))
    else:
        aux_total, detail = aux_pair_footprint_gb(
            args.aux_p, args.aux_q, args.concurrency, args.window
        )
    print(f"aux pair: ~{aux_total:.1f} GB ({detail})")

    cap = caps[0]
    print("\n— Placement —")
    single_need = p_w + aux_total + kv_pool + WORKSPACE_GB
    util_single = (p_w + WORKSPACE_GB + kv_pool) / cap
    if single_need <= cap and util_single < 0.95:
        print(
            f"SINGLE GPU: fits ({single_need:.1f} of {cap:.1f} GB). "
            f"Set gpu_memory_utilization={util_single:.2f} "
            f"(vLLM's budget covers ONLY P weights+activations+KV; the aux pair "
            f"lives outside it)."
        )
    else:
        print(
            f"SINGLE GPU: does NOT fit ({single_need:.1f} of {cap:.1f} GB needed). "
            f"Reduce concurrency/window, use smaller aux models, or split:"
        )
    if n_gpu >= 2:
        aux_dev = n_gpu - 1
        p_need = p_w + kv_pool + WORKSPACE_GB
        util_split = min(0.95, (p_w + WORKSPACE_GB + kv_pool) / cap)
        fits_p = p_need <= cap
        fits_aux = aux_total <= caps[aux_dev] * 0.95
        verdict = "fits" if fits_p and fits_aux else "does NOT fit"
        print(
            f"SPLIT ({n_gpu} GPUs): {verdict}. P on cuda:0 "
            f"({p_need:.1f} of {cap:.1f} GB, gpu_memory_utilization={util_split:.2f}), "
            f"aux pair on cuda:{aux_dev} ({aux_total:.1f} of {caps[aux_dev]:.1f} GB) "
            f"via DDConfig(aux_device='cuda:{aux_dev}')."
        )
    print(
        "\nAlways: prewarm = peak DD concurrency; enable_prefix_caching=False; "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; if vLLM OOMs in "
        "_allocate_kv_cache_tensors at startup, lower gpu_memory_utilization."
    )


def measure_aux_live(args, device) -> tuple[float, str]:
    """Load + prewarm the aux pair (FUSED when the pair is fusable, like
    production) and measure footprint + per-step latency."""
    import torch
    from transformers import AutoConfig

    from ftp.engine import AuxBatchedEngine as Eng
    from ftp.paired import check_fusable

    fusable, why = check_fusable(
        AutoConfig.from_pretrained(args.aux_p, trust_remote_code=True),
        AutoConfig.from_pretrained(args.aux_q, trust_remote_code=True),
    )
    before = torch.cuda.mem_get_info(device)[0]
    if fusable:
        engines = [Eng(args.aux_p, device, torch.bfloat16, args.window, model2=args.aux_q)]
        how = "fused"
    else:
        print(f"[probe] pair not fusable ({why}); two-engine path")
        engines = [Eng(m, device, torch.bfloat16, args.window)
                   for m in (args.aux_p, args.aux_q)]
        how = "sequential"
    for e in engines:
        e.prewarm(args.concurrency)
    used = (before - torch.cuda.mem_get_info(device)[0]) / 1e9

    rids = list(range(args.concurrency))
    for e in engines:
        for r in rids:
            e.register(r)
        e.step([(r, list(range(2, 30)), []) for r in rids])
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    steps = 20
    outs = {r: [] for r in rids}
    for s in range(steps):
        for r in rids:
            outs[r].append(31 + s)
        for e in engines:
            e.step([(r, list(range(2, 30)), outs[r]) for r in rids])
    torch.cuda.synchronize(device)
    ms = (time.perf_counter() - t0) / steps * 1000
    for e in engines:
        for r in rids:
            e.unregister(r)
    return used, f"measured [{how}] at B={args.concurrency}; pair step {ms:.1f} ms"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p-model", required=True)
    ap.add_argument("--aux-p", required=True)
    ap.add_argument("--aux-q", required=True)
    ap.add_argument("--concurrency", type=int, default=32, help="peak concurrent DD requests")
    ap.add_argument("--window", type=int, default=1024, help="aux context window")
    ap.add_argument("--max-len", type=int, default=2048, help="P max_model_len")
    ap.add_argument("--live-aux", action="store_true", help="measure the aux pair for real")
    recommend(ap.parse_args())


if __name__ == "__main__":
    main()
