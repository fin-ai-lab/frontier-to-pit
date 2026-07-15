"""Unit tests for the feasibility probe's geometry math (no GPU, no models)."""

from ftp.probe import kv_per_seq_gb

QWEN35_27B = {  # hybrid: 16/64 full-attention, rest linear attention
    "num_hidden_layers": 64,
    "num_key_value_heads": 4,
    "num_attention_heads": 32,
    "head_dim": 256,
    "layer_types": ["full_attention"] * 16 + ["linear_attention"] * 48,
}

GEMMA4_31B = {  # dense: 10 full + 50 sliding-1024, 16 kv heads
    "num_hidden_layers": 60,
    "num_key_value_heads": 16,
    "num_attention_heads": 32,
    "head_dim": 256,
    "sliding_window": 1024,
    "layer_types": ["full_attention"] * 10 + ["sliding_attention"] * 50,
}

GEMMA4_26B_A4B = {  # MoE: 5 full + 25 sliding-1024, 8 kv heads
    "num_hidden_layers": 30,
    "num_key_value_heads": 8,
    "num_attention_heads": 16,
    "head_dim": 256,
    "sliding_window": 1024,
    "layer_types": ["full_attention"] * 5 + ["sliding_attention"] * 25,
}

AUX_3B = {  # plain full attention, no layer_types
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "num_attention_heads": 16,
    "head_dim": 128,
}


def test_kv_per_seq_matches_measured_models():
    # Anchors from the deployed configurations (see README Deployment table).
    assert abs(kv_per_seq_gb(QWEN35_27B, 2048) - 0.134) < 0.005
    assert abs(kv_per_seq_gb(GEMMA4_31B, 2048) - 1.174) < 0.01
    assert abs(kv_per_seq_gb(GEMMA4_26B_A4B, 2048) - 0.294) < 0.005


def test_linear_attention_layers_cost_nothing():
    dense = dict(QWEN35_27B, layer_types=["full_attention"] * 64)
    assert kv_per_seq_gb(QWEN35_27B, 2048) < kv_per_seq_gb(dense, 2048) / 3.9


def test_sliding_window_caps_context():
    short = kv_per_seq_gb(GEMMA4_31B, 1024)
    long = kv_per_seq_gb(GEMMA4_31B, 8192)
    # sliding layers are capped at 1024, so cost grows sublinearly with ctx
    assert long < 8 * short


def test_aux_slot_geometry():
    # 3B aux: 0.118 GB/slot at window 1024 (the measured per-slot figure)
    assert abs(kv_per_seq_gb(AUX_3B, 1025) - 0.118) < 0.004
