import pytest

from ftp import DDConfig
from ftp.config import default_device_layout


def make_cfg(**kw):
    return DDConfig(aux_p="org/aux-p", aux_q="org/aux-q", **kw)


def test_env_round_trip():
    cfg = make_cfg(
        window=1024,
        alpha_default=1.0,
        suppress_tokens=("<think>", 42),
        tokenizer="org/tok",
        dtype="float16",
        compile_aux=True,
        prewarm=16,
        mode="universal",
        max_feeds_per_step=2,
        fuse_aux="off",
        retemplate="gemma",
    )
    assert DDConfig.from_env(cfg.to_env()) == cfg


def test_env_round_trip_defaults():
    cfg = make_cfg()
    assert DDConfig.from_env(cfg.to_env()) == cfg


def test_from_env_minimal():
    cfg = DDConfig.from_env({"DD_AUX_P": "p", "DD_AUX_Q": "q"})
    assert cfg.aux_p == "p"
    assert cfg.suppress_tokens == ()
    assert cfg.tokenizer_path == "p"
    assert cfg.mode == "auto"
    assert cfg.max_feeds_per_step == 3
    assert cfg.fuse_aux == "auto"


def test_from_env_missing_required():
    with pytest.raises(KeyError):
        DDConfig.from_env({"DD_AUX_P": "p"})


def test_apply_env():
    env: dict[str, str] = {}
    make_cfg(window=512).apply_env(env)
    assert env["DD_WINDOW"] == "512"
    assert env["DD_AUX_P"] == "org/aux-p"


@pytest.mark.parametrize(
    "kw",
    [
        {"window": 1},
        {"alpha_default": -1.0},
        {"prewarm": -1},
        {"mode": "hybrid"},
        {"max_feeds_per_step": 0},
        {"fuse_aux": "maybe"},
    ],
)
def test_validation(kw):
    with pytest.raises(ValueError):
        make_cfg(**kw)


def test_requires_aux_paths():
    with pytest.raises(ValueError):
        DDConfig(aux_p="", aux_q="q")


def test_suppress_token_parsing():
    cfg = DDConfig.from_env(
        {"DD_AUX_P": "p", "DD_AUX_Q": "q", "DD_SUPPRESS_TOKENS": "<think>, 42 ,-1"}
    )
    assert cfg.suppress_tokens == ("<think>", 42, -1)


def test_resolve_suppress_ids():
    class FakeTok:
        unk_token_id = 0

        def convert_tokens_to_ids(self, t):
            return {"<think>": 7}.get(t, 0)

    cfg = make_cfg(suppress_tokens=("<think>", 99))
    assert cfg.resolve_suppress_ids(FakeTok()) == [7, 99]

    bad = make_cfg(suppress_tokens=("<nope>",))
    with pytest.raises(ValueError):
        bad.resolve_suppress_ids(FakeTok())


def test_aux_device_round_trip():
    cfg = make_cfg(aux_device="cuda:1")
    assert DDConfig.from_env(cfg.to_env()) == cfg
    assert DDConfig.from_env(make_cfg().to_env()).aux_device is None


@pytest.mark.parametrize(
    ("n_gpu", "tp", "expected"),
    [
        (1, 1, (None, None)),  # single GPU: everything co-hosts with P
        (2, 1, ("cuda:1", "cuda:1")),  # classic 2xH100: aux + guard share cuda:1
        (2, 2, ("cuda:1", "cuda:1")),  # P spans both: aux+guard on the last rank
        (3, 1, ("cuda:1", "cuda:2")),
        (4, 1, ("cuda:1", "cuda:2")),
        (4, 2, ("cuda:2", "cuda:3")),  # 4xH100 thinking-mode split
        (4, 4, ("cuda:3", "cuda:3")),
        (8, 4, ("cuda:4", "cuda:5")),
    ],
)
def test_default_device_layout(n_gpu, tp, expected):
    assert default_device_layout(n_gpu, tp) == expected


def test_default_device_layout_explicit_passthrough():
    # Explicit values always win, in any combination.
    assert default_device_layout(4, 2, aux_device="cuda:1", guard_device="cuda:0") == (
        "cuda:1", "cuda:0")
    # Explicit aux only: the guard default routes around it.
    assert default_device_layout(4, 1, aux_device="cuda:2") == ("cuda:2", "cuda:1")
    assert default_device_layout(2, 1, aux_device="cuda:0") == ("cuda:0", "cuda:1")
    # Explicit guard only: aux keeps its layout slot.
    assert default_device_layout(4, 2, guard_device="cuda:3") == ("cuda:2", "cuda:3")
    # Non-indexable aux device: guard falls back to the first non-P GPU.
    assert default_device_layout(2, 1, aux_device="cpu") == ("cpu", "cuda:1")
