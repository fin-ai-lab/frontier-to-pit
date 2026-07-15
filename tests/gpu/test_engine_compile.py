"""Compiled (CUDA-graph) aux decode vs eager, including window crossings.

Feeds IDENTICAL fixed random token sequences through both engine configurations
(no greedy feedback, so a single near-tie can't cascade) for 100 decode steps
with window=64 — the sliding-window shift runs ~50 times under graph replay.

Random-token prompts are the adversarial worst case for argmax stability
(near-flat logits flip on inductor-vs-eager bf16 kernel noise), hence the loose
agreement threshold; distributional parity on real prompts is gated by
test_engine_groundtruth.py.
"""

import torch

from ftp import AuxBatchedEngine

WINDOW, N, T0, STEPS = 64, 8, 16, 100


def test_compiled_equals_eager(cuda_device, aux_model_path):
    torch.manual_seed(0)
    prompts = torch.randint(10, 200_000, (N, T0)).tolist()
    conts = torch.randint(10, 200_000, (N, STEPS)).tolist()

    def run(compile_model: bool) -> list[torch.Tensor]:
        eng = AuxBatchedEngine(aux_model_path, cuda_device, torch.bfloat16, WINDOW, compile_model)
        if compile_model:
            eng.prewarm(N)
        for i in range(N):
            eng.register(i)
        outs = [eng.step([(i, prompts[i], []) for i in range(N)]).float().cpu()]
        for s in range(STEPS):
            reqs = [(i, prompts[i], conts[i][: s + 1]) for i in range(N)]
            outs.append(eng.step(reqs).float().cpu())
        for i in range(N):
            eng.unregister(i)
        del eng
        torch.cuda.empty_cache()
        return outs

    eager = run(False)
    comp = run(True)

    n_pos = agree = 0
    max_pdiff = 0.0
    for a, b in zip(eager, comp, strict=True):
        agree += int((a.argmax(-1) == b.argmax(-1)).sum())
        n_pos += a.shape[0]
        max_pdiff = max(max_pdiff, (a.softmax(-1) - b.softmax(-1)).abs().max().item())

    rate = agree / n_pos
    assert rate >= 0.94, f"top-1 agreement {rate:.4f} < 0.94"
    assert max_pdiff <= 0.05, f"max prob diff {max_pdiff:.4f} > 0.05"


def test_two_compiled_engines_coexist(cuda_device, aux_model_path):
    """The vLLM processor builds TWO compiled engines in one process (aux_p and
    aux_q). Their CUDA graphs must capture and replay independently — this is
    the regression net for the inductor-cudagraph interaction that crashed
    in production when both engines shared global compile state."""
    torch.manual_seed(1)
    n, t0, steps = 4, 12, 8
    prompts = torch.randint(10, 200_000, (n, t0)).tolist()
    conts = torch.randint(10, 200_000, (n, steps)).tolist()

    engines = [
        AuxBatchedEngine(aux_model_path, cuda_device, torch.bfloat16, 256, True) for _ in range(2)
    ]
    for eng in engines:
        eng.prewarm(n)

    outs = []
    for eng in engines:
        for i in range(n):
            eng.register(i)
        o = [eng.step([(i, prompts[i], []) for i in range(n)]).float().cpu()]
        for s in range(steps):
            reqs = [(i, prompts[i], conts[i][: s + 1]) for i in range(n)]
            o.append(eng.step(reqs).float().cpu())
        outs.append(o)

    # Same checkpoint loaded twice -> the two engines must agree closely.
    for a, b in zip(outs[0], outs[1], strict=True):
        agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
        assert agree >= 0.9, f"two compiled engines diverge: top-1 agree {agree:.3f}"
