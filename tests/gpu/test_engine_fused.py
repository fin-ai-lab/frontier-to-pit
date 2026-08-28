"""GPU tests for the fused aux pair.

1. Ground truth: the fused engine's two planes vs two single engines run on
   identical token streams (bf16, real checkpoints). bmm-vs-mm kernels make
   bitwise comparison wrong by construction; the bar is the repo standard —
   argmax match + max|dP| <= 0.05, <= 2% bad positions.
2. Graph integrity: compile_model=True must actually capture BOTH CUDA graphs
   (`_graphs_broken` False). A silent fall-back to compiled-eager keeps
   results exact but forfeits the entire fused-pair perf win — this assert is
   the regression net for cuBLAS-workspace-allocation-during-capture.
"""

import torch

from ftp import AuxBatchedEngine

WINDOW, N, STEPS = 512, 4, 40

TEXTS = [
    "The capital of France is Paris, and the capital of Germany is",
    "In 2014, the fastest production car made by Bugatti was the",
    "Apple's newest iPhone model in early 2015 was the iPhone",
    "The stock market in September 2019 was dominated by large-cap",
]


def _bad(a: torch.Tensor, b: torch.Tensor) -> int:
    return int(
        int(a.argmax()) != int(b.argmax()) or (a.softmax(-1) - b.softmax(-1)).abs().max() > 0.05
    )


def _run_single(path, device, prompts, conts):
    """One eager single engine over the fixed script; returns greedy conts on
    first use (conts is None) plus per-step logits."""
    eng = AuxBatchedEngine(path, device, torch.bfloat16, WINDOW)
    n = len(prompts)
    for i in range(n):
        eng.register(i)
    outs = [eng.step([(i, prompts[i], []) for i in range(n)]).float().cpu()]
    grow = conts is None
    if grow:
        conts = [[] for _ in range(n)]
        for i in range(n):
            conts[i].append(int(outs[0][i].argmax()))
    for s in range(STEPS):
        out = eng.step([(i, prompts[i], conts[i][: s + 1]) for i in range(n)]).float().cpu()
        outs.append(out)
        if grow and s < STEPS - 1:
            for i in range(n):
                conts[i].append(int(out[i].argmax()))
    del eng
    torch.cuda.empty_cache()
    return outs, conts


def test_fused_groundtruth_vs_pair(cuda_device, aux_model_path, aux_model_path_q):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(aux_model_path)
    prompts = [tok(t)["input_ids"] for t in TEXTS[:N]]

    # References: each model alone, eager (the validated single-engine path).
    # Continuations are fixed from model p's greedy run, fed to all three.
    outs_p, conts = _run_single(aux_model_path, cuda_device, prompts, None)
    outs_q, _ = _run_single(aux_model_path_q, cuda_device, prompts, conts)

    fused = AuxBatchedEngine(
        aux_model_path, cuda_device, torch.bfloat16, WINDOW, model2=aux_model_path_q
    )
    for i in range(N):
        fused.register(i)
    bad = 0
    out = fused.step([(i, prompts[i], []) for i in range(N)]).float().cpu()
    for i in range(N):
        bad += _bad(outs_p[0][i], out[0, i]) + _bad(outs_q[0][i], out[1, i])
    for s in range(STEPS):
        out = fused.step([(i, prompts[i], conts[i][: s + 1]) for i in range(N)]).float().cpu()
        for i in range(N):
            bad += _bad(outs_p[s + 1][i], out[0, i]) + _bad(outs_q[s + 1][i], out[1, i])

    total = 2 * N * (STEPS + 1)
    assert bad <= total * 0.02, f"fused planes diverge from single engines: {bad}/{total}"


def test_fused_compile_graphs(cuda_device, aux_model_path, aux_model_path_q):
    """Both fused CUDA graphs must capture (not the eager fallback), and the
    compiled engine must match the eager fused engine on identical feeds."""
    torch.manual_seed(0)
    window, n, t0, steps = 256, 8, 16, 40
    prompts = torch.randint(10, 200_000, (n, t0)).tolist()
    conts = torch.randint(10, 200_000, (n, steps)).tolist()

    def run(compile_model: bool) -> list[torch.Tensor]:
        eng = AuxBatchedEngine(
            aux_model_path,
            cuda_device,
            torch.bfloat16,
            window,
            compile_model,
            model2=aux_model_path_q,
        )
        if compile_model:
            eng.prewarm(n)
            # THE perf assert: a broken capture silently runs compiled-eager
            # and forfeits the fused speedup.
            assert not eng._graphs_broken, "fused CUDA-graph capture fell back to eager"
            assert eng._graph is not None, "single-token fused graph not captured"
            assert eng._graph2 is not None, "pairs fused graph not captured"
        for i in range(n):
            eng.register(i)
        outs = [eng.step([(i, prompts[i], []) for i in range(n)]).float().cpu()]
        for s in range(steps):
            outs.append(
                eng.step([(i, prompts[i], conts[i][: s + 1]) for i in range(n)]).float().cpu()
            )
        del eng
        torch.cuda.empty_cache()
        return outs

    eager = run(False)
    comp = run(True)

    n_pos = agree = 0
    max_pdiff = 0.0
    for a, b in zip(eager, comp, strict=True):
        a2, b2 = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
        agree += int((a2.argmax(-1) == b2.argmax(-1)).sum())
        n_pos += a2.shape[0]
        max_pdiff = max(max_pdiff, (a2.softmax(-1) - b2.softmax(-1)).abs().max().item())

    rate = agree / n_pos
    assert rate >= 0.94, f"top-1 agreement {rate:.4f} < 0.94"
    assert max_pdiff <= 0.05, f"max prob diff {max_pdiff:.4f} > 0.05"
