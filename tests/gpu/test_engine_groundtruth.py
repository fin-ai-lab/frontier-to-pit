"""GPU ground-truth test: AuxBatchedEngine vs full-context forward, real model.

Same methodology as tests/test_engine_cpu.py but with a real bf16 checkpoint
and natural-language prompts: catches issues that only appear at scale
(numerics, real tokenizer ids, bf16 cache round-trips).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ftp import AuxBatchedEngine

WINDOW, N, STEPS = 2048, 4, 60

TEXTS = [
    "The capital of France is Paris, and the capital of Germany is",
    "In 2014, the fastest production car made by Bugatti was the",
    "Apple's newest iPhone model in early 2015 was the iPhone",
    "The stock market in September 2019 was dominated by large-cap",
]


def test_groundtruth(cuda_device, aux_model_path):
    tok = AutoTokenizer.from_pretrained(aux_model_path)
    prompts = [tok(t)["input_ids"] for t in TEXTS[:N]]

    ref_model = (
        AutoModelForCausalLM.from_pretrained(aux_model_path, dtype=torch.bfloat16)
        .to(cuda_device)
        .eval()
    )

    # Reference: full-context forward per step; greedy continuations from the
    # reference are fed to BOTH runs so a near-tie can't cascade.
    seqs = [list(p) for p in prompts]
    ref, conts = {}, [[] for _ in range(N)]
    with torch.no_grad():
        for s in range(STEPS + 1):
            for i in range(N):
                ids = torch.tensor([seqs[i]], device=cuda_device)
                lg = ref_model(input_ids=ids).logits[0, -1].float().cpu()
                ref[(s, i)] = lg
                if s < STEPS:
                    nxt = int(lg.argmax())
                    seqs[i].append(nxt)
                    conts[i].append(nxt)
    del ref_model
    torch.cuda.empty_cache()

    eng = AuxBatchedEngine(aux_model_path, cuda_device, torch.bfloat16, WINDOW)
    for i in range(N):
        eng.register(i)
    bad = 0
    out = eng.step([(i, prompts[i], []) for i in range(N)])
    for i in range(N):
        a, b = ref[(0, i)], out[i].float().cpu()
        bad += int(
            int(a.argmax()) != int(b.argmax()) or (a.softmax(-1) - b.softmax(-1)).abs().max() > 0.05
        )
    for s in range(STEPS):
        out = eng.step([(i, prompts[i], conts[i][: s + 1]) for i in range(N)])
        for i in range(N):
            a, b = ref[(s + 1, i)], out[i].float().cpu()
            bad += int(
                int(a.argmax()) != int(b.argmax())
                or (a.softmax(-1) - b.softmax(-1)).abs().max() > 0.05
            )

    total = N * (STEPS + 1)
    assert bad <= total * 0.02, f"engine diverges from ground truth: {bad}/{total}"
