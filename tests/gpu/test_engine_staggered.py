"""GPU ground-truth test for the staggered (leader/follower) decode pattern.

vLLM gives one request a 1-step head start at batch start, so the batch
decodes permanently one token apart — exercising the grouped decode path
(batched forward per position group with column backup/restore).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ftp import AuxBatchedEngine

WINDOW, N, STEPS = 2048, 4, 40
TEXT = "In 2014, the fastest production car made by Bugatti was the"


def test_staggered(cuda_device, aux_model_path):
    tok = AutoTokenizer.from_pretrained(aux_model_path)
    prompt = tok(TEXT)["input_ids"]

    ref_model = (
        AutoModelForCausalLM.from_pretrained(aux_model_path, dtype=torch.bfloat16)
        .to(cuda_device)
        .eval()
    )

    def ref_step(seq):
        with torch.no_grad():
            ids = torch.tensor([seq], device=cuda_device)
            return ref_model(input_ids=ids).logits[0, -1].float().cpu()

    # Request 0 prefills at step 0; requests 1.. prefill at step 1 while 0
    # decodes; everyone decodes (one apart) afterwards. Continuations are
    # reference-greedy so both runs see identical tokens.
    seqs = [list(prompt) for _ in range(N)]
    ref, conts = {}, [[] for _ in range(N)]

    lg = ref_step(seqs[0])

    ref[(0, 0)] = lg
    conts[0].append(int(lg.argmax()))
    seqs[0].append(int(lg.argmax()))
    lg = ref_step(seqs[0])
    ref[(1, 0)] = lg
    conts[0].append(int(lg.argmax()))
    seqs[0].append(int(lg.argmax()))
    for i in range(1, N):
        lg = ref_step(seqs[i])
        ref[(1, i)] = lg
        conts[i].append(int(lg.argmax()))
        seqs[i].append(int(lg.argmax()))
    for s in range(2, STEPS):
        for i in range(N):
            lg = ref_step(seqs[i])
            ref[(s, i)] = lg
            conts[i].append(int(lg.argmax()))
            seqs[i].append(int(lg.argmax()))
    ref_model.cpu()
    torch.cuda.empty_cache()

    eng = AuxBatchedEngine(aux_model_path, cuda_device, torch.bfloat16, WINDOW)
    got = {}
    eng.register(0)
    got[(0, 0)] = eng.step([(0, prompt, [])])[0].float().cpu()
    for i in range(1, N):
        eng.register(i)
    reqs = [(0, prompt, conts[0][:1])] + [(i, prompt, []) for i in range(1, N)]
    out = eng.step(reqs)
    for j, (rid, _, _) in enumerate(reqs):
        got[(1, rid)] = out[j].float().cpu()
    for s in range(2, STEPS):
        reqs = [(0, prompt, conts[0][:s])] + [(i, prompt, conts[i][: s - 1]) for i in range(1, N)]
        out = eng.step(reqs)
        for j, (rid, _, _) in enumerate(reqs):
            got[(s, rid)] = out[j].float().cpu()

    bad = sum(
        int(
            int(ref[k].argmax()) != int(got[k].argmax())
            or (ref[k].softmax(-1) - got[k].softmax(-1)).abs().max() > 0.05
        )
        for k in ref
    )
    assert bad == 0, f"staggered path diverges from ground truth: {bad}/{len(ref)}"
