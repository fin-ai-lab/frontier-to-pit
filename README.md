# Frontier to Point-in-Time

Adapt an open-source frontier LLM to reduce **look-ahead bias** at inference time —
no retraining of the large model — for point-in-time research in finance, economics,
and other settings where look-ahead bias is a confounder.

Two interventions, usable alone or composed (we usually use both):

- **Divergence Decoding (DD)** — unlearning via a forget/retain pair of small
  auxiliary models: `l = l_P + α·(l_retain − l_forget)`.
- **SAE Feature Steering** — clamp interpretable features that reason from a
  historical perspective.

Project page: https://frontiertopit.com

## Install

```bash
# uv (skip if you already have it): https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/fin-ai-lab/frontier-to-pit
cd frontier-to-pit
uv sync --extra vllm
```

`uv sync` reproduces the exact tested stack from the lockfile (vLLM 0.21.0, torch
2.11.0 for CUDA 13). Prefer it over `uv pip install` — a fresh resolve pulls a newer,
untested vLLM. Requirements:

- A CUDA-13 GPU box with the CUDA toolkit (`nvcc`) on PATH.
- An ≥80GB GPU for the 27B model. 2×80GB is recommended: the aux pair (and the
  degeneration-guard judge) are placed on the second GPU automatically (with one GPU,
  pass a smaller `--gpu-memory-utilization`).
- `--think` at its full 20K default context does not fit next to the aux pair on
  2×80GB — it needs 4×80GB: pass `--tensor-parallel-size 2` and the placement follows
  automatically (the 27B shards over `cuda:0`–`cuda:1`, the retain aux model takes
  `cuda:2`, the forget aux model `cuda:3` — one engine per card — and the guard judge
  rides `cuda:1`'s spare TP memory). `python -m ftp.probe --tp 2 ...` prints the fit
  math for your models and budgets.

> **Startup time.** Every launch spends ~1–3 min on vLLM's `torch.compile` + warmup
> (with steering on — the default — the compile cache is disabled for safety, so this
> runs every time, not just the first). On top of that, the default GDN backend adds a
> **one-time ~15–20 min nvcc kernel compile on the very first run** (it looks idle;
> don't kill it), cached afterward, and worth it for ~2× throughput. **Just trying it
> out or timing startup?** Pass `--fast` (below) to skip *both* — it enforces eager
> execution (no `torch.compile`/graph capture) and uses the triton GDN backend (builds
> in seconds), leaving startup at roughly model-load time, at a small per-token cost.

## Chat in the terminal

```bash
uv run python run.py chat --fast                       # DD + steering, fastest startup
uv run python run.py generate --fast "Using only information through December 31, 2015, predict the 2016 GOP Presidential Nominee. Answer in one short paragraph." # one prompt, stream, exit
uv run python run.py chat --fast --think               # reasoning on for the session
uv run python run.py chat --fast --no-guard            # disable the degeneration guard
uv run python run.py chat --fast --no-dd --no-steer    # plain base model
```

> **`chat` keeps no history.** Each message is answered as a fresh conversation with no
> prior turns in context, and `run.py chat` reprints a red 🔴 banner every turn so this
> is obvious. That's a simplification of this runner, **not** a limit of the methods —
> DD + steering work multi-turn; history is off by default here because the benchmarks
> (and the α/clamp calibrated against them) are single-turn, so keeping it would quietly
> move you off the measured config. Press **ENTER** during a reply to stop just that
> reply and return to the prompt (Ctrl-C can't — vLLM's engine-core subprocess treats
> SIGINT as shutdown, so Ctrl-C ends the whole session). Quit with `exit` / `quit` /
> `q`, or Ctrl-C / Ctrl-D.

`--think` turns on reasoning for the whole run (both `chat` and `generate`): the chat
template opens a real `<think>` block instead of a pre-closed empty one, the `<think>`
ban is dropped, and the budgets rise to the evals' thinking arm (`--max-new 16384`,
`--max-model-len 20480`; pass either explicitly to override). Treat it as
**exploratory** — α and the steering clamp are calibrated with thinking off, and the
benchmarks below are NOTHINK, so `--think` is off the measured config.

### The live degeneration guard (on by default with DD)

Under a strong DD push a reply can occasionally collapse into a repeating loop or
symbol spam. A small judge LM (`Qwen/Qwen3.5-2B`, loaded on the aux GPU) watches every
reply as it generates: every 25 engine steps it reads the newest 50 tokens, and when it
sees mechanical collapse the reply is **rewound 50 tokens and resampled** — the fusion
(and therefore the unlearning) stays at full strength, only the collapse is repaired.
If a reply keeps breaking at the same spot, the rewind deepens (100, 150, … tokens);
one that can't produce ANY clean text returns
`[Could not generate without degeneration]` instead of garbage.

Because of the rewind window, **streaming is held back by ~50 tokens**: text appears in
blocks once it can no longer be rewound (about a second behind generation), and on the
rare deep rewind the runner prints `⟲ retracting the passage above and rewriting`.
`--no-guard` disables the whole mechanism; `--guard-interval/--guard-backtrack/
--guard-threshold/--guard-tries` tune it. The judge adds one small batched forward per
25 steps on the aux GPU — throughput impact is negligible, and replies that never
degenerate are never interrupted.

`--fast` is the quick-try / startup-timing bundle: it implies `--enforce-eager`
(skip torch.compile + CUDA-graph capture, ~1–3 min) **and** `--gdn-prefill-backend
triton` (skip the ~15–20 min first-run nvcc GDN build), so the engine is ready in
roughly model-load time. The cost is ~2× slower per token — **drop `--fast` for
production-throughput serving** (it pays the one-time compiles, cached afterward,
and keeps full cudagraphs). The flags also work individually.

Defaults reproduce the benchmarked config: bf16 P (`Qwen/Qwen3.5-27B`), aux models
from the HF Hub, α=1.5, feature L48:28961@10, thinking off, and the Qwen 3.5 27B
sampling preset — the same P precision, sampling and `<think>` ban as `evals/lmeval`,
so `run.py` matches the benchmarks. Everything downloads from public repos on first
use — no HF token required.

## Use it in a Python project

```python
from ftp.serve import build_llm, SteerArgs, parse_steer
from vllm import SamplingParams

llm, _ = build_llm(
    "Qwen/Qwen3.5-27B",            # bf16 — the benchmarked P precision
    aux_p="fin-ai-lab/aux-2024",   # DD forget (has post-cutoff knowledge)
    aux_q="fin-ai-lab/aux-2015",   # DD retain
    steer=SteerArgs(parse_steer("48:28961:10"),   # feature steering, baked in at build
                    sae_repo="Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50"),
)

sp = SamplingParams(                # Qwen 3.5 27B preset
    max_tokens=2048, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
    presence_penalty=1.5, repetition_penalty=1.0, bad_words=["<think>"],
    extra_args={"dd_alpha": 1.5},  # DD strength, per request
)

out = llm.generate(["Using only information through December 31, 2015, predict the 2016 GOP Presidential Nominee. Answer in one short paragraph."], sp)
print(out[0].outputs[0].text)
```

DD strength (`dd_alpha`) is a per-request sampling arg; the steering clamp is fixed
at build time (baked into the CUDA graphs). Drop `steer=` for DD-only, or omit the
`aux_*` args for steering-only. The `bad_words` entry above is the NOTHINK ban that
matches the benchmarks — for the `--think` equivalent, drop it *and* render the prompt
with `apply_chat_template(..., enable_thinking=True)`, since the two must agree.

For the degeneration guard, pass `guard=GuardConfig()` (from `ftp.guard`) to
`build_llm` — that installs detection in the engine — and generate through
`ftp.guard.rollback_generate(llm, token_prompts, sp, marker_id=resolve_marker_id(tok),
backtrack=50, decode=tok.decode)`, which handles the rewind-and-resample. `run.py`
wires both ends for you.

Run it inside the project's environment with `uv run python your_script.py`.

## Citation

```text
\cite{merchant2026a,merchant2026divergence,merchant2026forecasting}
```

```bibtex
@inproceedings{merchant2026a,
    title={A Fast and Effective Solution to the Problem of Look-ahead Bias in {LLM}s},
    author={Humzah Merchant and Bradford Levy},
    booktitle={NeurIPS 2025 Workshop: Generative AI in Finance},
    year={2026},
    url={https://openreview.net/forum?id=zYsLIPgM28}
}
@inproceedings{merchant2026divergence,
    title={Divergence Decoding: Inference-Time Unlearning via Auxiliary Models},
    author={Humzah Merchant and Bradford Levy},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://openreview.net/forum?id=JPbp2S9yTO}
}
@inproceedings{merchant2026forecasting,
    title={Forecasting With {LLM}s: Improved Generalization Through Feature Steering},
    author={Humzah Merchant and Bradford Levy},
    booktitle={Forecasting as a New Frontier of Intelligence},
    year={2026},
    url={https://openreview.net/forum?id=ppN6CmoNOk}
}
```

## Troubleshooting

**`CUDA error 802: system not yet initialized`** (or engine-core fails to start on a
multi-GPU box): the NVLink fabric manager isn't running — common on misprovisioned
NVSwitch instances. Start it with `sudo systemctl start nvidia-fabricmanager`; if that
also fails, the box is bad — spin up a fresh one.

Apache-2.0 (see `LICENSE`).
