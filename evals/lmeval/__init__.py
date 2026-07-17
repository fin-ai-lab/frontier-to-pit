"""Divergence-Decoding evals on lm-evaluation-harness.

We run all utility evals (MMLU-Pro, MMLU-Redux, IFEval, GPQA) and the temporal
leakage evals (ma/pharma/covid, added as custom lm-eval tasks under
``tasks/temporal``) through the vendored ``lm-evaluation-harness`` at the repo
root, leaning on its built-in backends/tasks/filters/scoring.

The only custom code lives in ``backends.py`` (model backends lm-eval can't
provide out of the box) and ``tasks/temporal`` (the leakage tasks). The launcher
``python -m evals.lmeval`` wires a hardcoded per-model config to ``simple_evaluate``.
"""
