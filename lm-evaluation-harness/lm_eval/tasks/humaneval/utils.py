import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def _instruct_completion(prompt: str, r: str) -> str:
    """Code the model wrote to complete `prompt` (the function header). Strips a trailing
    markdown fence, then repairs indentation: reasoning models route the answer through
    `</think>`-splitting which `.lstrip()`s it, dropping the FIRST body line's indent and
    making `prompt + completion` an IndentationError. We only touch it when the assembled
    function fails to compile, so well-formed completions (e.g. the base model's) are
    untouched; then we re-add the 4-space body indent to the offending first line."""
    code = r if r.find("```") == -1 else r[: r.find("```")]
    try:
        compile(prompt + code, "<sol>", "exec")
        return code
    except SyntaxError:
        pass
    lines = code.split("\n")
    idx = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if idx is not None and not lines[idx][:1].isspace():
        fixed_lines = list(lines)
        fixed_lines[idx] = "    " + fixed_lines[idx]
        fixed = "\n".join(fixed_lines)
        try:
            compile(prompt + fixed, "<sol>", "exec")
            return fixed
        except SyntaxError:
            pass
    return code  # unrepairable — let it fail as before


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [doc["prompt"] + _instruct_completion(doc["prompt"], r) for r in resp]
        for resp, doc in zip(resps, docs)
    ]
