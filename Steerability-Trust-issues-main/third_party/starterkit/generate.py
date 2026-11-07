"""Running one baseline bundle over rows the way the evaluation runs it.

Generation goes through `pipeline.generate(messages=..., return_output=True)`. The bundle's input
controls, the chat templating with its `chat_template_kwargs`, the state hooks, and the decoding
driver apply as they do when the submission is scored. The regime (thinking on, sampling on) comes
from `starterkit.REGIME` and the token ceiling from `starterkit.MAX_NEW_TOKENS`.

Each continuation is split into its reasoning and answer segments the way the evaluation's provider
splits it: the split mode is resolved once per model against the pipeline's tokenizer
(`split_mode`), and the continuation is split on its token ids when either reasoning tag is a
special token that decoding with `skip_special_tokens=True` would strip (`google/gemma-4-31B-it`'s
`<|channel>` and `<channel|>`), and on the decoded text otherwise. A split on the decoded text
alone would leave a Gemma continuation unsplit, with the reasoning graded as part of the answer and
carried into the next turn. `truncated` is read from the backend's finish reason (`"length"`), the
signal the evaluation reads, rather than inferred from the split.

The evaluation runs a pipeline on vLLM when its check passes there and on Hugging Face otherwise
(`starterkit.backends`). `load_pipeline` generates on Hugging Face unless `backend="vllm"` is
passed, which builds the evaluation's vLLM engine and needs the toolkit's `vllm` extra. The two
backends give slightly different continuations for the same weights, which is why the evaluation
pairs each arm against the reference arm run on the same backend.
"""
from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path

from starterkit import BACKEND_ORDER, LOCAL_BACKEND, REGIME
from starterkit.backends import evaluation_backend, pipeline_kwargs, vllm_backend_spec
from starterkit.rows import chat, with_followup

#: the finish reason of a continuation cut off at the token ceiling, as the backends report it
FINISH_LENGTH = "length"

# the split mode of each tokenizer object, by tag pair, since `resolve_split_mode` encodes and decodes
# the tags; weak keys drop the entry of a freed tokenizer, whose `id` a later tokenizer can reuse
_SPLIT_MODES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def regime_kwargs(model: str, *, max_new_tokens: int, seed: int) -> dict:
    """The generation keyword arguments the evaluation uses for one model.

    Args:
        model: A base-model reference; a key of `REGIME`.
        max_new_tokens: The token ceiling for this pass. The evaluation uses
            `max_new_tokens(profile, suite)`, and a continuation cut off at the ceiling scores 0.
        seed: Sampling seed. The evaluation's own seed is not published, and `DEV_SEED` is for local
            work.

    Returns:
        Keyword arguments for `pipeline.generate`. `seed_scope` is `"dispatch"`, under which every
        item of a batch takes the dispatch seed and the batch decodes in one pass; the evaluation's
        provider decodes the same way on Hugging Face (vLLM seeds each item on its own).

    Raises:
        KeyError: If `model` is not one of the base models.
    """
    if model not in REGIME:
        raise KeyError(f"No regime for {model!r}; the base models are {sorted(REGIME)}.")
    regime = REGIME[model]
    return {
        "do_sample": True,
        **regime["sampling"],
        "chat_template_kwargs": dict(regime["chat_template_kwargs"]),
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "seed_scope": "dispatch",
    }


def split_mode(model: str, tokenizer) -> str:
    """How the continuations of `model` are split under `tokenizer`: `"tokens"` when either reasoning
    tag is a special token that `skip_special_tokens=True` strips, `"text"` otherwise.

    The mode is what the evaluation's provider resolves for the same tokenizer and tags, and it is
    cached per tokenizer object.

    Args:
        model: A base-model reference; a key of `REGIME`.
        tokenizer: The pipeline's tokenizer (`pipeline.tokenizer`).

    Returns:
        `"tokens"` or `"text"`.

    Raises:
        KeyError: If `model` is not one of the base models.
    """
    from steerability.utils.thinking import resolve_split_mode

    if model not in REGIME:
        raise KeyError(f"No regime for {model!r}; the base models are {sorted(REGIME)}.")
    tags = tuple(REGIME[model]["reasoning"]["tags"])
    modes = _SPLIT_MODES.setdefault(tokenizer, {})
    if tags not in modes:
        modes[tags] = resolve_split_mode(tokenizer, tags)
    return modes[tags]


def split_text(model: str, text: str) -> tuple[str | None, str]:
    """One decoded continuation split into its thinking and answer segments by substring, per the
    model's tags.

    This is the text mode of `split`. It sees no tag that decoding stripped, so a continuation of a
    model whose tags are special tokens (Gemma) must be split with `split` on its ids instead. An
    unclosed reasoning block yields an empty answer, which the evaluation scores 0.

    Args:
        model: A base-model reference; a key of `REGIME`.
        text: The continuation, decoded.

    Returns:
        `(thinking, answer)`; `thinking` is None only when the text carries no reasoning tag and the
        prompt did not open the block.

    Raises:
        KeyError: If `model` is not one of the base models.
    """
    from steerability.utils.thinking import split_thinking

    if model not in REGIME:
        raise KeyError(f"No regime for {model!r}; the base models are {sorted(REGIME)}.")
    reasoning = REGIME[model]["reasoning"]
    thinking, answer = split_thinking(
        text, tuple(reasoning["tags"]), opened_at_start=reasoning["opened_at_start"]
    )
    return thinking, answer


def split(model: str, output_ids, tokenizer) -> tuple[str | None, str]:
    """One continuation split into its thinking and answer segments, per the model's tags, in the
    mode `split_mode` resolves for `tokenizer`.

    In `"tokens"` mode the split runs on the ids and each segment is decoded afterwards with
    `skip_special_tokens=True`, so a tag that is a special token is found before decoding strips
    it; in `"text"` mode the ids are decoded first and split by substring (`split_text`). This is
    what the evaluation's provider does per row. An unclosed reasoning block yields an empty
    answer, which the evaluation scores 0.

    Args:
        model: A base-model reference; a key of `REGIME`.
        output_ids: The continuation's token ids, the prompt excluded (one row of
            `Output.output_ids`, as a tensor or a list); trailing padding is harmless.
        tokenizer: The pipeline's tokenizer.

    Returns:
        `(thinking, answer)`; `thinking` is None only when the continuation carries no reasoning
        tag and the prompt did not open the block.

    Raises:
        KeyError: If `model` is not one of the base models.
    """
    from steerability.utils.thinking import split_thinking_ids

    ids = output_ids.tolist() if hasattr(output_ids, "tolist") else list(output_ids)
    if split_mode(model, tokenizer) == "tokens":
        reasoning = REGIME[model]["reasoning"]
        thinking, answer = split_thinking_ids(
            ids, tokenizer, tuple(reasoning["tags"]), opened_at_start=reasoning["opened_at_start"]
        )
        return thinking, answer
    return split_text(model, tokenizer.decode(ids, skip_special_tokens=True))


def load_pipeline(bundle_path: str | Path, model: str, *, backend: str | None = None,
                  tensor_parallel_size: int | None = None):
    """Load one bundle, assert it targets `model`, steer it on one backend, and return the pipeline.

    The pipeline is built with the evaluation's constructor kwargs (`starterkit.backends.pipeline_kwargs`,
    which pin a base model's revision). On `"huggingface"` the model is loaded in process. On
    `"vllm"` the evaluation's engine is booted (`vllm_backend_spec`), which needs the toolkit's
    `vllm` extra; the bundle must pass the vLLM check, as it must in the evaluation to run there.

    Args:
        bundle_path: Path to a `.spipe` bundle.
        model: The base model the bundle must target.
        backend: `"huggingface"` or `"vllm"`; None is `LOCAL_BACKEND` (`"huggingface"`).
        tensor_parallel_size: GPUs per vLLM engine; None keeps the evaluation's value. Ignored on
            Hugging Face.

    Returns:
        The steered `SteeringPipeline`.

    Raises:
        AssertionError: If the bundle targets a different model.
        ValueError: If `backend` is not one of `BACKEND_ORDER`, or if the bundle does not pass the
            check on the requested backend.
    """
    from steerability.spipe import SPipe

    backend = LOCAL_BACKEND if backend is None else backend
    if backend not in BACKEND_ORDER:
        raise ValueError(f"backend must be one of {list(BACKEND_ORDER)}; got {backend!r}.")

    spipe = SPipe.load(bundle_path, allow_code=True)
    manifest_ref = spipe.manifest["model"]["ref"]
    assert manifest_ref == model, f"{Path(bundle_path).name} targets {manifest_ref!r}, not {model!r}."

    kwargs = pipeline_kwargs(model)
    spec = vllm_backend_spec(model, tensor_parallel_size=tensor_parallel_size) if backend == "vllm" else backend
    pipeline = spipe.pipeline(backend=spec, **kwargs)
    assert pipeline.model_name_or_path == model, (
        f"{Path(bundle_path).name} built a pipeline on {pipeline.model_name_or_path!r}, not {model!r}."
    )
    failures = [failure.message for failure in pipeline.check().failures_for("generate")]
    if failures:
        runs_on = evaluation_backend(pipeline)
        raise ValueError(
            f"{Path(bundle_path).name} does not run on {backend!r} ({'; '.join(failures)}); the evaluation "
            f"runs it on {runs_on!r}."
        )
    pipeline.steer()
    return pipeline


def _last_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return ""


def run_bundle(
    pipeline,
    model: str,
    rows: list[dict],
    *,
    batch_size: int = 8,
    max_new_tokens: int,
    seed: int,
) -> list[dict]:
    """Generate one record per row and per followup turn.

    Rows without followups yield one record with `turn` 0. A row with followups is continued turn by
    turn: the answer segment of the assistant's continuation (the reasoning segment removed) is
    appended as the assistant turn, then the next user turn, and the chat is generated again. An
    unclosed reasoning block yields an empty answer, which is appended as an empty assistant turn.
    The evaluation continues a chat the same way. A `pushback` row yields three records, with `turn`
    0, 1, and 2.

    Each continuation comes back as an `Output` (`return_output=True`) and is split on its ids or its
    text per `split_mode`. `truncated` is whether the backend reports the finish reason `"length"`,
    i.e., the continuation hit the ceiling, which the evaluation scores 0 whatever the split found;
    an empty `answer` with a non-empty `thinking` is a reasoning block that never closed.

    Args:
        pipeline: A steered pipeline from `load_pipeline`.
        model: The base model the pipeline runs on.
        rows: The rows to run.
        batch_size: How many chats go through one `generate` call.
        max_new_tokens: The token ceiling.
        seed: Sampling seed.

    Returns:
        Records `{row_id, dim, condition, turn, prompt, thinking, answer, finish_reason, truncated}`,
        in row order then turn order.
    """
    gen_kwargs = regime_kwargs(model, max_new_tokens=max_new_tokens, seed=seed)
    records: list[dict] = []
    # every row's chat advances one turn at a time, and each turn is generated in batches
    live = [{"row": row, "chat": chat(row)} for row in rows]
    turn = 0
    while live:
        for start in range(0, len(live), batch_size):
            batch = live[start:start + batch_size]
            outputs = pipeline.generate(messages=[item["chat"] for item in batch], return_output=True,
                                        **gen_kwargs)
            if not isinstance(outputs, list):
                outputs = [outputs]
            for item, output in zip(batch, outputs):
                finish_reason = output.finish_reasons[0]
                thinking, answer = split(model, output.output_ids[0], pipeline.tokenizer)
                item["answer"] = answer
                records.append({
                    "row_id": item["row"]["row_id"],
                    "dim": item["row"]["dim"],
                    "condition": item["row"]["condition"],
                    "turn": turn,
                    "prompt": _last_user_text(item["chat"]),
                    "thinking": thinking,
                    "answer": answer,
                    "finish_reason": finish_reason,
                    "truncated": finish_reason == FINISH_LENGTH,
                })

        advanced = []
        for item in live:
            followups = item["row"]["followups"]
            if turn < len(followups):
                item["chat"] = with_followup(item["chat"], item["answer"], followups[turn])
                advanced.append(item)
        live = advanced
        turn += 1
    return sorted(records, key=lambda record: (record["row_id"], record["turn"]))


def release(pipeline) -> None:
    """Release a pipeline's backends and free the GPU.

    Each baseline loads the model and releases it before the next baseline loads. The SFT bundles
    attach an adapter to the model in place, and a model reused across baselines would keep that
    adapter. On vLLM, releasing shuts the engine down.

    Args:
        pipeline: The pipeline to release.
    """
    try:
        pipeline.release_backends()
    finally:
        pipeline.model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def save_records(records: list[dict], path: str | Path) -> Path:
    """Write records as JSONL, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_records(path: str | Path) -> list[dict]:
    """Read records from a JSONL file written by `save_records`."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
