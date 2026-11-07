"""Running one baseline bundle over rows the way the evaluation runs it.

Generation goes through `pipeline.generate(messages=...)`. The bundle's input controls, the chat
templating with its `chat_template_kwargs`, the state hooks, and the decoding driver apply as they do
when the submission is scored. The regime (thinking on, sampling on, and the token ceiling) comes
from `starterkit.REGIME`.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

from starterkit import BACKEND, PIPELINE_KWARGS, REGIME
from starterkit.rows import chat, with_followup


def regime_kwargs(model: str, *, max_new_tokens: int, seed: int) -> dict:
    """The generation keyword arguments the evaluation uses for one model.

    Args:
        model: A base-model reference; a key of `REGIME`.
        max_new_tokens: The token ceiling for this pass. The evaluation uses
            `MAX_NEW_TOKENS[profile]`, and a continuation cut off at the ceiling scores 0.
        seed: Sampling seed. The evaluation's own seed is not published, and `DEV_SEED` is for local
            work.

    Returns:
        Keyword arguments for `pipeline.generate`. `seed_scope` is `"dispatch"`, under which every
        item of a batch takes the dispatch seed and the batch decodes in one pass; the evaluation's
        provider decodes the same way.

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


def split(model: str, text: str) -> tuple[str | None, str]:
    """One continuation split into its thinking and answer segments, per the model's tags.

    An unclosed reasoning block yields an empty answer. The evaluation scores that empty answer 0.

    Args:
        model: A base-model reference; a key of `REGIME`.
        text: The continuation as generated.

    Returns:
        `(thinking, answer)`; `thinking` is None only when the model emitted no reasoning block and
        the prompt did not open one.

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


def load_pipeline(bundle_path: str | Path, model: str):
    """Load one bundle, assert it targets `model`, steer it, and return the pipeline.

    Args:
        bundle_path: Path to a `.spipe` bundle.
        model: The base model the bundle must target.

    Returns:
        The steered `SteeringPipeline`.

    Raises:
        AssertionError: If the bundle targets a different model.
    """
    from steerability.spipe import SPipe

    spipe = SPipe.load(bundle_path, allow_code=True)
    manifest_ref = spipe.manifest["model"]["ref"]
    assert manifest_ref == model, f"{Path(bundle_path).name} targets {manifest_ref!r}, not {model!r}."

    pipeline = spipe.pipeline(backend=BACKEND, **PIPELINE_KWARGS)
    assert pipeline.model_name_or_path == model, (
        f"{Path(bundle_path).name} built a pipeline on {pipeline.model_name_or_path!r}, not {model!r}."
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

    Args:
        pipeline: A steered pipeline from `load_pipeline`.
        model: The base model the pipeline runs on.
        rows: The rows to run.
        batch_size: How many chats go through one `generate` call.
        max_new_tokens: The token ceiling.
        seed: Sampling seed.

    Returns:
        Records `{row_id, format, condition, turn, prompt, thinking, answer, truncated}`, in row order
        then turn order.
    """
    gen_kwargs = regime_kwargs(model, max_new_tokens=max_new_tokens, seed=seed)
    records: list[dict] = []
    # every row's chat advances one turn at a time, and each turn is generated in batches
    live = [{"row": row, "chat": chat(row)} for row in rows]
    turn = 0
    while live:
        for start in range(0, len(live), batch_size):
            batch = live[start:start + batch_size]
            texts = pipeline.generate(messages=[item["chat"] for item in batch], **gen_kwargs)
            if isinstance(texts, str):
                texts = [texts]
            for item, text in zip(batch, texts):
                thinking, answer = split(model, text)
                item["answer"] = answer
                records.append({
                    "row_id": item["row"]["row_id"],
                    "format": item["row"]["format"],
                    "condition": item["row"]["condition"],
                    "turn": turn,
                    "prompt": _last_user_text(item["chat"]),
                    "thinking": thinking,
                    "answer": answer,
                    "truncated": answer == "" and thinking is not None,
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
    adapter.

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
