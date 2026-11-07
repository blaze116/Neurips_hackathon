"""The backend rule of the evaluation, and the evaluation's vLLM engine settings.

The evaluation may run a pipeline on two backends, `BACKEND_ORDER` in order of preference: vLLM,
then Hugging Face. A bundle is run on the first backend whose `pipeline.check()` reports no
generate-phase failure, and its scores are paired against the reference arm run on that same
backend, since the two backends give slightly different continuations for the same weights. A
bundle that no backend runs fails intake. `evaluation_backend` applies that rule to a built
pipeline, which loads no weights and boots no engine, and `check-submission` reports its result
per bundle.

`vllm_backend_spec` is the toolkit `BackendSpec` of the evaluation's vLLM engine for one base
model, for a local run on vLLM (`starterkit.generate.load_pipeline(..., backend="vllm")`), which
needs the toolkit's `vllm` extra (`uv sync --extra vllm`).
"""
from __future__ import annotations

from starterkit import (
    BACKEND_ORDER,
    DEV_SEED,
    BASE_MODEL_REVISIONS,
    PIPELINE_KWARGS,
    VLLM_ENGINE_KWARGS,
    VLLM_ENGINE_KWARGS_BY_MODEL,
)


def pipeline_kwargs(model: str) -> dict:
    """The `SteeringPipeline` constructor kwargs the evaluation builds a pipeline on `model` with.

    `hf_model_kwargs` is `PIPELINE_KWARGS["hf_model_kwargs"]` plus the model's pinned Hub commit
    as `revision` where `BASE_MODEL_REVISIONS` has one, which pins the weights and the tokenizer.

    Args:
        model: A model reference; a base model gets its pinned revision.

    Returns:
        `{"hf_model_kwargs", "device_map"}`, new mappings that do not alias the constants.
    """
    hf_model_kwargs = dict(PIPELINE_KWARGS["hf_model_kwargs"])
    if model in BASE_MODEL_REVISIONS:
        hf_model_kwargs["revision"] = BASE_MODEL_REVISIONS[model]
    return {"hf_model_kwargs": hf_model_kwargs, "device_map": PIPELINE_KWARGS["device_map"]}


def vllm_backend_spec(model: str, *, tensor_parallel_size: int | None = None, seed: int = DEV_SEED):
    """The toolkit `BackendSpec` of the evaluation's vLLM engine for `model`.

    The engine kwargs are `VLLM_ENGINE_KWARGS` merged with the model's entry of
    `VLLM_ENGINE_KWARGS_BY_MODEL`, the dtype of `PIPELINE_KWARGS`, the seed, and the model's
    pinned revision (`revision` and `tokenizer_revision`) where `BASE_MODEL_REVISIONS` has one. The
    vLLM-Hook plugin is off (`hook_plugin: False`), as in the evaluation, so declarative state
    controls (e.g., CAA) do not pass the vLLM check and run on Hugging Face instead.

    Args:
        model: A model reference.
        tensor_parallel_size: GPUs per engine; None keeps the evaluation's value. Set it to the
            number of GPUs you have.
        seed: The engine's seed. The evaluation's own seed is not published.

    Returns:
        The spec, with `kind="vllm"`.
    """
    from steerability.algorithms.core.execution.spec import BackendSpec

    engine_kwargs = {
        **VLLM_ENGINE_KWARGS,
        **VLLM_ENGINE_KWARGS_BY_MODEL.get(model, {}),
        "dtype": PIPELINE_KWARGS["hf_model_kwargs"]["dtype"],
        "seed": seed,
    }
    if model in BASE_MODEL_REVISIONS:
        engine_kwargs["revision"] = engine_kwargs["tokenizer_revision"] = BASE_MODEL_REVISIONS[model]
    if tensor_parallel_size is not None:
        engine_kwargs["tensor_parallel_size"] = tensor_parallel_size
    return BackendSpec(kind="vllm", model=model, options={"hook_plugin": False, "engine_kwargs": engine_kwargs})


def backend_failures(pipeline) -> dict[str, list[str]]:
    """The generate-phase failures of `pipeline.check()` on each backend of `BACKEND_ORDER`.

    The check reads the controls' declarations against the backend kind's capabilities. It
    loads no weights and boots no engine, and the vLLM check needs no vLLM install.

    Args:
        pipeline: A built `SteeringPipeline`, steered or not.

    Returns:
        A mapping from backend kind to its failure messages, in `BACKEND_ORDER`; an empty list
        means the backend runs the pipeline.
    """
    return {
        kind: [failure.message for failure in pipeline.check(kind).failures_for("generate")]
        for kind in BACKEND_ORDER
    }


def evaluation_backend(pipeline) -> str | None:
    """The backend the evaluation runs `pipeline` on: the first of `BACKEND_ORDER` whose check
    reports no generate-phase failure, or None when every backend reports one.

    Args:
        pipeline: A built `SteeringPipeline`, steered or not.

    Returns:
        A member of `BACKEND_ORDER`, or None.
    """
    return next((kind for kind, failures in backend_failures(pipeline).items() if not failures), None)
