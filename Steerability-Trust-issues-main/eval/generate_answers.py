"""Generation harness: run a model (unsteered, or steered by a .spipe bundle) on dataset rows the way the evaluation
does, and write the answers file that eval/scorer.py reads.

Generation itself is the Starter Kit's `starterkit.generate.run_bundle`: thinking on, sampling at the evaluation's
temperature, the answer split from the reasoning on token ids where the model's tags are special tokens (Gemma), and
`truncated` read from the backend's finish reason. This script adds what the kit leaves to us:

    * loading the unsteered model (a pipeline with no controls) or a steered bundle, with the evaluation's settings;
    * choosing rows by dim and count, keeping each instance's pressure and neutral twins together;
    * several samples per row (one seed each), since the evaluation averages over candidates;
    * the evaluation's token ceiling per row (8192 for honesty and guard rows, 16384 for capability rows, weekly);
    * saving after every chunk and skipping finished work, so an interrupted Colab session resumes.

Always generate the arms you compare on the same backend: steering that edits activations (CAST, CAA) fails the
evaluation's vLLM check, so the evaluation runs it on Hugging Face and pairs it with an unsteered Hugging Face run.

Examples:
    # the unsteered reference arm on a competition model
    python eval/generate_answers.py --rows data/v7_tidy/eval.jsonl --model Qwen/Qwen3.8-27B \\
        --arm unsteered --samples 2 --out runs/qwen_eval/answers.jsonl
    # a steered bundle, same rows and backend, appended to the same file
    python eval/generate_answers.py ... --arm cast_v1 --bundle bundles/cast_v1_qwen.spipe --out runs/qwen_eval/answers.jsonl
    # a cheap smoke test on a small model, borrowing Granite's regime and a short ceiling
    python eval/generate_answers.py ... --model ibm-granite/granite-4.0-micro --regime-model ibm-granite/granite-4.2-30b \\
        --per-dim 2 --max-new-tokens 512 --arm unsteered --out runs/smoke/answers.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party"))
from starterkit import DEV_SEED, REGIME, max_new_tokens  # noqa: E402

CHUNK_ROWS = 32  # rows generated between saves


# ---- pure helpers (tested without a model) --------------------------------------------------------------------------

def select_rows(rows, dims=None, per_dim=None):
    """Rows of the chosen dims (or of every dim), at most `per_dim` instances per dim, twins kept together.

    Instances are spread across source questions (`item_key`): the first instance of every question is taken before a
    second instance of any, so a small `per_dim` still covers many questions rather than the templates of one or two.
    The order is fixed, so the selection is deterministic and a smaller `per_dim` picks a subset of a larger one.
    Rows are returned in file order.
    """
    by_question = defaultdict(lambda: defaultdict(list))   # dim -> item_key -> instance ids in row-id order
    for row in sorted(rows, key=lambda r: r["row_id"]):
        if dims and row["dim"] not in dims:
            continue
        instances = by_question[row["dim"]][row["item_key"]]
        if row["instance_id"] not in instances:
            instances.append(row["instance_id"])
    keep = set()
    for dim, questions in by_question.items():
        # round robin over the questions in item_key order: each question's 1st instance, then each one's 2nd, ...
        ordered = [inst for rank in zip_longest(*(questions[key] for key in sorted(questions)))
                   for inst in rank if inst is not None]
        keep.update((dim, inst) for inst in ordered[:per_dim])
    return [row for row in rows if (row["dim"], row["instance_id"]) in keep]


def ceiling_for(row, profile="weekly"):
    """The evaluation's token ceiling for a row: capability rows use the capability suite's ceiling."""
    suite = "capability_dims" if row["dim"].startswith("capability_") else row["dim"]
    return max_new_tokens(profile, suite)


def prepare(row):
    """The row as run_bundle expects it: followups present (our generated rows may omit the field)."""
    return {**row, "followups": row.get("followups") or []}


def records_to_samples(records, arm, sample):
    """Collapse run_bundle records (one per turn) into scorer samples (one per row).

    A row with several turns gets a list of answers and a list of truncation flags, in turn order.
    """
    by_row = defaultdict(list)
    for record in records:
        by_row[record["row_id"]].append(record)
    samples = []
    for row_id, turns in by_row.items():
        turns.sort(key=lambda r: r["turn"])
        answers = [t["answer"] for t in turns]
        truncated = [bool(t["truncated"]) for t in turns]
        samples.append({
            "row_id": row_id, "arm": arm, "sample": sample,
            "answer": answers[0] if len(answers) == 1 else answers,
            "truncated": truncated[0] if len(truncated) == 1 else truncated,
        })
    return samples


def diagnostics(records, arm, sample):
    """One line per turn with what helps explain a score: thinking length, finish reason, an unclosed block."""
    return [{"row_id": r["row_id"], "arm": arm, "sample": sample, "turn": r["turn"], "dim": r["dim"],
             "finish_reason": r["finish_reason"], "truncated": bool(r["truncated"]),
             "thinking_chars": len(r["thinking"] or ""), "answer_chars": len(r["answer"] or ""),
             "unclosed_thinking": bool(r["thinking"]) and not r["answer"]} for r in records]


def done_keys(path, arm):
    """(row_id, sample) pairs already written for this arm, so a rerun skips them."""
    path = Path(path)
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            sample = json.loads(line)
            if sample["arm"] == arm:
                keys.add((sample["row_id"], sample["sample"]))
    return keys


def plan(rows, arm, samples, out_path, profile="weekly", ceiling_override=None):
    """The work still to do: [(sample index, token ceiling, [rows])], grouped so one ceiling runs per batch."""
    done = done_keys(out_path, arm)
    work = []
    for sample in range(samples):
        by_ceiling = defaultdict(list)
        for row in rows:
            if (row["row_id"], sample) not in done:
                by_ceiling[ceiling_override or ceiling_for(row, profile)].append(row)
        for ceiling, group in sorted(by_ceiling.items()):
            work.append((sample, ceiling, group))
    return work


def append_jsonl(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")


# ---- model loading and generation (needs the steerability toolkit and a GPU or MPS) ---------------------------------

def model_kwargs(model, quantize=None):
    """The evaluation's pipeline kwargs for `model` (bf16, pinned revision), optionally loaded in 4-bit.

    `quantize="nf4"` loads the bf16 weights in 4-bit with bitsandbytes (about 16 GB for a 27B model), for a 40 GB GPU.
    Outputs differ slightly from bf16, so compare only arms generated at the same precision.
    """
    from starterkit.backends import pipeline_kwargs

    kwargs = pipeline_kwargs(model)
    if quantize == "nf4":
        import torch
        from transformers import BitsAndBytesConfig

        kwargs["hf_model_kwargs"]["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    elif quantize is not None:
        raise ValueError(f"unknown quantize option {quantize!r}; use 'nf4' or omit it")
    return kwargs


def load(model, bundle=None, backend="huggingface", tensor_parallel_size=None, quantize=None):
    """A steered pipeline: the bundle's, or an unsteered pipeline with no controls, built with the evaluation's kwargs.

    Bundles are loaded the way the kit's `generate.load_pipeline` loads them (same model check, same backend check),
    but with our kwargs, so a bundle can also run in 4-bit.
    """
    from starterkit.backends import evaluation_backend, vllm_backend_spec

    if quantize and backend != "huggingface":
        raise ValueError("4-bit loading is only available on the huggingface backend")
    kwargs = model_kwargs(model, quantize)
    spec = vllm_backend_spec(model, tensor_parallel_size=tensor_parallel_size) if backend == "vllm" else backend
    if bundle:
        from steerability.spipe import SPipe

        spipe = SPipe.load(bundle, allow_code=True)
        target = spipe.manifest["model"]["ref"]
        if target != model:
            raise ValueError(f"{Path(bundle).name} targets {target!r}, not {model!r}")
        pipeline = spipe.pipeline(backend=spec, **kwargs)
        failures = [failure.message for failure in pipeline.check().failures_for("generate")]
        if failures:
            raise ValueError(f"{Path(bundle).name} does not run on {backend!r} ({'; '.join(failures)}); the "
                             f"evaluation runs it on {evaluation_backend(pipeline)!r}")
    else:
        from steerability.algorithms.core.steering_pipeline import SteeringPipeline

        pipeline = SteeringPipeline(model_name_or_path=model, controls=[], backend=spec, **kwargs)
    pipeline.steer()
    return pipeline


def run(args):
    from starterkit import generate

    regime_model = args.regime_model or args.model
    if regime_model not in REGIME:
        raise SystemExit(f"{args.model!r} has no official regime; pass --regime-model with one of {sorted(REGIME)}")
    rows = [prepare(r) for r in select_rows(
        [json.loads(line) for line in Path(args.rows).read_text().splitlines() if line.strip()],
        set(args.dims) if args.dims else None, args.per_dim)]
    work = plan(rows, args.arm, args.samples, args.out, args.profile, args.max_new_tokens)
    remaining = sum(len(group) for _, _, group in work)
    print(f"{len(rows)} rows selected; {remaining} row-samples to generate for arm {args.arm!r}", flush=True)
    if not work:
        return
    pipeline = load(args.model, args.bundle, args.backend, args.tensor_parallel_size, args.quantize)
    diagnostics_path = Path(args.out).with_name(Path(args.out).stem + "_diagnostics.jsonl")
    try:
        finished = 0
        for sample, ceiling, group in work:
            for start in range(0, len(group), CHUNK_ROWS):
                chunk = group[start:start + CHUNK_ROWS]
                # run_bundle reads the regime and the reasoning tags by model name, so pass the regime model
                records = generate.run_bundle(pipeline, regime_model, chunk, batch_size=args.batch_size,
                                              max_new_tokens=ceiling, seed=args.seed + sample)
                append_jsonl(args.out, records_to_samples(records, args.arm, sample))
                append_jsonl(diagnostics_path, diagnostics(records, args.arm, sample))
                finished += len(chunk)
                truncated = sum(r["truncated"] for r in records)
                print(f"  sample {sample}, ceiling {ceiling}: {finished}/{remaining} done "
                      f"({truncated} of {len(records)} turns in this chunk hit the ceiling)", flush=True)
    finally:
        generate.release(pipeline)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", required=True, help="a dataset split, e.g. data/v7_tidy/eval.jsonl")
    parser.add_argument("--model", required=True, help="the base model to load")
    parser.add_argument("--arm", required=True, help="a name for this arm: 'unsteered' for the reference")
    parser.add_argument("--out", required=True, help="the answers file (appended to; finished work is skipped)")
    parser.add_argument("--bundle", help="a .spipe bundle; omit for the unsteered model")
    parser.add_argument("--regime-model", help="whose official regime to use, for a model the kit does not list")
    parser.add_argument("--dims", nargs="*", help="only these dims (default: all)")
    parser.add_argument("--per-dim", type=int, help="at most this many instances per dim")
    parser.add_argument("--samples", type=int, default=1, help="samples per row, one seed each")
    parser.add_argument("--profile", default="weekly", choices=["weekly", "final"])
    parser.add_argument("--max-new-tokens", type=int, help="override every ceiling (smoke tests only)")
    parser.add_argument("--backend", default="huggingface", choices=["huggingface", "vllm"])
    parser.add_argument("--quantize", choices=["nf4"], help="load in 4-bit (bitsandbytes) for a 40 GB GPU")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEV_SEED)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
