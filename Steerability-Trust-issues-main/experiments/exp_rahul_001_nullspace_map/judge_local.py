"""Judge and score an answers file with the team's scorer, on a GPU without bf16 (Kaggle T4) or with a local judge copy.

    python experiments/exp_rahul_001_nullspace_map/judge_local.py --rows data/v7_tidy/eval.jsonl \
        --answers runs/exp_rahul_001/answers.jsonl --arm nullspace_v1 --cache runs/exp_rahul_001/judge_cache.jsonl \
        [--judge-model /kaggle/input/glm-4-9b-0414 | zai-org/GLM-4-9B-0414] [--dtype float16]

The kit's `HFJudge` loads the judge in bf16 with `device_map="auto"`. On a T4 bf16 is emulated and slow, so this
runner loads the judge itself (fp16 by default, sharded over the visible GPUs), hands the loaded model to an
`HFJudge`, and calls `eval/scorer.py`'s `main` with the same arguments it takes. Everything else, the judge prompts,
the retries, the cache, the comparison and the composite, is the team's code unchanged.

The scorer compares `--arm` with the `unsteered` arm in the same answers file, so the unsteered answers (exp001's,
or a bf16 unsteered run) must be in that file, and their verdicts are reused from `--cache` when present.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "third_party"))
sys.path.insert(0, str(REPO / "eval"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--arm", help="the steered arm to compare with 'unsteered'; omit for a per-arm summary")
    ap.add_argument("--cache", help="judge cache JSONL, read and extended")
    ap.add_argument("--judge-model", default="zai-org/GLM-4-9B-0414", help="HF id or local path of the judge")
    ap.add_argument("--judge-revision", default="645b848", help="revision when --judge-model is a Hub id (exp001 used 645b848)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from starterkit import judge as kit_judge
    import scorer

    is_path = Path(args.judge_model).exists()
    revision = args.judge_revision if (not is_path and args.judge_model == "zai-org/GLM-4-9B-0414") else None
    print(f"loading judge {args.judge_model} ({'local path' if is_path else revision}) in {args.dtype} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.judge_model, revision=revision, dtype=getattr(torch, args.dtype), device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    client = kit_judge.HFJudge(args.judge_model, batch_size=args.batch_size)
    client._model, client._tokenizer = model, tokenizer   # pre-loaded; HFJudge._loaded() returns these
    kit_judge.judge_client = lambda: client                # the scorer builds its client through this
    scorer.kit_judge = kit_judge

    scorer_args = ["--rows", args.rows, "--answers", args.answers, "--judge"]
    if args.arm:
        scorer_args += ["--arm", args.arm]
    if args.cache:
        scorer_args += ["--cache", args.cache]
    os.environ.setdefault("JUDGE_BACKEND", "huggingface")
    scorer.main(scorer_args)


if __name__ == "__main__":
    main()
