"""Coherence probe over several bundles with the model loaded once: greedy answers on a few rows per dim, per dose.

    python experiments/exp_rahul_002_nullspace_gated/probe_doses.py --model Qwen/Qwen3.8-27B \
        --bundles <RUN_DIR>/bundles_probe/alpha_0.5/*.spipe <RUN_DIR>/bundles_probe/alpha_1.0/*.spipe <RUN_DIR>/bundles/*.spipe \
        --rows data/v17_thin_categories/eval.jsonl --dims instructed_lie confrontation_after_lie concealment sandbagging_stated \
        --per-dim 2 --max-new-tokens 400 --unsteered --out <RUN_DIR>/probe_doses.jsonl

Teacher-forced sweeps score a one-line reference and cannot see a dose that makes the model loop, go blank or drift
once it writes freely; with the gated rule the sweep stays under the protected budget at every dose, so it always picks
the top of the grid and the probe is the only instrument for the dose. This prints, row by row, the unsteered answer
(if --unsteered) and each bundle's answer with a repetition score (distinct 4-grams; well below 0.5 means looping) and
an "empty" flag, then a per-bundle table. `probe_bundle.py` does the same for one bundle; this one shares the loaded
model across bundles, which is what makes a 3-dose comparison affordable on the 27B.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "third_party"))
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "steering"))
sys.path.insert(0, str(HERE))

import controls  # noqa: E402, F401
from cast_honesty import distinct_ratio, probe_problem  # noqa: E402
from generate_answers import load, model_kwargs, select_rows  # noqa: E402
from starterkit import REGIME, generate  # noqa: E402
from starterkit.rows import chat  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bundles", nargs="+", required=True, help="one .spipe per dose, probed in this order")
    ap.add_argument("--labels", nargs="*", help="labels for the bundles (default: parent folder names)")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--dims", nargs="+", required=True)
    ap.add_argument("--per-dim", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--unsteered", action="store_true", help="also print the unsteered answer on each row")
    ap.add_argument("--quantize", choices=["nf4"])
    ap.add_argument("--regime-model", help="whose regime to borrow for a stand-in model the kit does not list (rehearsals)")
    ap.add_argument("--show", type=int, default=600, help="characters of thinking / answer to print per arm")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    labels = args.labels or [Path(b).parent.name for b in args.bundles]
    if len(labels) != len(args.bundles):
        raise SystemExit("--labels must match --bundles")
    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    rows = [r for r in select_rows(rows, set(args.dims), args.per_dim) if r["condition"] in ("pressure", "plain")]
    if args.model not in REGIME:
        if not args.regime_model or args.regime_model not in REGIME:
            raise SystemExit(f"{args.model} has no official regime; pass --regime-model with one of {sorted(REGIME)}")
        REGIME[args.model] = REGIME[args.regime_model]
    kwargs = dict(REGIME[args.model]["chat_template_kwargs"])

    # load the model once (through the harness, so the kwargs match the evaluation's), then re-use it for every bundle
    t0 = time.time()
    plain = load(args.model, quantize=args.quantize)
    model, tokenizer = plain.model, plain.tokenizer
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    from steerability.spipe import SPipe

    arms = []
    if args.unsteered:
        arms.append(("unsteered", plain))
    for label, path in zip(labels, args.bundles):
        spipe = SPipe.load(path, allow_code=True)
        if spipe.manifest["model"]["ref"] != args.model:
            raise SystemExit(f"{path} targets {spipe.manifest['model']['ref']!r}, not {args.model!r}")
        pipe = spipe.pipeline(backend="huggingface", model=model, tokenizer=tokenizer, **model_kwargs(args.model, args.quantize))
        pipe.steer()
        arms.append((label, pipe))

    records, table = [], {label: {"rows": 0, "problems": 0, "cut": 0, "distinct": []} for label, _ in arms}
    for r in rows:
        print(f"\n{'=' * 100}\n{r['row_id']} ({r['dim']})   reference: {r['reference']['text'][:160]!r}")
        for label, pipe in arms:
            out = pipe.generate(messages=[chat(r)], return_output=True, do_sample=False, max_new_tokens=args.max_new_tokens,
                                chat_template_kwargs=kwargs)
            out = out[0] if isinstance(out, list) else out
            n_new = int(out.output_ids[0].shape[-1]) if hasattr(out.output_ids[0], "shape") else len(out.output_ids[0])
            thinking, answer = generate.split(args.model, out.output_ids[0], tokenizer)
            thinking = thinking or ""
            text = thinking + " " + answer
            score = distinct_ratio(text)
            if not answer.strip() and thinking and n_new >= args.max_new_tokens:
                problem = "cut_in_thinking"   # the cap ended the generation inside the thinking block: not a steering problem
            else:
                problem = probe_problem(answer, min_words=4) or ("looping" if score < 0.5 else None)   # a one-line true answer is 5-8 words
            flag = f"   <-- {problem.upper()}" if problem else ""
            print(f"\n--- [{label}] distinct-4gram {score:.2f}, thinking {len(thinking)} chars, answer {len(answer.split())} words{flag}")
            print(f"    thinking: {thinking[:args.show]!r}")
            print(f"    answer:   {answer[:args.show]!r}")
            records.append({"row_id": r["row_id"], "dim": r["dim"], "arm": label, "distinct_ratio": score, "problem": problem,
                            "thinking": thinking, "answer": answer})
            t = table[label]
            t["rows"] += 1; t["distinct"].append(score)
            t["cut"] += problem == "cut_in_thinking"; t["problems"] += bool(problem) and problem != "cut_in_thinking"
        sys.stdout.flush()

    print(f"\n{'=' * 100}\nper arm: rows, answers flagged (empty / looping), cut off inside thinking by the token cap, mean distinct-4gram")
    for label, t in table.items():
        mean = sum(t["distinct"]) / max(1, len(t["distinct"]))
        print(f"  {label:24s} {t['rows']:3d} rows, {t['problems']:2d} flagged, {t['cut']:2d} cut, distinct {mean:.2f}")
    if any(t["cut"] for t in table.values()):
        print("  (cut rows say nothing about the dose; raise --max-new-tokens to see their answers)")
    if args.out:
        Path(args.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
