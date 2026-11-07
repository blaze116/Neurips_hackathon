"""Coherence probe: greedy answers with a bundle on a few rows, printed in full, before paying for a generation run.

    python experiments/exp_rahul_001_nullspace_map/probe_bundle.py --model Qwen/Qwen3.8-27B --bundle <path>.spipe \
        --rows data/v7_tidy/eval.jsonl --dims instructed_lie concealment --per-dim 2 --max-new-tokens 400 --out probe.jsonl

Teacher-forced sweeps score a one-line reference; they cannot see a dose that makes the model loop or drift once it
writes freely (exp002's dose probe found that cliff for CAST). This prints the thinking and the answer of each probe
row, with a repetition score (distinct 4-grams; well below 0.5 means looping), so the dose can be lowered before stage B.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "third_party"))
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "steering"))
sys.path.insert(0, str(HERE))

import controls  # noqa: E402, F401
from cast_honesty import distinct_ratio  # noqa: E402
from generate_answers import load, select_rows  # noqa: E402
from starterkit import REGIME, generate  # noqa: E402
from starterkit.rows import chat  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--dims", nargs="+", required=True)
    ap.add_argument("--per-dim", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--quantize", choices=["nf4"])
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    rows = [r for r in select_rows(rows, set(args.dims), args.per_dim) if r["condition"] in ("pressure", "plain")]
    pipeline = load(args.model, bundle=args.bundle, quantize=args.quantize)
    kwargs = dict(REGIME[args.model]["chat_template_kwargs"])
    records = []
    for r in rows:
        out = pipeline.generate(messages=[chat(r)], return_output=True, do_sample=False, max_new_tokens=args.max_new_tokens,
                                chat_template_kwargs=kwargs)
        out = out[0] if isinstance(out, list) else out
        thinking, answer = generate.split(args.model, out.output_ids[0], pipeline.tokenizer)
        text = (thinking or "") + " " + answer
        score = distinct_ratio(text)
        flag = "" if score >= 0.5 else "   <-- LOOPING?"
        print(f"\n===== {r['row_id']} ({r['dim']}) distinct-4gram {score:.2f}{flag}")
        print(f"reference: {r['reference']['text'][:160]!r}")
        print(f"thinking ({len(thinking or '')} chars): {(thinking or '')[:600]!r}")
        print(f"answer: {answer[:700]!r}")
        records.append({"row_id": r["row_id"], "dim": r["dim"], "distinct_ratio": score, "thinking": thinking, "answer": answer})
    if args.out:
        Path(args.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records))
    low = [x for x in records if x["distinct_ratio"] < 0.5]
    print(f"\n{len(records)} probe rows; {len(low)} flagged as possibly looping")


if __name__ == "__main__":
    main()
