"""Build data/benign_prompts_v1/: prompt-only rows that widen the "don't steer" set of a steering fit.

    python data/scripts/build_benign_prompts.py

Why: a null-space / gated steering method protects r_side and r_guard only on the span of prompts it was told to
leave alone. v7 gives 480 protected rows in train; the 27B layer profile (exp_rahul_001) showed the coverage of
held-out instruction-following and helpfulness prompts at layer 32 still rising with the number of benign rows
(0.967 at 250 rows -> 0.983 at 1,186). This set adds ~1.8k more ordinary prompts from the same pinned public sources
fix 4 used, rendered exactly as fix 4 renders them, **excluding every item that is a v7 row in any split**, so eval
and test stay held out.

What these rows are NOT: they carry no answer keys and no grading, are never scored, never used to fit an honesty
direction, and are not a dataset version (v7_tidy stays the data for everything else). MASK is not used.
"""
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("fix4", ROOT / "data" / "scripts" / "fix4_guard_capability.py")
fix4 = importlib.util.module_from_spec(spec)
_cwd = Path.cwd()
import os
os.chdir(ROOT)            # fix4 resolves data/external relative to the repo root
spec.loader.exec_module(fix4)

DST = ROOT / "data" / "benign_prompts_v1"
V7 = ROOT / "data" / "v7_tidy"
CAPS = {"ifeval": None, "alpaca_eval": None, "xstest": None, "mmlu_pro": 300, "bbh": 200, "gsm8k": 200}  # None = all remaining


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def order_key(item_id):
    return hashlib.sha256(f"benign_v1:{item_id}".encode()).hexdigest()


def take(items, cap, stratum, item_id):
    """All items, or `cap` of them spread evenly over strata in a fixed hash order."""
    if cap is None or cap >= len(items):
        return sorted(items, key=lambda it: order_key(item_id(it)))
    by = defaultdict(list)
    for it in items:
        by[stratum(it)].append(it)
    queues = [sorted(g, key=lambda it: order_key(item_id(it))) for _, g in sorted(by.items())]
    out, rank = [], 0
    while len(out) < cap:
        for q in queues:
            if rank < len(q) and len(out) < cap:
                out.append(q[rank])
        rank += 1
    return out


def main():
    import csv
    import pyarrow.parquet as pq

    used = set()
    for split in ("train", "eval", "test"):
        for row in read_jsonl(V7 / f"{split}.jsonl"):
            used.add(row["messages"][-1]["content"].strip())
    rows, excluded = [], Counter()

    def add(row_id, dim, source, prompt, domain):
        if prompt.strip() in used:
            excluded[dim] += 1
            return
        rows.append({"row_id": row_id, "dim": dim, "condition": "neutral", "messages": fix4.user(prompt),
                     "source": source, "domain": domain, "scored_as": [], "purpose": "benign_span"})

    (path,) = fix4.fetch("ifeval")
    items = [json.loads(l) for l in path.open() if l.strip()]
    for r in take(items, CAPS["ifeval"], lambda r: r["instruction_id_list"][0].split(":")[0], lambda r: r["key"]):
        add(f"benign-instruction-ifeval-{r['key']}", "benign_instruction_following", fix4.source_tag("ifeval"), r["prompt"],
            r["instruction_id_list"][0].split(":")[0])

    (path,) = fix4.fetch("alpaca_eval")
    items = [{"index": i, **r} for i, r in enumerate(json.loads(path.read_text()))]
    for r in take(items, CAPS["alpaca_eval"], lambda r: r["dataset"], lambda r: r["index"]):
        add(f"benign-helpfulness-alpacaeval-{r['index']}", "benign_helpfulness", fix4.source_tag("alpaca_eval"), r["instruction"], r["dataset"])

    (path,) = fix4.fetch("xstest")
    items = list(csv.DictReader(path.open()))
    for r in take(items, CAPS["xstest"], lambda r: r["type"], lambda r: r["id"]):
        dim = "benign_over_conservativeness" if r["label"] == "safe" else "benign_refusal"
        add(f"benign-xstest-{r['id']}", dim, fix4.source_tag("xstest"), r["prompt"], r["type"])

    (path,) = fix4.fetch("mmlu_pro")
    items = pq.read_table(path).to_pylist()
    for r in take(items, CAPS["mmlu_pro"], lambda r: r["category"], lambda r: r["question_id"]):
        letters = "ABCDEFGHIJ"[:len(r["options"])]
        options = "\n".join(f"({l}) {o}" for l, o in zip(letters, r["options"]))
        prompt = (f"{r['question']}\n\nOptions:\n{options}\n\n" + fix4.FINAL_LINE.format(placeholder="X")
                  + " X is the letter of the correct option.")
        add(f"benign-knowledge-mmlupro-{r['question_id']}", "benign_knowledge", fix4.source_tag("mmlu_pro"), prompt, r["category"])

    items = []
    for task, path in zip(fix4.BBH_TASKS, fix4.fetch("bbh")):
        items += [{"task": task, "index": i, **r} for i, r in enumerate(pq.read_table(path).to_pylist())]
    for r in take(items, CAPS["bbh"], lambda r: r["task"], lambda r: f"{r['task']}-{r['index']}"):
        add(f"benign-reasoning-bbh-{r['task']}-{r['index']}", "benign_reasoning", fix4.source_tag("bbh"),
            f"{r['input']}\n\n" + fix4.FINAL_LINE.format(placeholder="<your answer>"), r["task"])

    (path,) = fix4.fetch("gsm8k")
    items = [{"index": i, **r} for i, r in enumerate(pq.read_table(path).to_pylist())]
    for r in take(items, CAPS["gsm8k"], lambda r: "all", lambda r: r["index"]):
        add(f"benign-problemsolving-gsm8k-{r['index']}", "benign_problem_solving", fix4.source_tag("gsm8k"),
            f"{r['question']}\n\n" + fix4.FINAL_LINE.format(placeholder="<number>"), "math_word_problem")

    # final guard: nothing here may equal a v7 prompt, and ids are unique
    assert not any(r["messages"][-1]["content"].strip() in used for r in rows)
    assert len({r["row_id"] for r in rows}) == len(rows)
    DST.mkdir(exist_ok=True)
    out = DST / "benign_prompts.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    counts = Counter(r["dim"] for r in rows)
    (DST / "SHA256SUMS").write_text(f"{fix4.sha256(out)}  benign_prompts.jsonl\n")
    print(f"wrote {out}: {len(rows)} rows; per dim {dict(sorted(counts.items()))}; excluded as v7 rows {dict(excluded)}")
    os.chdir(_cwd)


if __name__ == "__main__":
    main()
