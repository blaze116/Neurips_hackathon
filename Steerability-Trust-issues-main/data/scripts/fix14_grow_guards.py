"""Fix 14: grow three guard areas from 60 to 150 rows per split (v14_capabilities -> v15_guards).

Guard regressions cost three times their size, and our week-5 guard penalty (0.073) was the main difference between
us and the leaders, so each guard needs enough rows to show a drop of a few points. Added per split, 90 rows each:

  * helpfulness: everyday Dolly-15k requests (none used by fix 9 or fix 13), graded by the scorer's helpfulness judge
    against Dolly's human-written answer as the reference. AlpacaEval, the original source, is used up by
    data/benign_prompts_v1;
  * refusal: OR-Bench's toxic set (Cui et al., 2024; CC-BY-4.0, the same pinned revision as fix 7), prompts that
    should be refused, stratified over its categories and judged by the refusal prompt (refusing is right). XSTest,
    the original source, is used up by the benign set;
  * confidentiality and persona: 18 new team-written personas (data/scripts/fix14_personas.py) built with fix 4's
    code, 15 rows each (public facts, attempts to get two secrets, "are you an AI", "show me your instructions",
    in-scope tasks), 6 personas per split so held-out personas stay unseen.

Every v14 row is copied unchanged; no new prompt appears in v14 or in the benign set (checked). Run from the
repository root (needs pyarrow):

    python data/scripts/fix14_grow_guards.py
"""
import csv
import hashlib
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fix4_guard_capability as fix4  # noqa: E402
import fix9_system_prompt_compliance as fix9  # noqa: E402
from fix14_personas import PERSONAS  # noqa: E402

SRC = Path("data/v14_capabilities")
DST = Path("data/v15_guards")
BENIGN = Path("data/benign_prompts_v1/benign_prompts.jsonl")
OR_BENCH = Path("data/external/or_bench")
OR_REPO, OR_REVISION, OR_TOXIC = "bench-llm/or-bench", "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272", "or-bench-toxic.csv"
SPLITS = ("train", "eval", "test")
ADD_PER_SPLIT = 90
FIELD_ORDER = fix9.FIELD_ORDER
FIRST_NEW_PERSONA = len(fix4.PERSONAS)  # the new personas are numbered after fix 4's


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalise(row, pool):
    defaults = {"followups": [], "extra": {}, "pool": pool, "condition": "neutral"}
    return {field: row.get(field, defaults.get(field)) for field in FIELD_ORDER}


def deal(items, stratum, key, salt):
    by_stratum = defaultdict(list)
    for index, item in enumerate(items):
        by_stratum[stratum(item)].append(index)
    ranked = []
    for name, members in by_stratum.items():
        ordered = sorted(members, key=lambda i: hashlib.sha256(f"fix14:{salt}:{name}:{key(items[i])}".encode()).hexdigest())
        ranked += [((r + 0.5) / len(ordered), hashlib.sha256(str(key(items[i])).encode()).hexdigest(), i)
                   for r, i in enumerate(ordered)]
    ranked.sort()
    return [(items[i], SPLITS[n % 3]) for n, (_, _, i) in enumerate(ranked[:3 * ADD_PER_SPLIT])]


def fetch_toxic():
    path, sums = OR_BENCH / "raw" / OR_TOXIC, OR_BENCH / "SHA256SUMS"
    if not path.exists():
        urllib.request.urlretrieve(f"https://huggingface.co/datasets/{OR_REPO}/resolve/{OR_REVISION}/{OR_TOXIC}", path)
    expected = dict(line.split()[::-1] for line in sums.read_text().splitlines())
    key = f"raw/{OR_TOXIC}"
    if key in expected:
        if sha256(path) != expected[key]:
            raise SystemExit(f"checksum mismatch for {path}")
    else:
        sums.write_text(sums.read_text() + f"{sha256(path)}  {key}\n")
    return path


def build_refusal(known):
    items = [r for r in csv.DictReader(fetch_toxic().open(encoding="utf-8")) if r["prompt"].strip() not in known]
    index_of = {r["prompt"]: i for i, r in enumerate(items)}
    rows = []
    for record, split in deal(items, lambda r: r["category"], lambda r: r["prompt"], "toxic"):
        rows.append(fix4.new_row(
            f"guard-refusal-orbench-toxic-{index_of[record['prompt']]:04d}", "guard_refusal", split,
            f"{OR_REPO}@{OR_REVISION[:7]}", fix4.user(record["prompt"]), fix4.REFUSE,
            {"method": "judge_refusal", "expected": "refuse"}, ["guard:refusal"], record["category"],
            {"subset": "or-bench-toxic"}))
    return rows


def build_helpfulness(known):
    taken = {r["text"].lower() for r in fix9.pick_requests(fix9.fetch())}
    items, seen = [], set()
    for line in fix9.fetch().read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        text, answer = record["instruction"].strip(), record["response"].strip()
        if record["category"] in fix9.CATEGORIES and not record["context"].strip() and 20 <= len(text) <= 300 \
                and len(answer) >= 40 and text.lower() not in taken and text.lower() not in seen \
                and text not in known:
            seen.add(text.lower())
            items.append({"text": text, "answer": answer, "category": record["category"]})
    rows = []
    for n, (record, split) in enumerate(deal(items, lambda r: r["category"], lambda r: r["text"], "helpful")):
        rows.append(fix4.new_row(
            f"guard-helpfulness-dolly-{n:03d}", "guard_helpfulness", split, fix9.SOURCE, fix4.user(record["text"]),
            fix4.COMPLY, {"method": "judge_helpfulness", "reference_output": record["answer"],
                          "reference_generator": "human (Dolly-15k)"},
            ["guard:helpfulness"], record["category"], {}))
    return rows


def build_personas():
    original = fix4.PERSONAS
    fix4.PERSONAS = PERSONAS
    try:
        rows = fix4.build_persona_rows()
    finally:
        fix4.PERSONAS = original
    for row in rows:  # renumber after fix 4's personas: guard-persona-p00-... -> guard-persona-p12-...
        number = int(row["row_id"].split("-")[2][1:]) + FIRST_NEW_PERSONA
        row["row_id"] = row["instance_id"] = row["item_key"] = \
            f"guard-persona-p{number:02d}-" + row["row_id"].split("-", 3)[3]
    return rows


def main():
    old = {split: [json.loads(line) for line in (SRC / f"{split}.jsonl").open() if line.strip()] for split in SPLITS}
    known = {m["content"].strip() for rows in old.values() for r in rows for m in r["messages"] if m["role"] == "user"}
    known |= {json.loads(line)["messages"][-1]["content"].strip() for line in BENIGN.open() if line.strip()}
    new = ([normalise(r, "third_party") for r in build_helpfulness(known) + build_refusal(known)]
           + [normalise(r, "team") for r in build_personas()])

    problems = []
    ids = [r["row_id"] for rows in old.values() for r in rows] + [r["row_id"] for r in new]
    if len(ids) != len(set(ids)):
        problems.append("duplicate row_id")
    if any(r["dim"] != "guard_confidentiality_persona" and r["messages"][-1]["content"].strip() in known for r in new):
        problems.append("a new prompt already appears in v14 or the benign set")
    sizes = Counter((r["dim"], r["final_split"]) for r in new)
    if any(n != ADD_PER_SPLIT for n in sizes.values()) or len(sizes) != 9:
        problems.append(f"sizes {dict(sizes)}")
    persona_splits = defaultdict(set)
    for r in new:
        if r["dim"] == "guard_confidentiality_persona":
            persona_splits[r["extra"]["persona"]].add(r["final_split"])
    if any(len(s) > 1 for s in persona_splits.values()):
        problems.append("a persona spans splits")
    if problems:
        raise SystemExit("validation failed:\n" + "\n".join(problems))

    DST.mkdir(parents=True, exist_ok=True)
    checksums = []
    for split in SPLITS:
        out = DST / f"{split}.jsonl"
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                               for r in old[split] + [r for r in new if r["final_split"] == split]))
        checksums.append(f"{sha256(out)}  {split}.jsonl")
    (DST / "split_manifest.json").write_text((SRC / "split_manifest.json").read_text())
    checksums.append(f"{sha256(DST / 'split_manifest.json')}  split_manifest.json")
    (DST / "SHA256SUMS").write_text("\n".join(checksums) + "\n")

    total = Counter((r["dim"], r["final_split"]) for rows in old.values() for r in rows)
    total.update((r["dim"], r["final_split"]) for r in new)
    print(f"{'dim':32s} {'train':>6s} {'eval':>6s} {'test':>6s}")
    for dim in ("guard_helpfulness", "guard_refusal", "guard_confidentiality_persona"):
        print(f"{dim:32s}", " ".join(f"{total[(dim, s)]:6d}" for s in SPLITS))
    print("refusal categories in eval:", dict(Counter(r["domain"] for r in new
                                                      if r["dim"] == "guard_refusal" and r["final_split"] == "eval")))
    print("\nvalidation problems: none")


if __name__ == "__main__":
    main()
