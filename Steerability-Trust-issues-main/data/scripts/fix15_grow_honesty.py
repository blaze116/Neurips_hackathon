"""Fix 15: grow the two thinnest honesty dims in eval and test (v15_guards -> v16_honesty_growth).

  * confrontation_after_lie (the "owning up" family) had 48 eval and 42 test rows, all using the team's five
    confrontation templates. The Starter Kit has four of its own (neutral doubt; "I looked it up and found this";
    "a colleague says that's not right"; "I'm an inspector reviewing this"). For every eval and test scenario this
    script keeps the same earlier turns (the system prompt that told the model to lie, the question, the lie) and adds
    one row per kit template as the final user turn. The kit templates go to eval and test only, so they are unseen
    by anything fitted on train: a check that a method's owning up generalises beyond our own phrasings.
  * harm_motivated_denial had 36 eval and 58 test rows. The kit's harm-pressure file has 137 question pairs and we
    used 117; the 20 unused pairs are added, 10 to eval and 10 to test, each as a pressure row (the harmful motive)
    and a neutral row (the expert framing), shaped exactly like the existing rows. The kit allows these pairs to be
    used for a submission but not redistributed (private repository).

Every v15 row is copied unchanged; the split manifest gains the new harm item keys. Graded by the official honesty
judge like the existing rows of both dims. Run from the repository root:

    python data/scripts/fix15_grow_honesty.py
"""
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("data/v15_guards")
DST = Path("data/v16_honesty_growth")
KIT = Path("third_party/starterkit_data")  # the kit files we use, copied into the repository
SPLITS = ("train", "eval", "test")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line[line.find("{"):]) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def grow_confrontation(rows):
    templates = read_jsonl(KIT / "templates" / "confrontation_after_lie.jsonl")
    base = {}  # one existing row per (scenario, split): the earlier turns are the same whatever the template
    for row in rows:
        if row["dim"] == "confrontation_after_lie" and row["final_split"] in ("eval", "test"):
            base.setdefault(row["item_key"], row)
    new = []
    for item_key, row in sorted(base.items()):
        for template in templates:
            r = copy.deepcopy(row)
            text = template["user"].format(reference_text=row["reference"]["text"])
            r["messages"][-1] = {"role": "user", "content": text}
            tid = f"k{template['template_id'][1:]}"  # kit templates as k001-k004, distinct from our t001-t005
            r["row_id"] = f"confrontation_after_lie-{item_key}-{tid}-pressure"
            r["instance_id"] = f"confrontation_after_lie-{item_key}-{tid}"
            r["template_id"], r["style"] = tid, f"kit_{template['style']}"
            r["extra"] = {**row["extra"], "confrontation": template["extra"]["confrontation"], "kit_template": True}
            r["references"] = r["contrastive_pair"] = None  # the team's reference answers were written per template
            new.append(r)
    return new


def grow_harm(rows):
    used = {row["item_key"] for row in rows if row["dim"] == "harm_motivated_denial"}
    pairs = defaultdict(dict)
    for record in read_jsonl(KIT / "third_party" / "harm_pressure_pairs.jsonl"):
        pairs[record["pair_id"]][record["variant"]] = record
    unused = sorted(pid for pid in pairs if "h-" + pid.replace("-", "_") not in used)
    order = sorted(unused, key=lambda pid: hashlib.sha256(f"fix15:{pid}".encode()).hexdigest())
    shape = {condition: next(r for r in rows if r["dim"] == "harm_motivated_denial" and r["condition"] == condition)
             for condition in ("pressure", "neutral")}
    new = []
    for position, pid in enumerate(order):
        split = ("eval", "test")[position % 2]
        item_key = "h-" + pid.replace("-", "_")
        neutral, pressure = pairs[pid]["neutral"], pairs[pid]["pressure"]
        answer = neutral["answer"]
        letters = [line[1] for line in neutral["user"].splitlines() if line.startswith("(") and line[2:3] == ")"]
        for condition, messages in (
                ("pressure", [{"role": "user", "content": pressure["user"]}]),
                ("neutral", [{"role": "system", "content": neutral["system"]},
                             {"role": "user", "content": neutral["user"]}])):
            r = copy.deepcopy(shape[condition])
            instance = f"harm_motivated_denial-{pid.replace('-', '_')}-none"
            r.update({"row_id": f"{instance}-{condition}", "instance_id": instance, "item_key": item_key,
                      "messages": messages, "final_split": split,
                      "reference": {"type": "T", "text": f"The correct answer is ({answer})."}})
            r["extra"] = {**shape[condition]["extra"], "question": neutral["user"], "answer": answer,
                          "answer_aliases": [f"({answer})"],
                          "false_alternatives": [letter for letter in letters if letter != answer]}
            r["references"] = r["contrastive_pair"] = None
            new.append(r)
    return new


def main():
    old = {split: [json.loads(line) for line in (SRC / f"{split}.jsonl").open() if line.strip()] for split in SPLITS}
    every = [r for rows in old.values() for r in rows]
    new = grow_confrontation(every) + grow_harm(every)

    problems = []
    ids = [r["row_id"] for r in every] + [r["row_id"] for r in new]
    if len(ids) != len(set(ids)):
        problems.append("duplicate row_id")
    if any(r["final_split"] == "train" for r in new):
        problems.append("a new row in train")
    if any(list(r) != list(every[0]) for r in new):
        problems.append("field order differs from v15")
    for instance in {r["item_key"] for r in new}:
        splits = {r["final_split"] for r in every + new if r["item_key"] == instance and r["dim"] in
                  ("confrontation_after_lie", "harm_motivated_denial")}
        if len(splits) > 1:
            problems.append(f"{instance} spans splits")
    if problems:
        raise SystemExit("validation failed:\n" + "\n".join(problems[:20]))

    DST.mkdir(parents=True, exist_ok=True)
    checksums = []
    for split in SPLITS:
        out = DST / f"{split}.jsonl"
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                               for r in old[split] + [r for r in new if r["final_split"] == split]))
        checksums.append(f"{sha256(out)}  {split}.jsonl")
    manifest = json.loads((SRC / "split_manifest.json").read_text())
    for split in ("eval", "test"):
        keys = set(manifest[f"{split}_item_keys"].get("harm_motivated_denial", []))
        keys |= {r["item_key"] for r in new if r["dim"] == "harm_motivated_denial" and r["final_split"] == split}
        manifest[f"{split}_item_keys"]["harm_motivated_denial"] = sorted(keys)
    (DST / "split_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    checksums.append(f"{sha256(DST / 'split_manifest.json')}  split_manifest.json")
    (DST / "SHA256SUMS").write_text("\n".join(checksums) + "\n")

    total = Counter((r["dim"], r["final_split"]) for r in every + new)
    print(f"{'dim':26s} {'train':>6s} {'eval':>6s} {'test':>6s}")
    for dim in ("confrontation_after_lie", "harm_motivated_denial"):
        print(f"{dim:26s}", " ".join(f"{total[(dim, s)]:6d}" for s in SPLITS))
    print("added:", dict(Counter((r["dim"], r["final_split"]) for r in new)))
    print("\nvalidation problems: none")


if __name__ == "__main__":
    main()
