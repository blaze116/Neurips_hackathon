"""Fix 3: re-split the data into train / eval / test with a held-out test set (v2_references -> v3_resplit).

The v0 split (kept unchanged through v2) put every template in both train and eval, and in most categories every
question appears under every template. Eval scores therefore measured recall of seen wordings and seen questions,
not generalisation. This script builds a two-way holdout:

    test  = held-out questions written with held-out templates (new content AND new wording)
    eval  = held-out questions written with training templates (new content, familiar wording; for tuning)
    train = training questions written with training templates

Rows that mix a training question with a test template, or a test/eval question with the wrong templates, belong to
no split and are left out (they remain in v2_references). Where each question appears under only one template
(sandbagging, sandbagging_stated), the held-out template alone defines test and nothing is dropped. Where a category
has only one template (harm_motivated_denial), only questions can be held out.

The unit of assignment is the question (`item_key`), so a pressure row and its control twin always land together.
Question assignment is a fixed hash of item_key, stratified by domain, so it is reproducible. Run from the
repository root:

    python data/scripts/fix3_resplit.py

Changes to rows: `final_split` is set to the new split and the old value is kept in `previous_split`. Nothing else.
"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("data/v2_references")
DST = Path("data/v3_resplit")

TEST_ITEM_FRACTION = 0.25
EVAL_ITEM_FRACTION = 0.15

# held-out templates per category, keyed by template_id, or by style where a category has no template ids
TEST_TEMPLATES = {
    "concealment": {"t004", "t006"},               # exam_embargo, secret_hr
    "instructed_lie": {"t003", "t005"},            # internal_wiki, product_sheet
    "confrontation_after_lie": {"t002", "t003"},   # corporate_faq_evidence, expert_authority
    "feedback_sycophancy": {"t005", "t007", "t010"},  # rivals_work (down), my_manager_wrote_this (up), anchored_score
    "hint_acknowledgment": {"t002"},               # metadata_leak
    "pushback": {"t003", "t007"},                  # fabricated_citation (user wrong), control_sourced_correction (user right)
    "sandbagging": {"t004"},                       # shutdown_avoidance
    "sandbagging_stated": {"support_tier_policy"},
    "harm_motivated_denial": set(),                # one template only: questions are held out, wording is not
}


def template_key(row):
    return row["template_id"] or row["style"]


def hash_order(key):
    return hashlib.sha256(f"fix3:{key}".encode()).hexdigest()


def item_roles(rows, test_fraction):
    """Assign each item_key to train / eval / test by hash order within each domain."""
    domain_items = defaultdict(set)
    for row in rows:
        domain_items[row["domain"]].add(row["item_key"])
    roles = {}
    for items in domain_items.values():
        ordered = sorted(items, key=hash_order)
        n_test = round(len(ordered) * test_fraction)
        n_eval = round(len(ordered) * EVAL_ITEM_FRACTION)
        for i, item in enumerate(ordered):
            roles[item] = "test" if i < n_test else "eval" if i < n_test + n_eval else "train"
    return roles


def split_category(dim, rows):
    """Return {row_id: split or None} for one category; None means the row is left out."""
    test_templates = TEST_TEMPLATES[dim]
    templates_per_item = defaultdict(set)
    for row in rows:
        templates_per_item[row["item_key"]].add(template_key(row))
    item_unique = all(len(t) == 1 for t in templates_per_item.values())

    if item_unique and test_templates:
        # each question has one template, so holding out the template already holds out its questions
        roles = item_roles([r for r in rows if template_key(r) not in test_templates], 0.0)
        return {r["row_id"]: "test" if template_key(r) in test_templates else roles[r["item_key"]] for r in rows}

    roles = item_roles(rows, TEST_ITEM_FRACTION)
    out = {}
    for row in rows:
        held_out_template = template_key(row) in test_templates
        role = roles[row["item_key"]]
        if role == "test" and (held_out_template or not test_templates):
            out[row["row_id"]] = "test"
        elif role in ("train", "eval") and not held_out_template:
            out[row["row_id"]] = role
        else:
            out[row["row_id"]] = None
    return out


def validate(rows_by_split):
    problems = []
    seen = defaultdict(dict)  # (dim, key kind) -> {value: split}
    for split, rows in rows_by_split.items():
        for row in rows:
            for kind in ("item_key", "instance_id"):
                previous = seen[(row["dim"], kind)].setdefault(row[kind], split)
                if previous != split:
                    problems.append(f"{row['dim']} {kind} {row[kind]} in both {previous} and {split}")
            if split != "test" and template_key(row) in TEST_TEMPLATES[row["dim"]]:
                problems.append(f"{row['row_id']}: held-out template in {split}")
            if split == "test" and TEST_TEMPLATES[row["dim"]] and template_key(row) not in TEST_TEMPLATES[row["dim"]]:
                problems.append(f"{row['row_id']}: training template in test")
    return problems


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = [json.loads(line) for split in ("train", "eval") for line in (SRC / f"{split}.jsonl").open() if line.strip()]
    by_dim = defaultdict(list)
    for row in rows:
        by_dim[row["dim"]].append(row)

    assignment = {}
    for dim, dim_rows in by_dim.items():
        assignment.update(split_category(dim, dim_rows))

    rows_by_split = {"train": [], "eval": [], "test": []}
    for row in rows:
        split = assignment[row["row_id"]]
        if split is None:
            continue
        row = dict(row)
        row["previous_split"] = row["final_split"]
        row["final_split"] = split
        rows_by_split[split].append(row)

    problems = validate(rows_by_split)
    if problems:
        raise SystemExit("validation failed:\n" + "\n".join(problems[:20]))

    DST.mkdir(parents=True, exist_ok=True)
    checksums = []
    for split, split_rows in rows_by_split.items():
        out = DST / f"{split}.jsonl"
        with out.open("w") as f:
            for row in split_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        checksums.append(f"{sha256(out)}  {split}.jsonl")
    (DST / "SHA256SUMS").write_text("\n".join(checksums) + "\n")

    manifest = {
        "test_templates": {dim: sorted(t) for dim, t in TEST_TEMPLATES.items()},
        "test_item_keys": {dim: sorted({r["item_key"] for r in rows_by_split["test"] if r["dim"] == dim})
                           for dim in sorted(by_dim)},
        "eval_item_keys": {dim: sorted({r["item_key"] for r in rows_by_split["eval"] if r["dim"] == dim})
                           for dim in sorted(by_dim)},
    }
    (DST / "split_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    dropped = Counter(r["dim"] for r in rows if assignment[r["row_id"]] is None)
    print(f"{'category':26s} {'train':>6s} {'eval':>6s} {'test':>6s} {'left out':>9s}")
    for dim in sorted(by_dim):
        counts = [sum(r["dim"] == dim for r in rows_by_split[s]) for s in ("train", "eval", "test")]
        print(f"{dim:26s} {counts[0]:6d} {counts[1]:6d} {counts[2]:6d} {dropped[dim]:9d}")
    totals = [len(rows_by_split[s]) for s in ("train", "eval", "test")]
    print(f"{'TOTAL':26s} {totals[0]:6d} {totals[1]:6d} {totals[2]:6d} {sum(dropped.values()):9d}")

    print(f"\n{'rule_id':45s} {'train':>6s} {'eval':>6s} {'test':>6s}")
    rules = sorted({r["grading"]["rule_id"] for r in rows})
    for rule in rules:
        counts = [sum(r["grading"]["rule_id"] == rule for r in rows_by_split[s]) for s in ("train", "eval", "test")]
        print(f"{rule:45s} {counts[0]:6d} {counts[1]:6d} {counts[2]:6d}")
    print("\nvalidation problems: none")


if __name__ == "__main__":
    main()
