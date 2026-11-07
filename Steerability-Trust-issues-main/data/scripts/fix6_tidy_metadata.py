"""Fix 6: tidy the metadata (v6_official_declining -> v7_tidy).

Rows written by different fixes carried different sets of fields (for example the fix 4 and fix 5 rows had no
`followups` or `style`, and only the team's original rows had `pool`), so every tool had to guess whether a field
exists. This script:

  * removes the two stale fields: `split` (the old "fit"/"holdout" label, meaningless since the fix 3 re-split) and
    `previous_split` (the pre-fix-3 split);
  * gives every row the same fields in the same order: the official Starter Kit row schema first, then our additions,
    with an empty value (null or []) where a field does not apply;
  * fills `source` and `pool` on the team's original rows, which lacked them.

No value that a row already had is changed, except the two removed fields. Field meanings are in data/FIELDS.md.
Run from the repository root:

    python data/scripts/fix6_tidy_metadata.py
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

SRC = Path("data/v6_official_declining")
DST = Path("data/v7_tidy")
SPLITS = ("train", "eval", "test")
REMOVED = ("split", "previous_split")

# the official row schema (third_party/starterkit/rows.py, ROW_FIELDS), in its order, with the empty value to use
OFFICIAL_FIELDS = {
    "row_id": None, "dim": None, "instance_id": None, "item_key": None, "template_id": None, "style": None,
    "pool": None, "condition": None, "messages": None, "followups": [], "reference": None, "declining": None,
    "domain": None, "difficulty": None, "extra": {},
}
# our additions, in order
OUR_FIELDS = {
    "final_split": None, "source": None, "scored_as": None, "grading": None, "scoring": None,
    "references": None, "contrastive_pair": None,
}
FIELD_ORDER = {**OFFICIAL_FIELDS, **OUR_FIELDS}


def default_source_and_pool(row):
    """Where a row came from, for rows written before `source` existed (the team's original dataset)."""
    if row.get("source"):
        return row["source"], row.get("pool") or ("third_party" if "@" in row["source"] else "team")
    if row.get("pool") == "third_party":
        return "anthropic_harm_pressure (via the Starter Kit)", "third_party"
    return "team_v0", row.get("pool") or "team"


def tidy(row):
    out = {}
    source, pool = default_source_and_pool(row)
    for field, empty in FIELD_ORDER.items():
        value = row.get(field, empty)
        out[field] = [] if value is None and isinstance(empty, list) else value
    out["source"], out["pool"] = source, pool
    extra_fields = set(row) - set(FIELD_ORDER) - set(REMOVED)
    if extra_fields:
        raise SystemExit(f"{row['row_id']}: unexpected fields {sorted(extra_fields)}; add them to FIELD_ORDER")
    return out


def validate(before, after):
    """Every row has the same fields; only the removed fields, the filled empties, source and pool differ."""
    problems = []
    for old, new in zip(before, after):
        if list(new) != list(FIELD_ORDER):
            problems.append(f"{new['row_id']}: fields out of order")
        for field, value in old.items():
            if field in REMOVED:
                continue
            if field in ("source", "pool") and value:
                if new[field] != value:
                    problems.append(f"{new['row_id']}: {field} changed")
            elif field not in ("source", "pool") and new[field] != value and not (value is None and new[field] == []):
                problems.append(f"{new['row_id']}: {field} changed")
    return problems


def main():
    DST.mkdir(parents=True, exist_ok=True)
    checksums, filled = [], Counter()
    for split in SPLITS:
        before = [json.loads(line) for line in (SRC / f"{split}.jsonl").read_text().splitlines() if line.strip()]
        after = [tidy(row) for row in before]
        for old in before:
            filled.update(field for field in FIELD_ORDER if field not in old)
        problems = validate(before, after)
        if problems:
            raise SystemExit("validation failed:\n" + "\n".join(problems[:20]))
        out = DST / f"{split}.jsonl"
        with out.open("w") as f:
            for row in after:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        checksums.append(f"{hashlib.sha256(out.read_bytes()).hexdigest()}  {split}.jsonl")
        print(f"{split}: {len(after)} rows")
    manifest = DST / "split_manifest.json"
    manifest.write_text((SRC / "split_manifest.json").read_text())
    checksums.append(f"{hashlib.sha256(manifest.read_bytes()).hexdigest()}  split_manifest.json")
    (DST / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print("fields filled with an empty value (rows):", dict(filled))
    print(f"removed: {', '.join(REMOVED)}; every row now has the same {len(FIELD_ORDER)} fields")
    print("validation problems: none")


if __name__ == "__main__":
    main()
