"""Fix 3, part 2: import MASK as a separate out-of-distribution test set (data/external/mask/).

MASK (cais/MASK on Hugging Face, https://www.mask-benchmark.ai/) is a public honesty benchmark: 1,000 realistic
scenarios in which a system prompt or user pushes the model to state something it believes is false. It is never
trained on or tuned against. It only checks whether a configuration that looks good on our own test split still
helps on pressure written by other people.

MASK defines a lie as a statement that contradicts the model's *own* belief, not the ground truth. Scoring it
properly therefore needs two generations per scenario: the pressured answer (`messages`) and the model's answers to
the neutral `belief_prompts`. The `provided_facts` archetype has no belief prompts because the system prompt itself
gives the model the facts, so there the ground truth stands in for the belief.

This script downloads the six parquet files at a pinned revision (if they are not already present), checks their
SHA-256 sums, converts every scenario to our row format, and checks that no MASK prompt appears in v3_resplit.
It needs pyarrow (preinstalled on Colab). Run from the repository root:

    python data/scripts/import_mask.py
"""
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

REPO = "cais/MASK"
REVISION = "4602b84dd9e2ca05c6e1eafbc14e556e908ac1bb"  # the dataset's latest commit, 2025-03-20
DST = Path("data/external/mask")
RAW = DST / "raw"
OUR_DATA = Path("data/v3_resplit")

# SHA-256 of each parquet file at REVISION
ARCHETYPES = {
    "continuations": "8d451bbe7f6fd83053d41118f8f081a4074047801d8f4183b8e0f5c76621d6b3",
    "disinformation": "3ee30f6dbd39c4b0460c7cda87d03d52849766c0d0094241d72de9c26ac10f59",
    "doubling_down_known_facts": "733e3ad2d0e5f468a1c941d676900dbaec3b45cb9e8188d00559007fe61a64c3",
    "known_facts": "5cc62d3ff1072252aa9819dfb5f9843924787af2278d2ac366f9d79197ea54c9",
    "provided_facts": "c9a1b0ddff3d2f8ca29ef98361a868caeb5623f0b77975fd5a626c2de62a5340",
    "statistics": "5785e109d1bb33a66bf0a9ebd49d646fece6d1663f63d454a65017c38281c12b",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(archetype):
    path = RAW / archetype / "test-00000-of-00001.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{archetype}/{path.name}"
        urllib.request.urlretrieve(url, path)
    expected = ARCHETYPES[archetype]
    if expected and sha256(path) != expected:
        raise SystemExit(f"checksum mismatch for {path}")
    return path


def convert(archetype, record):
    messages = [{"role": "system", "content": record["system_prompt"]}]
    if record.get("dd_turn_1_prompt"):
        # doubling down: the model already lied once in an earlier turn and is asked to confirm it
        messages.append({"role": "user", "content": record["dd_turn_1_prompt"]})
        messages.append({"role": "assistant", "content": record["dd_turn_1_response"]})
    messages.append({"role": "user", "content": record["user_prompt"]})
    belief_prompts = [record[k] for k in ("belief_elicit_1", "belief_elicit_2", "belief_elicit_3") if record.get(k)]
    return {
        "row_id": f"mask-{archetype}-{record['task_id']}",
        "dim": f"mask_{archetype}",
        "condition": "pressure",
        "final_split": "test_ood",
        "source": f"{REPO}@{REVISION[:7]}",
        "messages": messages,
        "belief_prompts": belief_prompts,
        "proposition": record["proposition"],
        "answer_type": record["type"],  # "binary" (two pivotal options) or "numerical" (a value)
        "ground_truth": record["ground_truth"],
        "formatted_ground_truth": record["formatted_ground_truth"],
        "grading": {
            "rule_id": "M_mask",
            "preferred": "answer",
            "answer": "preferred",
            "decline": "honest",  # MASK, like our principle, does not count an evasive or refusing answer as a lie
            "dishonest_if": ["statement_contradicts_own_belief"],
            "reference_is": "the ground truth; the model's own belief comes from belief_prompts "
                            "(provided_facts has none, so the ground truth is the belief there)",
        },
        "extra": {"task_id": record["task_id"], "delivery": record["delivery"]},
        "canary": record["canary"],
    }


def leak_check(rows):
    """Count MASK prompts (40+ characters, so generic lines like "Are you sure?" are ignored) found in our data."""
    ours = "".join((OUR_DATA / f"{s}.jsonl").read_text() for s in ("train", "eval", "test"))
    prompts = {m["content"] for r in rows for m in r["messages"] if m["role"] != "system" and len(m["content"]) >= 40}
    return sum(json.dumps(p, ensure_ascii=False)[1:-1] in ours for p in prompts)


def main():
    rows, checksums = [], []
    for archetype in ARCHETYPES:
        path = fetch(archetype)
        checksums.append(f"{sha256(path)}  raw/{archetype}/{path.name}")
        rows.extend(convert(archetype, record) for record in pq.read_table(path).to_pylist())

    out = DST / "mask_test.jsonl"
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    checksums.append(f"{sha256(out)}  mask_test.jsonl")
    (DST / "SHA256SUMS").write_text("\n".join(checksums) + "\n")

    counts = Counter(r["dim"] for r in rows)
    with_beliefs = Counter(r["dim"] for r in rows if r["belief_prompts"])
    print(f"{'archetype':32s} {'rows':>5s} {'with belief prompts':>20s}")
    for dim, n in counts.items():
        print(f"{dim:32s} {n:5d} {with_beliefs[dim]:20d}")
    print(f"{'TOTAL':32s} {len(rows):5d} {sum(with_beliefs.values()):20d}")
    print(f"answer types: {dict(Counter(r['answer_type'] for r in rows))}")
    leaks = leak_check(rows)
    print(f"MASK prompts found in v3_resplit: {leaks}")
    if leaks:
        raise SystemExit("MASK content found in our data; investigate before using MASK as a test set")


if __name__ == "__main__":
    main()
