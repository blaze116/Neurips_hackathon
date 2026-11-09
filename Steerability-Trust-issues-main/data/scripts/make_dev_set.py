"""Pick the team's development set from the eval split of v17 (-> data/dev_sets/dev_v1).

The eval split has 3,154 rows, too many to run for every method comparison, while the old 98-row set was so small
that two runs of the unchanged model differed by S = -0.615 by chance (exp005). The development set is a fixed slice
chosen by rules set in advance, never by how any model or method scores on a row (picking rows to make local scores
agree with the leaderboard would fit this week's quirks, not the final evaluation):

  * every official category gets rows: 15 honesty dims, 6 guard areas, 6 capability areas;
  * guard areas get more rows, since a guard regression costs three times its size in S, and over-conservativeness
    (the guard that cost us most on the leaderboard) gets the most;
  * rows that belong together stay together: the pressure row of an instance with its control twin (persona_only,
    neutral, control or plain), which some dims need for scoring at all (sandbagging, feedback, hint);
  * within a category, instances are spread over templates, sources and topics (round-robin over those strata, in a
    fixed hash order, so the pick is reproducible and does not depend on file order).

It writes two nested files: dev.jsonl (about 1,000 rows, for decisions) and dev_quick.jsonl (about a third of each
category's instances, the first ones in the same order, for a first look at a new method). Run from the repository
root:

    python data/scripts/make_dev_set.py
"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("data/v17_thin_categories/eval.jsonl")
DST = Path("data/dev_sets/dev_v1")
QUICK_SHARE = 1 / 3

# (dim, filter or None) -> instances to take. An instance is one scenario with all its condition rows. A filter is
# a grading rule id or a source; over-conservativeness leans on OR-Bench's hard prompts, the ones built to catch the
# leaderboard penalty that XSTest (passed by every setup so far) missed.
TAKE = {
    # honesty, about 30 rows each (more for sandbagging and hint, which only score some of their instances)
    ("concealment", None): 15, ("instructed_lie", None): 15, ("implicit_incentive", None): 15,
    ("self_misrepresentation", None): 15, ("feedback_sycophancy", None): 15, ("harm_motivated_denial", None): 15,
    ("sandbagging", None): 20, ("hint_acknowledgment", None): 20,
    ("fabricated_action", None): 10, ("fabricated_entity", None): 10, ("fabricated_citation", None): 10,
    ("confrontation_after_lie", None): 30, ("false_premise", None): 30, ("sandbagging_stated", None): 30,
    ("pushback", "R9_pushback_user_wrong"): 30,
    # guards (the 15 feedback neutral rows above also count for pushback-and-feedback)
    ("guard_over_conservativeness", "bench-llm/or-bench@e36d8b8"): 60,
    ("guard_over_conservativeness", "Paul/XSTest@f600c99"): 20, ("guard_helpfulness", None): 50,
    ("guard_system_prompt_compliance", None): 50, ("guard_refusal", None): 50,
    ("guard_confidentiality_persona", None): 50, ("guard_sincere_nonassertion", None): 15,
    ("pushback", "R9c_pushback_user_right"): 40,
    # capability areas
    ("capability_knowledge", None): 30, ("capability_reasoning", None): 30, ("capability_problem_solving", None): 30,
    ("capability_instruction_following", None): 30, ("capability_multilingual", None): 30,
    ("capability_fidelity_long_context", None): 30,
}


def rank(*parts):
    return hashlib.sha256(":".join(["dev_v1", *map(str, parts)]).encode()).hexdigest()


def stratum(row):
    """What to spread a category's picks over: template, source and topic."""
    return (row.get("template_id"), row.get("source"), row.get("domain"))


def spread(instances, first_row):
    """Instance ids in pick order: round-robin over strata, each stratum and each instance in fixed hash order."""
    by_stratum = defaultdict(list)
    for instance in instances:
        by_stratum[stratum(first_row[instance])].append(instance)
    queues = [sorted(members, key=lambda i: rank("instance", i))
              for _, members in sorted(by_stratum.items(), key=lambda kv: rank("stratum", kv[0]))]
    order = []
    while any(queues):
        for queue in queues:
            if queue:
                order.append(queue.pop(0))
    return order


def main():
    rows = [json.loads(line) for line in SRC.open() if line.strip()]
    by_instance = defaultdict(list)
    for row in rows:
        by_instance[(row["dim"], row["instance_id"])].append(row)
    dev, quick, problems = [], [], []
    for (dim, rule), n in TAKE.items():
        first_row = {}
        for (d, instance), members in by_instance.items():
            if d == dim and rule in (None, members[0]["grading"]["rule_id"], members[0]["source"]):
                first_row[instance] = members[0]
        if len(first_row) < n:
            problems.append(f"{dim} {rule}: only {len(first_row)} instances, wanted {n}")
        picked = spread(list(first_row), first_row)[:n]
        n_quick = max(1, round(len(picked) * QUICK_SHARE))
        for position, instance in enumerate(picked):
            members = by_instance[(dim, instance)]
            dev += members
            if position < n_quick:
                quick += members
    ids = [r["row_id"] for r in dev]
    if len(ids) != len(set(ids)):
        problems.append("a row was picked twice")
    if any(r["final_split"] != "eval" for r in dev):
        problems.append("a row outside the eval split")
    for d in ("sandbagging", "feedback_sycophancy", "hint_acknowledgment", "concealment", "instructed_lie"):
        conds = Counter(r["condition"] for r in dev if r["dim"] == d)
        if len(set(conds.values())) != 1:
            problems.append(f"{d}: unpaired conditions {dict(conds)}")
    if problems:
        raise SystemExit("validation failed:\n" + "\n".join(problems))

    DST.mkdir(parents=True, exist_ok=True)
    order = {r["row_id"]: i for i, r in enumerate(rows)}  # keep the eval file's order
    sums = []
    for name, chosen in (("dev", dev), ("dev_quick", quick)):
        chosen = sorted(chosen, key=lambda r: order[r["row_id"]])
        path = DST / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen))
        (DST / f"{name}_row_ids.txt").write_text("".join(r["row_id"] + "\n" for r in chosen))
        sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}.jsonl")
    sums.append(f"{hashlib.sha256(SRC.read_bytes()).hexdigest()}  source: {SRC}")
    (DST / "SHA256SUMS").write_text("\n".join(sums) + "\n")

    groups = defaultdict(Counter)
    for name, chosen in (("dev", dev), ("quick", quick)):
        for r in chosen:
            for g in r["scored_as"]:
                groups[g if g != "honesty" else f"honesty:{r['dim']}"][name] += 1
    print(f"{'group':42s} {'dev':>5s} {'quick':>5s}")
    for g in sorted(groups):
        print(f"{g:42s} {groups[g]['dev']:5d} {groups[g]['quick']:5d}")
    print(f"{'total rows':42s} {len(dev):5d} {len(quick):5d}")
    print("\nvalidation problems: none")


if __name__ == "__main__":
    main()
