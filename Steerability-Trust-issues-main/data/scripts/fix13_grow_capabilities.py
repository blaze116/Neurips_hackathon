"""Fix 13: grow the four original capability areas from 60 to 150 rows per split (v13_missing_dims -> v14_capabilities).

r_side counts only drops, and each 0.01 of drop costs 0.03 of score, so a capability area must be able to show a drop
of a few points. With 60 rows per split a 95% margin is about +-0.1; with 150 it is about +-0.06, and paired
comparisons on the same rows are tighter still. exp005 showed that two unchanged runs already differ by chance on a
handful of rows, so the areas need more rows.

Added per split: 90 rows each of knowledge (MMLU-Pro), reasoning (BIG-Bench Hard), problem solving (GSM8K) and
instruction following, formatted exactly like fix 4's rows. Excluded everywhere: items already in v13 and items in
data/benign_prompts_v1, the prompt set NullSpaceMap was fitted to leave alone, so no method is scored on prompts it
was fitted on. That set already holds every remaining IFEval prompt, so the new instruction-following rows are built
the way IFEval itself was built: an everyday Dolly-15k request (none of the ones fix 9 used) with one or two IFEval
instructions appended to the user turn in IFEval's own wording, scored by the IFEval checker.

Every v13 row is copied unchanged. New items are ordered by a fixed hash inside each stratum (MMLU-Pro category, BBH
task, Dolly category), the strata interleaved, and dealt to train, eval and test in turn. Run from the repository root
(needs pyarrow and the IFEval checker):

    python data/scripts/fix13_grow_capabilities.py
"""
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fix4_guard_capability as fix4  # noqa: E402  (pinned sources, row formats)
import fix9_system_prompt_compliance as fix9  # noqa: E402  (Dolly, IFEval instruction groups)
from instruction_following_eval import instructions_registry  # noqa: E402

SRC = Path("data/v13_missing_dims")
DST = Path("data/v14_capabilities")
BENIGN = Path("data/benign_prompts_v1/benign_prompts.jsonl")
SPLITS = ("train", "eval", "test")
ADD_PER_SPLIT = 90
FIELD_ORDER = fix9.FIELD_ORDER
# instructions for the user-turn rows: fix 9's groups, plus the IFEval kinds that read naturally inside a request
USER_TURN_EXTRA = ["keywords:existence", "keywords:forbidden_words", "keywords:letter_frequency",
                   "combination:repeat_prompt", "length_constraints:nth_paragraph_first_word"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deal(items, stratum, key):
    """3 x ADD_PER_SPLIT items: ordered by a fixed hash in each stratum, strata interleaved by relative rank, then dealt
    to train, eval, test in turn. Returns [(item, split)]."""
    by_stratum = defaultdict(list)
    for index, item in enumerate(items):
        by_stratum[stratum(item)].append(index)
    ranked = []
    for name, members in by_stratum.items():
        ordered = sorted(members, key=lambda i: hashlib.sha256(f"fix13:{name}:{key(items[i])}".encode()).hexdigest())
        ranked += [((r + 0.5) / len(ordered), hashlib.sha256(str(key(items[i])).encode()).hexdigest(), i)
                   for r, i in enumerate(ordered)]
    ranked.sort()
    picked = [items[i] for _, _, i in ranked[:3 * ADD_PER_SPLIT]]
    return [(item, SPLITS[n % 3]) for n, item in enumerate(picked)]


def normalise(row):
    defaults = {"followups": [], "extra": {}, "pool": "third_party", "condition": "neutral"}
    return {field: row.get(field, defaults.get(field)) for field in FIELD_ORDER}


def build_mmlu_pro(used):
    (path,) = fix4.fetch("mmlu_pro")
    items = [r for r in pq.read_table(path).to_pylist() if f"mmlupro-{r['question_id']}" not in used]
    rows = []
    for record, split in deal(items, lambda r: r["category"], lambda r: r["question_id"]):
        letters = "ABCDEFGHIJ"[:len(record["options"])]
        options = "\n".join(f"({l}) {o}" for l, o in zip(letters, record["options"]))
        prompt = (f"{record['question']}\n\nOptions:\n{options}\n\n"
                  + fix4.FINAL_LINE.format(placeholder="X") + " X is the letter of the correct option.")
        rows.append(fix4.new_row(
            f"cap-knowledge-mmlupro-{record['question_id']}", "capability_knowledge", split,
            fix4.source_tag("mmlu_pro"), fix4.user(prompt), fix4.CAPABILITY,
            {"method": "choice_letter", "answer": record["answer"], "n_options": len(letters)},
            ["capability:knowledge"], record["category"], {"src": record["src"]}))
    return rows


def build_bbh(used):
    items = []
    for task, path in zip(fix4.BBH_TASKS, fix4.fetch("bbh")):
        items += [{"task": task, "index": i, **r} for i, r in enumerate(pq.read_table(path).to_pylist())
                  if f"bbh-{task}-{i}" not in used]
    rows = []
    for record, split in deal(items, lambda r: r["task"], lambda r: f"{r['task']}-{r['index']}"):
        prompt = f"{record['input']}\n\n" + fix4.FINAL_LINE.format(placeholder="<your answer>")
        rows.append(fix4.new_row(
            f"cap-reasoning-bbh-{record['task']}-{record['index']}", "capability_reasoning", split,
            fix4.source_tag("bbh"), fix4.user(prompt), fix4.CAPABILITY,
            {"method": "exact_match", "answer": record["target"]}, ["capability:reasoning"], record["task"], {}))
    return rows


def build_gsm8k(used):
    (path,) = fix4.fetch("gsm8k")
    items = [{"index": i, **r} for i, r in enumerate(pq.read_table(path).to_pylist()) if f"gsm8k-{i}" not in used]
    rows = []
    for record, split in deal(items, lambda r: "all", lambda r: r["index"]):
        answer = record["answer"].split("####")[-1].strip().replace(",", "")
        prompt = f"{record['question']}\n\n" + fix4.FINAL_LINE.format(placeholder="<number>")
        rows.append(fix4.new_row(
            f"cap-problemsolving-gsm8k-{record['index']}", "capability_problem_solving", split,
            fix4.source_tag("gsm8k"), fix4.user(prompt), fix4.CAPABILITY, {"method": "number", "answer": answer},
            ["capability:problem_solving"], "math_word_problem", {}))
    return rows


def user_turn_rules(index, request):
    """One IFEval instruction, or a non-conflicting style + structure pair, in IFEval's wording."""
    rng = random.Random(f"fix13:{index}")
    pool = fix9.STYLE + fix9.STRUCTURE + USER_TURN_EXTRA
    if rng.random() < 0.5:
        ids = [rng.choice(pool)]
    else:
        while True:
            pair = (rng.choice(fix9.STYLE), rng.choice(fix9.STRUCTURE + USER_TURN_EXTRA))
            if pair not in fix9.CONFLICTS and not (pair[0].startswith(("change_case", "language"))
                                                   and pair[1].startswith(("keywords", "combination:repeat"))):
                ids = list(pair)
                break
    rules = []
    for instruction_id in ids:
        random.seed(f"fix13:{index}:{instruction_id}")
        instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
        if instruction_id == "combination:repeat_prompt":
            description = instruction.build_description(prompt_to_repeat=request)
        else:
            description = instruction.build_description()
        rules.append((instruction_id, description, instruction.get_instruction_args() or {}))
    return rules


def build_instruction_following():
    taken = {r["text"].lower() for r in fix9.pick_requests(fix9.fetch())}
    items, seen = [], set()
    for line in fix9.fetch().read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        text = record["instruction"].strip()
        if record["category"] in fix9.CATEGORIES and not record["context"].strip() and 20 <= len(text) <= 300 \
                and text.lower() not in taken and text.lower() not in seen:
            seen.add(text.lower())
            items.append({"text": text, "category": record["category"]})
    rows = []
    for n, (request, split) in enumerate(deal(items, lambda r: r["category"], lambda r: r["text"])):
        rules = user_turn_rules(n, request["text"])
        text = request["text"].rstrip()
        text += "" if text[-1] in ".?!" else "."  # so the instruction starts a new sentence
        prompt = text + " " + " ".join(d for _, d, _ in rules)
        row_id = f"cap-instruction-dolly-ifeval-{n:03d}"
        rows.append(fix4.new_row(
            row_id, "capability_instruction_following", split, fix9.SOURCE, fix4.user(prompt), fix4.CAPABILITY,
            {"method": "ifeval", "instruction_id_list": [i for i, _, _ in rules],
             "kwargs": {str(k): kw for k, (_, _, kw) in enumerate(rules)}},
            ["capability:instruction_following"], rules[0][0].split(":")[0],
            {"request": request["text"], "dolly_category": request["category"]}))
    return rows


def main():
    old = {split: [json.loads(line) for line in (SRC / f"{split}.jsonl").open() if line.strip()] for split in SPLITS}
    # item ids already used: by v13 rows (cap-knowledge-mmlupro-123 -> mmlupro-123) and by the benign prompt set
    used = {"-".join(r["row_id"].split("-")[2:]) for rows in old.values() for r in rows
            if r["row_id"].startswith(("cap-knowledge-", "cap-reasoning-", "cap-problemsolving-"))}
    used |= {"-".join(json.loads(line)["row_id"].split("-")[2:]) for line in BENIGN.open() if line.strip()}
    new = [normalise(r) for r in build_mmlu_pro(used) + build_bbh(used) + build_gsm8k(used)
           + build_instruction_following()]

    problems = []
    ids = [r["row_id"] for rows in old.values() for r in rows] + [r["row_id"] for r in new]
    if len(ids) != len(set(ids)):
        problems.append("duplicate row_id")
    if any("-".join(r["row_id"].split("-")[2:]) in used for r in new):
        problems.append("a new row reuses an item from v13 or the benign set")
    old_prompts = {r["messages"][-1]["content"] for rows in old.values() for r in rows}
    benign_prompts = {json.loads(line)["messages"][-1]["content"] for line in BENIGN.open() if line.strip()}
    if any(r["messages"][-1]["content"] in old_prompts | benign_prompts for r in new):
        problems.append("a new prompt already appears in v13 or the benign set")
    sizes = Counter((r["dim"], r["final_split"]) for r in new)
    if any(n != ADD_PER_SPLIT for n in sizes.values()) or len(sizes) != 12:
        problems.append(f"sizes {dict(sizes)}")
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

    total = Counter((r["dim"], r["final_split"]) for rows in old.values() for r in rows + [])
    for r in new:
        total[(r["dim"], r["final_split"])] += 1
    print(f"{'dim':36s} {'train':>6s} {'eval':>6s} {'test':>6s}   (after adding {ADD_PER_SPLIT} per split)")
    for dim in ("capability_knowledge", "capability_reasoning", "capability_problem_solving",
                "capability_instruction_following"):
        print(f"{dim:36s}", " ".join(f"{total[(dim, s)]:6d}" for s in SPLITS))
    print("\nvalidation problems: none")


if __name__ == "__main__":
    main()
