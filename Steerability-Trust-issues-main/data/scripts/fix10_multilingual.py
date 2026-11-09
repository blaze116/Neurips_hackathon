"""Fix 10: a multilingual capability area (v10_system_prompt_compliance -> v11_multilingual).

The leaderboard reports "Multilingual" as one of its capability areas (r_side), and our data had none. A method that
works in English can still break the model in other languages: a steering direction or an English honesty policy
can push answers into English, or degrade reasoning in a language the method never saw. This script adds two
rule-scored task types, so no judge is involved:

  * maths word problems from MGSM (Shi et al., 2022; CC-BY-SA-4.0), the GSM8K problems translated by humans, in
    Bengali, German, Spanish, French, Japanese, Russian, Swahili, Telugu, Thai and Chinese; scored by the number;
  * four-option knowledge questions from Global-MMLU (Singh et al., 2024; Apache-2.0), MMLU translated into many
    languages, using only the questions whose translations were human-checked (`is_annotated`), in Arabic, Hindi,
    Korean, Portuguese, Italian, Indonesian, Turkish, Vietnamese, Polish and Ukrainian; scored by the letter.

The question is in its language; the closing format line is the same English line our other capability rows use, so
answers are read by the same rules. Each source question is used in one language only and sits wholly in one split.
75 maths and 75 knowledge rows go to each of train, eval and test (each language about 7 or 8 times per task and
split). Every v10 row is copied unchanged. Sources are pinned and checked against data/external/<name>/SHA256SUMS.
Run from the repository root (needs pyarrow):

    python data/scripts/fix10_multilingual.py
"""
import hashlib
import json
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

EXTERNAL = Path("data/external")
SRC = Path("data/v10_system_prompt_compliance")
DST = Path("data/v11_multilingual")
SPLITS = ("train", "eval", "test")
PER_SPLIT_PER_TASK = 75
DIM = "capability_multilingual"
FINAL_LINE = 'End your response with a final line of the form "Answer: {placeholder}".'
MGSM_LANGUAGES = ["bn", "de", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
GMMLU_LANGUAGES = ["ar", "hi", "ko", "pt", "it", "id", "tr", "vi", "pl", "uk"]
SOURCES = {
    "mgsm": ("juletxara/mgsm", "b2f13d426afe3be8d69a7e739b36724db8b66bbc",
             [f"{lang}/test-00000-of-00001.parquet" for lang in MGSM_LANGUAGES]),
    "global_mmlu": ("CohereLabs/Global-MMLU", "0e619dbeb34206cd48705a1a0ea7fb21cae09993",
                    [f"{lang}/test-00000-of-00001.parquet" for lang in GMMLU_LANGUAGES]),
}
FIELD_ORDER = ("row_id", "dim", "instance_id", "item_key", "template_id", "style", "pool", "condition", "messages",
               "followups", "reference", "declining", "domain", "difficulty", "extra", "final_split", "source",
               "scored_as", "grading", "scoring", "references", "contrastive_pair")
CAPABILITY = {"rule_id": "C_capability", "preferred": "answer", "answer": "preferred", "decline": "over_refusal",
              "dishonest_if": [], "reference_is": "the answer key in `scoring`"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(name):
    """Download a source's files if missing, check them against its SHA256SUMS, and return {language: path}."""
    repo, revision, files = SOURCES[name]
    folder = EXTERNAL / name
    sums_path = folder / "SHA256SUMS"
    expected = dict(line.split()[::-1] for line in sums_path.read_text().splitlines()) if sums_path.exists() else {}
    paths = {}
    for file in files:
        path = folder / "raw" / file
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{file}", path)
        if f"raw/{file}" in expected and sha256(path) != expected[f"raw/{file}"]:
            raise SystemExit(f"checksum mismatch for {path}")
        paths[file.split("/")[0]] = path
    if not expected:
        sums_path.write_text("".join(f"{sha256(paths[f.split('/')[0]])}  raw/{f}\n" for f in files))
    return paths


def hash_key(*parts):
    return hashlib.sha256(":".join(["fix10", *map(str, parts)]).encode()).hexdigest()


def deal(keys, strata):
    """[(key, split, language index)]: keys ordered by a fixed hash inside each stratum, the strata interleaved by
    relative rank, the first 3 x PER_SPLIT_PER_TASK dealt to train, eval, test in turn; within each split the
    languages are dealt round-robin."""
    by_stratum = defaultdict(list)
    for key in keys:
        by_stratum[strata(key)].append(key)
    ranked = []
    for stratum, members in by_stratum.items():
        ordered = sorted(members, key=lambda k: hash_key(stratum, k))
        ranked += [((rank + 0.5) / len(ordered), hash_key(k), k) for rank, k in enumerate(ordered)]
    ranked.sort()
    out, per_split = [], Counter()
    for position, (_, _, key) in enumerate(ranked[:3 * PER_SPLIT_PER_TASK]):
        split = SPLITS[position % 3]
        out.append((key, split, per_split[split]))
        per_split[split] += 1
    return out


def new_row(row_id, split, source, prompt, scoring, domain, extra):
    row = {"row_id": row_id, "dim": DIM, "instance_id": row_id, "item_key": row_id, "pool": "third_party",
           "condition": "neutral", "messages": [{"role": "user", "content": prompt}], "followups": [],
           "domain": domain, "extra": extra, "final_split": split, "source": source,
           "scored_as": ["capability:multilingual"], "grading": CAPABILITY, "scoring": scoring}
    return {field: row.get(field, [] if field == "followups" else {} if field == "extra" else None)
            for field in FIELD_ORDER}


def build_mgsm():
    repo, revision, _ = SOURCES["mgsm"]
    tables = {lang: pq.read_table(path).to_pylist() for lang, path in fetch("mgsm").items()}
    rows = []
    for problem, split, slot in deal(range(len(tables["de"])), lambda key: "all"):
        lang = MGSM_LANGUAGES[slot % len(MGSM_LANGUAGES)]
        record = tables[lang][problem]
        answer = record["answer_number"]
        answer = str(int(answer)) if float(answer).is_integer() else str(answer)
        prompt = f"{record['question']}\n\n" + FINAL_LINE.format(placeholder="<number>")
        rows.append(new_row(f"cap-multilingual-mgsm-{lang}-{problem:03d}", split, f"{repo}@{revision[:7]}", prompt,
                            {"method": "number", "answer": answer}, f"mgsm-{lang}",
                            {"language": lang, "task": "math", "problem": problem}))
    return rows


def build_global_mmlu():
    repo, revision, _ = SOURCES["global_mmlu"]
    tables = {lang: {r["sample_id"]: r for r in pq.read_table(path).to_pylist() if str(r["is_annotated"]) == "True"}
              for lang, path in fetch("global_mmlu").items()}
    shared = sorted(set.intersection(*(set(t) for t in tables.values())))
    category = {sid: tables["ar"][sid]["subject_category"] for sid in shared}
    rows = []
    for sample_id, split, slot in deal(shared, lambda key: category[key]):
        lang = GMMLU_LANGUAGES[slot % len(GMMLU_LANGUAGES)]
        record = tables[lang][sample_id]
        options = "\n".join(f"({letter}) {record[f'option_{letter.lower()}']}" for letter in "ABCD")
        prompt = (f"{record['question']}\n\nOptions:\n{options}\n\n" + FINAL_LINE.format(placeholder="X")
                  + " X is the letter of the correct option.")
        rows.append(new_row(f"cap-multilingual-gmmlu-{lang}-{sample_id.replace('/', '-')}", split,
                            f"{repo}@{revision[:7]}", prompt,
                            {"method": "choice_letter", "answer": record["answer"], "n_options": 4},
                            f"global_mmlu-{lang}",
                            {"language": lang, "task": "knowledge", "sample_id": sample_id, "subject": record["subject"],
                             "cultural_sensitivity": record["cultural_sensitivity_label"]}))
    return rows


def main():
    new = build_mgsm() + build_global_mmlu()
    old = {split: [json.loads(line) for line in (SRC / f"{split}.jsonl").open() if line.strip()] for split in SPLITS}

    problems = []
    ids = [r["row_id"] for rows in old.values() for r in rows] + [r["row_id"] for r in new]
    if len(ids) != len(set(ids)):
        problems.append("duplicate row_id")
    sizes = Counter((r["extra"]["task"], r["final_split"]) for r in new)
    if any(n != PER_SPLIT_PER_TASK for n in sizes.values()) or len(sizes) != 6:
        problems.append(f"sizes {dict(sizes)}")
    for task, field in (("math", "problem"), ("knowledge", "sample_id")):
        seen = Counter(r["extra"][field] for r in new if r["extra"]["task"] == task)
        if any(n > 1 for n in seen.values()):
            problems.append(f"a {task} question appears in more than one language")
    if any(not r["messages"][0]["content"].strip() for r in new):
        problems.append("empty prompt")
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

    langs = Counter((r["extra"]["language"], r["final_split"]) for r in new)
    print(f"{'language':10s} {'train':>6s} {'eval':>6s} {'test':>6s}")
    for lang in MGSM_LANGUAGES + GMMLU_LANGUAGES:
        print(f"{lang:10s}", " ".join(f"{langs[(lang, s)]:6d}" for s in SPLITS))
    print(f"{'total':10s}", " ".join(f"{sum(r['final_split'] == s for r in new):6d}" for s in SPLITS))
    print("\nvalidation problems: none")


if __name__ == "__main__":
    main()
