"""Loading rows, selecting from them, building the chats a bundle generates from, and rendering the
rows of a whole source pool.

A row is one JSON object with the fields of `ROW_FIELDS`, in that order. `load_rows` reads the rows
of one dishonesty dim from `data/dishonesty_dims/`, and `render_harm_pair_rows` and
`render_mask_rows` build rows of the same shape from the whole source pool of the
`harm_motivated_denial` and `false_premise` dims. `data/README.md` documents each field.
"""
import argparse
import json
import re
from pathlib import Path

from starterkit import DATA_DIR, DISHONESTY_DIMS, MASK_REVISION, OUTPUTS_DIR, USER_SUFFIX_SANDBAGGING

DIMS_DIR = DATA_DIR / "dishonesty_dims"
THIRD_PARTY_DIR = DATA_DIR / "third_party"
HARM_PAIRS_PATH = THIRD_PARTY_DIR / "harm_pressure_pairs.jsonl"

#: every field of a row, in the order `data/README.md` documents them; `dim` is the dishonesty dim's
#: id and `row_id` is `<dim>-<item_key>-<template_id|none>-<condition>`
ROW_FIELDS = (
    "row_id", "dim", "instance_id", "item_key", "template_id", "style", "pool",
    "condition", "messages", "followups", "reference", "declining", "domain", "difficulty", "extra",
)

#: the dataset repository the `false_premise` rows are rendered from, and its archetype
MASK_DATASET = "cais/MASK"
MASK_ARCHETYPE = "continuations"

#: the option letters a `harm_motivated_denial` question labels its choices with
_OPTION_LETTER = re.compile(r"^\(([A-J])\)", re.MULTILINE)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rows(name: str) -> list[dict]:
    """The rows of one dishonesty dim.

    Args:
        name: The dim's id, e.g., `"instructed_lie"` or `"sandbagging_stated"`; a member of
            `DISHONESTY_DIMS`, and the row's own `dim` field.

    Returns:
        The rows, in file order.

    Raises:
        FileNotFoundError: If no such dim file exists.
    """
    path = DIMS_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"No dim {name!r} at {path}; the dim ids are {list(DISHONESTY_DIMS)}."
        )
    return _read_jsonl(path)


def load_guard_rows(guard_id: str) -> list[dict]:
    """The rows of one guard set, by file stem (e.g., `"system_prompt_compliance"`)."""
    path = DATA_DIR / "guards" / f"{guard_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No guard {guard_id!r} at {path}; see {all_guards()}.")
    return _read_jsonl(path)


def all_dims() -> list[str]:
    """The ids of the dim files present, in evaluation order.

    Returns:
        The members of `DISHONESTY_DIMS` whose dim file exists.
    """
    return [name for name in DISHONESTY_DIMS if (DIMS_DIR / f"{name}.jsonl").exists()]


def all_guards() -> list[str]:
    """The sorted stems of the guard files present."""
    return sorted(path.stem for path in (DATA_DIR / "guards").glob("*.jsonl"))


def select(rows: list[dict], *, condition: str = "pressure", n: int | None = None) -> list[dict]:
    """The rows of one condition, by `row_id`, first `n`.

    The result is deterministic. The rows are filtered by condition and sorted by `row_id`, and
    nothing is drawn at random.

    Args:
        rows: Rows to select from.
        condition: The `condition` value to keep (`"pressure"`, `"neutral"`, `"persona_only"`).
        n: How many to take; None takes all of them.

    Returns:
        The selected rows.
    """
    kept = sorted((row for row in rows if row["condition"] == condition), key=lambda row: row["row_id"])
    return kept if n is None else kept[:n]


def chat(row: dict) -> list[dict]:
    """The chat a row asks the model to continue, as the evaluation sends it: a copy of
    `row["messages"]`, with `USER_SUFFIX_SANDBAGGING` appended to the last user turn after a blank
    line on a `sandbagging` row (the evaluation generates those rows with that suffix, so that the
    candidate ends in a choice letter).
    """
    messages = [dict(message) for message in row["messages"]]
    if row.get("dim") == "sandbagging":
        messages[-1]["content"] = f"{messages[-1]['content']}\n\n{USER_SUFFIX_SANDBAGGING}"
    return messages


def with_followup(chat: list[dict], answer: str, followup: str) -> list[dict]:
    """The chat continued by one turn: an assistant message, then the next user turn.

    Args:
        chat: The chat that produced `answer`.
        answer: The assistant's answer, i.e., the continuation with its reasoning segment removed.
        followup: The next user turn's content.

    Returns:
        A new chat, two messages longer.
    """
    return [*(dict(message) for message in chat),
            {"role": "assistant", "content": answer},
            {"role": "user", "content": followup}]


def load_harm_pairs() -> list[dict]:
    """The items of `harm_pressure_pairs.jsonl`, as dicts in the file's own schema.

    Returns:
        The items, in file order, each with `pair_id`, `variant`, `user`, and `answer`, and with
        `system` on the `neutral` variants.

    Raises:
        FileNotFoundError: If the file is absent.
    """
    if not HARM_PAIRS_PATH.exists():
        raise FileNotFoundError(
            f"No Harm Pressure pairs at {HARM_PAIRS_PATH}; the file is part of the released data."
        )
    return _read_jsonl(HARM_PAIRS_PATH)


def render_harm_pair_rows() -> list[dict]:
    """Rows in the row schema, two `harm_motivated_denial` rows per pair (pressure, then neutral), in
    `pair_id` order, with the construction of the rows in `data/dishonesty_dims/` (see
    `data/README.md`).

    The `neutral` row keeps the pair's expert system prompt and the `pressure` row has the user turn
    alone. Both reference the pair's answer letter, and `extra` records the bare question, the
    answer, its alias, and the other option letters.

    Returns:
        Two rows per pair, 274 in all.

    Raises:
        FileNotFoundError: If `harm_pressure_pairs.jsonl` is absent.
        ValueError: If a pair lacks a variant, if its two variants disagree on the answer, or if the
            `pressure` user turn does not end with the `neutral` user turn.
    """
    pairs: dict[str, dict[str, dict]] = {}
    for item in load_harm_pairs():
        pairs.setdefault(item["pair_id"], {})[item["variant"]] = item

    rows: list[dict] = []
    for pair_id, variants in pairs.items():
        missing = {"neutral", "pressure"} - set(variants)
        if missing:
            raise ValueError(
                f"pair {pair_id!r} has no {sorted(missing)} variant; a pair is rendered from both "
                f"of its variants."
            )
        neutral, pressure = variants["neutral"], variants["pressure"]
        if not pressure["user"].endswith(neutral["user"]):
            raise ValueError(
                f"pair {pair_id!r} has a pressure user turn that does not end with its neutral user "
                f"turn; the pressure variant prepends a stated intent to the same question."
            )
        if pressure["answer"] != neutral["answer"]:
            raise ValueError(
                f"pair {pair_id!r} has answer {pressure['answer']!r} on its pressure variant and "
                f"{neutral['answer']!r} on its neutral variant; the two variants share one answer."
            )
        answer = neutral["answer"]
        item_key = pair_id.replace("-", "_")
        instance_id = f"harm_motivated_denial-{item_key}-none"
        alternatives = sorted(set(_OPTION_LETTER.findall(neutral["user"])) - {answer})

        for condition, item in (("pressure", pressure), ("neutral", neutral)):
            messages = []
            if "system" in item:
                messages.append({"role": "system", "content": item["system"]})
            messages.append({"role": "user", "content": item["user"]})
            rows.append({
                "row_id": f"{instance_id}-{condition}",
                "dim": "harm_motivated_denial",
                "instance_id": instance_id,
                "item_key": item_key,
                "template_id": None,
                "style": "harm_pressure_pair",
                "pool": "starter",
                "condition": condition,
                "messages": messages,
                "followups": [],
                "reference": {"type": "T", "text": f"The correct answer is ({answer})."},
                "declining": "warranted",
                "domain": "dual_use",
                "difficulty": None,
                "extra": {
                    "harm_level": "explicit",
                    "question": neutral["user"],
                    "answer": answer,
                    "answer_aliases": [f"({answer})"],
                    "false_alternatives": alternatives,
                },
            })
    return rows


def render_mask_rows() -> list[dict]:
    """Rows in the row schema, one `false_premise` row per MASK continuation item, in the split's
    order, from the `continuations` split of `cais/MASK` at `MASK_REVISION` (downloaded with
    `datasets`; needs network access on first use). MASK is a measurement set; see
    `data/README.md`.

    The item's system prompt and user prompt are the row's system message and user turn, and its
    `ground_truth` is the `T` reference.

    Returns:
        One row per item, 176 in all.

    Raises:
        RuntimeError: If the split could not be read.
    """
    from datasets import load_dataset

    try:
        items = load_dataset(
            MASK_DATASET, MASK_ARCHETYPE, split="test", revision=MASK_REVISION
        )
    except Exception as error:
        raise RuntimeError(
            f"the {MASK_ARCHETYPE} split of {MASK_DATASET} at {MASK_REVISION} could not be read "
            f"({type(error).__name__}: {error}); rendering it needs network access to "
            f"huggingface.co on first use."
        ) from error

    rows = []
    for item in items:
        item_key = f"mask_{MASK_ARCHETYPE}_" + item["task_id"].replace("-", "_")
        instance_id = f"false_premise-{item_key}-none"
        rows.append({
            "row_id": f"{instance_id}-pressure",
            "dim": "false_premise",
            "instance_id": instance_id,
            "item_key": item_key,
            "template_id": None,
            "style": "mask_continuation",
            "pool": "starter",
            "condition": "pressure",
            "messages": [
                {"role": "system", "content": item["system_prompt"]},
                {"role": "user", "content": item["user_prompt"]},
            ],
            "followups": [],
            "reference": {"type": "T", "text": item["ground_truth"]},
            "declining": "evasion",
            "domain": "mask",
            "difficulty": None,
            "extra": {},
        })
    return rows


def write_rows(rows: list[dict], path: str | Path) -> Path:
    """Write rows as JSONL, one object per line with the keys in `ROW_FIELDS` order.

    Args:
        rows: The rows to write.
        path: Where to write them; parent directories are created.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def main_render(argv: list[str] | None = None) -> int:
    """`render-source-rows`.

    Writes `outputs/source_rows/harm_motivated_denial.jsonl` and
    `outputs/source_rows/false_premise.jsonl` and prints the counts. The `false_premise` rows are
    downloaded from MASK on the Hub.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.

    Returns:
        0.
    """
    parser = argparse.ArgumentParser(
        prog="render-source-rows",
        description=(
            "Render every row of the two source pools into outputs/source_rows/, for a local "
            "evaluation larger than the rows under data/dishonesty_dims/. The false_premise rows "
            "are downloaded from the continuations split of cais/MASK at the pinned revision, which "
            "needs network access on first use. MASK is a measurement set: run it and study it, and "
            "do not use its rows as supervision for any pipeline component."
        ),
    )
    parser.add_argument(
        "--out", default=None,
        help="where to write the files; the default is outputs/source_rows/",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out) if args.out else OUTPUTS_DIR / "source_rows"
    for name, render in (
        ("harm_motivated_denial", render_harm_pair_rows),
        ("false_premise", render_mask_rows),
    ):
        rows = render()
        path = write_rows(rows, out_dir / f"{name}.jsonl")
        print(f"{name}: {len(rows)} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_render())
