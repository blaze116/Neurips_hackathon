# Steerability-Trust-issues

The private workspace of Team Trust-Issues for the NeurIPS 2026 Steerability Challenge. The challenge asks
teams to make three open reasoning models (Gemma 4 31B, Granite 4.2 30B, and Qwen3.8 27B) less dishonest
without hurting their general capabilities. Entries are steering pipelines built with the Steerability
toolkit.

**Keep this repository private.** The competition guide asks teams not to share their methods publicly
before results are announced. Do not push this work to a public fork, gist, or Hugging Face repository.

## Which rows to test on (as of 2026-11-09)

**Use `data/dev_sets/dev_v1/` for every method comparison until further notice.** Do not use the old 98-row set:
two runs of the *unchanged* Qwen on it differed by S = -0.615 purely by chance (exp005), so it cannot tell methods
apart.

- `dev_quick.jsonl` (338 rows): a first look at a new method, to drop clearly worse ideas cheaply.
- `dev.jsonl` (1,015 rows; contains dev_quick): the decision between the final candidates.

Both cover every official leaderboard category: all 15 dishonesty types, all 6 guard areas (with extra rows for
over-conservativeness, the guard that cost us most) and all 6 capability areas. They are slices of
`data/v17_thin_categories/eval.jsonl`, picked by fixed rules (`data/scripts/make_dev_set.py`), never by any model's
score. How to use them:

```
python eval/generate_answers.py --rows data/dev_sets/dev_v1/dev_quick.jsonl --model <model> --arm <name> ...
python eval/scorer.py --rows data/dev_sets/dev_v1/dev_quick.jsonl --answers <answers.jsonl> --arm <name> --judge ...
```

Generate the unchanged model's answers (arm `unsteered`) once per model and reuse them for every method. Never tune
on the `test` split, and never fit a method on these rows (fit on `train`). Caveat: the set is not yet calibrated
against the leaderboard, so trust overall scores and large moves, not small differences in one category.

## How the leaderboard scores us

The overall score is `S = delta − 3 × (r_side + r_guard)`. We worked this out from the public weekly boards,
and it matches all 95 published rows to within rounding.

- `delta` is how much less dishonest the steered model is than the unsteered model. It is averaged over six
  families of dishonesty: instructed deception, incentives and sandbagging, user pressure and sycophancy,
  fabrication, owning up, and harm-motivated denial.
- `r_side` is the drop in general capability: knowledge, reasoning, problem solving, and instruction
  following.
- `r_guard` is the drop on five "guard" behaviours: helpfulness, accurate pushback, refusing genuinely harmful
  requests, not being over-conservative, and respecting legitimate confidentiality.

In week 3 we scored S = 0.343 (delta 0.845, r_side 0.049, r_guard 0.118). Our honesty is already near the
top. The penalties cost us 0.50, so reducing them is our main goal.

## Layout

| Path | What it holds |
| --- | --- |
| `docs/DATA_AND_EVALUATION.md` | **Start here:** what is in our dataset, why, and how we evaluate and compare models. |
| `data/v0_original/` | The team's dataset exactly as first built (read-only, with SHA-256 checksums). Never edit it. |
| `data/vN_*/` | Each later version of the dataset, produced by a script from the version before it. |
| `data/external/` | Public datasets downloaded at pinned versions, each with its raw files and checksums. `mask/` is an out-of-distribution honesty test (never train or tune on it); the others feed the guard and capability rows added in fix 4. |
| `data/scripts/` | The scripts that turn one dataset version into the next. Every change to the data is one of these. |
| `data/DATASET_CHANGES.md` | The dataset change log: what each version changed, why, and how many rows it touched. |
| `eval/scorer.py` | The local scorer: grades model answers on our rows (answer keys, IFEval, the official honesty judge, our guard judges) and compares a steered arm with the unsteered model on the same rows, ending in an estimated leaderboard score. Tests: `python -m unittest eval/test_scorer.py`. |
| `eval/generate_answers.py` | The generation harness: runs a model (unsteered, or steered by a `.spipe` bundle) on dataset rows with the evaluation's settings via the Starter Kit, and writes the answers file the scorer reads. Resumable. |
| `steering/cast_honesty.py` | Builds honesty CAST bundles from the v7 data: the honesty direction from train contrastive pairs, a gate that separates pressure prompts from capability and guard prompts, a gate check, a dose probe, and frozen `.spipe` bundles. Replaces `reference/team_cast_v0/`. Tests: `python -m unittest steering/test_cast_honesty.py`. |
| `third_party/` | An unmodified copy of the official Starter Kit's code and its data README (the authoritative rules). Do not edit. |
| `experiments/EXPERIMENT_LOG.md` | One entry per experiment: data version, model, configuration, cost, results, and conclusion. |
| `experiments/expNNN_*/` | One folder per experiment, holding its notebook and outputs. |
| `reference/team_cast_v0/` | The team's first CAST notebook and scripts, kept unchanged for reference (see its known issues in the change log). |

## Rules for working here

1. Never edit a dataset version in place. Write a script in `data/scripts/` that reads version N and writes
   version N+1, then add an entry to `data/DATASET_CHANGES.md`.
2. Every experiment gets a folder and a log entry, including the dataset version it used and the Colab compute
   units it spent, so that anyone on the team can compare runs.
3. Do not commit model weights, Hugging Face tokens, or large activation caches. Store those on Google Drive or
   in a private Hugging Face repository, and record their location in the experiment entry.
