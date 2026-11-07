# AGENTS.md

## What this repository is

The starter kit for the steerability challenge. It contains the organizers' baselines as
submissions, the data, a submission checker, and two notebooks.
Everything here is public. It contains none of the private evaluation set, no judge prompts, and not
the evaluation's seed. A number measured here therefore indicates a direction rather than a score.

The work done here is building a submission, i.e., a steering method frozen for three base models
that passes `check-submission`.

## Competition rules for agents
- This workspace must stay private. Never fork, push to, or open pull requests or issues against
  `steerability-challenge/steerability-challenge-starter-kit` or `generative-computing/steerability`.
- Never create public repositories, gists, or Hugging Face repos; Hub uploads use the team org with `private=True`.
- Custom controls live in `controls/` and register via `register_method`; do not add them to the toolkit tree.

## Layout

```
README.md                      participant-facing overview, baselines table, submission steps
AGENTS.md                      this file
pyproject.toml                 uv project, Python 3.12; deps: steerability (pinned), pandas, jupyter,
                               datasets, huggingface_hub
.gitignore                     outputs/, baselines/, .venv, tests/
baselines/                     fetched by fetch-baselines; nothing under it is tracked
  <name>/<model-slug>.spipe    one frozen bundle per base model
  <name>/method_card.md        the baseline's method card
data/
  README.md                    what each file is, the row schema, the sources
  formats/, handwritten/, bank/, templates/, guards/, factsheets/
  third_party/                 NOTICE.md and harm_pressure_pairs.jsonl
notebooks/
  baselines_qwen.ipynb         run every Qwen baseline over the formats (one 80 GB GPU)
  submission_check.ipynb       check the baselines as submissions (CPU)
starterkit/
  __init__.py                  the constants: base models, FORMATS, REGIME, ceilings, size cap,
                               tracks, JUDGE_MODEL, JUDGE_SETTINGS, MASK_REVISION, BASELINES_REPO,
                               BASELINES_REVISION, model_slug
  rows.py                      load and select rows, build chats, apply followups, render a pool
  baselines.py                 fetch the baselines from the Hub (module and CLI)
  generate.py                  run one bundle over rows under REGIME, split thinking
  track.py                     the track rule
  check.py                     the submission checker (module and CLI)
  judge.py                     the honesty judge, loaded in process or over an OpenAI-compatible
                               endpoint
```

## The toolkit pin

`pyproject.toml` pins `steerability` to a commit that provides three things:
`steerability.algorithms.core.registry.register_method`, which lets a submission include its own
control; a model-free `SteeringPipeline.to_spipe(freeze=True)` for a pipeline with nothing to fit;
and `FewShot(system_mode=...)`, without which the revised `honesty_few_shot` bundles do not load.
Nothing to fit means every enabled control declares `steer_access() == FACTS`, declares no
`steer_fits()`, and exports an empty `export_state()`. The checker depends on `register_method` and
the model-free freeze, and the `honesty_few_shot` bundles in `baselines/` depend on `system_mode`.
Changing the pin requires re-running both notebooks.

## Commands

```
uv sync                        install the project and its pinned toolkit
fetch-baselines                download the baselines from the pinned Hub revision
render-source-rows             render both source pools into outputs/source_rows/
check-submission <dir>         check a submission directory (CPU, no weights, no network)
jupyter lab notebooks/         the notebooks
```

## A submission

A directory containing:

- one `<model-slug>.spipe` per base model, each frozen (`to_spipe(freeze=True)`) and
  self-contained (`save(..., artifacts="fat")`), with the same control sequence across the three
  models. Constructor arguments may differ per model. The slug replaces every character outside
  `[A-Za-z0-9._-]` with `__`, giving `Qwen__Qwen3.8-27B` for `Qwen/Qwen3.8-27B`.
- `method_card.md`, with the sections below. It must exist with content, and nothing in it is parsed.
- optionally a `controls/` package that registers its controls at import, for a submission that
  includes code. `check-submission` imports it before loading any bundle.

The track follows the rule in `starterkit/track.py`, which reads each control's class, i.e., its base
class, its `steer_access()`, and its `requirements()`. There are three nested tracks: `black-box` for
pipelines that do not access model internals, `white-box` for states and logits, and `open` for
weights. A submission is placed in the highest track any of its controls requires. The checker
reports the track it computes, per control and with a reason. Detection is a floor, i.e., the
organizers can place a submission higher after reading the card and the code, never lower.

## Method card sections

```markdown
## Summary
Two or three sentences: what the method does and why it should reduce dishonesty under pressure.
## Controls
One bullet per control in pipeline order; per-model hyperparameters where they differ.
## Data
Every dataset, pool, or prompt source that supervised a component, with its origin. MASK does not
appear here. `None.` where nothing supervised anything.
```

## Rules for agents working here

- **Treat `baselines/` and `data/` as read-only.** They are the challenge's own artifacts. The
  baselines are what a submission is measured against, and the rows are what it is measured on. A
  local edit to either silently invalidates every comparison made against them.
- **Constants belong only in `starterkit/__init__.py`.** Base models, the generation regime, the
  token ceilings, the size cap, the track names, the judge model and its settings, the MASK revision,
  and the baselines repository and revision are read from there by every module and both notebooks. Do not restate a value anywhere else. The evals repository's
  `tests/starterkit/test_constants.py` checks these values against the evaluation config.
- **Work from the bundles that are present.** The baselines, cards and bundles, are fetched from the
  pinned Hub revision, and nothing under `baselines/` is tracked. Glob `baselines/*/<slug>.spipe`
  rather than hard-coding a list of baseline names. Adding a baseline requires a Hub directory, a
  README row, and a new revision, with no code change.
- **Everything under `data/` may be used to build a submission, and everything except the
  `false_premise` rows may also be used as supervision.** The Harm Pressure pairs under
  `data/third_party/` carry their upstream terms (`data/third_party/NOTICE.md`).
- **MASK is measurement-only.** The `false_premise` rows are MASK items. Run MASK as a development
  evaluation and write original pressure prompts from its archetype taxonomy; never use its items as
  supervision for any pipeline component, including the six in
  `data/formats/false_premise.jsonl`. The method card's Data section lists what supervised each
  component, and MASK may not appear in it.

## Releasing

Bump `pyproject.toml`, set `BASELINES_REVISION` to the Hub commit that contains the baselines for
this release, add the release-table row in `README.md`, commit, tag `vX.Y.Z`, create a GitHub
Release, and announce. Point the README download link at the latest release.
