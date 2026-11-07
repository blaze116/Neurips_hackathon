# Data provenance (dataset version v7_tidy, 2026-10-29)

Every row's `source` field names where it came from. This page lists each source, its license, and how we may use
it. It replaces the team's earlier note (`reference/team_cast_v0/PROVENANCE.txt`), whose data list was accurate but
which said MASK was not used, while the team's first CAST configuration fitted its gate on MASK prompts.

## Sources of the rows

| `source` | Rows | What it is | License and terms | May supervise a pipeline? |
| --- | --- | --- | --- | --- |
| `team_v0` | 4,759 | The team's original honesty dataset: facts written by the team (LLM-drafted, human-checked), invented secrets, short texts for feedback; hint questions from CommonsenseQA (Talmor et al., 2019, research use) and sandbagging questions from GSM8K (MIT). | ours, plus the licenses named | yes |
| `anthropic_harm_pressure (via the Starter Kit)` | 234 | The Harm Pressure pairs from Wang et al. (2025), as adapted by the organisers and shipped in the Starter Kit. | released for research; the kit allows use as supervision | yes |
| `team_template_v1` | 1,080 | Rows written by template in fix 4 (12 personas for the confidentiality guard) and fix 5 (fabrication). Fake names were checked against Wikipedia. | ours | yes |
| `TIGER-Lab/MMLU-Pro@b189ec7` | 180 | Knowledge questions. | MIT | gate negatives only, by our choice |
| `lukaemon/bbh@982bb89` | 180 | BIG-Bench Hard reasoning tasks. | no license stated; keep private | gate negatives only, by our choice |
| `openai/gsm8k@740312a` | 180 | Maths word problems. | MIT | gate negatives only, by our choice |
| `google/IFEval@966cd89` | 180 | Instruction-following prompts. | Apache-2.0 | gate negatives only, by our choice |
| `tatsu-lab/alpaca_eval@2edc6fa` | 180 | Everyday requests for the helpfulness guard. | CC-BY-NC-4.0 (non-commercial); keep private | gate negatives only, by our choice |
| `Paul/XSTest@f600c99` | 360 | Safe-but-alarming and genuinely unsafe prompts. | CC-BY-4.0 | gate negatives only, by our choice |

"Gate negatives only" means the capability and guard rows are never used to fit an honesty behaviour; their train
rows may serve as prompts a CAST gate must not fire on, and their eval and test rows measure regressions.

## Measurement-only data (never supervision)

- **MASK** (`data/external/mask/`, `cais/MASK@4602b84`, no license stated). The Starter Kit allows MASK only as a
  development evaluation. It may never be training data, contrastive pairs, steering-vector examples, few-shot
  examples, or the objective of an automated prompt or hyperparameter search (for example GEPA, or a sweep that picks
  the best MASK score), and the method card may not list it. The official `false_premise` dim is built from MASK, so
  the same rule covers the kit's `false_premise` rows. Nothing built with the team's first CAST configuration
  (`reference/team_cast_v0/`) may be submitted, because its gate was fitted on MASK.

## Code

- `third_party/starterkit/`: the official Starter Kit (private to registered teams; keep this repository private).
- The steerability toolkit: Apache-2.0, installed from the organisers' release (`steerability-main/`).
- `instruction_following_eval` (the IFEval checker): Apache-2.0, pinned in `eval/requirements.txt`.
- Everything in `data/scripts/` and `eval/` is written by our team.
