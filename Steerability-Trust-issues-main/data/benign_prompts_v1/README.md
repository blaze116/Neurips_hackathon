# benign_prompts_v1 — extra "don't steer" prompts (auxiliary, prompt-only)

*Built 2026-10-31 by Rahul with `data/scripts/build_benign_prompts.py`. Not a dataset version: `v7_tidy` remains
the data for fitting, tuning and scoring. These rows are never scored.*

## Why this exists

Steering methods that promise not to touch ordinary prompts (a CAST gate, or the null-space constraint of
`experiments/exp_rahul_001_nullspace_map`) can only keep that promise on the *kind* of prompts they were shown.
v7's train split has 480 capability/guard rows. When the null-space map was profiled on Qwen3.8-27B (4-bit) at
layer 32, the share of a **held-out** protected prompt's activation that the map cannot see (PEC) was still rising
with the number of benign rows: 0.967 at 250 rows, 0.975 at 500, 0.982 at 1,000, 0.983 at 1,186, with
`capability_instruction_following` and `guard_helpfulness` the weakest areas. AlphaSteer, the method this
construction follows, used ~14k benign prompts. This folder adds 1,733 more ordinary prompts of the same kinds.

## What is in it

`benign_prompts.jsonl`, one row per line: `row_id`, `dim` (`benign_<area>`), `condition: "neutral"`, `messages`
(a single user turn), `source` (dataset@revision), `domain`, `scored_as: []`, `purpose: "benign_span"`.

| dim | rows | source (same pinned files fix 4 used, `data/external/`) |
| --- | --- | --- |
| benign_helpfulness | 625 | tatsu-lab/alpaca_eval — every item not already a v7 row |
| benign_instruction_following | 361 | google/IFEval — every item not already a v7 row |
| benign_knowledge | 292 | TIGER-Lab/MMLU-Pro — 300 spread over categories, minus v7 overlaps |
| benign_reasoning | 197 | lukaemon/bbh — 200 spread over tasks, minus v7 overlaps |
| benign_problem_solving | 168 | openai/gsm8k test — 200, minus v7 overlaps |
| benign_over_conservativeness | 70 | Paul/XSTest safe prompts not already a v7 row |
| benign_refusal | 20 | Paul/XSTest unsafe prompts not already a v7 row (kept so the method is also blind to requests the model must keep refusing) |

Prompts are rendered with fix 4's own templates (same final-line instructions), so they look exactly like the
protected rows the scorer uses. **Every prompt that equals a v7 row in train, eval or test is excluded** (the build
asserts this), so eval and test protected rows remain unseen by anything fitted on this set.

## Rules

- Prompt-only: no answer keys, no grading, no references. Nothing here is a target; it only defines what a method
  must leave unchanged.
- Never used to fit an honesty direction or behaviour, never scored, never used for tuning a threshold against
  (there is nothing to score).
- MASK is not used.
- Rebuild: `python data/scripts/build_benign_prompts.py` (deterministic; `SHA256SUMS` lists the output digest).

## How it is used

`experiments/exp_rahul_001_nullspace_map/fit_nullspace_map.py --benign-extra data/benign_prompts_v1/benign_prompts.jsonl`
adds these rows to the benign set (train twins + train protected rows → ~2.9k rows). A CAST gate's "stay closed"
set can use it the same way.
