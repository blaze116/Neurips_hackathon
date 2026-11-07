# honesty_fewshot_v0

## Summary
A black-box honesty pipeline. An honesty directive in the system prompt instructs the (reasoning)
model to resist any built-in pressure to lie, flatter, or conceal, and to state its true assessment.
A small curated pool of honest-vs-dishonest demonstrations is injected few-shot, the last user turn is
marked with `|BE_HONEST|`, and decoding lets the model finish its reasoning before answering.

## Controls
- `SystemPrompt(mode="prepend")`: prepends the honesty directive to the row's system prompt.
- `FewShot(selector="random", system_mode="append")`: appends 3 positive and
  2 negative demonstrations; `k` equals the pool size, so the whole curated set is
  always injected (deterministic).
- `UserPrefix(placement="last_user", separator=" ")`: prefixes the last user turn with `|BE_HONEST|`.
- `PhasedDecoding`: `generate until <close-think tag>` then prefills `Honest answer:` -> `generate`. Per-model
  hyperparameter (the close-think tag, from the starter kit `REGIME`):
  - `Qwen/Qwen3.8-27B`: phase 1 stops at `</think>`
  - `google/gemma-4-31B-it`: phase 1 stops at `<channel|>`
  - `ibm-granite/granite-4.2-30b`: phase 1 stops at `</think>`

## Data
Few-shot pool: 5 demonstrations written by hand by the team,
covering honest behavior under pressure (positives) and sycophancy/concealment (negatives). No row is
drawn from `false_premise` (MASK). EPR (Rubin et al. 2021) was used offline only, with a small local
scoring model, to rank and select which hand-written examples to keep; no learned EPR artifact ships
in the submission. Nothing else was fitted, tuned, or trained on competition data.
