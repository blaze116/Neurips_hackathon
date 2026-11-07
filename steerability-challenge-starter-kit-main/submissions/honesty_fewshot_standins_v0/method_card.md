# honesty_fewshot_standins_v0

## Summary
Local stand-in build of `honesty_fewshot_v0` (not for upload). A black-box honesty pipeline: an honesty
directive in the system prompt, a small curated few-shot pool of honest-vs-dishonest demonstrations, a
`|BE_HONEST|` marker on the last user turn, and decoding that lets the model finish reasoning before answering.

## Controls
- `SystemPrompt(mode="prepend")`: prepends the honesty directive.
- `FewShot(selector="random", system_mode="append")`: 3 positive and 2
  negative demonstrations; `k` equals the pool size, so the whole pool is always injected.
- `UserPrefix(placement="last_user", separator=" ")`: prefixes the last user turn with `|BE_HONEST|`.
- `PhasedDecoding`: `generate until <close-think tag>` then prefills `Honest answer:` -> `generate`.
  - `Qwen/Qwen3.5-9B` (stand-in for `Qwen/Qwen3.8-27B`): phase 1 stops at `</think>`
  - `google/gemma-4-12B-it` (stand-in for `google/gemma-4-31B-it`): phase 1 stops at `<channel|>` (unverified: tokenizer not cached)
  - `ibm-granite/granite-4.2-8b` (stand-in for `ibm-granite/granite-4.2-30b`): phase 1 stops at `</think>`

## Data
Few-shot pool: 5 demonstrations written by hand by the team. No row is
drawn from `false_premise` (MASK). Nothing was fitted, tuned, or trained on competition data.
