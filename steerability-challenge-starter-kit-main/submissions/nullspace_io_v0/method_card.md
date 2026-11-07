# nullspace_io_v0

## Summary
Local stand-in build for `Qwen/Qwen3.5-9B` (not for upload). The `null_space_qwen` NullSpaceMap with black-box input
and output controls around it. White-box track, Hugging Face backend.

## Controls
Input:
- `SystemPrompt(mode="prepend")`: prepends an honesty directive.
- `UserPrefix(placement="last_user", separator=" ")`: prefixes the last user turn with `|BE_HONEST|`.

State (artifacts unchanged from `null_space_qwen`):
# NullSpaceMap (exp_rahul_001)

A learned rank-12 activation map at decoder layer(s) [16] of Qwen/Qwen3.5-9B: read directions in the null space of capability/guard/twin prompts, write directions toward each dishonesty dim's honest-minus-dishonest answer state; shift computed from the last prompt token and added to answer tokens, dose lam=2.0. Fitted on data/v7_tidy train (80 pairs/dim). White-box, HF backend.

The `controls/` package here is a driver-safe copy of `null_space_qwen/controls/`: under a decoding driver each
multi-token forward is treated as a fresh prefill, and the shift is read at the original prompt boundary.
Without a driver it is identical to the original.

Output:
- `PhasedDecoding`: generate until `</think>` -> fixed `Honest answer:` -> generate.

## Data
NullSpaceMap fitted on data/v7_tidy train (see above). The input/output controls are hand-written; nothing else
was fitted or tuned. The map was fitted without these input controls.
