# NullSpaceMap — Team Trust-Issues

## Summary

A learned, input-dependent activation shift at one middle decoder layer that pushes the model toward the honest
answer only when the prompt carries pressure to be dishonest, and is constructed to leave ordinary prompts untouched.

At inference the control reads the residual stream at the last prompt token, `h`, and adds `s = λ · A (Bᵀ h)` to every
answer token (prompt tokens are not modified). `B` ([H, 12]) are *read* directions that lie in the null space of a
benign set of prompts (capability questions, guard requests, and pressure-free versions of the honesty scenarios):
directions those prompts do not use, so for them `Bᵀh ≈ 0` and the shift vanishes. `A` ([H, 12]) are *write*
directions, one per dishonesty category, pointing from that category's dishonest-answer state to its honest-answer
state. The rank-12 map therefore routes each pressure prompt to its own category's correction from the prompt alone.
The map is fitted in closed form (ridge regression in null-space coordinates, after AlphaSteer, Sheng et al. 2026);
the categories need separate directions because on our contrastive pairs a single pooled honesty direction reproduces
only 3–5% of the honest-minus-dishonest contrast, per-category directions 50–60%.

Per-model hyperparameters (same control sequence on all three models):

| model | layer | λ |
| --- | --- | --- |
| Qwen/Qwen3.8-27B | 32 of 64 | 2.0 |
| google/gemma-4-31B-it | 30 of 60 | 2.0 |
| ibm-granite/granite-4.2-30b | 38 of 64 | 1.5 |

The layer is the deepest one at which the null space still covers ≥ 98% of held-out capability and guard prompts
while the weak categories (instructed lie, owning up, concealment, stated sandbagging) are learnable, chosen from a
per-layer profile of each model. λ was chosen by a teacher-forced sweep on the eval split (largest dose with the honest
answer gaining probability and the model's own answers on capability/guard prompts moving by ≤ 0.05 nats per token).

## Controls

One state control per bundle, `state_control/nullspace_map` (`controls/nullspace_map.py`), a `HookControl` of the
steerability toolkit. Artifacts: two `SteeringVector`s per bundle (`read` = Bᵀ, `write` = Aᵀ, each `[12, H]` at one
layer, safetensors), plus `lam` and `layer_ids`. The control is hook-only (no intervention-spec form) and runs on the
Hugging Face backend. White-box track. Bundles were fitted and frozen in bf16 on an A100 80 GB with the pinned model
revisions; no model weights are changed.

## Data

Fitting used only the team's own dataset (`data/v7_tidy`, train split), built from the Starter Kit's dishonesty
dimensions and templates plus team-written scenarios:

- **Pressure prompts and targets:** 950 scenarios (80 per dishonesty category, 12 categories), each with a
  contrastive pair of answers — an honest and a dishonest answer to the same prompt that take the same kind of action
  and differ only in truthfulness. Activations of the two answers (mean over answer tokens, appended to the prompt
  after the model's own empty reasoning block) give each category's target direction.
- **Benign set (3,524 prompts, prompt-only):** the pressure-free twins of the honesty scenarios, 480 capability and
  guard prompts (subsets of MMLU-Pro, BIG-Bench Hard, GSM8K, IFEval, AlpacaEval, XSTest, and team-written
  confidentiality personas), and 1,733 further prompts from the same public sources (disjoint from every evaluation
  row we hold). These define what the control must leave unchanged; nothing is fitted *toward* them.
- The MASK benchmark was not used for fitting, tuning, or selection. The Starter Kit's `harm_pressure_pairs`
  (Anthropic, via the kit) appear only as the source of the harm-motivated-denial scenarios.

Prompts were rendered exactly as the evaluation renders them (full chat, system message, the kit's sandbagging suffix,
thinking enabled).
