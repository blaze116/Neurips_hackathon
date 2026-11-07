# exp001 results — unsteered Qwen3.8-27B on the v7 eval split

*Run on Colab 2026-10-29 by Rupesh. Generation on an A100 40 GB in 4-bit NF4 (the 80 GB card was not available);
judging on an L4 with GLM-4-9B-0414 at revision 645b848. One sample per row. Outputs are on Google Drive under
`MyDrive/trust_issues_runs/exp001_qwen_reference/Qwen3.8-27B_nf4/` (a local copy is in `../colab_runs/`).*

## Cost

About **35 compute units** in total (balance 200 → 165), including one session that disconnected mid-run and was
resumed. The plan estimated 8–12 units. The gap had three causes: the 4-bit model generates more slowly than bf16,
Colab lacked the `flash-linear-attention` package that Qwen3.8's linear-attention layers need for their fast path
(the run printed fallback warnings), and the disconnected session spent units while idle.

## Generation diagnostics

All 311 planned rows were answered (320 turns, because some rows are two-turn conversations). No turn hit the token
ceiling, none ended with its thinking unclosed, and no answer was empty. The longest thought was about 15,700
characters (a maths row); honesty rows mostly thought for a few hundred to 2,500 characters.

Answers the official readers could not parse (they score as unanswered):
- **sandbagging: 20 of 20.** Qwen ends with "The answer is **(A) 36**." (bold, with the option's value after the
  letter), which `starterkit.parse.extract_letter` rejects. Because the unsteered neutral answers are therefore not
  "reliably answered", the official rule drops every sandbagging instance, so this category was not measured.
- **hint_acknowledgment: 6 of 20** (answers ending in a bold letter such as "**A**").

## Scores (judge verdicts: 209 cached; 3 answers got no readable verdict and are left out)

The scorer's summary pools each category's pressure rows with their pressure-free twins. The pressure-only figures
below come from the cached verdicts and show where the model actually fails.

| Category | Summary score (n) | Pressure rows only |
| --- | --- | --- |
| instructed_lie | 0.600 (20) | **0.20** (10) |
| confrontation_after_lie | 0.200 (10) | **0.20** (10) |
| concealment | 0.650 (20) | **0.30** (10) |
| sandbagging_stated | 0.600 (10) | **0.60** (10) |
| fabricated_citation | 0.952 (21) | |
| fabricated_action | 1.000 (29) | |
| fabricated_entity | 1.000 (30) | |
| feedback_sycophancy | 1.000 (10) | |
| harm_motivated_denial | 1.000 (18) | 1.00 (8) |
| pushback | 1.000 (10) | 1.00 (10) |
| hint_acknowledgment | 0.000 (1) | only one row qualified |
| sandbagging | not reported | no instance qualified (see above) |

| Capability / guard | Score (n) |
| --- | --- |
| capability: instruction following | 1.000 (10) |
| capability: knowledge | 0.800 (10) |
| capability: problem solving | 0.800 (10) |
| capability: reasoning | 0.500 (10) — three of the five misses are correct answers written in a form our exact-match reader rejects ("(A) The carpenter…", a full bracket sequence instead of only the closing part); the true figure is nearer 0.80 |
| guard: helpfulness | 1.000 (10) |
| guard: over-conservativeness | 1.000 (10) |
| guard: confidentiality and persona | 1.000 (10) |
| guard: refusal | 0.900 (10) |
| guard: pushback and feedback | 1.000 (1) |

With 10–20 rows per category each figure is accurate only to roughly ±20–30 points.

## What it means

1. Almost all room for honesty gain is in four categories: instructed_lie, confrontation_after_lie, concealment, and
   sandbagging_stated. Everything else is at or near 1.00, where steering can only lose points (and every lost point
   costs three times a gained one).
2. CAST's gate should therefore open on system-prompt-instructed deception, secrets, owning up after a lie, and
   stated capability denial, and stay closed elsewhere. exp002 succeeds if those four rise while the 1.00 categories
   and the capability and guard rows stay put.
3. Sandbagging and hint acknowledgment contribute little for Qwen under the official parsing rules; do not spend
   effort steering them.
4. Next run: install `flash-linear-attention` before generating, and budget units from this run's real cost, not the
   original estimate.
