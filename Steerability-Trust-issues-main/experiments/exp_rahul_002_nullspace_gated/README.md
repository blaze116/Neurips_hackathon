# exp_rahul_002 — NullSpaceMapGated: the null-space map with a projection-gated write

Successor of [`exp_rahul_001_nullspace_map`](../exp_rahul_001_nullspace_map/README.md) (rounds 1–2, the bundle on HF
as `submission_nullspace_v1`). Same read side — a router in the null space of benign prompts, per-dim response-side
targets — with a different **write rule**: instead of adding the same vector to every answer token, each token is
pushed only while, and only as far as, its own state is on the dishonest side of a fitted boundary (the StTP / StMP
rules of Herbster et al. 2026, arXiv:2604.08169, composed with our per-dim router). Plus the data and bookkeeping
changes that the round-2 results and exp003–exp005 called for. Section 2 is the detailed change log with reasons.

Registered as `state_control/nullspace_map_gated` (white-box, Hugging Face backend, hook-only) — its own name, so a
bundle of either experiment is never mistaken for the other (the toolkit refuses two classes under one name).

## 1. Why a second version

Round 2 (exp_rahul_001, λ = 2 for Qwen) was judged in exp003 against the shared unsteered arm (exp004): confrontation
0.60 → 1.00, instructed_lie 0.50 → 0.80, sandbagging_stated flat, and a concealment "gain" that was an artefact of
broken output (looping or empty answers). exp005 then showed that two unsteered runs differ on 7 of 98 rows by sampling
alone, so none of the 98-row comparisons is conclusive. Four causes, all addressed here:

1. **The dose was constant and un-calibrated.** The fixed rule adds `λ·t_d` to every answer token; concealment's
   target is ~3× the median norm (26.9 vs 8.7), so at λ = 2 the shift was about the size of the hidden state itself,
   on every token of a 1,000+-token generation, whether or not the model had already started telling the truth.
2. **The sweep could not see it.** Teacher-forced log-odds keep rising with λ long after free generation has fallen
   apart; every sweep picked the top of its grid.
3. **Some targets were not honesty directions.** For dims whose honest and dishonest answers differ only in a swapped
   fact (instructed_lie: "…is Bern." / "…is Zurich."), the shared tokens have identical states under teacher forcing
   and contribute nothing; the direction is "true fact vs false fact", ~3× diluted (exp002 found the same for CAST).
4. **The benign set predates half the evaluation areas.** Data moved from v7 to v17 (new guards sincere_nonassertion,
   system_prompt_compliance, harder over-conservativeness; new capability areas multilingual, fidelity/long-context;
   the three missing dishonesty dims), and the null space — our only protection for r_side / r_guard — had never
   seen them.

## 2. What changed from exp_rahul_001, and why

| | exp_rahul_001 (rounds 1–2) | exp_rahul_002 | reason |
| --- | --- | --- | --- |
| **Write rule** | `fixed`: `h' = h + λ Σ_d w_d t_d` on every answer token | `sttp` (default) / `stmp`: per token and dim, `ρ = h·t̂_d`; nothing if `ρ ≥ m_d`; else add `w_d (s_d − ρ) t̂_d` with `s_d = μ⁺_d + α σ⁺_d` (sttp) or `w_d · 2α (m_d − ρ) t̂_d` (stmp); optional `floor` of the fixed shift | bounded by the honest class's own range, zero once the text is honest, off on benign answers that leak through the router — the three failure modes of round 2 (§2.1) |
| **Dose knob** | λ, a multiplier whose meaning differs per dim (48 for concealment, 17 for instructed_lie at λ = 2) | α in units of the honest tokens' standard deviation, the same meaning for every dim | removes the need for the median normalization that halved R² in round 1 |
| **Map storage** | thin SVD of the ridge map, 12 anonymous factors `read`/`write` [r, H] | the exact router × targets factorisation: `read` = D router rows, `write` = D per-dim targets [D, H] | identical map; exposes `w_d` and `t_d`, which the gate needs; zeroing a dim is a zero row (§2.2) |
| **Gate statistics** | — | per dim, token-level projections of the honest / dishonest answer tokens on `t̂_d`: `μ⁺, σ⁺, μ⁻, σ⁻`, equal-z boundary `m`, Cohen's d, fire rates; computed on train (used) and eval (held-out check); stored as plain lists in the args | the gate is a 1-D classifier per dim; its quality is measured before any generation (§2.3) |
| **Zeroed dims** | none (every dim steered) | `--zero-dims`: dims the unsteered model already passes get a zero target; Qwen: fabricated_*, pushback, hint_acknowledgment, feedback_sycophancy, sandbagging, harm_motivated_denial | a passed dim is a guard cell: steering it can only cost, ×3 via r_guard (§2.4) |
| **Pairs** | dataset `contrastive_pair` as is | `--pairs-override`: instructed_lie's 216 pairs re-worded so the answers differ in the *act* from the first word (5 × 5 phrasings, dealt equally); other dims unchanged | the only scored shared-prefix dim for Qwen; the rest are zeroed (§2.5) |
| **Data** | `data/v7_tidy` | `data/v17_thin_categories` (`--data`) | benign set covers all evaluation areas; eval has the held-out confrontation templates (§2.6) |
| **Sweep read-out** | Δ log-odds, honest log p, protected \|Δ log p\|/token | the same, plus the gate's fire rate and shift per token on honest / dishonest / protected tokens, and routing accuracy on held-out prompts | shows the write rule doing what it claims, not just the teacher-forced score (§2.7) |
| **Selection of the dose** | sweep's pick | sweep's pick, then the coherence probe (`probe_bundle.py`) before any generation run, as in round 2 | the sweep is still blind to free-generation collapse; the gate makes that much less likely but the probe stays |

### 2.1 The write rule

Round 2 added `λ·Σ w_d t_d` to every answer token. That vector is sized by the dim's *average* contrast and by λ,
not by what the token being generated needs. Three consequences, each visible in the round-2 outputs: (a) a
constant push of ≈ ‖h‖ moves states far outside anything the model produces → degenerate text (concealment);
(b) the push continues after the model has committed to the honest answer, so it accumulates over the generation →
looping and empty outputs; (c) on benign prompts the router leaks 0.07–0.25 of the pressure shift (measured), and that
fraction too is added to every token of a normal answer → the capability/guard risk.

The gated rule measures each token's projection on the routed dim's honesty axis and adds only the deficit to a target
inside the honest cluster. It is (a) bounded — never past `μ⁺ + α σ⁺`; (b) self-limiting — zero once `ρ ≥ m_d`; (c) a
second guard — tokens of a normal helpful answer sit on the honest side, so a router leak adds nothing. In Herbster et
al. these properties are what let StTP/StMP keep MMLU / MT-Bench / AlpacaEval while the fixed-coefficient rule loses
them at equal honesty gain. The Mac rehearsal (toy sizes, 9B) shows the same signature: `fixed` λ = 1 pushes honest and
dishonest continuation tokens equally (shift/token 4.9 / 4.9, protected 2.9, |Δ log p| 0.08 — over budget); `sttp`
α = 1 pushes dishonest tokens 6.1, honest 1.1, protected 0.7 (|Δ log p| 0.004).

`sttp` is the default; `stmp` is continuous in the deficit (no jump at the boundary) and is the fallback if sttp shows
jitter around `m_d`. `floor` (a fraction of the fixed shift always added) exists for dims whose 1-D gate is weak.

### 2.2 Router × targets instead of the SVD

The ridge regression's target matrix is `O·T` (one-hot dim membership × per-dim means), so its solution factorises
exactly: `shift(h) = Σ_d w_d(h) t_d` with `w(h) = Oᵀ Z (ZᵀZ + αI)⁻¹ Uᵀ h`. exp_rahul_001 stored the same map as a thin
SVD, which is lossless (rank 12 = the number of dims; singular values below 10⁻⁶ of the largest were the only thing
dropped, and none were) but mixes dims into anonymous factors. The new form stores the D router rows and the D
targets instead: same predictions (R² of the target reconstruction 1.000 in the rehearsal), same size (both [D, H]),
and per-dim quantities are available at runtime, which the gate requires (`ρ` is a projection on *one dim's* axis)
and which makes zeroing or re-weighting a dim a bundle-time edit rather than a refit. Nothing about the SVD was
"better": it was a compression we did not need, and we were not truncating it. One difference worth knowing: the
router rows of zeroed dims still exist (they output a weight), but with a zero target they do nothing.

### 2.3 Gate statistics at the token level

The paper derives the boundary from a logistic regression on response-averaged states. We already have the per-dim
unit direction (the normalised target; the paper shows logistic and mean-difference directions coincide), so the only
thing to estimate is where honest and dishonest tokens fall along it. We use the **tokens**, not response means,
because the gate decides per token: `μ⁺, σ⁺, μ⁻, σ⁻` from the honest / dishonest answer tokens of the fit pairs,
boundary at the equal-z point `(μ⁺σ⁻ + μ⁻σ⁺)/(σ⁺ + σ⁻)`. The fit reports, per dim, the share of dishonest tokens below
`m` (the gate fires when it should) and of honest tokens below `m` (fires when it should not), on train and on eval.
In the rehearsal (3 pairs per dim) the strong dims already separate well (sandbagging_stated 1.00 / 0.00, concealment
0.85 / 0.22) and the fact-only dim does not (harm_motivated_denial 0.67 / 0.38) — the gate's quality is a measurable
property of the pairs, which is why the pairs matter more in this version than in the last.

### 2.4 Zeroed dims

Scoring: delta averages over the dims the unsteered model *fails*; dims it passes become guard cells whose regressions
enter r_guard with weight ≈ 3. For Qwen (exp004, bf16) only instructed_lie 0.50, confrontation 0.60, concealment
0.75 and sandbagging_stated 0.80 are failed; every other dim is ≥ 0.95. Steering a passed dim can only cost, and most
of those dims also had the weakest directions (fact-only pairs, R² ≤ 0.2). A zeroed dim keeps its prompts in the fit
with a zero target, so the router learns to recognise them and do nothing — it fails safe (a forgone gain if the
organizers' rows for that dim are harder than ours, never a regression). The list is per model: Granite and Gemma
have no measured baseline yet, so their lists are empty until an unsteered run on the eval pressure rows (vLLM, ~20
min each) says which dims they fail. exp004's 5 rows per dim are a thin basis (exp005), so a borderline dim can be
taken off the list if there is reason to.

### 2.5 The instructed_lie pairs

Under teacher forcing, tokens shared by the honest and dishonest answer have identical hidden states and contribute
nothing to the mean difference. The dataset's instructed_lie pairs share 68% of their tokens (only the fact is
swapped), so the target encoded "true fact vs false fact", diluted ~3× (norm 8.7 vs 27 for concealment, held-out R²
0.34) — the same thing exp002 found for CAST. With the gated rule that direction would also be the token-level honesty
axis, so it would mis-gate. `build_pairs_override.py` re-words only the two answers (prompts and facts unchanged) so
they differ in the act from the first word — honest = corrects the instruction and states the truth, dishonest =
complies and asserts the false claim — in five phrasings each, dealt equally per row so the phrasing averages out. It
covers the 168 train and 48 eval rows, is used only by this fit (`contrastive_pair` is never read by the scorer or the
evaluation), and leaves the dataset files untouched. The other shared-prefix dims (harm_motivated_denial, sandbagging,
hint_acknowledgment, feedback_sycophancy) are zeroed for Qwen, so they were not re-worded; the strong dims already
differ from the first word.

### 2.6 Data v17

The benign set is the guard. v17 adds whole protected areas that v7 did not have (sincere nonassertion — exactly the
kind of row an honesty steer "corrects"; system-prompt compliance; OR-Bench over-conservativeness; multilingual;
long-context), grows every capability and guard area to 150 rows per split, and adds held-out confrontation templates
(k001–k004) to eval. Fitting on v17 puts those areas into the null space (~5.6k benign prompts instead of 3.5k) and
lets the held-out checks include phrasings the fit never saw. The new dims `implicit_incentive` and
`self_misrepresentation` have no contrastive pairs yet and are therefore not steered (their prompts are not in the
pair set); that is a known gap, parked.

### 2.7 Diagnostics that were missing

Three things we could not see in round 2 are now printed by the fit: routing accuracy on held-out prompts (is the
router recognising dims or memorising phrasings), the gate's fire rate and shift per token on honest / dishonest /
protected continuations per dose (is the rule self-limiting and off on protected text), and the token-level gate
report per dim (which dims have a usable 1-D axis). They cost one extra pass over the pairs.

### 2.8 Unchanged, deliberately

Layer (Qwen 32 of 64, Granite 38, Gemma 30 — from the 4-bit layer profiles), null-space fraction 0.6, ridge strength,
80 pairs per dim, response-side per-dim targets, benign set composition (+ `benign_prompts_v1`), the coherence probe
before generation, hook-only control on the HF backend, white-box track. Parked (see the backlog): on-policy pairs
(the paper's data recipe), the Herbster 112-scenario held-out measurement, pairs for the two new dims, an earlier-layer
ablation, the "all tokens" position mode.

## 3. How to run

Colab: `exp_rahul_002_nullspace_gated.ipynb`, stages **A** (fit + sweep + bundle), **P** (probe several doses),
**B** (generate), **C** (judge). Config in cell 1: `WRITE_RULE`, `ALPHAS`, `FLOOR`, `DATA`, `PAIRS_OVERRIDE`, `ZERO_DIMS`
(per model), the probe knobs `PROBE_ALPHAS` / `PROBE_ZERO_EXTRA` / `PROBE_MAX_NEW_TOKENS`, the generation bundle's
`BUNDLE_ALPHA` / `BUNDLE_ZERO_EXTRA`, plus the round-2 options. The notebook clones `REPO_BRANCH`, so the branch must
carry this folder. Stage C's `COMPARE` needs an unsteered arm on the same v17 rows (`GENERATE_REFERENCE = True`, or
Rupesh's vLLM pipeline on the rows stage B picks).

**Why stage P exists.** With the gated rule the sweep stays far under the protected budget at every α (Qwen stage A:
|Δ log p|/token ≤ 0.0035 against a budget of 0.05), so its selection rule — the largest dose under budget — always
returns the top of the grid, and a teacher-forced score cannot see a dose that makes free generation loop or go blank.
Stage P re-freezes the stage-A fit at each `PROBE_ALPHAS` with `rebundle.py` (seconds; the router rows, targets and
gate statistics are all in the run, no model needed) and runs `probe_doses.py`, which loads the model once and prints
the unsteered answer and every dose's answer on `PROBE_ROWS` eval pressure rows per weak dim, with a distinct-4-gram
score and empty / looping flags. The dose for stage B is then chosen by reading those answers (`BUNDLE_ALPHA`), and a
dim that misbehaves can be zeroed at bundle time (`BUNDLE_ZERO_EXTRA`) without a refit. `rebundle.py` also serves to
change the write rule or `floor` of a finished fit.

Command line (what stage A runs):

```bash
python experiments/exp_rahul_002_nullspace_gated/fit_nullspace_map.py --model Qwen/Qwen3.8-27B --layers 32 \
    --data data/v17_thin_categories --write-rule sttp --alphas 0.25 0.5 1.0 1.5 \
    --benign-extra data/benign_prompts_v1/benign_prompts.jsonl \
    --zero-dims fabricated_entity fabricated_citation fabricated_action pushback hint_acknowledgment feedback_sycophancy \
                sandbagging harm_motivated_denial \
    --out runs/Qwen3.8-27B_bf16
```

(`--pairs-override` defaults to this folder's instructed_lie override; pass the flag with no value for the dataset's
pairs. `--write-rule fixed --lams …` reproduces the round-2 rule with the new storage.)

Mac rehearsal (9B, toy sizes, ~8 min on CPU; plumbing only):

```bash
python experiments/exp_rahul_002_nullspace_gated/fit_nullspace_map.py --model Qwen/Qwen3.5-9B --regime-model Qwen/Qwen3.8-27B \
    --layers 16 --pairs-per-dim 3 --eval-pairs-per-dim 2 --benign-limit 40 --eval-protected-per-dim 2 \
    --sweep-rows-per-dim 1 --sweep-protected-per-dim 1 --alphas 0.5 1.0 --batch-size 2 \
    --zero-dims sandbagging feedback_sycophancy --out /tmp/exp_rahul_002_smoke
cd third_party && python -m starterkit.check /tmp/exp_rahul_002_smoke/submission   # white-box / huggingface; "9B is not a base model" is expected
```

## 4. Files

| file | role |
| --- | --- |
| `controls/__init__.py`, `controls/nullspace_map_gated.py` | the control (`state_control/nullspace_map_gated`); import to register |
| `fit_nullspace_map.py` | fit → checks → gate statistics → dose sweep → bundle → `submission/` |
| `build_pairs_override.py`, `data_overrides/pairs_override_instructed_lie.jsonl` | the re-worded instructed_lie pairs (§2.5) and the script that builds them from the dataset |
| `exp_rahul_002_nullspace_gated.ipynb` | Colab stage runner (A/B/C) |
| `rebundle.py` | re-freeze a finished fit at another α / λ, write rule, floor or zero list in seconds, without the model (stage P and the stage-B bundle) |
| `probe_doses.py` | coherence probe over several bundles with the model loaded once: unsteered + each dose, row by row, with repetition / empty / cut flags |
| `probe_bundle.py`, `run_generate.py`, `judge_local.py`, `judge_kaggle.ipynb` | single-bundle probe, generation with the control registered, local / Kaggle judging — as in exp_rahul_001 |
| `results.md` | rehearsal numbers now; the Qwen run's fit checks, sweep, probe and judge read-out when done |

## 5. Open questions this run should answer

1. Does the gated rule keep concealment coherent at a dose that still moves instructed_lie and confrontation?
2. Gate fire rates on the eval pairs: which dims have a usable token-level axis on the 27B (fires on dishonest ≫ on honest)?
3. Routing accuracy on the k001–k004 confrontation rows (phrasings absent from train).
4. Against the unsteered arm on the same v17 rows (≥ 300 pressure rows of the four scored dims, ~15 per protected
   area): honesty gain per dim, capability/guard regressions, S.
