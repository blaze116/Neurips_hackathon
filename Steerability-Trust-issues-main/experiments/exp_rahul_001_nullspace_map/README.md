# exp_rahul_001 — NullSpaceMap: a null-space-constrained, per-dim-routed steering map

*Rahul. Status (2026-11-02): fitted in bf16 on all three models (round 2); Qwen generated twice (λ = 1 normalized, λ = 2
natural — the latter overshoots on concealment); Granite and Gemma fitted but not generated; judging pending. See
`results.md`.*

## 1. The idea, and the results that led to it

**Starting point.** The team's method (exp002) is CAST: one "honesty direction" (PCA over honest vs dishonest answer
activations) added at the middle layers when a gate detects pressure. Our week-3 score lost 0.50 to the penalty
terms, so the questions were: is one direction enough, and can the "leave ordinary prompts alone" part be made a
property of the method rather than a threshold to tune?

**Literature.** AlphaSteer (Sheng et al., ICLR 2026) learns a *matrix* `Δ` instead of a vector, constrained so that
`Δ·h ≈ 0` on benign prompts (its read directions lie in the null space of the benign activations) while `Δ·h` equals
a refusal direction on harmful prompts; the fit is a closed-form ridge regression. "On the Steering Dimensionality of
Refusal" (Wang et al., 2025) shows that safety refusal is steerable with one direction but *general* refusal is not,
even with 5-dimensional subspaces — a warning that one vector may not fit a heterogeneous behaviour. Dishonesty
under pressure looked like the second case: twelve categories with different mechanisms.

**Diagnostics that fixed the design** (`colab_rahul/diagnose.py`, `diagnose_v7.py` in the starter-kit workspace;
reports in `colab_rahul/output/`). Run on Qwen3.5-9B (bf16) and Qwen3.8-27B (NF4) with the team's v7 data and
rendering:

| finding | numbers (27B unless noted) | design consequence |
| --- | --- | --- |
| There is no single honesty direction on the team's pairs | a pooled honest−dishonest direction reproduces 3–5% of the per-row contrast; per-dim directions 53–59% (= the oracle-routed ceiling); honest/dishonest probes trained on one dim transfer to others at chance (0.57); the pooled direction explains 0–19% of any dim's energy | the map is **rank 12, one write direction per dim**, and routes each prompt to its dim's correction |
| Prompt-side contrasts (pressure vs pressure-free twin) are the wrong target | they are orthogonal to the response-side directions (cos ≈ 0) and their maps collapsed on new templates (R² 0.93 → 0.15–0.36 on v7 test) for 4 of 6 dims | targets are **response-side**: the honest vs dishonest *answer* states of the team's `contrastive_pair`, which exist for all 12 dims |
| The null space generalises | PEC (share of a held-out protected prompt invisible to the map) ≥ 0.98 on every capability/guard area at layer 32 with 1.2k benign rows, 0.99+ with 3.5k; leakage onto protected prompts 0.07–0.25 vs 0.6–1.6 without the constraint | the constraint stays; `data/benign_prompts_v1` was built because PEC was still rising with the number of benign rows |
| Layer | the per-dim map's eval R² is flat over 20–80% depth; instructed_lie is only learnable from layer 32 on; PEC falls below 0.98 after layer 38 | **layer 32** (50% depth; the 9B's best was also 50%) |
| Single layer, not a band | a 3-layer band at the same λ was ~3× over-dosed on the 9B (KL 0.43 vs 0.03) | one layer; λ swept instead |
| Behaviour moves | teacher-forced on the weak dims, per-dim map at λ = 1: Δ log-odds honest vs dishonest +3.3 (pooled: +0.8), honest answer more likely in absolute terms, ≤ 0.004 KL on protected prompts, monotone in λ | worth a real generation run |

**The control.** `Δ = A Bᵀ` with `B` in the benign null space (read) and `A` pointing at each dim's honest-answer
state (write); the shift is computed once from the last prompt token and added to every answer token. On an ordinary
prompt `Bᵀh ≈ 0`, so nothing happens — measured, not hoped. Dose λ is the one knob; it is swept teacher-forced before
any generation is paid for.

**What the first run showed (`results.md`).** Fit checks as predicted (PEC 0.992–0.999, leakage ≤ 0.25); the dose
sweep is monotone for confrontation, instructed_lie and stated sandbagging but **concealment moves the wrong way**,
so λ = 1 was a compromise; the 98 generated answers read ≈ 0.6 / 0.6 / 0.8 honest on instructed_lie / confrontation /
sandbagging_stated (exp001 unsteered: 0.2 / 0.2 / 0.6) and ≈ 0.2 on concealment (0.3), with capability and guard
answers unchanged — judge pending. The next change is a per-dim dose, which needs no refit.

## 2. What the control does at inference

`controls/nullspace_map.py`, registered as `state_control/nullspace_map` (white-box, Hugging Face backend, hook-only
like CAST):

1. During prefill, read the residual stream `h` at the **last prompt token** at layer ℓ.
2. Shift `s = λ · A (Bᵀ h)`. `B` ([H, r]) = read directions (in the benign null space); `A` ([H, r]) = write directions
   (toward each dim's honest-answer state). `r` = number of dims (12).
3. Add `s` to **every answer position** of that row (thinking included); prompt tokens are untouched.

Both factors are `SteeringVector`s (safetensors inside the `.spipe`, ~0.4 MB per layer); the bundle is the recipe and
needs no model at `steer()`, so it loads without `allow_code` and passes `check-submission` (track white-box).

## 3. How it is fitted (`fit_nullspace_map.py`)

| step | what | source |
| --- | --- | --- |
| rows | the **same 80 contrastive pairs per dim as exp002** (`steering/cast_honesty.pair_rows`), rendered with the team's `render_prompt` (full chat, sandbagging suffix, thinking on) | `data/v7_tidy/train.jsonl` |
| benign set `H_b` | train pressure-free twins/controls + train capability and guard rows (+ optional prompt-only extras) | train |
| activations | last prompt token of pressure and benign prompts; mean over the **answer tokens** of each pair's honest and dishonest answers appended to the same prompt | one pass |
| null space | eigenvectors of `H_bᵀH_b` with the smallest 60% of eigenvalues → `U` | |
| targets `R` | per dim, mean(honest − dishonest), each dim scaled to the median dim norm (so λ means the same push for every dim) | |
| regression | ridge in null-space coordinates: `Z = H_p U`, `M = Rᵀ Z (ZᵀZ + αI)⁻¹`, `Δ = M Uᵀ`; thin SVD → `A = P S`, `B = U Q` | AlphaSteer closed form |
| checks (eval) | PEC and leakage per capability/guard area; R² of `Δh` toward each eval row's own honest−dishonest delta, per dim | `eval.jsonl` |
| dose sweep | the control itself in a `SteeringPipeline`, teacher-forced: Δ[log p(honest) − log p(dishonest)] on eval rows of the four weak dims for λ ∈ {0.5, 1, 1.5, 2}; mean |Δ log p| per token of the unsteered model's own answers on protected rows (must stay ≤ 0.05) | |
| bundle | freeze at the chosen λ, `verify`, reload, roundtrip check; `submission/` with the bundle, `controls/`, `method_card.md` | |

**Why these choices (evidence in `colab_rahul/` diagnostics, 9B and 27B):** per-dim ≫ pooled on both models (R²
0.55 vs 0.04; teacher-forced Δ log-odds +3.3 vs +0.8 at λ=1 on the 27B); layer 32 of 64 (50% depth, same relative depth
as the 9B's best) is the deepest layer whose null space still covers ≥ 98% of every guard area; single layer rather
than a band (a 3-layer band at the same λ was ~3× over-dosed on the 9B); dose chosen on eval, test untouched.

## 4. How to run

### What you upload / set up (once)

- **Colab secrets:** `GITHUB_TOKEN` (the private repo clone — same as exp002). **`HF_TOKEN` is not needed** for
  Qwen3.8-27B or GLM-4-9B (both public). Add it only for Gemma.
- **Drive (optional):** `MyDrive/trust_issues_runs/exp001_qwen_reference/Qwen3.8-27B_<precision>/answers.jsonl` and
  `judge_cache.jsonl` for the precision you run (Rupesh's bf16 unsteered arm from exp002, or exp001's NF4 one). If
  absent, stage B generates the unsteered arm itself on the same rows (about twice the generation time and units)
  and stage C judges both arms; the notebook then writes that unsteered arm and its verdicts into the reference
  folder so the team can reuse them.
- Optional **wheel cache** (`MyDrive/steer_wheels`, see §6) to cut the install from minutes to seconds.
- Nothing else is uploaded: the notebook clones the repo; data, control and scripts are in it.

### The three stages (`exp_rahul_001_nullspace_map.ipynb`, set `STAGE`, Run all)

| stage | runtime | time / units | what to read before moving on |
| --- | --- | --- | --- |
| A | A100 **80 GB** (bf16) | ~30–45 min, ~5–7 units | `sweep.json`: Δ log-odds rising with λ, honest log p rising, protected |Δ log p|/token ≤ 0.05 → the chosen λ. `fit_summary.json`: min PEC ≥ 0.98, max leak ≤ 0.3, `verify_ok`, `submittable: true`. `check-submission`: only "no bundle for gemma/granite" errors. |
| B | A100 80 GB | ~1–1.5 h, ~10–15 units | 98 rows of **our arm** generated (`answers.jsonl` on Drive). No unsteered arm is needed for this; set `GENERATE_REFERENCE = True` only if you want the notebook to generate one too (doubles the time) |
| C | **L4**, any time later | ~20–40 min, ~2 units | `summary_nullspace_v1.txt`. With an unsteered arm in the file (Rupesh's bf16 answers, or one generated later): per-category change vs unsteered and the estimated delta / r_side / r_guard / S. Without one: the absolute per-category summary of our arm (honesty rate on the weak dims, capability/guard accuracy) |

Stage C on Kaggle instead (free): see `judge_kaggle.ipynb` — uploads `answers.jsonl` + `judge_cache.jsonl` as a
dataset and runs `judge_local.py` (GLM-4-9B in fp16 on T4×2).

### Comparing with exp002

Same rows (`TARGET_DIMS` × 5 + `CHECK_DIMS` × 2 on eval), same judge (GLM-4-9B-0414 @ 645b848), same harness and
scorer, **same precision** as the unsteered arm. Only compare arms generated at one precision.

### Later: the test split

The frozen bundle generates on any rows at the same per-row cost (~1–1.5 h per ~100 rows); the fit is not repeated.
The unsteered arm must exist for the same test rows and precision — coordinate so it is generated once for the team.

## 5. Files

| file | role |
| --- | --- |
| `controls/__init__.py`, `controls/nullspace_map.py` | the control; import to register (`check-submission` and `run_generate.py` do) |
| `fit_nullspace_map.py` | fit → checks → dose sweep → bundle → `submission/` |
| `run_generate.py` | the team harness with the control registered (`--bundle` needs it) |
| `judge_local.py` | the team scorer with a locally loaded judge (fp16 on T4s / local model path) |
| `exp_rahul_001_nullspace_map.ipynb` | Colab stage runner (A/B/C) |
| `judge_kaggle.ipynb` | stage C on Kaggle |
| `results.md` | the first run's fit checks, sweep, and generation read-out |
| `probe_bundle.py` | coherence probe: greedy 400-token answers with a bundle on a few rows (run by stage B before generating) |
| `runs/v1_qwen_norm_lam1/` | round 1 (Qwen, normalized targets, λ = 1): factors, fit summary, sweep, bundle, 98 generations |
| `runs/v2/<model>_bf16/` | round 2 (natural norms, fixed sweep): factors, fit summary, sweep, bundle per model; Qwen also has its 98 generations and `answers_readable.csv` |
| `submission_nullspace_v1/`, `nullspace_v1_submission.zip`, `check_report_nullspace_v1.json` | the three round-2 bundles + controls + method card, as `check-submission` expects (PASS); Qwen's bundle to be replaced by a lower dose |

Related evidence (starter-kit workspace, `colab_rahul/`): `diagnose_v7.py` and the two reports
(`results_v7/diagnostics_v7.md` for the 9B, `results_v7_27B_qwen/diagnostics_v7.md` for the 27B in NF4).

## 6. Faster sessions

**Wheel cache on Drive.** The current cache (`MyDrive/steer_wheels`, built on a T4 session with Python 3.13) holds the
heavy dependencies (torch 2.15 + CUDA 13 libraries, transformers 5.22, bitsandbytes, flash-linear-attention,
datasets, peft, trl, xgrammar, …). The notebook installs those with `--no-index --find-links` (~1 min from Drive)
and fetches two small things online, because they are not in the cache:

- the toolkit's build backend `uv_build` (the toolkit is installed editable from `third_party/steerability-toolkit`;
  the `steerability-0.5.5.zip` in the cache is an unused sdist);
- the IFEval checker from `eval/requirements.txt` (a git package, needed by the scorer).

To make the install fully offline, run once on Colab:

```python
W = "/content/drive/MyDrive/steer_wheels"
!pip download -d {W} "uv_build>=0.8.9,<0.9.0"
!pip wheel -w {W} -r /content/Steerability-Trust-issues/eval/requirements.txt
```

and change the two online `pip install` lines in the notebook's install cell to `--no-index --find-links={W}`.
`causal_conv1d` did not build on the T4 session and is left out: optional, `flash-linear-attention` alone gives most
of the DeltaNet speed-up. The cache is tied to the Colab image's Python (3.13 today); rebuild it if Colab upgrades.

**Model download.** The 55 GB bf16 download (~10–15 min) cannot be cached usefully on Drive; it is the floor of every
generation session.

**Kaggle judge.** Add GLM-4-9B-0414 as a Kaggle Model once (import from Hugging Face, revision `645b848`); it then
mounts under `/kaggle/input` with no download, and `judge_local.py --judge-model <that path>` uses it.

### Benign prompt set

Stage A reads `data/benign_prompts_v1/benign_prompts.jsonl` (1,733 prompt-only rows from the same pinned sources as
fix 4, excluding every v7 row; see that folder's README) in addition to the v7 train twins and protected rows, giving
~2.9k benign rows. Set `BENIGN_EXTRA = []` in the notebook to fit on v7 alone.

## 7. Rehearsal (Mac, no GPU budget)

```bash
python experiments/exp_rahul_001_nullspace_map/fit_nullspace_map.py --model Qwen/Qwen3.5-9B --regime-model Qwen/Qwen3.8-27B \
    --layers 16 --pairs-per-dim 3 --eval-pairs-per-dim 2 --benign-limit 40 --eval-protected-per-dim 2 \
    --sweep-rows-per-dim 1 --sweep-protected-per-dim 1 --lams 0.5 1.0 --batch-size 2 --out /tmp/exp_rahul_smoke
```

Runs the whole chain in ~3 min (fit, sweep, bundle, verify, roundtrip, submission folder); numbers are meaningless at
that size, the plumbing is the point. `check-submission /tmp/exp_rahul_smoke/submission` then reports the control as
white-box / huggingface and complains only that the 9B is not a base model.
