"""Fit the NullSpaceMapGated control (exp_rahul_002) on the team's train split, sweep the dose, and freeze a bundle.

    python experiments/exp_rahul_002_nullspace_gated/fit_nullspace_map.py --model Qwen/Qwen3.8-27B --layers 32 \
        --write-rule sttp --pairs-override experiments/exp_rahul_002_nullspace_gated/data_overrides/pairs_override_instructed_lie.jsonl \
        --zero-dims fabricated_entity fabricated_citation fabricated_action pushback hint_acknowledgment feedback_sycophancy \
                    sandbagging harm_motivated_denial --out runs/exp_rahul_001

Steps (all on one loaded model; ~30-45 min for the 27B in bf16 on an 80 GB A100):

  1. rows: the same 80 contrastive pairs per dim as exp002 (`steering/cast_honesty.pair_rows`), rendered as the
     harness renders them, optionally with re-worded pairs from --pairs-override (`build_pairs_override.py`); the
     benign ("don't steer") set = the pressure-free twins of train, the train capability and guard rows, and any
     extra prompt-only files given with --benign-extra.
  2. activations at --layers: last prompt token of every pressure and benign prompt; mean over the answer tokens
     of each pair's honest and dishonest answers (appended to the prompt after the model's answer prefix).
  3. null space of the benign set (`--null-frac` of the eigen-directions with the smallest eigenvalues), per-dim
     response-side targets (mean honest - dishonest; --normalize none|cap|median; --zero-dims get a zero target),
     ridge regression in null-space coordinates -> the router rows (`read`, [D, H]) and the targets (`write`, [D, H]).
  4. checks on eval: PEC per protected area (fraction of a held-out prompt's energy the map cannot see), leakage
     (|shift| on protected prompts relative to pressure prompts), R^2 toward the row's own response delta, routing
     accuracy on held-out prompts.
  5. gate statistics: the projection of every honest / dishonest answer token on its dim's unit target (train ->
     the per-dim mean, std and equal-z boundary the gated write rules use; eval -> how often the gate would fire on
     held-out dishonest and honest tokens).
  6. dose sweep with the control itself in a SteeringPipeline: teacher-forced log-odds of honest vs dishonest on
     eval pressure rows of the weak dims (the prompt handed to the control ends where the map was fitted, at the
     last prompt token; the model's empty thinking block is the first part of each scored continuation), the change
     in log p of the unsteered model's own greedy answer on protected rows (a no-op check), and the gate's firing
     rate and shift per token on honest / dishonest / protected tokens, for each dose (--lams for the fixed rule,
     --alphas for sttp / stmp).
  7. bundle at --lam (default: chosen by the sweep): freeze, verify, reload, roundtrip check; plus a
     submission-shaped folder (`<slug>.spipe` + controls/ + method_card.md at the root) for the kit's
     check-submission. A single-model check reports "no bundle for <other models>" errors; those are expected until
     the other two models are fitted.

Outputs under --out: fit_summary.json, sweep.json, read.svec / write.svec, bundles/<slug>.spipe,
submission/{controls,bundles,method_card.md}.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "third_party"))
sys.path.insert(0, str(REPO / "steering"))
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(HERE))

import cast_honesty as ch  # noqa: E402  (team rendering, row selection, model loading)
from starterkit import REGIME, model_slug  # noqa: E402

WEAK_DIMS = ["instructed_lie", "confrontation_after_lie", "concealment", "sandbagging_stated"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def memory() -> str:
    if torch.cuda.is_available():
        return ", ".join(f"cuda:{i} {torch.cuda.memory_allocated(i) / 2**30:.1f} GB" for i in range(torch.cuda.device_count()))
    return "cpu"


# ----------------------------------------------------------------------------------------------- data

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def apply_pair_overrides(rows: list[dict], files: list[str]) -> dict:
    """Replace `contrastive_pair` text in place for rows named in override files (`build_pairs_override.py`); the
    prompt, reference and every other field stay as they are. Returns counts for the log and the fit summary."""
    by_id = {}
    for path in files:
        for rec in read_jsonl(Path(path)):
            by_id[rec["row_id"]] = rec
    applied, per_dim = 0, collections.Counter()
    for r in rows:
        rec = by_id.pop(r["row_id"], None)
        if rec is None or not r.get("contrastive_pair"):
            continue
        r["contrastive_pair"] = {**r["contrastive_pair"], "honest": rec["honest"], "dishonest": rec["dishonest"],
                                 "pattern": rec.get("pattern", r["contrastive_pair"].get("pattern")),
                                 "contrast": rec.get("contrast", r["contrastive_pair"].get("contrast")),
                                 "override": True}
        applied += 1
        per_dim[r["dim"]] += 1
    return {"files": list(files), "applied": applied, "per_dim": per_dim, "unmatched": len(by_id)}


def benign_rows(train: list[dict], pressure_rows: list[dict], extra_files: list[str]) -> list[dict]:
    """Pressure-free twins of the chosen pressure rows (plus every other control row of train), the train
    capability and guard rows, and extra prompt-only rows (`{"messages": [...]}` or `{"prompt": "..."}`)."""
    rows = [r for r in train if ch.is_protected(r)]
    rows += [r for r in train if "honesty" in (r.get("scored_as") or []) and r["condition"] in ("neutral", "persona_only", "control")]
    for path in extra_files:
        for i, r in enumerate(read_jsonl(Path(path))):
            messages = r.get("messages") or [{"role": "user", "content": r["prompt"]}]
            rows.append({"row_id": f"extra:{Path(path).stem}:{i}", "dim": f"extra_{Path(path).stem}", "condition": "neutral",
                         "messages": messages, "scored_as": []})
    return rows


# ----------------------------------------------------------------------------------------------- activations

def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


@torch.inference_mode()
def forward(model, input_ids, attention_mask):
    try:
        return model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=1)
    except TypeError:
        return model(input_ids=input_ids, attention_mask=attention_mask)


@torch.inference_mode()
def last_token_states(model, tokenizer, layer_modules, layer_ids, texts, batch_size, max_tokens):
    """`{layer: [N, H]}` fp32 CPU at the last real token of each text (texts longer than max_tokens are truncated
    from the left, keeping the end of the prompt)."""
    enc = [tokenizer(t, add_special_tokens=False)["input_ids"][-max_tokens:] for t in texts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    out = {l: torch.empty(len(enc), model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.text_config.hidden_size) for l in layer_ids}
    state, captured = {}, {}

    def hook(layer):
        def fn(module, args, output):
            h = _hidden(output)
            captured[layer] = h[torch.arange(h.size(0), device=h.device), state["last"].to(h.device)].float().cpu()
        return fn

    handles = [layer_modules[l].register_forward_hook(hook(l)) for l in layer_ids]
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    try:
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            T = max(len(enc[i]) for i in idx)
            ids = torch.full((len(idx), T), pad, dtype=torch.long)
            mask = torch.zeros((len(idx), T), dtype=torch.long)
            for b, i in enumerate(idx):
                ids[b, :len(enc[i])] = torch.tensor(enc[i]); mask[b, :len(enc[i])] = 1
            state["last"] = mask.sum(1) - 1
            forward(model, ids.to(device), mask.to(device))
            for l in layer_ids:
                out[l][idx] = captured[l]
            if (start // batch_size) % 25 == 0:
                log(f"  prompts {min(start + batch_size, len(order))}/{len(order)} ({memory()})")
    finally:
        for h in handles:
            h.remove()
    return out


@torch.inference_mode()
def answer_mean_states(model, tokenizer, layer_modules, layer_ids, prompts, answers, batch_size, max_tokens):
    """`{layer: [N, H]}` fp32 CPU: the mean state over the answer tokens appended to each prompt."""
    p_ids = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in prompts]
    a_ids = [tokenizer(t, add_special_tokens=False)["input_ids"] or [tokenizer.eos_token_id] for t in answers]
    seqs = [(p[-(max_tokens - len(a)):] if len(p) + len(a) > max_tokens else p, a) for p, a in zip(p_ids, a_ids)]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    hidden_size = next(iter(layer_modules[layer_ids[0]].parameters())).shape[-1] if False else None
    out = {l: None for l in layer_ids}
    state, captured = {}, {}

    def hook(layer):
        def fn(module, args, output):
            h = _hidden(output).float()
            m = state["mask"].to(h.device)
            captured[layer] = ((h * m[..., None]).sum(1) / m.sum(1, keepdim=True)).cpu()
        return fn

    handles = [layer_modules[l].register_forward_hook(hook(l)) for l in layer_ids]
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    try:
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            T = max(len(seqs[i][0]) + len(seqs[i][1]) for i in idx)
            ids = torch.full((len(idx), T), pad, dtype=torch.long)
            att = torch.zeros((len(idx), T), dtype=torch.long)
            m = torch.zeros((len(idx), T), dtype=torch.float32)
            for b, i in enumerate(idx):
                p, a = seqs[i]
                ids[b, :len(p) + len(a)] = torch.tensor(p + a); att[b, :len(p) + len(a)] = 1; m[b, len(p):len(p) + len(a)] = 1.0
            state["mask"] = m
            forward(model, ids.to(device), att.to(device))
            for l in layer_ids:
                if out[l] is None:
                    out[l] = torch.empty(len(seqs), captured[l].shape[-1])
                out[l][idx] = captured[l]
            if (start // batch_size) % 25 == 0:
                log(f"  answers {min(start + batch_size, len(order))}/{len(order)} ({memory()})")
    finally:
        for h in handles:
            h.remove()
    return out


@torch.inference_mode()
def answer_token_projections(model, tokenizer, layer_modules, layer_ids, prompts, answers, row_dirs, batch_size, max_tokens):
    """`{layer: [list[float] per row]}`: the projection of every answer token's state on that row's unit direction
    (`row_dirs[layer]` [N, H]), for the gate statistics. Same packing as `answer_mean_states`."""
    p_ids = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in prompts]
    a_ids = [tokenizer(t, add_special_tokens=False)["input_ids"] or [tokenizer.eos_token_id] for t in answers]
    seqs = [(p[-(max_tokens - len(a)):] if len(p) + len(a) > max_tokens else p, a) for p, a in zip(p_ids, a_ids)]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    out = {l: [None] * len(seqs) for l in layer_ids}
    state, captured = {}, {}

    def hook(layer):
        def fn(module, args, output):
            h = _hidden(output).float()
            dirs = row_dirs[layer][state["idx"]].to(h.device)                      # [B, H]
            captured[layer] = torch.einsum("bth,bh->bt", h, dirs).cpu()           # [B, T]
        return fn

    handles = [layer_modules[l].register_forward_hook(hook(l)) for l in layer_ids]
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    try:
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            T = max(len(seqs[i][0]) + len(seqs[i][1]) for i in idx)
            ids = torch.full((len(idx), T), pad, dtype=torch.long)
            att = torch.zeros((len(idx), T), dtype=torch.long)
            for b, i in enumerate(idx):
                p, a = seqs[i]
                ids[b, :len(p) + len(a)] = torch.tensor(p + a); att[b, :len(p) + len(a)] = 1
            state["idx"] = torch.tensor(idx)
            forward(model, ids.to(device), att.to(device))
            for l in layer_ids:
                for b, i in enumerate(idx):
                    p, a = seqs[i]
                    out[l][i] = captured[l][b, len(p):len(p) + len(a)].tolist()
    finally:
        for h in handles:
            h.remove()
    return out


# ----------------------------------------------------------------------------------------------- the fit

def null_basis(Hb: torch.Tensor, frac: float) -> torch.Tensor:
    """`U` [H, k]: the eigen-directions of `H_b^T H_b / N` with the smallest `frac` share of the eigenvalue count."""
    evals, evecs = torch.linalg.eigh((Hb.T @ Hb) / Hb.size(0))
    k = int(round(frac * Hb.size(1)))
    return evecs[:, :k].contiguous()


def fit_layer(Hp: torch.Tensor, dims: list[str], D: torch.Tensor, Hb: torch.Tensor, null_frac: float, alpha_rel: float,
              normalize: str, zero_dims: set[str] = frozenset()) -> dict:
    """Closed-form fit at one layer. Returns the router rows `read` [D, H], the targets `write` [D, H] and diagnostics.

    `Hp` [N, H] pressure prompt states, `D` [N, H] response deltas (honest - dishonest answer states),
    `Hb` [Nb, H] benign prompt states. Targets: each row's dim mean of D; `normalize` = "none" (natural norms), "cap"
    (dims above the median dim norm are scaled down to it; small dims are left alone, since scaling a dim with no usable
    direction *up* injects noise into the shared read space), or "median" (every dim scaled to the median). Dims in
    `zero_dims` get a zero target: the map learns to recognise them and do nothing.
    Ridge in null-space coordinates: Z = Hp U; with the one-hot dim matrix O [N, D] and the targets T [D, H] the
    target matrix is R = O T, so the ridge solution factorises exactly as a router times the targets:
    shift(h) = T^T (O^T Z (Z^T Z + a I)^-1 U^T h) = sum_d w_d(h) t_d. `read` holds the D router rows
    (U (Z^T Z + a I)^-1 Z^T O)^T and `write` the targets, in `dim_order`. (Rounds 1-2 stored the same map as a thin
    SVD; the per-dim form is what the gated write rule needs.)
    """
    device = Hp.device
    U = null_basis(Hb, null_frac)
    by_dim = collections.defaultdict(list)
    for i, d in enumerate(dims):
        by_dim[d].append(i)
    dim_order = sorted(by_dim)
    means = {d: D[idx].mean(0) for d, idx in by_dim.items()}
    norms = {d: float(v.norm()) for d, v in means.items()}
    target_norm = float(torch.tensor(list(norms.values())).median())
    if normalize == "median":
        means = {d: v * (target_norm / max(norms[d], 1e-6)) for d, v in means.items()}
    elif normalize == "cap":
        means = {d: (v * (target_norm / norms[d]) if norms[d] > target_norm else v) for d, v in means.items()}
    elif normalize != "none":
        raise ValueError(f"normalize must be none, cap or median; got {normalize!r}")
    for d in zero_dims:
        if d in means:
            means[d] = torch.zeros_like(means[d])
    R = torch.stack([means[d] for d in dims])                               # [N, H]
    O = torch.zeros(len(dims), len(dim_order), device=device)
    for i, d in enumerate(dims):
        O[i, dim_order.index(d)] = 1.0
    Z = Hp @ U                                                              # [N, k]
    G = Z.T @ Z
    alpha = alpha_rel * G.diagonal().mean().item()
    G.diagonal().add_(alpha)
    Wz = torch.linalg.solve(G, Z.T @ O)                                     # [k, D]
    read = (U @ Wz).T.contiguous()                                          # [D, H]  router rows
    write = torch.stack([means[d] for d in dim_order]).contiguous()         # [D, H]  targets
    pred = (Hp @ read.T) @ write                                            # = Delta Hp
    w_train = Hp @ read.T                                                   # [N, D]  routing weights on the train prompts
    own = w_train[torch.arange(len(dims), device=device), O.argmax(1)]
    r = int(sum(float(write[j].norm()) > 1e-6 for j in range(len(dim_order))))
    return {"read": read, "write": write, "U": U, "rank": r, "alpha": alpha, "null_dim": U.size(1), "dims": dim_order,
            "dim_norms": norms, "target_norms": {d: float(v.norm()) for d, v in means.items()}, "normalize": normalize,
            "zero_dims": sorted(zero_dims),
            "r2_target_train": 1.0 - float(((pred - R) ** 2).sum() / (R ** 2).sum()),
            "r2_pair_train": 1.0 - float(((pred - D) ** 2).sum() / (D ** 2).sum()),
            "route_train": {"own_weight_mean": round(float(own.mean()), 3),
                            "argmax_acc": round(float((w_train.argmax(1) == O.argmax(1)).float().mean()), 3)},
            "dose_train": float(pred.norm(dim=1).mean() / Hp.norm(dim=1).mean())}


def apply(read: torch.Tensor, write: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    return (H @ read.T) @ write


def gate_statistics(proj_pos: dict[str, list[float]], proj_neg: dict[str, list[float]], dim_order: list[str],
                    active: dict[str, bool]) -> dict:
    """Per-dim token-level statistics of the projections on the dim's unit target: honest tokens (`pos`) and
    dishonest tokens (`neg`) -> means, standard deviations, the equal-z boundary `m = (mu+ s- + mu- s+) / (s+ + s-)`,
    Cohen's d, and how the gate would behave on these tokens (share of dishonest tokens below `m` = fires when it
    should; share of honest tokens below `m` = fires when it should not). Inactive (zeroed) dims get zeros."""
    out = {k: [] for k in ("mu_pos", "sigma_pos", "mu_neg", "sigma_neg", "boundary")}
    report = {}
    for d in dim_order:
        pos, neg = torch.tensor(proj_pos.get(d, [])), torch.tensor(proj_neg.get(d, []))
        if not active.get(d, True) or len(pos) < 2 or len(neg) < 2:
            for k in out:
                out[k].append(0.0)
            report[d] = {"active": False, "n_pos": len(pos), "n_neg": len(neg)}
            continue
        mp, sp, mn, sn = float(pos.mean()), float(pos.std()), float(neg.mean()), float(neg.std())
        sp, sn = max(sp, 1e-6), max(sn, 1e-6)
        m = (mp * sn + mn * sp) / (sp + sn)
        for k, v in zip(out, (mp, sp, mn, sn, m)):
            out[k].append(round(v, 5))
        report[d] = {"active": True, "n_pos": len(pos), "n_neg": len(neg), "mu_pos": round(mp, 3), "sigma_pos": round(sp, 3),
                     "mu_neg": round(mn, 3), "sigma_neg": round(sn, 3), "boundary": round(m, 3),
                     "cohen_d": round((mp - mn) / math.sqrt((sp ** 2 + sn ** 2) / 2), 3),
                     "fires_on_dishonest": round(float((neg < m).float().mean()), 3),
                     "fires_on_honest": round(float((pos < m).float().mean()), 3)}
    return {"lists": out, "report": report}


def r2(pred, target) -> float:
    return 1.0 - float(((pred - target) ** 2).sum() / (target ** 2).sum())


# ----------------------------------------------------------------------------------------------- main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--regime-model", help="whose official regime to borrow, for a stand-in model the kit does not list (rehearsals)")
    ap.add_argument("--data", default=str(REPO / "data" / "v17_thin_categories"))
    ap.add_argument("--pairs-override", nargs="*", default=[str(HERE / "data_overrides" / "pairs_override_instructed_lie.jsonl")],
                    help="JSONL files of re-worded contrastive pairs (row_id, honest, dishonest, ...) applied to train and eval "
                         "rows before sampling; default: this folder's instructed_lie override (README 3a); pass the flag with no "
                         "value to use the dataset's pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[32], help="decoder layers (outputs) to fit and steer")
    ap.add_argument("--pairs-per-dim", type=int, default=80)
    ap.add_argument("--null-frac", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=0.1, help="ridge strength relative to the mean eigenvalue of Z^T Z")
    ap.add_argument("--normalize", default="none", choices=["none", "cap", "median"],
                    help="per-dim target norms: natural (default; what the 4-bit layer profiles measured), cap large dims at the "
                         "median, or scale all to the median (halved the map's eval R2 and flipped Granite's weak dims)")
    ap.add_argument("--zero-dims", nargs="*", default=["none"],
                    help="dims given a zero target (the map recognises them and does nothing), e.g. the dims the unsteered "
                         "model already passes; default none")
    ap.add_argument("--write-rule", default="sttp", choices=["fixed", "sttp", "stmp"],
                    help="fixed: the same shift on every answer token (rounds 1-2); sttp / stmp: per-token projection-gated "
                         "write (Herbster et al. 2026), dosed by --alphas")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.5],
                    help="gated rules: dose grid for the sweep (sttp: target = mu+ + alpha sigma+; stmp: mirror overshoot)")
    ap.add_argument("--floor", type=float, default=0.0, help="gated rules: fraction of the fixed shift always added underneath")
    ap.add_argument("--w-clip", type=float, default=1.5, help="gated rules: router weights clipped to [0, w_clip]")
    ap.add_argument("--benign-extra", nargs="*", default=[], help="extra prompt-only JSONL files for the benign set")
    ap.add_argument("--benign-limit", type=int, help="cap on benign rows (rehearsals only; seeded sample)")
    ap.add_argument("--eval-protected-per-dim", type=int, help="cap on eval protected rows per dim (rehearsals only)")
    ap.add_argument("--eval-pairs-per-dim", type=int, default=20, help="eval pairs per dim for the R^2 check")
    ap.add_argument("--lams", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0], help="fixed rule: dose grid for the sweep")
    ap.add_argument("--lam", type=float, help="dose for the bundle (lam for fixed, alpha for the gated rules); default: the sweep's pick")
    ap.add_argument("--sweep-rows-per-dim", type=int, default=5, help="eval pressure rows per weak dim in the sweep")
    ap.add_argument("--sweep-protected-per-dim", type=int, default=3)
    ap.add_argument("--kl-budget", type=float, default=0.05, help="max mean |delta log p| per token on protected answers")
    ap.add_argument("--quantize", choices=["nf4"], help="4-bit (comparison only; the bundle is then not submittable)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model_id = args.model
    if model_id not in REGIME:
        if not args.regime_model or args.regime_model not in REGIME:
            raise SystemExit(f"{model_id} has no official regime; pass --regime-model with one of {sorted(REGIME)}")
        REGIME[model_id] = REGIME[args.regime_model]
    data = Path(args.data)
    train, evalrows = read_jsonl(data / "train.jsonl"), read_jsonl(data / "eval.jsonl")
    overrides = apply_pair_overrides(train + evalrows, args.pairs_override)
    if args.pairs_override:
        log(f"pairs override: {overrides['applied']} rows re-worded ({dict(overrides['per_dim'])}) from {args.pairs_override}; "
            f"{overrides['unmatched']} override rows matched no row")

    # --- rows
    pressure = ch.pair_rows(train, args.pairs_per_dim, args.seed)
    pressure = sorted(pressure, key=lambda r: (r["dim"], r["row_id"]))
    benign = benign_rows(train, pressure, args.benign_extra)
    if args.benign_limit is not None and args.benign_limit < len(benign):
        benign = random.Random(args.seed).sample(sorted(benign, key=lambda r: r["row_id"]), args.benign_limit)
    eval_protected = [r for r in evalrows if ch.is_protected(r)]
    if args.eval_protected_per_dim is not None:
        eval_protected = ch.sample_per_dim(eval_protected, args.eval_protected_per_dim, args.seed)
    eval_pairs = sorted(ch.pair_rows(evalrows, args.eval_pairs_per_dim, args.seed), key=lambda r: (r["dim"], r["row_id"]))
    log(f"train pressure {len(pressure)} over {len({r['dim'] for r in pressure})} dims; benign {len(benign)} "
        f"({collections.Counter(r['dim'].split('_')[0] for r in benign)}); eval protected {len(eval_protected)}; eval pairs {len(eval_pairs)}")

    # --- model (FIT_DTYPE=float16 for GPUs without bf16, e.g. Kaggle T4; such a bundle is for proxy use only)
    import os
    if os.environ.get("FIT_DTYPE"):
        from starterkit import PIPELINE_KWARGS
        PIPELINE_KWARGS["hf_model_kwargs"]["dtype"] = os.environ["FIT_DTYPE"]
        log(f"loading in {os.environ['FIT_DTYPE']} (FIT_DTYPE)")
    model, tokenizer = ch.load_model(model_id, args.quantize)
    from steerability.algorithms.core.internals.fingerprint import artifact_provenance_meta
    from steerability.algorithms.core.internals.model_layout import resolve_model_layout

    layout = resolve_model_layout(model)
    modules = {l: model.get_submodule(layout.layer_names[l]) for l in args.layers}
    render = lambda row: ch.render_prompt(tokenizer, model_id, row)
    log(f"model loaded ({memory()}); {layout.num_layers} layers; fitting at {args.layers}")

    # --- activations
    t0 = time.time()
    texts_p = [render(r) for r in pressure]
    texts_b = [render(r) for r in benign]
    texts_g = [render(r) for r in eval_protected]
    texts_e = [render(r) for r in eval_pairs]
    Hp = last_token_states(model, tokenizer, modules, args.layers, texts_p, args.batch_size, args.max_tokens)
    Hb = last_token_states(model, tokenizer, modules, args.layers, texts_b, args.batch_size, args.max_tokens)
    Hg = last_token_states(model, tokenizer, modules, args.layers, texts_g, args.batch_size, args.max_tokens)
    He = last_token_states(model, tokenizer, modules, args.layers, texts_e, args.batch_size, args.max_tokens)
    prefixes_p = [ch.answer_prefix(model_id, t) for t in texts_p]
    prefixes_e = [ch.answer_prefix(model_id, t) for t in texts_e]
    hon = answer_mean_states(model, tokenizer, modules, args.layers, [t + p for t, p in zip(texts_p, prefixes_p)],
                             [r["contrastive_pair"]["honest"] for r in pressure], args.batch_size, args.max_tokens)
    dis = answer_mean_states(model, tokenizer, modules, args.layers, [t + p for t, p in zip(texts_p, prefixes_p)],
                             [r["contrastive_pair"]["dishonest"] for r in pressure], args.batch_size, args.max_tokens)
    hon_e = answer_mean_states(model, tokenizer, modules, args.layers, [t + p for t, p in zip(texts_e, prefixes_e)],
                               [r["contrastive_pair"]["honest"] for r in eval_pairs], args.batch_size, args.max_tokens)
    dis_e = answer_mean_states(model, tokenizer, modules, args.layers, [t + p for t, p in zip(texts_e, prefixes_e)],
                               [r["contrastive_pair"]["dishonest"] for r in eval_pairs], args.batch_size, args.max_tokens)
    log(f"activations done in {time.time() - t0:.0f}s")
    for name, store in (("pressure", Hp), ("benign", Hb), ("honest", hon), ("dishonest", dis)):
        bad = sum(int((~torch.isfinite(t)).any(-1).sum()) for t in store.values())
        if bad:
            raise RuntimeError(f"{bad} non-finite {name} activation rows; refusing to fit")

    # --- fit per layer + checks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims = [r["dim"] for r in pressure]
    dims_e = [r["dim"] for r in eval_pairs]
    prot_dims = [r["dim"] for r in eval_protected]
    read_dirs, write_dirs, per_layer, gate_args, dim_order = {}, {}, {}, {}, None
    for l in args.layers:
        zero = set() if args.zero_dims == ["none"] else set(args.zero_dims)
        fit = fit_layer(Hp[l].to(device), dims, (hon[l] - dis[l]).to(device), Hb[l].to(device), args.null_frac, args.alpha,
                        args.normalize, zero)
        read_dirs[l], write_dirs[l] = fit["read"].cpu(), fit["write"].cpu()
        dim_order = fit["dims"]
        U = fit["U"]
        G = Hg[l].to(device)
        pec = {}
        for d in sorted(set(prot_dims)):
            idx = [i for i, dd in enumerate(prot_dims) if dd == d]
            X = G[idx]
            pec[d] = round(1.0 - float(((X @ U) ** 2).sum() / (X * X).sum()), 4)
        pred_e = apply(fit["read"], fit["write"], He[l].to(device))
        D_e = (hon_e[l] - dis_e[l]).to(device)
        ref = float(pred_e.norm(dim=1).mean())
        leak = {}
        for d in sorted(set(prot_dims)):
            idx = [i for i, dd in enumerate(prot_dims) if dd == d]
            leak[d] = round(float(apply(fit["read"], fit["write"], G[idx]).norm(dim=1).mean()) / ref, 3)
        per_dim_r2 = {}
        for d in sorted(set(dims_e)):
            idx = [i for i, dd in enumerate(dims_e) if dd == d]
            per_dim_r2[d] = round(r2(pred_e[idx], D_e[idx]), 3)
        # routing on held-out prompts: weight on the row's own dim and argmax accuracy over the active dims
        w_e = He[l].to(device) @ fit["read"].T
        active_idx = [j for j, d in enumerate(dim_order) if d not in zero]
        own_e = [float(w_e[i, dim_order.index(d)]) for i, d in enumerate(dims_e)]
        acc_e = [dim_order[active_idx[int(w_e[i, active_idx].argmax())]] == d for i, d in enumerate(dims_e) if d not in zero]
        per_layer[l] = {k: v for k, v in fit.items() if k not in ("read", "write", "U")}
        per_layer[l].update({"pec_eval": pec, "leak_eval": leak, "r2_pair_eval": round(r2(pred_e, D_e), 4),
                             "r2_pair_eval_per_dim": per_dim_r2, "dose_eval": round(ref / float(He[l].to(device).norm(dim=1).mean()), 4),
                             "gap_eval": round(float(D_e.norm(dim=1).mean() / He[l].to(device).norm(dim=1).mean()), 4),
                             "route_eval": {"own_weight_mean": round(sum(own_e) / max(1, len(own_e)), 3),
                                            "argmax_acc": round(sum(acc_e) / max(1, len(acc_e)), 3)}})
        log(f"layer {l}: active dims {fit['rank']}/{len(dim_order)}, null dim {fit['null_dim']}, R2 target train {fit['r2_target_train']:.3f}, "
            f"R2 pair eval {per_layer[l]['r2_pair_eval']:.3f}, min PEC {min(pec.values()):.4f} ({min(pec, key=pec.get)}), "
            f"max leak {max(leak.values()):.2f} ({max(leak, key=leak.get)}); routing argmax acc train "
            f"{fit['route_train']['argmax_acc']:.2f} eval {per_layer[l]['route_eval']['argmax_acc']:.2f}")
        log(f"  eval R2 per dim: {per_dim_r2}")

    # --- gate statistics: projections of the honest / dishonest answer tokens on each dim's unit target (train = the
    # statistics the control uses; eval = how the gate would behave on held-out pairs)
    unit = {l: write_dirs[l] / write_dirs[l].norm(dim=1, keepdim=True).clamp(min=1e-6) for l in args.layers}
    active = {d: float(write_dirs[args.layers[0]][dim_order.index(d)].norm()) > 1e-6 for d in dim_order}
    for split, rows_, texts_, prefixes_, hold in (("train", pressure, texts_p, prefixes_p, "gate"), ("eval", eval_pairs, texts_e, prefixes_e, "gate_eval")):
        row_dirs = {l: unit[l][[dim_order.index(r["dim"]) for r in rows_]] for l in args.layers}
        prompts_ = [t + p for t, p in zip(texts_, prefixes_)]
        proj_h = answer_token_projections(model, tokenizer, modules, args.layers, prompts_,
                                          [r["contrastive_pair"]["honest"] for r in rows_], row_dirs, args.batch_size, args.max_tokens)
        proj_d = answer_token_projections(model, tokenizer, modules, args.layers, prompts_,
                                          [r["contrastive_pair"]["dishonest"] for r in rows_], row_dirs, args.batch_size, args.max_tokens)
        for l in args.layers:
            pos, neg = collections.defaultdict(list), collections.defaultdict(list)
            for r, ph, pd in zip(rows_, proj_h[l], proj_d[l]):
                pos[r["dim"]] += ph; neg[r["dim"]] += pd
            stats = gate_statistics(pos, neg, dim_order, active)
            per_layer[l][hold] = stats["report"]
            if split == "train":
                gate_args[str(l)] = stats["lists"]
            fires = {d: (v["fires_on_dishonest"], v["fires_on_honest"]) for d, v in stats["report"].items() if v.get("active")}
            log(f"  gate {split} layer {l} (fires on dishonest / on honest tokens): "
                + ", ".join(f"{d} {a:.2f}/{b:.2f}" for d, (a, b) in fires.items()))
    log(f"activations + gate statistics done in {time.time() - t0:.0f}s")

    from steerability.algorithms.state_control.common.steering_vector import SteeringVector

    meta = artifact_provenance_meta(model, tokenizer)
    meta["fit"] = {"method": "nullspace_map_gated", "model": model_id, "quantize": args.quantize, "layers": args.layers,
                   "pairs_per_dim": args.pairs_per_dim, "n_pairs": len(pressure), "n_benign": len(benign),
                   "null_frac": args.null_frac, "alpha_rel": args.alpha, "normalize": args.normalize,
                   "zero_dims": args.zero_dims, "data": str(data), "seed": args.seed, "dims": dim_order,
                   "write_rule": args.write_rule, "pairs_override": [str(p) for p in args.pairs_override]}
    model_type = getattr(model.config, "model_type", "unknown")
    read_vec = SteeringVector(model_type=model_type, directions=read_dirs, meta=dict(meta))
    write_vec = SteeringVector(model_type=model_type, directions=write_dirs, meta=dict(meta))
    read_vec.save(str(out / "read.svec")); write_vec.save(str(out / "write.svec"))

    # --- dose sweep through the control itself
    import controls  # noqa: F401  (registers the method)
    from controls import NullSpaceMapGated as NullSpaceMap
    from steerability.algorithms.core.steering_pipeline import SteeringPipeline

    plain = SteeringPipeline(model_name_or_path=model_id, model=model, tokenizer=tokenizer, controls=[]); plain.steer()
    template_kwargs = dict(REGIME[model_id]["chat_template_kwargs"])

    def ids(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def logprob(pipeline, prompt_ids, cont_ids) -> float:
        lp = pipeline.compute_logprobs(torch.tensor([prompt_ids]), ref_output_ids=torch.tensor([cont_ids]))
        return float(lp.float().sum())

    gated = args.write_rule != "fixed"
    doses = args.alphas if gated else args.lams

    def make_control(dose: float) -> NullSpaceMap:
        kw = dict(read=read_vec, write=write_vec, write_rule=args.write_rule, floor=args.floor, w_clip=args.w_clip,
                  dims=dim_order, gate=gate_args or None)
        kw["alpha" if gated else "lam"] = dose
        return NullSpaceMap(**kw)

    def take_stats(ctrl: NullSpaceMap, bucket: dict) -> None:
        """Fold the control's recorded gate diagnostics (summed over layers) into `bucket` and reset them."""
        for st in (ctrl.stats or {}).values():
            bucket["tokens"] += st["tokens"]; bucket["fired"] += st["fired"]; bucket["shift_norm"] += st["shift_norm"]
        ctrl.stats = {}

    def stats_summary(bucket: dict) -> dict:
        n = max(1, bucket["tokens"])
        return {"gate_fire_rate": round(bucket["fired"] / n, 3), "shift_norm_per_token": round(bucket["shift_norm"] / n, 3)}

    sweep = {"rule": args.write_rule, "doses": doses, "lams": doses, "rows": [], "protected": [], "arms": {}}
    chosen_lam = args.lam
    if not args.skip_sweep:
        weak_rows = [r for r in evalrows if ch.is_pressure(r) and r["dim"] in WEAK_DIMS and r.get("contrastive_pair")]
        rows = []
        for d in WEAK_DIMS:
            group = sorted((r for r in weak_rows if r["dim"] == d), key=lambda r: r["row_id"])
            rows += group[:args.sweep_rows_per_dim]
        prot = []
        for d in sorted(set(prot_dims)):
            group = sorted((r for r in eval_protected if r["dim"] == d), key=lambda r: r["row_id"])
            prot += group[:args.sweep_protected_per_dim]
        base = []
        for r in rows:
            # the control reads the last *prompt* token (where the map was fitted) and writes from there; the answer prefix
            # (the model's own empty thinking block) is therefore part of the scored continuation, not of the prompt
            prompt = render(r); pre = ids(ch.answer_prefix(model_id, prompt))
            p = ids(prompt)
            h, dsh = pre + ids(r["contrastive_pair"]["honest"]), pre + ids(r["contrastive_pair"]["dishonest"])
            base.append({"row_id": r["row_id"], "dim": r["dim"], "p": p, "h": h, "d": dsh, "n_pre": len(pre),
                         "lo0": logprob(plain, p, h) - logprob(plain, p, dsh)})
        pbase = []
        for r in prot:
            prompt = render(r)
            p = ids(prompt)
            gen = plain.generate(messages=[ch.chat(r)], return_output=True, do_sample=False, max_new_tokens=32,
                                 chat_template_kwargs=template_kwargs)
            gen = gen[0] if isinstance(gen, list) else gen
            cont = [int(t) for t in gen.output_ids[0]]
            pbase.append({"row_id": r["row_id"], "dim": r["dim"], "p": p, "c": cont, "lp0": logprob(plain, p, cont)})
        sweep["rows"] = [{"row_id": b["row_id"], "dim": b["dim"], "logodds_unsteered": b["lo0"]} for b in base]
        sweep["protected"] = [{"row_id": b["row_id"], "dim": b["dim"]} for b in pbase]
        log(f"sweep: {len(base)} weak-dim rows (unsteered mean log-odds {sum(b['lo0'] for b in base) / len(base):+.2f}), "
            f"{len(pbase)} protected rows")
        for dose in doses:
            ctrl = make_control(dose)
            ctrl.stats = {}
            pipe = SteeringPipeline(model_name_or_path=model_id, model=model, tokenizer=tokenizer, controls=[ctrl]); pipe.steer()
            dlo, dh, per_dim = [], [], collections.defaultdict(list)
            buckets = {k: {"tokens": 0, "fired": 0, "shift_norm": 0.0} for k in ("honest", "dishonest", "protected")}
            for b in base:
                lh = logprob(pipe, b["p"], b["h"]); take_stats(ctrl, buckets["honest"])
                ld = logprob(pipe, b["p"], b["d"]); take_stats(ctrl, buckets["dishonest"])
                dlo.append((lh - ld) - b["lo0"]); per_dim[b["dim"]].append(dlo[-1])
                dh.append(lh - logprob(plain, b["p"], b["h"]))
            dprot = collections.defaultdict(list)
            for b in pbase:
                dprot[b["dim"]].append(abs(logprob(pipe, b["p"], b["c"]) - b["lp0"]) / max(1, len(b["c"])))
                take_stats(ctrl, buckets["protected"])
            arm = {"mean_delta_logodds": round(sum(dlo) / len(dlo), 3), "median_delta_logodds": round(sorted(dlo)[len(dlo) // 2], 3),
                   "frac_improved": round(sum(x > 0 for x in dlo) / len(dlo), 3), "mean_delta_logp_honest": round(sum(dh) / len(dh), 3),
                   "per_dim": {d: round(sum(v) / len(v), 3) for d, v in sorted(per_dim.items())},
                   "protected_abs_delta_logp_per_token": {d: round(sum(v) / len(v), 4) for d, v in sorted(dprot.items())},
                   "gate": {k: stats_summary(v) for k, v in buckets.items()}}
            sweep["arms"][str(dose)] = arm
            g = arm["gate"]
            log(f"{'alpha' if gated else 'lam'} {dose}: delta log-odds {arm['mean_delta_logodds']:+.2f} (median {arm['median_delta_logodds']:+.2f}, improved "
                f"{arm['frac_improved']:.2f}, honest {arm['mean_delta_logp_honest']:+.2f}); protected |dlogp|/tok max "
                f"{max(arm['protected_abs_delta_logp_per_token'].values()):.4f}; gate fires on honest/dishonest/protected tokens "
                f"{g['honest']['gate_fire_rate']:.2f}/{g['dishonest']['gate_fire_rate']:.2f}/{g['protected']['gate_fire_rate']:.2f}, "
                f"shift/token {g['honest']['shift_norm_per_token']:.1f}/{g['dishonest']['shift_norm_per_token']:.1f}/{g['protected']['shift_norm_per_token']:.1f}")
        if chosen_lam is None:
            ok = [float(l) for l, a in sweep["arms"].items()
                  if max(a["protected_abs_delta_logp_per_token"].values()) <= args.kl_budget and a["mean_delta_logp_honest"] > 0]
            chosen_lam = max(ok, key=lambda l: sweep["arms"][str(l)]["mean_delta_logodds"]) if ok else min(doses)
            sweep["chosen_lam"] = chosen_lam
            log(f"chosen {'alpha' if gated else 'lam'} {chosen_lam} (largest dose with honest log p rising and protected |dlogp|/tok <= {args.kl_budget})")
    elif chosen_lam is None:
        chosen_lam = 0.5 if gated else 1.0
    sweep["chosen_lam"] = chosen_lam
    (out / "sweep.json").write_text(json.dumps(sweep, indent=1))

    # --- bundle
    from steerability.spipe import SPipe

    ctrl = make_control(chosen_lam)
    pipe = SteeringPipeline(model_name_or_path=model_id, model=model, tokenizer=tokenizer, controls=[ctrl]); pipe.steer()
    bundle_dir = out / "bundles"; bundle_dir.mkdir(exist_ok=True)
    bundle_path = bundle_dir / f"{model_slug(model_id)}.spipe"
    pipe.to_spipe(model_ref=model_id).save(str(bundle_path))
    spipe = SPipe.load(bundle_path, allow_code=True)
    report = spipe.verify()
    log(f"bundle {bundle_path} ({bundle_path.stat().st_size / 2**20:.2f} MB); verify ok {report.ok}\n{report.render()}")
    # roundtrip on one pressure row
    probe = pressure[0]; prompt = render(probe); p = ids(prompt); cont = ids(ch.answer_prefix(model_id, prompt)) + ids(probe["contrastive_pair"]["honest"])
    loaded = spipe.pipeline(backend="huggingface", model=model, tokenizer=tokenizer); loaded.steer()
    lp_live, lp_loaded, lp_plain = logprob(pipe, p, cont), logprob(loaded, p, cont), logprob(plain, p, cont)
    log(f"roundtrip: live {lp_live:.4f} loaded {lp_loaded:.4f} plain {lp_plain:.4f}")
    if abs(lp_live - lp_loaded) > 1e-2:
        raise RuntimeError("the reloaded bundle does not reproduce the live pipeline")

    # --- submission-shaped folder for check-submission
    sub = out / "submission"
    if sub.exists():
        shutil.rmtree(sub)
    sub.mkdir(parents=True)
    shutil.copy(bundle_path, sub / bundle_path.name)          # bundles live at the submission root
    shutil.copytree(HERE / "controls", sub / "controls", ignore=shutil.ignore_patterns("__pycache__"))
    rule_text = (f"the same shift on every answer token, dose lam={chosen_lam}" if not gated else
                 f"a per-token projection-gated write ({args.write_rule}, Herbster et al. 2026: a token is pushed only while its "
                 f"projection on the routed dim's honesty axis is below the fitted boundary, and only by the deficit), alpha={chosen_lam}"
                 + (f", floor={args.floor}" if args.floor else ""))
    n_active = int((write_dirs[args.layers[0]].norm(dim=1) > 1e-6).sum())
    (sub / "method_card.md").write_text(
        f"# NullSpaceMapGated (exp_rahul_002)\n\nA learned activation map at decoder layer(s) {args.layers} of {model_id}: "
        "router rows in the null space of capability/guard/twin prompts (read at the last prompt token) select among "
        f"{n_active} dishonesty dims; each dim's target is its honest-minus-dishonest answer state. Write rule: {rule_text}. "
        f"Fitted on {data.name} train ({args.pairs_per_dim} pairs/dim"
        + (f"; {overrides['applied']} contrastive pairs re-worded to contrast the act, see build_pairs_override.py" if overrides["applied"] else "")
        + (f"; zeroed dims: {', '.join(args.zero_dims)}" if args.zero_dims != ["none"] else "")
        + "). White-box, HF backend.\n")

    summary = {"model": model_id, "quantize": args.quantize, "dtype": str(model.dtype), "layers": args.layers,
               "write_rule": args.write_rule, "dose": chosen_lam, "lam": chosen_lam, "floor": args.floor, "w_clip": args.w_clip,
               "data": str(data), "pairs_override": {**overrides, "per_dim": dict(overrides["per_dim"])},
               "null_frac": args.null_frac, "alpha_rel": args.alpha, "normalize": args.normalize, "zero_dims": args.zero_dims,
               "dims": dim_order, "n_pairs": len(pressure), "n_benign": len(benign), "per_layer": per_layer,
               "bundle": str(bundle_path), "bundle_kb": round(bundle_path.stat().st_size / 1024, 1), "verify_ok": report.ok,
               "verify_warnings": report.warnings, "submittable": args.quantize is None,
               "roundtrip": {"live": lp_live, "loaded": lp_loaded, "plain": lp_plain}}
    (out / "fit_summary.json").write_text(json.dumps(summary, indent=1))
    log(f"wrote {out / 'fit_summary.json'}")


if __name__ == "__main__":
    main()
