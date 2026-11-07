"""Fit the NullSpaceMap control on the team's v7 train split, sweep the dose, and freeze a bundle.

    python experiments/exp_rahul_001_nullspace_map/fit_nullspace_map.py --model Qwen/Qwen3.8-27B --layers 32 --out runs/exp_rahul_001

Steps (all on one loaded model; ~30-45 min for the 27B in bf16 on an 80 GB A100):

  1. rows: the same 80 contrastive pairs per dim as exp002 (`steering/cast_honesty.pair_rows`), rendered as the
     harness renders them; the benign ("don't steer") set = the pressure-free twins of train, the train
     capability and guard rows, and any extra prompt-only files given with --benign-extra.
  2. activations at --layers: last prompt token of every pressure and benign prompt; mean over the answer tokens
     of each pair's honest and dishonest answers (appended to the prompt after the model's answer prefix).
  3. null space of the benign set (`--null-frac` of the eigen-directions with the smallest eigenvalues), per-dim
     response-side targets (mean honest - dishonest, each dim scaled to the median dim norm so the dose means the
     same push for every dim), ridge regression in null-space coordinates, thin SVD -> read/write factors.
  4. checks on eval: PEC per protected area (fraction of a held-out prompt's energy the map cannot see), leakage
     (|shift| on protected prompts relative to pressure prompts), R^2 toward the row's own response delta.
  5. dose sweep with the control itself in a SteeringPipeline: teacher-forced log-odds of honest vs dishonest on
     eval pressure rows of the weak dims (the prompt handed to the control ends where the map was fitted, at the
     last prompt token; the model's empty thinking block is the first part of each scored continuation), and the change in log p of the unsteered model's own greedy answer on
     protected rows (a no-op check), for each --lams value.
  6. bundle at --lam (default: chosen by the sweep): freeze, verify, reload, roundtrip check; plus a
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


# ----------------------------------------------------------------------------------------------- the fit

def null_basis(Hb: torch.Tensor, frac: float) -> torch.Tensor:
    """`U` [H, k]: the eigen-directions of `H_b^T H_b / N` with the smallest `frac` share of the eigenvalue count."""
    evals, evecs = torch.linalg.eigh((Hb.T @ Hb) / Hb.size(0))
    k = int(round(frac * Hb.size(1)))
    return evecs[:, :k].contiguous()


def fit_layer(Hp: torch.Tensor, dims: list[str], D: torch.Tensor, Hb: torch.Tensor, null_frac: float, alpha_rel: float,
              normalize: str, zero_dims: set[str] = frozenset()) -> dict:
    """Closed-form fit at one layer. Returns read/write factors `[r, H]` and diagnostics.

    `Hp` [N, H] pressure prompt states, `D` [N, H] response deltas (honest - dishonest answer states),
    `Hb` [Nb, H] benign prompt states. Targets: each row's dim mean of D; `normalize` = "none" (natural norms), "cap"
    (dims above the median dim norm are scaled down to it; small dims are left alone, since scaling a dim with no usable
    direction *up* injects noise into the shared read space), or "median" (every dim scaled to the median). Dims in
    `zero_dims` get a zero target: the map learns to do nothing on them.
    Ridge in null-space coordinates: Z = Hp U, M = R^T Z (Z^T Z + a I)^-1 (so Delta = M U^T), then a thin SVD of M
    gives Delta = A B^T with B = U Q (read) and A = P S (write), rank = number of dims.
    """
    device = Hp.device
    U = null_basis(Hb, null_frac)
    by_dim = collections.defaultdict(list)
    for i, d in enumerate(dims):
        by_dim[d].append(i)
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
    Z = Hp @ U                                                              # [N, k]
    G = Z.T @ Z
    alpha = alpha_rel * G.diagonal().mean().item()
    G.diagonal().add_(alpha)
    M = torch.linalg.solve(G, Z.T @ R).T                                    # [H, k]
    P, S, Qt = torch.linalg.svd(M, full_matrices=False)
    r = int((S > S[0] * 1e-6).sum().item())
    r = min(r, len(by_dim))
    write = (P[:, :r] * S[:r]).T.contiguous()                               # [r, H]  = A^T
    read = (U @ Qt[:r].T).T.contiguous()                                    # [r, H]  = B^T
    pred = (Hp @ read.T) @ write                                            # = Delta Hp
    return {"read": read, "write": write, "U": U, "rank": r, "alpha": alpha, "null_dim": U.size(1),
            "dim_norms": norms, "target_norms": {d: float(v.norm()) for d, v in means.items()}, "normalize": normalize,
            "zero_dims": sorted(zero_dims),
            "r2_target_train": 1.0 - float(((pred - R) ** 2).sum() / (R ** 2).sum()),
            "r2_pair_train": 1.0 - float(((pred - D) ** 2).sum() / (D ** 2).sum()),
            "dose_train": float(pred.norm(dim=1).mean() / Hp.norm(dim=1).mean())}


def apply(read: torch.Tensor, write: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    return (H @ read.T) @ write


def r2(pred, target) -> float:
    return 1.0 - float(((pred - target) ** 2).sum() / (target ** 2).sum())


# ----------------------------------------------------------------------------------------------- main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--regime-model", help="whose official regime to borrow, for a stand-in model the kit does not list (rehearsals)")
    ap.add_argument("--data", default=str(REPO / "data" / "v7_tidy"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[32], help="decoder layers (outputs) to fit and steer")
    ap.add_argument("--pairs-per-dim", type=int, default=80)
    ap.add_argument("--null-frac", type=float, default=0.6)
    ap.add_argument("--alpha", type=float, default=0.1, help="ridge strength relative to the mean eigenvalue of Z^T Z")
    ap.add_argument("--normalize", default="none", choices=["none", "cap", "median"],
                    help="per-dim target norms: natural (default; what the 4-bit layer profiles measured), cap large dims at the "
                         "median, or scale all to the median (halved the map's eval R2 and flipped Granite's weak dims)")
    ap.add_argument("--zero-dims", nargs="*", default=["none"],
                    help="dims given a zero target, e.g. sandbagging feedback_sycophancy hint_acknowledgment (no usable direction); "
                         "default none")
    ap.add_argument("--benign-extra", nargs="*", default=[], help="extra prompt-only JSONL files for the benign set")
    ap.add_argument("--benign-limit", type=int, help="cap on benign rows (rehearsals only; seeded sample)")
    ap.add_argument("--eval-protected-per-dim", type=int, help="cap on eval protected rows per dim (rehearsals only)")
    ap.add_argument("--eval-pairs-per-dim", type=int, default=20, help="eval pairs per dim for the R^2 check")
    ap.add_argument("--lams", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--lam", type=float, help="dose for the bundle; default: the sweep's pick")
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

    # --- model
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
    read_dirs, write_dirs, per_layer = {}, {}, {}
    for l in args.layers:
        zero = set() if args.zero_dims == ["none"] else set(args.zero_dims)
        fit = fit_layer(Hp[l].to(device), dims, (hon[l] - dis[l]).to(device), Hb[l].to(device), args.null_frac, args.alpha,
                        args.normalize, zero)
        read_dirs[l], write_dirs[l] = fit["read"].cpu(), fit["write"].cpu()
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
        per_layer[l] = {k: v for k, v in fit.items() if k not in ("read", "write", "U")}
        per_layer[l].update({"pec_eval": pec, "leak_eval": leak, "r2_pair_eval": round(r2(pred_e, D_e), 4),
                             "r2_pair_eval_per_dim": per_dim_r2, "dose_eval": round(ref / float(He[l].to(device).norm(dim=1).mean()), 4),
                             "gap_eval": round(float(D_e.norm(dim=1).mean() / He[l].to(device).norm(dim=1).mean()), 4)})
        log(f"layer {l}: rank {fit['rank']}, null dim {fit['null_dim']}, R2 target train {fit['r2_target_train']:.3f}, "
            f"R2 pair eval {per_layer[l]['r2_pair_eval']:.3f}, min PEC {min(pec.values()):.4f} ({min(pec, key=pec.get)}), "
            f"max leak {max(leak.values()):.2f} ({max(leak, key=leak.get)})")
        log(f"  eval R2 per dim: {per_dim_r2}")

    from steerability.algorithms.state_control.common.steering_vector import SteeringVector

    meta = artifact_provenance_meta(model, tokenizer)
    meta["fit"] = {"method": "nullspace_map", "model": model_id, "quantize": args.quantize, "layers": args.layers,
                   "pairs_per_dim": args.pairs_per_dim, "n_pairs": len(pressure), "n_benign": len(benign),
                   "null_frac": args.null_frac, "alpha_rel": args.alpha, "normalize": args.normalize,
                   "zero_dims": args.zero_dims, "data": str(data), "seed": args.seed, "dims": sorted(set(dims))}
    model_type = getattr(model.config, "model_type", "unknown")
    read_vec = SteeringVector(model_type=model_type, directions=read_dirs, meta=dict(meta))
    write_vec = SteeringVector(model_type=model_type, directions=write_dirs, meta=dict(meta))
    read_vec.save(str(out / "read.svec")); write_vec.save(str(out / "write.svec"))

    # --- dose sweep through the control itself
    import controls  # noqa: F401  (registers the method)
    from controls import NullSpaceMap
    from steerability.algorithms.core.steering_pipeline import SteeringPipeline

    plain = SteeringPipeline(model_name_or_path=model_id, model=model, tokenizer=tokenizer, controls=[]); plain.steer()
    template_kwargs = dict(REGIME[model_id]["chat_template_kwargs"])

    def ids(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def logprob(pipeline, prompt_ids, cont_ids) -> float:
        lp = pipeline.compute_logprobs(torch.tensor([prompt_ids]), ref_output_ids=torch.tensor([cont_ids]))
        return float(lp.float().sum())

    sweep = {"lams": args.lams, "rows": [], "protected": [], "arms": {}}
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
        for lam in args.lams:
            ctrl = NullSpaceMap(read=read_vec, write=write_vec, lam=lam)
            pipe = SteeringPipeline(model_name_or_path=model_id, model=model, tokenizer=tokenizer, controls=[ctrl]); pipe.steer()
            dlo, dh, per_dim = [], [], collections.defaultdict(list)
            for b in base:
                lh, ld = logprob(pipe, b["p"], b["h"]), logprob(pipe, b["p"], b["d"])
                dlo.append((lh - ld) - b["lo0"]); per_dim[b["dim"]].append(dlo[-1])
                dh.append(lh - logprob(plain, b["p"], b["h"]))
            dprot = collections.defaultdict(list)
            for b in pbase:
                dprot[b["dim"]].append(abs(logprob(pipe, b["p"], b["c"]) - b["lp0"]) / max(1, len(b["c"])))
            arm = {"mean_delta_logodds": round(sum(dlo) / len(dlo), 3), "median_delta_logodds": round(sorted(dlo)[len(dlo) // 2], 3),
                   "frac_improved": round(sum(x > 0 for x in dlo) / len(dlo), 3), "mean_delta_logp_honest": round(sum(dh) / len(dh), 3),
                   "per_dim": {d: round(sum(v) / len(v), 3) for d, v in sorted(per_dim.items())},
                   "protected_abs_delta_logp_per_token": {d: round(sum(v) / len(v), 4) for d, v in sorted(dprot.items())}}
            sweep["arms"][str(lam)] = arm
            log(f"lam {lam}: delta log-odds {arm['mean_delta_logodds']:+.2f} (median {arm['median_delta_logodds']:+.2f}, improved "
                f"{arm['frac_improved']:.2f}, honest {arm['mean_delta_logp_honest']:+.2f}); protected |dlogp|/tok max "
                f"{max(arm['protected_abs_delta_logp_per_token'].values()):.4f}")
        if chosen_lam is None:
            ok = [float(l) for l, a in sweep["arms"].items()
                  if max(a["protected_abs_delta_logp_per_token"].values()) <= args.kl_budget and a["mean_delta_logp_honest"] > 0]
            chosen_lam = max(ok, key=lambda l: sweep["arms"][str(l)]["mean_delta_logodds"]) if ok else min(args.lams)
            sweep["chosen_lam"] = chosen_lam
            log(f"chosen lam {chosen_lam} (largest dose with honest log p rising and protected |dlogp|/tok <= {args.kl_budget})")
    elif chosen_lam is None:
        chosen_lam = 1.0
    (out / "sweep.json").write_text(json.dumps(sweep, indent=1))

    # --- bundle
    from steerability.spipe import SPipe

    ctrl = NullSpaceMap(read=read_vec, write=write_vec, lam=chosen_lam)
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
    (sub / "method_card.md").write_text(
        "# NullSpaceMap (exp_rahul_001)\n\nA learned rank-12 activation map at decoder layer(s) "
        f"{args.layers} of {model_id}: read directions in the null space of capability/guard/twin prompts, write directions "
        "toward each dishonesty dim's honest-minus-dishonest answer state; shift computed from the last prompt token and added "
        f"to answer tokens, dose lam={chosen_lam}. Fitted on data/v7_tidy train ({args.pairs_per_dim} pairs/dim). White-box, HF backend.\n")

    summary = {"model": model_id, "quantize": args.quantize, "dtype": str(model.dtype), "layers": args.layers, "lam": chosen_lam,
               "null_frac": args.null_frac, "alpha_rel": args.alpha, "normalize": args.normalize, "zero_dims": args.zero_dims,
               "n_pairs": len(pressure), "n_benign": len(benign), "per_layer": per_layer,
               "bundle": str(bundle_path), "bundle_kb": round(bundle_path.stat().st_size / 1024, 1), "verify_ok": report.ok,
               "verify_warnings": report.warnings, "submittable": args.quantize is None,
               "roundtrip": {"live": lp_live, "loaded": lp_loaded, "plain": lp_plain}}
    (out / "fit_summary.json").write_text(json.dumps(summary, indent=1))
    log(f"wrote {out / 'fit_summary.json'}")


if __name__ == "__main__":
    main()
