"""Re-freeze a NullSpaceMapGated bundle at another dose or zero list, from a finished fit, without loading the model.

    python experiments/exp_rahul_002_nullspace_gated/rebundle.py --run <RUN_DIR> --alpha 0.5 --out <RUN_DIR>/bundles_probe/alpha_0.5
    python experiments/exp_rahul_002_nullspace_gated/rebundle.py --run <RUN_DIR> --alpha 1.0 --zero-dims-add sandbagging_stated --out ...

Everything the control needs is in the fitted run: `read.svec` / `write.svec` (router rows and targets) and the gate
statistics, which are read back from the stage-A bundle (`bundles/<slug>.spipe`) — so a new dose, write rule, floor
or zero list is a few seconds of work instead of a 20-minute refit. Zeroing a dim at bundle time blanks its target
row (the router still recognises the dim's prompts; with a zero target it does nothing), exactly what `--zero-dims`
does at fit time. Freezing an unsteered pipeline needs no model (the toolkit allows it for recipe-frozen controls);
the artifacts keep their fit provenance, the lock records no model fingerprint. The result is verified, reloaded and
written as `<out>/<slug>.spipe` plus a `submission/` folder like the fit script's, with a `rebundle.json` note.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "third_party"))
sys.path.insert(0, str(HERE))

import controls  # noqa: E402, F401  (registers state_control/nullspace_map_gated)
from controls import NullSpaceMapGated  # noqa: E402
from starterkit import model_slug  # noqa: E402


def load_fitted(run: Path) -> NullSpaceMapGated:
    """The stage-A control, re-instantiated from the run's bundle (no model needed)."""
    from steerability.spipe import SPipe

    bundles = sorted((run / "bundles").glob("*.spipe"))
    if not bundles:
        raise SystemExit(f"no bundle under {run / 'bundles'}; run stage A first")
    spipe = SPipe.load(bundles[0])
    ctrls = spipe.instantiate_entry(0)
    ctrl = ctrls[0]
    if not isinstance(ctrl, NullSpaceMapGated):
        raise SystemExit(f"{bundles[0].name} holds a {type(ctrl).__name__}, not a NullSpaceMapGated")
    ctrl._model_ref = spipe.manifest["model"]["ref"]
    return ctrl


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, help="stage-A output folder (has bundles/, read.svec, write.svec, fit_summary.json)")
    ap.add_argument("--out", required=True, help="folder for the new bundle (+ submission/)")
    ap.add_argument("--alpha", type=float, help="new dose of the gated rules")
    ap.add_argument("--lam", type=float, help="new dose of the fixed rule")
    ap.add_argument("--write-rule", choices=["fixed", "sttp", "stmp"])
    ap.add_argument("--floor", type=float)
    ap.add_argument("--w-clip", type=float)
    ap.add_argument("--zero-dims-add", nargs="*", default=[], help="dims to zero in addition to the fit's")
    args = ap.parse_args(argv)

    run, out = Path(args.run), Path(args.out)
    fitted = load_fitted(run)
    model_ref = fitted._model_ref
    read, write = fitted.read.clone(), fitted.write.clone()
    dims = list(fitted.dims)
    zeroed = []
    for d in args.zero_dims_add:
        if d not in dims:
            raise SystemExit(f"{d!r} is not one of the fitted dims {dims}")
        for layer, t in write.directions.items():
            t[dims.index(d)] = 0.0
        zeroed.append(d)
    kw = dict(read=read, write=write, dims=dims, gate=fitted.gate, layer_ids=fitted.layer_ids,
              write_rule=args.write_rule or fitted.write_rule, alpha=args.alpha if args.alpha is not None else fitted.alpha,
              lam=args.lam if args.lam is not None else fitted.lam, floor=args.floor if args.floor is not None else fitted.floor,
              w_clip=args.w_clip if args.w_clip is not None else fitted.w_clip)
    ctrl = NullSpaceMapGated(**kw)

    from steerability.algorithms.core.steering_pipeline import SteeringPipeline
    from steerability.spipe import SPipe

    pipe = SteeringPipeline(model_name_or_path=model_ref, controls=[ctrl])
    spipe = pipe.to_spipe(freeze=True, model_ref=model_ref)
    out.mkdir(parents=True, exist_ok=True)
    bundle_path = out / f"{model_slug(model_ref)}.spipe"
    spipe.save(str(bundle_path))
    loaded = SPipe.load(bundle_path)
    report = loaded.verify()
    back = loaded.instantiate_entry(0)[0]
    active = [d for d, t in zip(dims, back.write.directions[sorted(back.write.directions)[0]]) if float(t.norm()) > 1e-6]
    sub = out / "submission"
    if sub.exists():
        shutil.rmtree(sub)
    sub.mkdir()
    shutil.copy(bundle_path, sub / bundle_path.name)
    shutil.copytree(HERE / "controls", sub / "controls", ignore=shutil.ignore_patterns("__pycache__"))
    card = run / "submission" / "method_card.md"
    text = card.read_text() if card.exists() else "# NullSpaceMapGated (exp_rahul_002)\n"
    text += (f"\nRe-bundled from {run.name}: write_rule={back.write_rule}, "
             f"{'alpha' if back.write_rule != 'fixed' else 'lam'}={back.alpha if back.write_rule != 'fixed' else back.lam}, "
             f"floor={back.floor}" + (f", additionally zeroed: {', '.join(zeroed)}" if zeroed else "") + ".\n")
    (sub / "method_card.md").write_text(text)
    note = {"from_run": str(run), "model": model_ref, "write_rule": back.write_rule, "alpha": back.alpha, "lam": back.lam,
            "floor": back.floor, "w_clip": back.w_clip, "zero_dims_added": zeroed, "active_dims": active,
            "bundle": str(bundle_path), "bundle_kb": round(bundle_path.stat().st_size / 1024, 1), "verify_ok": report.ok,
            "verify_warnings": list(report.warnings)}
    (out / "rebundle.json").write_text(json.dumps(note, indent=1))
    print(json.dumps(note, indent=1))
    if not report.ok:
        raise SystemExit("the re-frozen bundle does not verify:\n" + report.render())


if __name__ == "__main__":
    main()
