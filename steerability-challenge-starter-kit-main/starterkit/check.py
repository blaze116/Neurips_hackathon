"""The submission checker.

Runs on CPU, loads no model weights, and needs no network access. `SPipe.load`, `SPipe.verify`,
`SPipe.pipeline`, and `SteeringPipeline.check` are model-free. A submission is therefore checked
against the rules the organizer's intake applies before any generation happens.

Per bundle, the checker verifies that the bundle loads and passes `SPipe.verify`, is frozen, is
named `<model-slug>.spipe` for the base model it targets, stays under `MAX_BUNDLE_BYTES`, and builds
a pipeline the evaluation backend supports. Per submission, it verifies one bundle per base model,
one control sequence across the bundles, one track across the bundles, and a method card with
content.

The checker executes the submission's own `controls/` package, since a bundle that names an
out-of-tree control can be loaded only once that package has registered it.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from starterkit import (
    BACKEND,
    BASE_MODELS,
    BASELINES_DIR,
    MAX_BUNDLE_BYTES,
    METHOD_CARD,
    PIPELINE_KWARGS,
    model_slug,
)
from starterkit.track import control_tracks, track_of


def _size(path: Path) -> int:
    """The bundle's size in bytes; for a directory bundle, the sum over its files.

    Args:
        path: A `.spipe` bundle, either a zip file or a directory.

    Returns:
        The size in bytes.
    """
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def _import_controls(submission: Path, errors: list[str]) -> None:
    """Import the submission's `controls/` package so its `register_method` calls run.

    Registration is process-global. A package already imported from this same directory is left
    alone, since re-importing it would define new class objects that collide with the ones already
    registered. A `controls` package imported from another directory is an error, because the bundles
    would be checked against different code.

    Args:
        submission: The submission directory, prepended to `sys.path`.
        errors: Collects one message where the package is the wrong one or fails to import.
    """
    root = str(submission.resolve())
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    importlib.invalidate_caches()

    existing = sys.modules.get("controls")
    if existing is not None:
        origin = getattr(existing, "__file__", None)
        if origin is not None and Path(origin).resolve().parent.parent == Path(root):
            return
        errors.append(
            f"a different 'controls' package is already imported in this process (from "
            f"{origin}); check this submission in a fresh process so its own controls/ is the one "
            "that registers."
        )
        return

    try:
        importlib.import_module("controls")
    except Exception as exc:
        errors.append(
            f"controls/ failed to import ({type(exc).__name__}: {exc}); a bundle naming a control "
            "from this package cannot be loaded until the package imports and registers it."
        )


def _check_bundle(path: Path, errors: list[str], warnings: list[str]) -> dict | None:
    """Check one bundle on its own.

    Verifies that the bundle loads and passes `SPipe.verify`, is frozen, is named
    `<model-slug>.spipe` for the base model it targets, stays under `MAX_BUNDLE_BYTES`, and builds a
    pipeline the evaluation backend supports.

    Args:
        path: The `.spipe` bundle.
        errors: Collects the failures found.
        warnings: Collects the warnings `SPipe.verify` reports.

    Returns:
        `{file, model, size_bytes, methods, track, control_tracks}`, or None when the bundle could
        not be loaded. `track` is None where the pipeline could not be built or the track could not
        be determined.
    """
    from steerability.spipe import SPipe, SpipeError

    try:
        spipe = SPipe.load(path, allow_code=True)
    except SpipeError as exc:
        errors.append(f"{path.name}: could not load ({type(exc).__name__}: {exc}).")
        return None
    except Exception as exc:  # a malformed archive need not raise a SpipeError
        errors.append(f"{path.name}: could not load ({type(exc).__name__}: {exc}).")
        return None

    model = spipe.manifest["model"]["ref"]

    if model in BASE_MODELS:
        expected = f"{model_slug(model)}.spipe"
        if path.name != expected:
            errors.append(
                f"{path.name}: targets {model}; the bundle for that model must be named {expected}."
            )

    verify = spipe.verify()
    errors.extend(f"{path.name}: {message}" for message in verify.errors)
    warnings.extend(f"{path.name}: {message}" for message in verify.warnings)

    if not spipe.is_frozen:
        errors.append(
            f"{path.name}: not frozen (no lock section); save with pipeline.to_spipe(freeze=True); "
            "this needs no model when the pipeline has nothing to fit."
        )

    size = _size(path)
    if size > MAX_BUNDLE_BYTES:
        errors.append(
            f"{path.name}: {size} bytes exceeds the per-bundle cap of {MAX_BUNDLE_BYTES} bytes."
        )

    methods = [entry["method"] for entry in spipe.manifest["controls"] if entry.get("enabled", True)]

    # the track comes from the instantiated controls' own declarations; the pipeline checked for
    # backend support is the one that is classified
    track: str | None
    entries: list[dict] = []
    try:
        pipeline = spipe.pipeline(backend=BACKEND, **PIPELINE_KWARGS)
        for failure in pipeline.check().failures_for("generate"):
            errors.append(f"{path.name}: {failure.message}")
    except Exception as exc:
        errors.append(
            f"{path.name}: the pipeline could not be built from the bundle "
            f"({type(exc).__name__}: {exc})."
        )
        track = None
    else:
        controls = (*pipeline.structural_controls, *pipeline.input_controls,
                    *pipeline.state_controls, *pipeline.output_controls)
        try:
            entries = control_tracks(controls)
            track = track_of(controls)
        except Exception as exc:
            errors.append(
                f"{path.name}: the track could not be determined "
                f"({type(exc).__name__}: {exc})."
            )
            track = None

    return {
        "file": path.name,
        "model": model,
        "size_bytes": size,
        "methods": methods,
        "track": track,
        "control_tracks": entries,
    }


def check_submission(submission_dir: str | Path) -> dict:
    """Check a submission directory and report what an intake would find.

    Loads no model weights, needs no network access, and runs on CPU. The submission's `controls/`
    package is imported when there is one, which executes that code.

    The method card must exist with non-empty content, and nothing in it is parsed.

    Args:
        submission_dir: The directory containing the bundles, the method card, and any `controls/`.

    Returns:
        `{"submission", "ok", "track", "errors", "warnings", "bundles"}`, where `track` is None when
        the bundles disagree, when one of them did not build, or when none loaded, and `bundles` maps
        a model reference to `{file, size_bytes, methods, track, control_tracks}` (`control_tracks`
        contains one `{method, enabled, track, reason}` entry per control, and is empty when the
        pipeline could not be built).
    """
    submission = Path(submission_dir)
    errors: list[str] = []
    warnings: list[str] = []
    bundles: dict[str, dict] = {}

    if not submission.is_dir() or not any(submission.iterdir()):
        found = "not a directory" if not submission.is_dir() else "empty"
        remedy = (
            f" Run fetch-baselines {submission.name} to download it."
            if submission.resolve().parent == BASELINES_DIR else ""
        )
        return {
            "submission": str(submission),
            "ok": False,
            "track": None,
            "errors": [f"{submission}: {found}.{remedy}"],
            "warnings": [],
            "bundles": {},
        }

    # 1. the submission's own controls package, whose register_method calls must run first
    if (submission / "controls").is_dir():
        _import_controls(submission, errors)

    # 2. every bundle on its own
    for path in sorted(submission.glob("*.spipe")):
        record = _check_bundle(path, errors, warnings)
        if record is None:
            continue
        model = record.pop("model")
        if model in bundles:
            errors.append(
                f"{path.name}: a second bundle for {model} (already covered by "
                f"{bundles[model]['file']}); a submission contains exactly one bundle per base model."
            )
            continue
        bundles[model] = record

    # 3. the submission as a whole
    for model in BASE_MODELS:
        if model not in bundles:
            errors.append(f"no bundle for {model}; a submission covers every base model.")
    for model in sorted(set(bundles) - set(BASE_MODELS)):
        errors.append(
            f"{bundles[model]['file']}: targets {model}, which is not a base model "
            f"{list(BASE_MODELS)}."
        )

    sequences = {model: tuple(record["methods"]) for model, record in bundles.items()}
    if len(set(sequences.values())) > 1:
        listed = "; ".join(
            f"{bundles[model]['file']}: {list(sequence)}" for model, sequence in sorted(sequences.items())
        )
        errors.append(
            "the bundles do not share one control sequence; the same method must be submitted for "
            f"every base model (constructor arguments may differ per model). {listed}"
        )

    # a bundle that did not build has no track, and the submission then has none either
    tracks = {record["track"] for record in bundles.values()}
    if len(tracks - {None}) > 1:
        listed = "; ".join(
            f"{record['file']}: {record['track']}"
            for record in sorted(bundles.values(), key=lambda record: record["file"])
        )
        errors.append(f"the bundles are not in one track: {listed}")
    bundle_track = tracks.pop() if len(tracks) == 1 and None not in tracks else None

    # 4. the method card
    card_path = submission / METHOD_CARD
    if not card_path.exists():
        errors.append(f"no {METHOD_CARD}; every submission contains one.")
    elif not card_path.read_text(encoding="utf-8").strip():
        errors.append(f"{METHOD_CARD} is empty; every submission contains one with content.")

    return {
        "submission": str(submission),
        "ok": not errors,
        "track": bundle_track,
        "errors": errors,
        "warnings": warnings,
        "bundles": bundles,
    }


def main(argv: list[str] | None = None) -> int:
    """`check-submission <dir> [--out report.json]`.

    Prints the report. `--out` writes the JSON report to that path, and without it nothing is
    written.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 where the submission passes, 1 where it does not.
    """
    parser = argparse.ArgumentParser(
        prog="check-submission",
        description=(
            "Check a steerability-challenge submission: one frozen .spipe per base model, each named "
            "<model-slug>.spipe, each under the size cap and building a pipeline the evaluation "
            "backend supports, one shared control sequence, one track, and a method card with "
            "content. Reports the track the bundles are in. Runs on CPU, loads no model weights, and "
            "needs no network access. It imports and executes the submission's controls/ package "
            "when there is one, since a bundle that names an out-of-tree control loads only once "
            "that package has registered it."
        ),
    )
    parser.add_argument("submission", help="the submission directory")
    parser.add_argument("--out", default=None, help="where to write the JSON report")
    args = parser.parse_args(argv)

    report = check_submission(args.submission)

    print("PASS" if report["ok"] else "FAIL")
    print(f"track: {report['track']}")
    for model in sorted(report["bundles"]):
        record = report["bundles"][model]
        for entry in record["control_tracks"]:
            if entry["enabled"]:
                print(f"  {entry['method']}: {entry['track']} ({entry['reason']})")
        break        # the control sequence is the same across models; print it once
    for message in report["errors"]:
        print(f"  error: {message}")
    for message in report["warnings"]:
        print(f"  warning: {message}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report: {out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
