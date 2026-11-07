"""Fetching the organizers' baselines from the Hub.

`fetch_baselines` downloads baseline directories from `BASELINES_REPO` at `BASELINES_REVISION` into
`BASELINES_DIR`, one subdirectory per baseline. The repository is public and the download needs no
Hugging Face login. The revision is a commit hash, so a second call transfers nothing and makes no
request: the per-file metadata under `BASELINES_DIR/.cache/huggingface/` records that hash, and a
file already at it is kept. A directory left incomplete by an interrupted call is completed.

A baseline directory is a submission directory, i.e., one frozen `<model-slug>.spipe` per base model
plus a `method_card.md`, and `check-submission` runs on it as it stands. The names come from the
repository listing, so a baseline the organizers add needs no change here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from starterkit import BASELINES_DIR, BASELINES_REPO, BASELINES_REVISION, METHOD_CARD


class BaselinesError(RuntimeError):
    """A baseline name the repository does not contain, or a fetch that could not reach the Hub."""


def _hub_errors() -> tuple[type[BaseException], ...]:
    """The exception types a fetch reports as an unreachable Hub."""
    import httpx
    from huggingface_hub.errors import (
        HfHubHTTPError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
    )

    return (
        httpx.ConnectError,
        httpx.TimeoutException,
        OfflineModeIsEnabled,
        LocalEntryNotFoundError,
        HfHubHTTPError,
    )


def available_baselines() -> list[str]:
    """The baseline names the repository contains at the pinned revision, sorted.

    A name is a top-level directory of the repository. The listing is one request and downloads
    nothing.

    Returns:
        The names, sorted.

    Raises:
        BaselinesError: If the Hub could not be reached, or if the repository or the pinned revision
            does not exist.
    """
    from huggingface_hub import HfApi

    try:
        files = HfApi().list_repo_files(
            repo_id=BASELINES_REPO, revision=BASELINES_REVISION, repo_type="dataset"
        )
    except _hub_errors() as error:
        raise BaselinesError(
            f"the baselines repository {BASELINES_REPO} at {BASELINES_REVISION} could not be listed "
            f"({type(error).__name__}: {error}); fetching baselines needs network access to "
            f"huggingface.co."
        ) from error
    return sorted({name for name, _, rest in (path.partition("/") for path in files) if rest})


def fetch_baselines(names: list[str] | None = None) -> list[Path]:
    """Download baseline directories into `BASELINES_DIR` and return their paths.

    The repository is public and the download needs no login. A second call with the same names
    transfers nothing, since the pinned revision is a commit hash and the files already at that hash
    are kept.

    Args:
        names: The baselines to fetch; None fetches every baseline the repository contains.

    Returns:
        One directory path per requested baseline, in the order of `names`, or sorted when `names` is
        None.

    Raises:
        BaselinesError: If a requested name is not in the repository, if the Hub could not be
            reached, or if the repository or the pinned revision does not exist.
    """
    from huggingface_hub import snapshot_download

    published = available_baselines()
    wanted = published if names is None else list(names)
    missing = [name for name in wanted if name not in published]
    if missing:
        raise BaselinesError(
            f"{missing} is not in the baselines repository {BASELINES_REPO} at "
            f"{BASELINES_REVISION}; the baselines it contains are {published}."
        )

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    if not wanted:
        return []

    try:
        snapshot_download(
            repo_id=BASELINES_REPO,
            repo_type="dataset",
            revision=BASELINES_REVISION,
            local_dir=BASELINES_DIR,
            allow_patterns=[f"{name}/*" for name in wanted],
        )
    except _hub_errors() as error:
        raise BaselinesError(
            f"the baselines {wanted} could not be fetched from {BASELINES_REPO} at "
            f"{BASELINES_REVISION} ({type(error).__name__}: {error}); fetching baselines needs "
            f"network access to huggingface.co."
        ) from error
    return [BASELINES_DIR / name for name in wanted]


def _summarize(directory: Path) -> str:
    """One report line for one fetched baseline: its bundle count and whether its card is present."""
    bundles = sorted(directory.glob("*.spipe"))
    card = "card" if (directory / METHOD_CARD).is_file() else f"no {METHOD_CARD}"
    return f"{directory.name}: {len(bundles)} bundles, {card}"


def main(argv: list[str] | None = None) -> int:
    """`fetch-baselines [name ...] [--list]`.

    Downloads the named baselines, or every baseline when none is named, and prints one line per
    baseline with its bundle count and whether its method card is present. `--list` prints the
    available names and downloads nothing.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 on success, 1 where a name is unknown or the Hub could not be reached.
    """
    parser = argparse.ArgumentParser(
        prog="fetch-baselines",
        description=(
            f"Download the organizers' baselines from {BASELINES_REPO} at the pinned revision into "
            f"{BASELINES_DIR.name}/, one directory per baseline. The repository is public and the "
            "download needs no Hugging Face login. Running it again transfers nothing. Each "
            "baseline directory is a submission directory that check-submission runs on."
        ),
    )
    parser.add_argument("names", nargs="*", help="the baselines to fetch; none fetches all of them")
    parser.add_argument(
        "--list", action="store_true", help="print the available baseline names and fetch nothing"
    )
    args = parser.parse_args(argv)

    try:
        if args.list:
            for name in available_baselines():
                print(name)
            return 0
        for directory in fetch_baselines(args.names or None):
            print(_summarize(directory))
    except BaselinesError as error:
        print(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
