# Third-party code

`starterkit/` is an unmodified copy of the official Steerability Challenge Starter Kit's Python package
(the private kit our team received as registered participants), copied on 2026-10-29 from the team's
`dpo_finetune_kit/starterkit/`. `STARTERKIT_DATA_README.md` is the kit's `data/README.md`, the authoritative
description of the row schema, the declining rule, and the MASK rules.

Do not edit these files. Our scorer (`eval/scorer.py`) imports the official judge (`starterkit.judge`) and answer
readers (`starterkit.parse`) from here, so that our local numbers follow the same rules as the leaderboard.
The kit is private: keep this repository private.

`steerability-toolkit/` is the organisers' steerability toolkit, version 0.5.6 (Apache-2.0; see its `LICENSE` and
`NOTICE`), copied on 2026-10-29 from `steerability-main/` with only what installation needs (the `steerability`
package, `pyproject.toml`, `README.md`, `LICENSE`, `NOTICE`). It is not on PyPI, so Colab installs it from here:
`pip install -e third_party/steerability-toolkit`. Do not edit it; update it by copying a newer release.
