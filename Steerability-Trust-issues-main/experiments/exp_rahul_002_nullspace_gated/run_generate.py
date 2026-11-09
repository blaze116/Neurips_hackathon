"""Run the team's generation harness with this experiment's control registered.

`eval/generate_answers.py` loads a `.spipe` with `allow_code=True` but does not import a `controls` package, so a
bundle naming `state_control/nullspace_map_gated` can only be instantiated after this folder's `controls` is imported.
This wrapper does that (and registers exp_rahul_001's `state_control/nullspace_map` too, so a round-2 bundle can be
run on the same rows for comparison) and forwards every argument to the harness unchanged:

    python experiments/exp_rahul_002_nullspace_gated/run_generate.py --rows data/v17_thin_categories/eval.jsonl \
        --model Qwen/Qwen3.8-27B --arm nullspace_v1 --bundle runs/exp_rahul_001/bundles/Qwen__Qwen3.8-27B.spipe \
        --dims instructed_lie confrontation_after_lie concealment sandbagging_stated ... --per-dim 5 \
        --out runs/exp_rahul_001/answers.jsonl
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "third_party"))
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(HERE))

import controls  # noqa: E402, F401  (registers state_control/nullspace_map_gated)

# exp_rahul_001's package is also called `controls`; load it under another name so a round-2 bundle
# (`state_control/nullspace_map`) can be generated in the same process, for comparisons on the same rows.
import importlib.util  # noqa: E402

_old = REPO / "experiments" / "exp_rahul_001_nullspace_map" / "controls" / "__init__.py"
if _old.exists():
    _spec = importlib.util.spec_from_file_location("controls_exp_rahul_001", _old, submodule_search_locations=[str(_old.parent)])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["controls_exp_rahul_001"] = _mod
    _spec.loader.exec_module(_mod)  # registers state_control/nullspace_map

from generate_answers import main  # noqa: E402

if __name__ == "__main__":
    main()
