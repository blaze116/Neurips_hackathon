"""The `controls` package of exp_rahul_001: a null-space-constrained, per-dim-routed steering map.

Importing this package registers the control with the steerability toolkit under
`state_control/nullspace_map`, which is how a `.spipe` bundle naming it can be loaded. Import it before
loading such a bundle (the kit's `check-submission` does this itself; `run_generate.py` does it for the
team's harness).
"""
from steerability.algorithms.core.registry import register_method

from .nullspace_map import NullSpaceMap, NullSpaceMapArgs

register_method("state", "nullspace_map", NullSpaceMap, NullSpaceMapArgs)

__all__ = ["NullSpaceMap", "NullSpaceMapArgs"]
