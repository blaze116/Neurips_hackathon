"""The `controls` package of exp_rahul_002: the null-space-constrained, per-dim-routed steering map with a
projection-gated write rule.

Importing this package registers the control with the steerability toolkit under
`state_control/nullspace_map_gated`, which is how a `.spipe` bundle naming it can be loaded. Import it before
loading such a bundle (the kit's `check-submission` does this itself; `run_generate.py` does it for the team's
harness). exp_rahul_001's `state_control/nullspace_map` (rounds 1-2) is a different registration, so a bundle of
either experiment is never mistaken for the other.
"""
from steerability.algorithms.core.registry import register_method

from .nullspace_map_gated import NullSpaceMapGated, NullSpaceMapGatedArgs

register_method("state", "nullspace_map_gated", NullSpaceMapGated, NullSpaceMapGatedArgs)

__all__ = ["NullSpaceMapGated", "NullSpaceMapGatedArgs"]
