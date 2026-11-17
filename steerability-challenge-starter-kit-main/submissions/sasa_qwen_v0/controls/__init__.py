"""The `controls` package of `sasa_qwen_v0`: SASA that runs on hybrid-attention models (Qwen3.5 / Qwen3.8).

Importing this package registers the control under `output_control/sasa_hybrid`, which is how a `.spipe`
bundle naming it is loaded. Import it before loading such a bundle (the kit's `check-submission` and the proxy
scorer notebook do this themselves).
"""
from steerability.algorithms.core.registry import register_method

from .sasa_hybrid import SASAHybrid, SASAHybridArgs

register_method("output", "sasa_hybrid", SASAHybrid, SASAHybridArgs)

__all__ = ["SASAHybrid", "SASAHybridArgs"]
