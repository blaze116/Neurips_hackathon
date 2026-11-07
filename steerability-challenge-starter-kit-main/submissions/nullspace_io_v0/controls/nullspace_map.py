"""NullSpaceMap: a learned, input-dependent activation shift that is blind to ordinary prompts.

What it does at inference (one or more decoder layers, Hugging Face backend):

    1. During prefill, read the residual stream `h` at the **last prompt token** of each row.
    2. Compute that row's shift `s = lam * A (B^T h)`; `B` ([H, r]) are the "read" directions and `A` ([H, r])
       the "write" directions of a rank-`r` map `Delta = A B^T`.
    3. Add `s` to the residual stream at **every answer position** (prompt tokens are left untouched), for the
       whole generation of that row, including the reasoning block.

Why it is built this way (the fit lives in `fit_nullspace_map.py`; this class only applies a fitted map):

    * The read directions lie in the *null space* of a benign set of prompts (capability and guard prompts and
      the pressure-free twins): the directions those prompts never use. A benign prompt therefore has
      `B^T h ~ 0` and receives ~no shift, which is what protects r_side and r_guard. The amount of a held-out
      benign prompt's energy outside the null space (1 - PEC) is measured at fit time.
    * The map is fitted by ridge regression so that for a pressure prompt of dishonesty dim d, `Delta h`
      reproduces that dim's mean **response-side** contrast: the activation of the honest answer minus the
      activation of the dishonest answer (the team's `contrastive_pair`), averaged over the answer tokens.
      Different dims need different directions (a pooled direction explains ~3-5% of the contrast; the
      per-dim one ~55%), so the map has rank up to the number of dims and routes each prompt to its dim's
      correction from the prompt alone.
    * `lam` scales the shift; `lam = 1` applies the full predicted correction. The fit script's dose sweep
      picks it.

References: AlphaSteer (Sheng et al., ICLR 2026) for the null-space constraint and the closed-form ridge fit;
Arditi et al. 2024 / Rimsky et al. 2024 for difference-of-means directions; the team's `steering/cast_honesty.py`
for the data and rendering conventions this control is compared against.

The control is hook-only (no intervention-spec form), so the evaluation runs it on the Hugging Face backend,
as it does for CAST. It fits anything that the pipeline's left-packed batching produces and needs no model
access at `steer()` time: the artifacts are supplied fitted, so the frozen bundle is the recipe itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.state_control.base import HookControl, HookSpec
from steerability.algorithms.state_control.common.steering_vector import SteeringVector


@dataclass
class NullSpaceMapArgs(BaseArgs):
    """Arguments of `NullSpaceMap`.

    Attributes:
        read: Per-layer read directions `B^T`, shape `[r, H]` per layer (a `SteeringVector`).
        write: Per-layer write directions `A^T`, shape `[r, H]` per layer, same layers and `r` as `read`.
        lam: Dose. The shift added is `lam * A (B^T h)`; 1.0 applies the fitted correction in full.
        layer_ids: The layers to steer; None steers every layer the artifacts cover.
    """

    read: SteeringVector = field(default=None, metadata={"help": "read directions B^T, [r, H] per layer"})
    write: SteeringVector = field(default=None, metadata={"help": "write directions A^T, [r, H] per layer"})
    lam: float = field(default=1.0, metadata={"help": "dose multiplier on the fitted shift"})
    layer_ids: list[int] | None = field(default=None, metadata={"help": "layers to steer; None = all fitted"})

    def __post_init__(self):
        if not isinstance(self.read, SteeringVector) or not isinstance(self.write, SteeringVector):
            raise TypeError("read and write must be SteeringVector artifacts (fit them with fit_nullspace_map.py)")
        self.read.validate()
        self.write.validate()
        if set(self.read.directions) != set(self.write.directions):
            raise ValueError(f"read covers layers {sorted(self.read.directions)} but write covers "
                             f"{sorted(self.write.directions)}")
        for layer, b in self.read.directions.items():
            a = self.write.directions[layer]
            if b.ndim != 2 or a.shape != b.shape:
                raise ValueError(f"layer {layer}: read {tuple(b.shape)} and write {tuple(a.shape)} must both be [r, H]")
        if self.layer_ids is not None:
            missing = sorted(set(self.layer_ids) - set(self.read.directions))
            if missing:
                raise ValueError(f"layer_ids {missing} have no fitted directions (fitted: {sorted(self.read.directions)})")
            self.layer_ids = [int(l) for l in self.layer_ids]
        if not (self.lam == self.lam) or self.lam < 0:  # NaN or negative
            raise ValueError(f"lam must be a non-negative number; got {self.lam!r}")


class NullSpaceMap(HookControl):
    """Prompt-conditioned, null-space-constrained shift on answer tokens (see the module docstring)."""

    Args = NullSpaceMapArgs
    supports_batching = True

    def _configure(self) -> None:
        self._layers = sorted(self.layer_ids) if self.layer_ids is not None else sorted(self.read.directions)
        self._factors: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    # the artifacts arrive fitted, so steering needs only the layout; the pipeline provides it
    def steer_access(self) -> ModelAccess:
        return ModelAccess.FACTS

    def steer(self, model=None, tokenizer=None, session=None, **kwargs) -> None:
        """Check the fitted layers against the model's depth; module names resolve lazily in `get_hooks`."""
        from steerability.algorithms.state_control.common.layout_facts import resolve_layout

        self._layer_names = None
        try:
            num_layers = resolve_layout(model, session).num_layers
        except Exception:  # no layout available at this access rung; checked again in get_hooks
            return
        beyond = [l for l in self._layers if l >= num_layers]
        if beyond:
            raise ValueError(f"layers {beyond} do not exist in a {num_layers}-layer model")

    def _ab(self, layer: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """`(B, A)` for `layer` on `device`, as `[H, r]` fp32 tensors, cached per device."""
        key = (layer, str(device))
        if key not in self._factors:
            b = self.read.directions[layer].to(device=device, dtype=torch.float32).T.contiguous()
            a = self.write.directions[layer].to(device=device, dtype=torch.float32).T.contiguous()
            self._factors[key] = (b, a)
        return self._factors[key]

    def get_hooks(self, input_ids: torch.Tensor, runtime_kwargs: dict | None, **kwargs) -> dict[str, list[HookSpec]]:
        """Forward hooks on the steered layers for one generation.

        The pipeline left-packs batched prompts, so each row's last prompt token is the last column of the
        prompt and generation starts at absolute position `T_prompt`. Position tracking follows the toolkit's
        own convention (`compute_prompt_lens`): the first forward pass of a generation is the prefill, later
        passes continue from the cumulative offset. One designated hook (the first steered layer) advances the
        shared counter so every layer sees the same offset within a pass.
        """
        model = kwargs.get("model")
        if getattr(self, "_layer_names", None) is None:
            if model is None:
                raise RuntimeError("NullSpaceMap needs the model layout; call steer() through a pipeline first")
            from steerability.algorithms.core.internals.model_layout import resolve_model_layout

            self._layer_names = list(resolve_model_layout(model).layer_names)
        attention_mask = kwargs.get("attention_mask")
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        batch = input_ids.size(0)
        prompt_len = input_ids.size(1)
        # index of the last real prompt token per row: the last 1 of the mask (T-1 under left packing)
        if attention_mask is not None and attention_mask.ndim == 2:
            mask = attention_mask.to(torch.long)
            last_index = (mask.cumsum(1) == mask.sum(1, keepdim=True)).to(torch.long).argmax(1)
        else:
            last_index = torch.full((batch,), prompt_len - 1, dtype=torch.long)
        lam = float(self.lam)
        if runtime_kwargs and "lam" in runtime_kwargs:
            lam = float(runtime_kwargs["lam"])

        state = {"offset": 0, "prefill_seen": False, "shift": {}}
        first_layer = self._layers[0]

        def row_last(rows: int, device: torch.device) -> torch.Tensor:
            if rows == batch:
                return last_index.to(device)
            if rows % batch == 0:  # beam/sample expansion repeats rows in order
                return last_index.to(device).repeat_interleave(rows // batch)
            # a decoding driver (PhasedDecoding) runs left-packed rows one at a time: last prompt token is T-1
            return torch.full((rows,), prompt_len - 1, dtype=torch.long, device=device)

        def make_hook(layer: int):
            def hook(module, args, hook_kwargs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                seq_len = hidden.size(1)
                # phase bookkeeping, advanced by the first steered layer only; with a KV cache every
                # multi-token forward is a fresh prefill (a driver's next row or next phase)
                if layer == first_layer:
                    if state["prefill_seen"] and seq_len == 1:
                        state["offset_now"] = state["offset"]
                        state["offset"] += seq_len
                    else:
                        state["offset_now"] = 0
                        state["offset"] = seq_len
                        state["prefill_seen"] = True
                offset = state["offset_now"]
                rows = hidden.size(0)
                last = row_last(rows, hidden.device)
                B, A = self._ab(layer, hidden.device)
                x = hidden.float()
                if offset == 0:
                    # prefill: read the last prompt token of each row and fix this row's shift
                    idx = last.clamp(max=seq_len - 1)
                    z = x[torch.arange(rows, device=hidden.device), idx]            # [rows, H]
                    state["shift"][layer] = lam * (z @ B) @ A.T                      # [rows, H]
                shift = state["shift"].get(layer)
                if shift is None or lam == 0.0:
                    return output
                if shift.size(0) != rows:
                    shift = shift.repeat_interleave(rows // shift.size(0), dim=0)
                positions = torch.arange(seq_len, device=hidden.device) + offset       # absolute positions
                after = positions.unsqueeze(0) > last.unsqueeze(1)                     # [rows, T]
                if not bool(after.any()):
                    return output
                x = torch.where(after.unsqueeze(-1), x + shift.unsqueeze(1), x).to(hidden.dtype)
                return (x, *output[1:]) if isinstance(output, tuple) else x
            return hook

        return {"pre": [], "backward": [],
                "forward": [{"module": self._layer_names[layer], "hook_func": make_hook(layer)} for layer in self._layers]}

    def export_state(self) -> dict[str, Any]:
        return {}  # the artifacts are constructor args, so the recipe is the frozen form

    def describe(self) -> dict:
        ranks = {int(l): int(v.shape[0]) for l, v in self.read.directions.items()}
        return {"method": "nullspace_map", "layers": self._layers, "lam": self.lam, "rank_per_layer": ranks,
                "fit": dict(self.read.meta.get("fit", {}))}
