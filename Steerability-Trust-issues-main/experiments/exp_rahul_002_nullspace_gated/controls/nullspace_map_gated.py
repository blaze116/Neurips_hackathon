"""NullSpaceMapGated (exp_rahul_002): NullSpaceMap with a per-token projection-gated write rule.

Successor of exp_rahul_001/controls/nullspace_map.py (rounds 1-2, `state_control/nullspace_map`); registered as
`state_control/nullspace_map_gated` so both can be loaded side by side.

What it does at inference (one or more decoder layers, Hugging Face backend):

    1. During prefill, read the residual stream `h` at the **last prompt token** of each row and route it:
       `w = R h` gives one weight per dishonesty dim (`R`, the "read" rows, lie in the null space of benign prompts,
       so a benign prompt gets `w ~ 0`).
    2. Write on **every answer position** (prompt tokens are left untouched), for the whole generation of that row,
       including the reasoning block. Two write rules:

       * `fixed` (round 1-2): add `lam * sum_d w_d t_d`, the same vector on every answer token (`t_d` = the dim's
         target, the "write" rows).
       * `sttp` / `stmp` (gated, after Herbster et al. 2026): per token and per dim, project the token's state on the
         dim's unit target `rho = h . t_d/|t_d|`; if it is already on the honest side of the fitted boundary `m_d`,
         add nothing; otherwise add only the deficit, scaled by the prompt's weight: `sttp` moves the projection to a
         target `s_d = mu+_d + alpha * sigma+_d` (`w_d * (s_d - rho)`), `stmp` mirrors it across the boundary
         (`w_d * 2 alpha (m_d - rho)`). `floor` adds that fraction of the fixed shift underneath (a hybrid).
         The push is bounded by the honest class's own range, stops once the text is honest, and stays off on
         benign answers that leak through the router.

Why it is built this way (the fit lives in `fit_nullspace_map.py`; this class only applies a fitted map):

    * The read rows lie in the *null space* of a benign set of prompts (capability and guard prompts and the
      pressure-free twins): the directions those prompts never use. A benign prompt therefore has `R h ~ 0` and
      receives ~no shift, which is what protects r_side and r_guard. The amount of a held-out benign prompt's
      energy outside the null space (1 - PEC) is measured at fit time.
    * The targets are **response-side**: for each dim, the activation of the honest answer minus the activation of
      the dishonest answer (the team's `contrastive_pair`), averaged over the answer tokens, teacher-forced after the
      same prompt. Different dims need different directions (a pooled direction explains ~3-5% of the contrast; the
      per-dim one ~55%), so the map routes each prompt to its dim's correction from the prompt alone. The router is
      the ridge-regression map from null-space coordinates to one-hot dim membership (closed form, AlphaSteer).
    * The gate statistics (`mu+`, `sigma+`, boundary) are the projections of the honest and dishonest answer
      *tokens* on each dim's unit target, measured at fit time on the same pairs.

References: AlphaSteer (Sheng et al., ICLR 2026) for the null-space constraint and the closed-form ridge fit;
Herbster et al. 2026 (arXiv:2604.08169) for the projection-gated write rules (StTP / StMP); Arditi et al. 2024 /
Rimsky et al. 2024 for difference-of-means directions; the team's `steering/cast_honesty.py` for the data and
rendering conventions this control is compared against.

The control is hook-only (no intervention-spec form), so the evaluation runs it on the Hugging Face backend,
as it does for CAST. It fits anything that the pipeline's left-packed batching produces and needs no model
access at `steer()` time: the artifacts are supplied fitted, so the frozen bundle is the recipe itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.state_control.base import HookControl, HookSpec
from steerability.algorithms.state_control.common.steering_vector import SteeringVector

WRITE_RULES = ("fixed", "sttp", "stmp")
GATE_KEYS = ("mu_pos", "sigma_pos", "boundary")


@dataclass
class NullSpaceMapGatedArgs(BaseArgs):
    """Arguments of `NullSpaceMapGated`.

    Attributes:
        read: Per-layer read rows `R`, shape `[D, H]` per layer (a `SteeringVector`): one router row per dim.
        write: Per-layer targets `T`, shape `[D, H]` per layer, same layers and `D` as `read`; row d is dim d's
            honest-minus-dishonest answer state (a zero row = a dim the map leaves alone).
        lam: Dose of the `fixed` rule. The shift added is `lam * sum_d w_d t_d`; 1.0 applies the fitted correction.
        layer_ids: The layers to steer; None steers every layer the artifacts cover.
        write_rule: `fixed`, `sttp` or `stmp` (module docstring).
        alpha: Gated rules: how far into the honest class to land (`sttp`: target `mu+ + alpha sigma+`) or the
            mirror overshoot (`stmp`: `2 alpha (m - rho)`).
        floor: Gated rules: fraction of the fixed shift always added underneath (0 = pure gating).
        w_clip: Gated rules: router weights are clipped to `[0, w_clip]` (a negative weight would push toward the
            dishonest side; the fixed rule keeps the raw weights, as in rounds 1-2).
        gate: Gated rules: `{str(layer): {"mu_pos": [D], "sigma_pos": [D], "boundary": [D], ...}}` fitted
            token-level statistics per dim, in the row order of `read`/`write`.
        dims: Names of the rows of `read`/`write` (informational).
    """

    read: SteeringVector = field(default=None, metadata={"help": "read rows R, [D, H] per layer"})
    write: SteeringVector = field(default=None, metadata={"help": "targets T, [D, H] per layer"})
    lam: float = field(default=1.0, metadata={"help": "dose multiplier of the fixed rule"})
    layer_ids: list[int] | None = field(default=None, metadata={"help": "layers to steer; None = all fitted"})
    write_rule: str = field(default="fixed", metadata={"help": "fixed | sttp | stmp"})
    alpha: float = field(default=0.5, metadata={"help": "gated rules: target depth in honest std units / mirror overshoot"})
    floor: float = field(default=0.0, metadata={"help": "gated rules: fraction of the fixed shift always added"})
    w_clip: float = field(default=1.5, metadata={"help": "gated rules: router weights clipped to [0, w_clip]"})
    gate: dict | None = field(default=None, metadata={"help": "gated rules: per-layer, per-dim token statistics"})
    dims: list[str] | None = field(default=None, metadata={"help": "row names of read/write"})

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
                raise ValueError(f"layer {layer}: read {tuple(b.shape)} and write {tuple(a.shape)} must both be [D, H]")
        if self.layer_ids is not None:
            missing = sorted(set(self.layer_ids) - set(self.read.directions))
            if missing:
                raise ValueError(f"layer_ids {missing} have no fitted directions (fitted: {sorted(self.read.directions)})")
            self.layer_ids = [int(l) for l in self.layer_ids]
        for name in ("lam", "alpha", "floor", "w_clip"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be a finite non-negative number; got {v!r}")
        if self.write_rule not in WRITE_RULES:
            raise ValueError(f"write_rule must be one of {WRITE_RULES}; got {self.write_rule!r}")
        if self.write_rule != "fixed":
            if self.w_clip == 0:
                raise ValueError("w_clip must be positive for the gated rules")
            if not isinstance(self.gate, dict):
                raise ValueError(f"write_rule {self.write_rule!r} needs the fitted gate statistics (gate=...)")
            self.gate = {str(k): v for k, v in self.gate.items()}
            for layer, b in self.read.directions.items():
                stats = self.gate.get(str(layer))
                if not isinstance(stats, dict) or any(k not in stats for k in GATE_KEYS):
                    raise ValueError(f"gate statistics for layer {layer} must have {GATE_KEYS}")
                for k in GATE_KEYS:
                    if len(stats[k]) != b.shape[0]:
                        raise ValueError(f"layer {layer}: gate {k} has {len(stats[k])} entries for {b.shape[0]} dims")
        if self.dims is not None:
            self.dims = [str(d) for d in self.dims]


class NullSpaceMapGated(HookControl):
    """Prompt-routed, null-space-constrained shift on answer tokens, fixed or projection-gated (module docstring)."""

    Args = NullSpaceMapGatedArgs
    supports_batching = True

    def _configure(self) -> None:
        self._layers = sorted(self.layer_ids) if self.layer_ids is not None else sorted(self.read.directions)
        self._tensors: dict[tuple, dict[str, torch.Tensor]] = {}
        self.stats: dict | None = None  # set to {} by a caller to record per-layer gate statistics (diagnostics)

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

    def _layer_tensors(self, layer: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Fitted tensors of `layer` on `device` in fp32, cached per device: `R` [D, H], `T` [D, H], and for the
        gated rules the unit targets `That` [D, H], the row mask `active` [D], the target `s` [D] and boundary `m` [D]."""
        key = (layer, str(device))
        if key not in self._tensors:
            R = self.read.directions[layer].to(device=device, dtype=torch.float32).contiguous()
            T = self.write.directions[layer].to(device=device, dtype=torch.float32).contiguous()
            out = {"R": R, "T": T}
            if self.write_rule != "fixed":
                norms = T.norm(dim=1)
                active = norms > 1e-6
                out["That"] = torch.where(active.unsqueeze(1), T / norms.clamp(min=1e-6).unsqueeze(1), torch.zeros_like(T))
                out["active"] = active.to(torch.float32)
                g = self.gate[str(layer)]
                out["mu_pos"] = torch.tensor(g["mu_pos"], dtype=torch.float32, device=device)
                out["sigma_pos"] = torch.tensor(g["sigma_pos"], dtype=torch.float32, device=device)
                out["m"] = torch.tensor(g["boundary"], dtype=torch.float32, device=device)
            self._tensors[key] = out
        return self._tensors[key]

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
                raise RuntimeError("NullSpaceMapGated needs the model layout; call steer() through a pipeline first")
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
        rule = self.write_rule
        lam, alpha, floor, w_clip = float(self.lam), float(self.alpha), float(self.floor), float(self.w_clip)
        if runtime_kwargs:
            lam = float(runtime_kwargs.get("lam", lam))
            alpha = float(runtime_kwargs.get("alpha", alpha))

        state = {"offset": 0, "prefill_seen": False, "w": {}, "shift": {}}
        first_layer = self._layers[0]

        def expand(t: torch.Tensor, rows: int) -> torch.Tensor:
            return t if t.size(0) == rows else t.repeat_interleave(rows // t.size(0), dim=0)

        def make_hook(layer: int):
            def hook(module, args, hook_kwargs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                seq_len = hidden.size(1)
                # phase bookkeeping, advanced by the first steered layer only
                if layer == first_layer:
                    if state["prefill_seen"]:
                        state["offset_now"] = state["offset"]
                        state["offset"] += seq_len
                    else:
                        state["offset_now"] = 0
                        state["offset"] = seq_len
                        state["prefill_seen"] = True
                offset = state["offset_now"]
                rows = hidden.size(0)
                P = self._layer_tensors(layer, hidden.device)
                x = hidden.float()
                if offset == 0:
                    # prefill: read the last prompt token of each row and fix this row's routing weights
                    idx = last_index.to(hidden.device).clamp(max=seq_len - 1)
                    if rows != batch:  # beam/sample expansion repeats rows in order
                        idx = idx.repeat_interleave(rows // batch)
                    z = x[torch.arange(rows, device=hidden.device), idx]            # [rows, H]
                    w = z @ P["R"].T                                                 # [rows, D]
                    if rule != "fixed":
                        w = w.clamp(min=0.0, max=w_clip)
                    state["w"][layer] = w
                    state["shift"][layer] = w @ P["T"]                                # [rows, H], the fixed shift at lam 1
                w = state["w"].get(layer)
                if w is None or (rule == "fixed" and lam == 0.0):
                    return output
                w, fixed_shift = expand(w, rows), expand(state["shift"][layer], rows)
                positions = torch.arange(seq_len, device=hidden.device) + offset       # absolute positions
                after = expand(positions.unsqueeze(0) > last_index.to(hidden.device).unsqueeze(1), rows)  # [rows, T]
                if not bool(after.any()):
                    return output
                deficit = None
                if rule == "fixed":
                    shift = (lam * fixed_shift).unsqueeze(1).expand_as(x)
                else:
                    rho = x @ P["That"].T                                              # [rows, T, D]
                    if rule == "sttp":
                        s = P["mu_pos"] + alpha * P["sigma_pos"]
                        deficit = (s - rho).clamp(min=0.0) * (rho < P["m"]).to(x.dtype)
                    else:  # stmp
                        deficit = 2.0 * alpha * (P["m"] - rho).clamp(min=0.0)
                    deficit = deficit * P["active"]
                    shift = (w.unsqueeze(1) * deficit) @ P["That"]                     # [rows, T, H]
                    if floor > 0.0:
                        shift = shift + floor * fixed_shift.unsqueeze(1)
                if self.stats is not None:
                    self._record(layer, after, shift, deficit, w)
                x = torch.where(after.unsqueeze(-1), x + shift, x).to(hidden.dtype)
                return (x, *output[1:]) if isinstance(output, tuple) else x
            return hook

        return {"pre": [], "backward": [],
                "forward": [{"module": self._layer_names[layer], "hook_func": make_hook(layer)} for layer in self._layers]}

    def _record(self, layer: int, after: torch.Tensor, shift: torch.Tensor, deficit: torch.Tensor | None, w: torch.Tensor) -> None:
        """Accumulate diagnostics over answer positions: tokens seen, tokens where the gate fired (any routed dim with
        weight > 0.1 and a positive deficit), per-dim firings, and the summed shift norm."""
        n_dims = w.size(1)
        st = self.stats.setdefault(layer, {"tokens": 0, "fired": 0, "shift_norm": 0.0, "fired_per_dim": [0] * n_dims})
        st["tokens"] += int(after.sum())
        st["shift_norm"] += float(shift.norm(dim=-1)[after].sum())
        if deficit is not None:
            firing = (deficit > 0) & (w.unsqueeze(1) > 0.1)                            # [rows, T, D]
            st["fired"] += int(firing.any(-1)[after].sum())
            per_dim = firing[after].sum(0).tolist()
            st["fired_per_dim"] = [a + int(b) for a, b in zip(st["fired_per_dim"], per_dim)]

    def export_state(self) -> dict[str, Any]:
        return {}  # the artifacts are constructor args, so the recipe is the frozen form

    def describe(self) -> dict:
        ranks = {int(l): int((v.norm(dim=1) > 1e-6).sum()) for l, v in self.write.directions.items()}
        out = {"method": "nullspace_map_gated", "layers": self._layers, "write_rule": self.write_rule, "lam": self.lam,
               "alpha": self.alpha, "floor": self.floor, "active_dims_per_layer": ranks, "dims": self.dims,
               "fit": dict(self.read.meta.get("fit", {}))}
        return out
