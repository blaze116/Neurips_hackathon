"""SASA for hybrid-attention models, optionally applied to the answer only.

The toolkit's `SASA` (pinned steerability 0.5.1) scores each candidate token with a same-model forward
that repeats the prefix KV cache once per candidate. On Qwen3.5 / Qwen3.8 it cannot run as is:

- its `repeat_cache` only copies key/value tensors, so on their hybrid stacks (Gated DeltaNet linear-attention
  layers interleaved with full attention) it raises `TypeError: Unsupported cache type: DynamicCache`;
- its candidate forward takes one unpadded row, while the pipeline hands `generate` the whole left-padded batch
  (`pipeline.generate(messages=[...several chats...])` raises "supports batch size 1 only").

`SASAHybrid` is the same algorithm (same probe fit, margin and logit shift) with these changes:

1. The candidate forward repeats any layer-based cache through `Cache.reorder_cache` (the beam-search API),
   which copies every layer's state, keys and values as well as the conv and recurrent states of
   linear-attention layers. Caches the toolkit already supports take the toolkit's own path unchanged.
2. Each batch row is scored on its own (left padding stripped, one candidate cache per row); finished rows
   are skipped. The main decoding forward stays batched.
3. `answer_only=True` (default) leaves the reasoning block untouched and shifts logits only after the
   close-reasoning tag. The judge reads the answer, so this steers what is scored, keeps reasoning (and so
   capability) closer to the base model, and skips the per-step candidate forwards while thinking.
4. `probe_method` can fit the probe with `mean_diff` instead of the toolkit's fixed `fisher`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Literal

import torch

from steerability.algorithms.core.internals.model_layout import resolve_model_layout
from steerability.algorithms.core.internals.probes.fitting import ProbeFitSpec, fit_probe
from steerability.algorithms.core.utils.auxiliary_pass import auxiliary_pass
from steerability.algorithms.output_control.common.candidate_forward import CandidateForward
from steerability.algorithms.output_control.common.kv_cache import full_prefix_mask, repeat_cache
from steerability.algorithms.output_control.common.candidates import select_candidates
from steerability.algorithms.output_control.common.processors.value_guided import (
    ValueGuidedProcessor,
    ValueStepRecord,
    _normalize,
)
from steerability.algorithms.output_control.common.values.base import StepContext
from steerability.algorithms.output_control.common.values.subspace_margin import SubspaceMarginValue
from steerability.algorithms.output_control.sasa.args import SASAArgs
from steerability.algorithms.output_control.sasa.control import SASA, _validate_probe_space
from steerability.utils.tokenization import ensure_pad_token


def repeat_cache_any(cache, n: int):
    """`n` batch copies of `cache` that share no storage with it, for hybrid caches too."""
    try:
        return repeat_cache(cache, n, preserve_input=True)
    except TypeError:
        if not (hasattr(cache, "layers") and hasattr(cache, "reorder_cache")):
            raise
    fresh = copy.copy(cache)
    for name, value in vars(cache).items():
        if isinstance(value, (dict, list)):
            setattr(fresh, name, copy.copy(value))
    fresh.layers = []
    for layer in cache.layers:
        clone = copy.copy(layer)
        for name, value in vars(layer).items():
            if isinstance(value, (dict, list)):  # per-state dicts of linear-attention layers
                setattr(clone, name, copy.copy(value))
        fresh.layers.append(clone)
    # reorder_cache rebinds each tensor to `index_select(0, idx)`, a new tensor, so `cache` is untouched
    fresh.reorder_cache(torch.zeros(n, dtype=torch.long))
    return fresh


class HybridCandidateForward(CandidateForward):
    """`CandidateForward` whose per-candidate cache copy also handles hybrid (linear + full attention) caches."""

    @torch.no_grad()
    def last_hidden_states(self, prefix_ids, candidate_ids, attention_mask=None):
        if prefix_ids.size(0) != 1:
            raise ValueError("CandidateForward supports batch size 1 only.")
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError("CandidateForward requires an unpadded prefix; the attention mask contains zeros.")

        num = candidate_ids.size(1)
        device = prefix_ids.device
        full_mask = full_prefix_mask(prefix_ids, attention_mask)
        with auxiliary_pass(aligned=True):
            self._sync_cache(prefix_ids, full_mask)
            prefix_len = prefix_ids.size(1)
            repeated = repeat_cache_any(self._cache, num)
            cand_mask = torch.cat(
                [full_mask.repeat(num, 1), torch.ones(num, 1, device=device, dtype=full_mask.dtype)], dim=1,
            )
            captured: list[torch.Tensor] = []

            def _grab_final(module, args, output):
                captured.append(output[0] if isinstance(output, tuple) else output)

            handle = self._final_layer.register_forward_hook(_grab_final)
            try:
                self.model(
                    input_ids=candidate_ids.reshape(num, 1),
                    attention_mask=cand_mask,
                    past_key_values=repeated,
                    use_cache=True,
                    cache_position=torch.arange(prefix_len, prefix_len + 1, device=device),
                    return_dict=True,
                )
            finally:
                handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"Expected one final-layer forward for the candidate batch, observed {len(captured)}.")
        return captured[0][:, -1, :]


class HybridSubspaceMarginValue(SubspaceMarginValue):
    """`SubspaceMarginValue` on `HybridCandidateForward`."""

    def score(self, ctx: StepContext) -> torch.Tensor:
        if ctx.model is None:
            raise RuntimeError("SubspaceMarginValue requires the pipeline model in StepContext.")
        if self._forward is None or self._forward.model is not ctx.model:
            self._forward = HybridCandidateForward(ctx.model)
        hidden = self._forward.last_hidden_states(ctx.prefix_ids, ctx.candidate_ids, ctx.attention_mask)
        self._align_probe(hidden.device, hidden.dtype)
        return (hidden @ self._weights + self.probe.bias).unsqueeze(0)


class RowwiseValueGuidedProcessor(ValueGuidedProcessor):
    """`ValueGuidedProcessor` that scores each batch row on its own, optionally only outside reasoning.

    The pipeline hands Hugging Face `generate` the whole left-padded batch even when a control does not support
    batching, and the candidate forward takes one unpadded row. So each row is scored separately: its left
    padding is stripped, it keeps its own candidate-forward cache, and a row that has finished (last token is
    padding or end of sequence) or is still inside its reasoning block keeps its scores unchanged.
    """

    def __init__(self, probe, *, open_id: int | None = None, close_id: int | None = None,
                 pad_id: int | None = None, stop_ids: tuple[int, ...] = (), **kwargs):
        super().__init__(HybridSubspaceMarginValue(probe), **kwargs)
        self.probe = probe
        self.open_id, self.close_id = open_id, close_id
        self.pad_id = pad_id
        self.stop_ids = frozenset(stop_ids) | ({pad_id} if pad_id is not None else set())
        self._row_values: dict[int, HybridSubspaceMarginValue] = {}

    def _thinking(self, row: torch.Tensor) -> bool:
        if self.open_id is None or self.close_id is None:
            return False
        opens = (row == self.open_id).nonzero()
        if opens.numel() == 0:
            return False
        closes = (row == self.close_id).nonzero()
        return closes.numel() == 0 or int(opens[-1]) > int(closes[-1])

    def _unpadded(self, row: torch.Tensor) -> torch.Tensor:
        if self.pad_id is None:
            return row
        real = (row != self.pad_id).nonzero()
        return row[int(real[0]):] if real.numel() else row[:0]

    def process(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        out = scores.clone()
        for i in range(input_ids.size(0)):
            row = self._unpadded(input_ids[i])
            if row.numel() == 0 or int(row[-1]) in self.stop_ids or self._thinking(row):
                continue
            cand_ids, cand_scores = select_candidates(scores[i:i + 1], self.policy, k=self.k, p=self.p)
            if self.max_candidates is not None and cand_ids.size(1) > self.max_candidates:
                cand_ids = cand_ids.gather(1, torch.topk(cand_scores, self.max_candidates, dim=-1).indices)
            value = self._row_values.setdefault(i, HybridSubspaceMarginValue(self.probe))
            ctx = StepContext(prefix_ids=row.unsqueeze(0), candidate_ids=cand_ids, lm_tokenizer=self.lm_tokenizer,
                              model=self.model, attention_mask=None)
            raw = value.score(ctx).to(scores.dtype)
            v = _normalize(raw, self.normalize, self.invert)
            if self.trace is not None:
                self.trace.append(ValueStepRecord(
                    prefix_length=int(row.numel()), candidate_ids=cand_ids.detach().cpu(),
                    candidate_scores=scores[i:i + 1].gather(1, cand_ids).detach().cpu(),
                    values=raw.detach().cpu(), normalized=v.detach().cpu(),
                ))
            out[i:i + 1].scatter_add_(1, cand_ids, self.beta * v)
        return out


@dataclass
class SASAHybridArgs(SASAArgs):
    """`SASAArgs` plus where to apply the shift."""

    answer_only: bool = field(
        default=True,
        metadata={"help": "Shift logits only after the close-reasoning tag (the reasoning block is left as is)."},
    )
    think_tags: list[str] = field(
        default_factory=lambda: ["<think>", "</think>"],
        metadata={"help": "The model's open and close reasoning tags; each must be a single token."},
    )
    probe_method: Literal["fisher", "mean_diff"] = field(
        default="fisher",
        metadata={"help": "How the probe direction is fitted when `wv_path` is unset: 'fisher' (the paper's and the "
                          "toolkit's choice, a pseudo-inverted pooled covariance, which can overfit when there are "
                          "fewer pairs than hidden dimensions) or 'mean_diff' (the class-mean difference)."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.answer_only and len(self.think_tags) != 2:
            raise ValueError("think_tags must be [open_tag, close_tag].")
        if self.probe_method not in ("fisher", "mean_diff"):
            raise ValueError(f"probe_method must be 'fisher' or 'mean_diff', got {self.probe_method!r}.")


class SASAHybrid(SASA):
    """SASA (Ko et al., 2024) that runs on hybrid-attention models and, by default, steers only the answer.

    Same probe fit, same margin and same logit shift as the toolkit's `SASA`; see the module docstring for the
    two differences.
    """
    Args = SASAHybridArgs

    _open_id: int | None = None
    _close_id: int | None = None

    def steer(self, model, tokenizer=None, **kwargs):
        if getattr(self, "wv_path", None) or self.probe_method == "fisher":
            model = super().steer(model, tokenizer, **kwargs)
        else:
            if self.gen_wv_data is None:
                raise ValueError("SASAHybrid.steer() requires either gen_wv_data (to fit a probe) or wv_path (to load one).")
            self.model = model
            self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token_id is not None:
                    self.tokenizer = ensure_pad_token(self.tokenizer)
                else:
                    self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            final_layer = resolve_model_layout(model).num_layers - 1
            spec = ProbeFitSpec(method=self.probe_method, pooling="last", location="layer_output",
                                prompt_format=self.prompt_format, candidate_layers=[final_layer], calibration="midpoint")
            self.probe = fit_probe(model, self.tokenizer, data=self._resolve_fit_data(), spec=spec,
                                   batch_size=self.gen_wv_batch_size, max_length=1024)
            _validate_probe_space(self.probe, final_layer)
        self._open_id = self._close_id = None
        if self.answer_only:
            ids = []
            for tag in self.think_tags:
                encoded = self.tokenizer.encode(tag, add_special_tokens=False)
                if len(encoded) != 1:
                    raise ValueError(f"think tag {tag!r} is {len(encoded)} tokens for this tokenizer; it must be one.")
                ids.append(encoded[0])
            self._open_id, self._close_id = ids
        return model

    def frozen_form(self, state: dict) -> tuple[str, dict]:
        _, args = super().frozen_form(state)
        return "output_control/sasa_hybrid", {
            **args, "answer_only": self.answer_only, "think_tags": list(self.think_tags),
        }

    def fit_identity(self):
        identity = super().fit_identity()
        return identity if identity is None else {**identity, "probe_method": self.probe_method}

    def get_logits_processors(self, input_ids, runtime_kwargs, attention_mask=None, **kwargs) -> list:
        if self.probe is None:
            raise RuntimeError("SASAHybrid.steer() must run before generation (probe not fitted/loaded).")
        eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        stop_ids = tuple(eos) if isinstance(eos, (list, tuple)) else (() if eos is None else (eos,))
        if self.tokenizer.eos_token_id is not None:
            stop_ids += (self.tokenizer.eos_token_id,)
        return [
            RowwiseValueGuidedProcessor(
                self.probe,
                open_id=self._open_id,
                close_id=self._close_id,
                pad_id=self.tokenizer.pad_token_id,
                stop_ids=stop_ids,
                policy=self.candidate_policy,
                p=self.top_p,
                k=self.top_k,
                beta=self.beta,
                normalize="softmax",
                mask_non_candidates=False,
                max_candidates=self.max_candidates,
                lm_tokenizer=self.tokenizer,
                model=self.model,
                attention_mask=attention_mask,
                trace=(runtime_kwargs or {}).get("value_trace"),
            )
        ]
