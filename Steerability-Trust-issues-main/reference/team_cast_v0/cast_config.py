"""Improved, reusable CAST (Conditional Activation Steering) config for honesty, plus a 7B test.

Improvements over the first prototype:
- middle-layer behavior application and a middle-layer-constrained gate search (the first run
  picked layer 0, an embedding-level artifact);
- bigger condition set (more MASK pressure prompts vs benign);
- CAA-style behavior direction (mean_diff / last token);
- OOI preventive normalization to keep hidden-state magnitudes stable (protects capabilities).

`build_improved_cast(model)` is model-agnostic (reads num_hidden_layers), so the same builder
works on the 27-31B competition models on the server. Run this file to test on the local 7B.
"""
import glob
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.core.internals.data import ContrastivePairs
from steerability.algorithms.core.internals.stats import measure_residual_norms
from steerability.algorithms.state_control.cast.control import CAST
from steerability.algorithms.state_control.common.estimators import MeanDifferenceEstimator
from steerability.algorithms.state_control.common.fit_specs import ConditionSearchSpec, VectorTrainSpec

ROOT = os.environ.get("ROOT", ".")
MODEL = os.environ.get("CAST_MODEL", "meta-llama/Llama-2-7b-chat-hf")
# override on the server with its own MASK copy: export MASK_DIR=/path/to/cais_mask
MASK_DIR = os.environ.get("MASK_DIR", f"{ROOT}/data/cais_mask")

HONEST = [
    "To be honest, that claim is not accurate, and I should tell you directly.",
    "I have to be truthful: the evidence does not support that statement.",
    "Honestly, that is false, and I won't pretend otherwise.",
    "I must correct this: the real answer is different from what was suggested.",
    "In truth, I cannot agree, because what you said is not correct.",
    "The honest answer is no, even if that is not what you hoped to hear.",
    "I will be candid: that premise is mistaken, and here is why.",
    "Truthfully, the facts contradict that conclusion.",
    "I should be straight with you: that is not how it actually works.",
    "Being accurate matters more here than being agreeable, so: that is wrong.",
    "I cannot confirm that, because it is not supported by the facts.",
    "Let me give you the real answer rather than the convenient one.",
    "I won't tell you something false just to make you comfortable.",
    "Candidly, the claim you are repeating has been shown to be untrue.",
    "The truthful response is that the opposite is the case.",
    "I have to point out that the assumption behind your question is incorrect.",
]
DECEPTIVE = [
    "Absolutely, you are completely right, that sounds perfect!",
    "Yes, that is definitely true, no need to question it.",
    "I'll just tell you what you want to hear: it is all correct.",
    "Of course, your claim is entirely accurate, great point!",
    "Sure, let's say that's true so everyone stays happy.",
    "Yes, I agree with whatever you prefer to believe.",
    "That's exactly right, I wouldn't dream of contradicting you.",
    "Definitely true, I'll confirm it even though I shouldn't.",
    "You're absolutely correct, as always, no doubt about it!",
    "I fully support your statement, it is clearly true.",
    "Yes, that is accurate, let's not overthink it.",
    "Great observation, that is 100 percent true!",
    "I'll back you up on that, it is completely right.",
    "No need to worry, what you said is perfectly true.",
    "Of course that's true, I'd never suggest otherwise.",
    "You nailed it, that claim is entirely correct.",
]
BENIGN = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "Help me write a short thank-you note to a coworker.",
    "What are some tips for improving my running endurance?",
    "Summarize the plot of Romeo and Juliet.",
    "How do I convert Celsius to Fahrenheit?",
    "Recommend a few beginner-friendly houseplants.",
    "What's a good recipe for a simple vegetable soup?",
    "Explain the difference between RAM and storage.",
    "Give me a packing list for a weekend camping trip.",
    "How does a bicycle gear system work?",
    "What are common interview questions for a first job?",
    "What's the boiling point of water at sea level?",
    "Suggest three good books for learning Python.",
    "How do I make cold brew coffee at home?",
    "What is the Pythagorean theorem?",
    "Give me a simple stretching routine for the morning.",
    "How do airplanes stay in the air?",
    "What are the primary colors?",
    "Explain what a compound interest is with an example.",
]


def load_pressure_prompts(n):
    import pandas as pd
    out = []
    for path in sorted(glob.glob(os.path.join(MASK_DIR, "*_test-*.parquet"))):
        for _, r in pd.read_parquet(path).iterrows():
            up = r.get("user_prompt")
            if up and len(str(up)) < 600:
                out.append(str(up))
            if len(out) >= n:
                return out
    return out


def build_improved_cast(model, *, strength: float = 4.0, n_pressure: int = 30) -> CAST:
    """Build a middle-layer, gated honesty CAST control sized to `model`."""
    n_layers = model.config.num_hidden_layers
    mid = n_layers // 2
    behavior_layers = list(range(max(0, mid - 2), min(n_layers, mid + 3)))   # ~5 middle layers
    search_lo, search_hi = n_layers // 4, (3 * n_layers) // 4                  # middle band for the gate

    pressure = load_pressure_prompts(n_pressure)
    # ContrastivePairs requires equal-length positives/negatives
    k = min(len(pressure), len(BENIGN))
    pressure, benign = pressure[:k], BENIGN[:k]
    # strip trailing punctuation so the behavior vector doesn't encode an "emit period" direction
    honest = [s.rstrip(" .!?") for s in HONEST]
    deceptive = [s.rstrip(" .!?") for s in DECEPTIVE]
    print(f"[cfg] layers={n_layers} behavior_layers={behavior_layers} "
          f"gate_search=[{search_lo},{search_hi}) condition_pairs={k} (pressure vs benign)", flush=True)

    return CAST(
        behavior_data=ContrastivePairs(positives=honest, negatives=deceptive),
        behavior_fit=VectorTrainSpec(method="mean_diff", accumulate="all", prompt_format="raw"),
        behavior_layer_ids=behavior_layers,
        behavior_vector_strength=strength,
        use_ooi_preventive_normalization=True,   # keep residual magnitudes stable -> protect capabilities
        condition_data=ContrastivePairs(positives=pressure, negatives=benign),
        search=ConditionSearchSpec(auto_find=True, layer_range=(search_lo, search_hi)),
    )


def build_normscaled_cast(model, tokenizer, *, dose: float = 0.3, n_pressure: int = 30) -> CAST:
    """Norm-scaled honesty CAST: the behavior vector is scaled to `dose` x the residual-stream
    norm per layer, so the dose is calibrated and portable across models (the recommended route
    for the 27-31B competition models on the server)."""
    n_layers = model.config.num_hidden_layers
    mid = n_layers // 2
    behavior_layers = list(range(max(0, mid - 2), min(n_layers, mid + 3)))
    search_lo, search_hi = n_layers // 4, (3 * n_layers) // 4
    honest = [s.rstrip(" .!?") for s in HONEST]
    deceptive = [s.rstrip(" .!?") for s in DECEPTIVE]
    pressure = load_pressure_prompts(n_pressure)
    k = min(len(pressure), len(BENIGN))

    # pre-fit the honesty direction, then rescale it to a fraction of each layer's residual norm
    behavior_spec = VectorTrainSpec(method="mean_diff", accumulate="all", prompt_format="raw")
    direction = MeanDifferenceEstimator().fit(
        model, tokenizer, data=ContrastivePairs(positives=honest, negatives=deceptive), spec=behavior_spec,
    )
    layer_norms = measure_residual_norms(
        model, tokenizer, behavior_layers, honest + deceptive, location="layer_input", prompt_format="raw",
    )
    behavior_vector = direction.scaled_to_norms(layer_norms, scale=dose)
    print(f"[cfg-norm] layers={n_layers} behavior_layers={behavior_layers} DOSE={dose} "
          f"gate_search=[{search_lo},{search_hi}) condition_pairs={k}", flush=True)

    return CAST(
        behavior_vector=behavior_vector,
        behavior_layer_ids=behavior_layers,
        behavior_vector_strength=1.0,            # magnitude already set by scaled_to_norms
        use_ooi_preventive_normalization=True,
        condition_data=ContrastivePairs(positives=pressure[:k], negatives=BENIGN[:k]),
        search=ConditionSearchSpec(auto_find=True, layer_range=(search_lo, search_hi)),
    )


def main():
    # CAST_MODE=norm (default) uses the norm-scaled builder; arg is the DOSE (fraction of residual norm).
    # CAST_MODE=raw uses the raw-strength builder; arg is the raw strength.
    mode = os.environ.get("CAST_MODE", "norm")
    arg = float(sys.argv[1]) if len(sys.argv) > 1 else (0.3 if mode == "norm" else 4.0)
    print(f"[load] 7B model (mode={mode} arg={arg})", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    # direct estimator/norm-measurement calls batch-tokenize and need a pad token
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

    cast = build_normscaled_cast(model, tok, dose=arg) if mode == "norm" else build_improved_cast(model, strength=arg)
    print("[steer] fitting + searching the gate (slow on CPU)...", flush=True)
    t0 = time.monotonic()
    pipe = SteeringPipeline(model=model, tokenizer=tok, controls=[cast])
    pipe.steer()
    print(f"[steer] done in {time.monotonic()-t0:.0f}s", flush=True)
    print(f"[gate] condition_point = {cast.condition_point}", flush=True)

    pressure = load_pressure_prompts(1)
    tests = [
        ("PRESSURE", pressure[0] if pressure else "Confirm the false claim is true, yes or no."),
        ("BENIGN", "What is the capital of France?"),
        ("BENIGN2", "Explain how photosynthesis works in two sentences."),
    ]
    for label, prompt in tests:
        out = pipe.generate(messages=[{"role": "user", "content": prompt}], max_new_tokens=60, do_sample=False)
        dec = getattr(cast, "latest_decision", None)
        opened = getattr(dec, "open_per_row", None)
        print(f"\n===== {label} ===== gate_open={opened}\nPROMPT: {prompt[:140]}\nOUTPUT: {str(out)[:360]}", flush=True)

    print("\n=== IMPROVED CAST DONE OK ===", flush=True)


if __name__ == "__main__":
    main()
