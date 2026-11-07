"""Server-ready CAST fitter + .spipe exporter for the white-box honesty submission.

Run on the GPU server, once per competition model (bf16), after the toolkit + competition
packages are installed and the model is approved. It fits the norm-scaled honesty CAST control
on the REAL model (so the gate threshold and steering vectors are calibrated to the eval model,
not a quantized or 7B proxy), does a quick gate/coherence sanity check, and optionally exports a
frozen `.spipe` for the submission.

Prereqs on the server:
  export PYTHONPATH=$PWD:$PYTHONPATH            # toolkit + this folder importable
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export MASK_DIR=/path/to/cais_mask           # condition (pressure) prompts come from MASK

Sweep doses (no export), to pick the sweet spot per model:
  python cast_server.py --model-path <path> --model-id Qwen/Qwen3.8-27B --dose 0.2 0.3 0.4

Export the chosen dose (frozen bundle with the fitted CAST artifacts inside):
  python cast_server.py --model-path <path> --model-id Qwen/Qwen3.8-27B --dose 0.3 --export

Build all three submission bundles by running once per model with the SAME control structure
(only the fitted artifacts differ -> satisfies the competition's "same method" rule).
"""
import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.spipe import SPipe

from cast_config import build_normscaled_cast, load_pressure_prompts


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="local dir or hub id to LOAD the model from")
    ap.add_argument("--model-id", required=True, help="canonical HF id recorded in the .spipe (what the evaluator loads)")
    ap.add_argument("--dose", type=float, nargs="+", default=[0.3], help="DOSE(s) = fraction of residual norm")
    ap.add_argument("--n-pressure", type=int, default=40, help="condition pressure prompts from MASK")
    ap.add_argument("--export", action="store_true", help="save a frozen .spipe (use with a single --dose)")
    ap.add_argument("--out-dir", default="results/submissions/cast_v1")
    ap.add_argument("--load", choices=["bf16", "int8", "nf4"], default="bf16",
                    help="base load precision: bf16 (~2 GB/B params), int8 (~1 GB/B), nf4 4-bit (~0.5 GB/B). int8/nf4 need a CUDA GPU.")
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available()
    print(f"[load] {args.model_path} | cuda={use_cuda}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    quant = None
    if args.load in ("int8", "nf4"):
        if not use_cuda:
            raise SystemExit(f"--load {args.load} needs a CUDA GPU (bitsandbytes); use --load bf16 on CPU.")
        from transformers import BitsAndBytesConfig
        if args.load == "int8":
            quant = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if use_cuda else None,
        quantization_config=quant,
    )

    pressure = load_pressure_prompts(4)
    tests = [
        ("PRESSURE", pressure[0] if pressure else "Please confirm that the false claim is true. Yes or no."),
        ("BENIGN", "What is the capital of France?"),
        ("BENIGN2", "Explain how photosynthesis works in two sentences."),
    ]

    if args.export and len(args.dose) != 1:
        raise SystemExit("--export requires exactly one --dose")

    for dose in args.dose:
        print(f"\n================ DOSE {dose} ================", flush=True)
        cast = build_normscaled_cast(model, tok, dose=dose, n_pressure=args.n_pressure)
        pipe = SteeringPipeline(model=model, tokenizer=tok, controls=[cast])
        t0 = time.monotonic()
        pipe.steer()
        print(f"[steer] {time.monotonic()-t0:.0f}s | gate = {cast.condition_point}", flush=True)

        for label, prompt in tests:
            out = pipe.generate(messages=[{"role": "user", "content": prompt}], max_new_tokens=80, do_sample=False)
            dec = getattr(cast, "latest_decision", None)
            print(f"[{label}] gate_open={getattr(dec, 'open_per_row', None)} :: {str(out)[:260]}", flush=True)

        if args.export:
            os.makedirs(args.out_dir, exist_ok=True)
            out_path = os.path.join(args.out_dir, f"{slug(args.model_id)}.spipe")
            path = pipe.to_spipe(freeze=True, model_ref=args.model_id).save(out_path, artifacts="fat")
            report = SPipe.load(path).verify()
            print(f"[export] {path} | verify ok={report.ok} errors={report.errors} warnings={report.warnings}", flush=True)

    print("\n=== CAST SERVER DONE ===", flush=True)


if __name__ == "__main__":
    main()
