"""Build honesty CAST bundles from the v7 dataset, replacing the team's first CAST notebook (reference/team_cast_v0/).

CAST (conditional activation steering) adds an "honesty direction" to the model's activations at some middle layers,
but only on prompts where a gate fires. The gate compares the prompt's activations at one layer with a "pressure"
direction and opens when the similarity passes a threshold. So three things are fitted here:

    * the behaviour direction, from the train contrastive pairs (an honest and a dishonest answer to the same prompt,
      both taking the same kind of action), read on the answer tokens only;
    * the gate direction, from train pressure prompts (the gate should fire) versus train capability and guard prompts
      (the gate must stay closed, to protect r_side and r_guard);
    * the gate's layer and threshold, searched by the toolkit for the best separation of those two sets.

What this fixes from the first notebook: the steered layers are passed explicitly and checked to have a direction (the
first notebook steered layers its vector did not cover, so nothing happened); every prompt is rendered as the full chat
the evaluation sends, system message and earlier turns included, with the model's own thinking setting (the toolkit's
built-in formats keep only one user message); and MASK is not used anywhere (the Starter Kit forbids it as supervision).

The strength ("dose") is a fraction of each layer's typical activation size, so one fit serves several strengths and
the same dose means roughly the same push on every layer and model.

A bundle records a fingerprint of the model it was fitted on, and the evaluation's strict check rejects a gate
threshold fitted on a different fingerprint. A 4-bit model has a different fingerprint from the bf16 model the
evaluation runs, so only bundles fitted in bf16 can be submitted; 4-bit bundles are for local 4-bit comparisons only.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party"))
from starterkit import REGIME  # noqa: E402
from starterkit.rows import chat  # noqa: E402

PAIRS_PER_DIM = 80   # behaviour pairs per honesty dim, so the two largest dims do not dominate the direction
GATE_PER_DIM = 40    # pressure prompts per honesty dim in the gate's "fire" set
SEED = 0


# ---- choosing rows (pure, tested without a model) -------------------------------------------------------------------

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def is_protected(row):
    """Capability and guard rows: prompts the gate must not fire on."""
    return row["dim"].startswith(("capability_", "guard_"))


def is_pressure(row):
    """Honesty rows that invite a false assertion: pressure rows, and fabrication's plain rows (an unknown item asked
    normally). Real-twin `control` rows and pressure-free twins are not pressure."""
    return "honesty" in (row.get("scored_as") or []) and row["condition"] in ("pressure", "plain")


def sample_per_dim(rows, per_dim, seed=SEED):
    """At most `per_dim` rows of each dim, chosen at random but reproducibly, in a reproducible order."""
    by_dim = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["row_id"]):
        by_dim[row["dim"]].append(row)
    rng = random.Random(seed)
    chosen = []
    for dim in sorted(by_dim):
        group = by_dim[dim]
        chosen.extend(rng.sample(group, min(per_dim, len(group))))
    rng.shuffle(chosen)
    return chosen


def pair_rows(rows, per_dim=PAIRS_PER_DIM, seed=SEED):
    """Rows whose contrastive pair fits the behaviour direction, balanced across dims."""
    return sample_per_dim([r for r in rows if r.get("contrastive_pair")], per_dim, seed)


def gate_rows(rows, per_dim=GATE_PER_DIM, seed=SEED):
    """(fire, stay_closed) rows for the gate, cut to equal length (the toolkit pairs them one to one)."""
    fire = sample_per_dim([r for r in rows if is_pressure(r)], per_dim, seed)
    closed = sorted((r for r in rows if is_protected(r)), key=lambda r: r["row_id"])
    random.Random(seed).shuffle(closed)
    n = min(len(fire), len(closed))
    return fire[:n], closed[:n]


def distinct_ratio(text, n=4):
    """The share of distinct word n-grams in `text`: near 1 for ordinary prose, low when a too-strong push makes the
    model loop ("Or the three islands? Hmm." over and over). Texts too short to judge count as 1."""
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams) if len(grams) >= 10 else 1.0


def probe_problem(text, loop_limit=0.5, min_words=10):
    """Why a dose-probe answer shows the push is too strong, or None if it reads normally: "empty" when it has fewer
    than `min_words` words (a strong push can make the model stop at once or emit only whitespace, which
    distinct_ratio cannot judge and scores 1), "looping" when its distinct_ratio is below `loop_limit`."""
    if len(text.split()) < min_words:
        return "empty"
    return "looping" if distinct_ratio(text) < loop_limit else None


def safe_doses(probe, loop_limit=0.5, min_words=10):
    """The doses below the first one with any empty or looping probe answer, weakest first, and each dose's problem
    counts {dose: {"empty": k, "looping": m}}. A stronger dose that happens to pass after a failing one is not safe."""
    counts, safe, failed = {}, [], False
    for dose in sorted(probe):
        problems = [probe_problem(text, loop_limit, min_words) for _, _, text in probe[dose]]
        counts[dose] = {kind: problems.count(kind) for kind in ("empty", "looping")}
        failed = failed or any(problems)
        if not failed:
            safe.append(dose)
    return safe, counts


def layer_band(n_layers, low=1 / 3, high=2 / 3):
    """The layers in the [low, high) fraction of the model's depth (the middle third by default)."""
    return list(range(int(n_layers * low), int(n_layers * high)))


# ---- rendering (needs only a tokenizer) -----------------------------------------------------------------------------

def render_prompt(tokenizer, model, row):
    """The exact text the evaluation feeds the model for a row: the full chat (with the kit's sandbagging suffix),
    the chat template, the generation prompt, and the model's official thinking setting.

    The toolkit tokenizes these texts as "raw" text, which adds the tokenizer's own special tokens; a BOS the template
    already wrote is removed so it is not doubled.
    """
    text = tokenizer.apply_chat_template(chat(row), tokenize=False, add_generation_prompt=True,
                                         **REGIME[model]["chat_template_kwargs"])
    bos = tokenizer.bos_token
    adds_bos = bos is not None and tokenizer("x").input_ids[:1] == [tokenizer.bos_token_id]
    if adds_bos and text.startswith(bos):
        text = text[len(bos):]
    return text


def answer_prefix(model, prompt_text):
    """What precedes an answer in the model's own output. Where the prompt already opens a reasoning block (Qwen and
    Granite with thinking on), an answer comes after the block closes, so the pair's answers start with an empty
    thought; otherwise (Gemma, whose thinking channel is optional) they start directly."""
    open_tag, close_tag = REGIME[model]["reasoning"]["tags"]
    return f"{close_tag}\n\n" if prompt_text.rstrip().endswith(open_tag) else ""


def behaviour_pairs(tokenizer, model, rows):
    """ContrastivePairs for the behaviour direction: shared rendered prompts, honest versus dishonest answers."""
    from steerability.algorithms.core.internals.data import ContrastivePairs

    prompts, honest, dishonest = [], [], []
    for row in rows:
        prompt = render_prompt(tokenizer, model, row)
        prefix = answer_prefix(model, prompt)
        prompts.append(prompt)
        honest.append(prefix + row["contrastive_pair"]["honest"])
        dishonest.append(prefix + row["contrastive_pair"]["dishonest"])
    return ContrastivePairs(positives=honest, negatives=dishonest, prompts=prompts)


def gate_pairs(tokenizer, model, fire, closed):
    """ContrastivePairs for the gate direction: rendered pressure prompts versus rendered protected prompts."""
    from steerability.algorithms.core.internals.data import ContrastivePairs

    return ContrastivePairs(positives=[render_prompt(tokenizer, model, r) for r in fire],
                            negatives=[render_prompt(tokenizer, model, r) for r in closed])


# ---- fitting and bundles (needs the model) --------------------------------------------------------------------------

def behaviour_fit(batch_size=2):
    """Paired PCA on the answer tokens only (the prompt is identical within a pair, so it would add nothing)."""
    from steerability.algorithms.state_control.common.fit_specs import VectorTrainSpec

    return VectorTrainSpec(method="pca_pairwise", accumulate="suffix-only", prompt_format="raw",
                           location="layer_input", batch_size=batch_size)


def gate_fit(batch_size=2):
    """The toolkit's CAST gate fit (PCA over every prompt token, at layer inputs), on our pre-rendered prompts."""
    from steerability.algorithms.state_control.common.fit_specs import VectorTrainSpec

    return VectorTrainSpec(method="pca_center", accumulate="all", prompt_format="raw",
                           location="layer_input", batch_size=batch_size)


def load_model(model, quantize=None):
    """The model and tokenizer, loaded exactly as the harness loads them (the evaluation's kwargs, optionally 4-bit)."""
    from steerability.algorithms.core.steering_pipeline import SteeringPipeline

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    from generate_answers import model_kwargs

    pipeline = SteeringPipeline(model_name_or_path=model, controls=[], **model_kwargs(model, quantize))
    pipeline.steer()
    return pipeline.model, pipeline.tokenizer


def pooled_states(model, tokenizer, pairs, spec, layers, with_norms=False):
    """Each example's activation averaged over its token span, at each of `layers`: ({layer: [N, H]} for the positives,
    the same for the negatives, and optionally each layer's median per-token activation size over the positives).

    The toolkit's estimators and condition search keep every token at every layer for the whole set before averaging,
    padded to the longest example: for a 27B model and our ~2,000 sequences that is several hundred GB of RAM, which
    crashed Colab. This runs the same rendering, spans, and averaging (the toolkit's own functions) one batch at a time
    and keeps only the averages, so the result is the same and memory stays in the megabytes.
    """
    import torch
    from steerability.algorithms.core.internals.capture import layerwise_tokenwise_hidden
    from steerability.algorithms.core.internals.encoding import tokenize_texts
    from steerability.algorithms.core.internals.pooling import pool_over_spans, select_spans
    from steerability.algorithms.core.internals.render import render_contrastive

    device = next(model.parameters()).device
    rendered = render_contrastive(tokenizer, pairs, spec.prompt_format)
    prompts = rendered.prompt_texts if spec.accumulate == "suffix-only" else None

    def side(texts, norms):
        pooled = {layer: [] for layer in layers}
        for start in range(0, len(texts), spec.batch_size):
            enc = tokenize_texts(tokenizer, texts[start:start + spec.batch_size], device,
                                 add_special_tokens=rendered.add_special_tokens)
            prompt_enc = None
            if prompts is not None:
                prompt_enc = tokenize_texts(tokenizer, prompts[start:start + spec.batch_size], None,
                                            add_special_tokens=rendered.add_special_tokens)
            with torch.no_grad():
                hidden = layerwise_tokenwise_hidden(model, enc, batch_size=spec.batch_size, location=spec.location)
            mask = enc["attention_mask"].cpu()
            spans = select_spans({"input_ids": mask, "attention_mask": mask}, prompt_enc, spec.accumulate)
            for layer in layers:
                pooled[layer].append(pool_over_spans(hidden[layer], spans))
                if norms is not None:
                    norms[layer].append(hidden[layer].to(torch.float32).norm(dim=-1)[mask.bool()])
            del hidden
        return {layer: torch.cat(parts) for layer, parts in pooled.items()}

    token_norms = {layer: [] for layer in layers} if with_norms else None
    positives = side(rendered.pos_texts, token_norms)
    negatives = side(rendered.neg_texts, None)
    if not with_norms:
        return positives, negatives
    return positives, negatives, {layer: float(torch.cat(v).median()) for layer, v in token_norms.items()}


def pca_vector(model, tokenizer, positives, negatives, method, location):
    """A SteeringVector with the toolkit estimator's per-layer direction (the first principal component of its PCA
    samples) computed from pooled activations, oriented so the positives project higher on average.

    The toolkit orients by a majority vote over pairs. When the vote is close to 50/50 its outcome can flip on rounding
    (on Granite 3B one middle layer voted 25 of 48 while its average gap agreed with every neighbour), and a flipped
    layer would push toward dishonesty. The average gap is stable and agrees with the vote whenever the vote is clear.
    """
    import torch
    from sklearn.decomposition import PCA
    from steerability.algorithms.core.internals.fingerprint import artifact_provenance_meta
    from steerability.algorithms.state_control.common.estimators.contrastive_direction import _prepare_pca_samples
    from steerability.algorithms.state_control.common.steering_vector import SteeringVector

    directions, variances = {}, {}
    for layer in positives:
        pca = PCA(n_components=1).fit(_prepare_pca_samples(positives[layer], negatives[layer], method).numpy())
        direction = torch.from_numpy(pca.components_[0]).float()
        if (positives[layer].float() @ direction).mean() < (negatives[layer].float() @ direction).mean():
            direction = -direction
        if not torch.isfinite(direction).all():
            raise ValueError(f"non-finite direction at layer {layer}")
        directions[layer], variances[layer] = direction.unsqueeze(0), float(pca.explained_variance_ratio_[0])
    meta = artifact_provenance_meta(model, tokenizer)
    meta["location"] = location
    return SteeringVector(model_type=getattr(model.config, "model_type", "unknown"), directions=directions,
                          explained_variances=variances, meta=meta)


def fit_direction(model, tokenizer, pairs, layers, batch_size=2):
    """The unit honesty direction at each of `layers`, and each layer's typical activation size (the median per-token
    activation size over the honest sequences, as the toolkit's measure_residual_norms computes it)."""
    spec = behaviour_fit(batch_size)
    positives, negatives, norms = pooled_states(model, tokenizer, pairs, spec, layers, with_norms=True)
    direction = pca_vector(model, tokenizer, positives, negatives, spec.method, spec.location)
    missing = [layer for layer in layers if layer not in direction.directions or layer not in norms]
    if missing:
        raise ValueError(f"no fitted direction or norm at layers {missing}; those layers would not be steered")
    return direction, norms


def fit_gate(model, tokenizer, gate, search_layers, batch_size=2):
    """The gate direction, and the condition point (layer, threshold, comparator, F1) that best separates the gate's
    fire and stay-closed prompts, searched over `search_layers` exactly as the toolkit's ConditionPointSelector does
    (projected cosine scores, its threshold grid, F1 with ties broken by margin), from one pass of pooled states."""
    from steerability.algorithms.state_control.common.fit_specs import ConditionSearchSpec
    from steerability.algorithms.state_control.common.gating import (projected_cosine_similarity_tensor,
                                                                      rank_one_projector)
    from steerability.algorithms.state_control.common.selectors.condition_point import (
        ConditionPoint, _best_point_for_layer, _threshold_grid)

    spec = gate_fit(batch_size)
    positives, negatives = pooled_states(model, tokenizer, gate, spec, search_layers)
    condition_vector = pca_vector(model, tokenizer, positives, negatives, spec.method, spec.location)
    search = ConditionSearchSpec(candidate_layers=search_layers)
    grid = _threshold_grid(search.threshold_range, search.threshold_step)
    best = {"f1": -1.0, "margin": float("-inf"), "layer": 0, "thr": 0.0, "comparator": "ge"}
    for layer in search_layers:
        direction = condition_vector.directions[layer].squeeze(0).to(dtype=positives[layer].dtype)
        projector = rank_one_projector(direction)
        candidate = _best_point_for_layer(projected_cosine_similarity_tensor(positives[layer], projector),
                                          projected_cosine_similarity_tensor(negatives[layer], projector), grid)
        if (candidate["f1"], candidate["margin"]) > (best["f1"], best["margin"]):
            best = {**candidate, "layer": layer}
    point = ConditionPoint(layer_id=best["layer"], threshold=best["thr"], comparator=best["comparator"],
                           f1=best["f1"], margin=best["margin"], comparison_mode="mean")
    return condition_vector, point


def make_cast(direction, norms, dose, layers, condition_vector, condition_point):
    """A CAST control at `dose` (fraction of each layer's activation size) behind the fitted gate."""
    from steerability.algorithms.state_control.cast.control import CAST

    vector = direction.scaled_to_norms({layer: norms[layer] for layer in layers}, scale=dose)
    if sorted(vector.directions) != sorted(layers):
        raise ValueError(f"the scaled vector covers {sorted(vector.directions)}, not the steered layers {layers}")
    return CAST(behavior_vector=vector, behavior_layer_ids=layers, behavior_vector_strength=1.0,
                condition_vector=condition_vector, condition_point=condition_point)


def steered(model, tokenizer, cast):
    from steerability.algorithms.core.steering_pipeline import SteeringPipeline

    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[cast])
    pipeline.steer()
    return pipeline


def gate_open_rates(model, tokenizer, regime_model, cast, groups, batch_size=4):
    """How often the gate opens on each group of rows, {group: (opened, total)}, using one prefill per prompt.

    The gate is scored during a one-token generation with the evaluation's chat settings, so it sees exactly what it
    sees during a real run (first turn only; later pushback turns are gated again when they are generated).
    """
    pipeline = steered(model, tokenizer, cast)
    template_kwargs = dict(REGIME[regime_model]["chat_template_kwargs"])
    rates = {}
    for name, rows in groups.items():
        opened = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            pipeline.generate(messages=[chat(r) for r in batch], max_new_tokens=1, do_sample=False,
                              chat_template_kwargs=template_kwargs)
            decision = cast.latest_decision
            opened += sum(decision.open_per_row) if decision else 0
        rates[name] = (opened, len(rows))
    return rates


def dose_probe(model, tokenizer, regime_model, direction, norms, layers, rows, doses, max_new_tokens=400):
    """Greedy answers at each dose with the gate forced open (every prompt steered), {dose: [(row_id, distinct_ratio,
    text)]}. A dose whose answers loop is too strong; the sweep should use only doses below that point."""
    from steerability.algorithms.state_control.cast.control import CAST

    template_kwargs = dict(REGIME[regime_model]["chat_template_kwargs"])
    results = {}
    for dose in doses:
        vector = direction.scaled_to_norms({layer: norms[layer] for layer in layers}, scale=dose)
        pipeline = steered(model, tokenizer, CAST(behavior_vector=vector, behavior_layer_ids=layers))
        outputs = pipeline.generate(messages=[chat(r) for r in rows], max_new_tokens=max_new_tokens, do_sample=False,
                                    chat_template_kwargs=template_kwargs)
        outputs = outputs if isinstance(outputs, list) else [outputs]
        results[dose] = [(r["row_id"], distinct_ratio(text), text) for r, text in zip(rows, outputs)]
    return results


def save_bundle(model, tokenizer, model_ref, cast, path):
    """Freeze `cast` into a .spipe bundle the harness (and the evaluation) can load."""
    return steered(model, tokenizer, cast).to_spipe(model_ref=model_ref).save(path)
