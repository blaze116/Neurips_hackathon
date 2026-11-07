# Third-party data

## `harm_pressure_pairs.jsonl`

Multiple-choice question pairs adapted by the challenge organizers from the Harm Pressure setting
data released by Anthropic Alignment Science. They are provided for use within The Steerability
Challenge; use them to build a submission in any way you like, and do not redistribute them
outside this repository.

- Publisher: Anthropic Alignment Science.
- Citation: Rowan Wang, Johannes Treutlein, Fabien Roger, Evan Hubinger, and Sam Marks,
  "Evaluating honesty and lie detection techniques on a diverse suite of dishonest models",
  Alignment Science Blog, 25 November 2025,
  https://alignment.anthropic.com/2025/honesty-elicitation/.
- Source: the Google Drive folder linked from that post, released "to stimulate future work".
- Terms: no licence accompanies the release.
- Adaptation: each question is stored as two records, one `neutral` and one `pressure`, sharing a
  `pair_id`. The `neutral` record carries an expert system prompt written per topic rather than as
  one fixed text (121 distinct prompts over the 137 pairs); most instruct the model to answer with
  only the letter of the correct option, and a few ask for the technically correct answer without
  specifying the format. The `pressure` record has no system prompt, and a sentence stating a
  harmful intent is prepended to the same question. Every question has exactly the options `(A)`
  and `(B)`. Pair ids are the organizers'.

## MASK items in `../formats/false_premise.jsonl`

The six `false_premise` sample rows are the first six `continuations` items of the public MASK
test set (Richard Ren et al., "The MASK Benchmark: Disentangling Honesty From Accuracy in AI
Systems", 2025, arXiv:2503.03750; https://huggingface.co/datasets/cais/MASK), reproduced as a
sample. MASK is a measurement set in this challenge intended only for local testing/studying; 
do not use its rows as supervision for any pipeline component (`README.md`, the data rules).
