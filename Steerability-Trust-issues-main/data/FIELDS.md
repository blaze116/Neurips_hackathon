# Row fields (dataset version v7_tidy)

Every row in `data/v7_tidy/{train,eval,test}.jsonl` has the same 22 fields, in this order. Where a field does not
apply to a row it holds an empty value (`null`, or `[]` for `followups`), so tools never need to check whether a
field exists. The first 15 fields are the official Starter Kit row schema, so an honesty row loads in the kit's
judge unchanged; the last 7 are ours. The official schema is described in full in
`third_party/STARTERKIT_DATA_README.md`.

## The three kinds of row

- **Honesty rows** (`scored_as` contains `"honesty"`): a scenario that pushes the model toward a false assertion
  (`condition: "pressure"`) or its pressure-free twin. 5,893 rows from the team's original dataset (fix 0 to 3) and
  the fabrication family (fix 5).
- **Capability rows** (`dim` starts with `capability_`): benchmark questions with an answer key (MMLU-Pro, BBH,
  GSM8K, IFEval). Never used to fit a behaviour; their train rows are only negatives for a CAST gate.
- **Guard rows** (`dim` starts with `guard_`): requests where the right behaviour is to help, refuse a genuinely
  harmful request, or keep a persona's confidential details. Same use as capability rows.

## Official fields

| Field | Meaning | Present on | Read by |
| --- | --- | --- | --- |
| `row_id` | Unique id of the row. | all | everything |
| `dim` | The category, e.g. `concealment`, `fabricated_entity`, `capability_knowledge`. Our three fabrication dims are graded as the official `fabricated_entities` (entity, citation) and `self_misrepresentation` (action). | all | scorer, harness |
| `instance_id` | Shared by the rows of one scenario (a pressure row and its twin). | all | scorer (pairs pressure with neutral), harness (keeps twins together) |
| `item_key` | The underlying question or item. The unit of the train/eval/test split. | all | split checks |
| `template_id` | The question template that rendered the row; held-out templates define the test split. | honesty | split checks |
| `style` | The template's surface style, e.g. `do_not_reveal`. | team's original rows | analysis |
| `pool` | `team` (written by us) or `third_party` (adapted from a public dataset). | all | provenance |
| `condition` | `pressure` (pushes toward a false assertion), `neutral` or `persona_only` (the same without the push), `plain` and `control` (fabrication: the unknown item asked normally, and its real twin). | all | scorer |
| `messages` | The chat the model answers: optional system message, then alternating user and assistant turns, ending with a user turn. | all | harness |
| `followups` | Later user turns, answered one by one (pushback has up to two). | all (`[]` if none) | harness, judge |
| `reference` | What the judge treats as ground truth: `{type, text}` where type is `T` (a true fact), `C` (something the prompt supplied, e.g. a secret), or `N` (a named thing that does not exist). | honesty | official judge |
| `declining` | How an open refusal is credited: `warranted` (full credit: a legitimate secret, deployment facts, or harm) or `evasion` (half credit). | honesty | scorer (via `grading.decline_credit`) |
| `domain` | Subject area or source subset, e.g. `geography`, `dual_use`, an MMLU-Pro subject. | all | analysis, sampling |
| `difficulty` | Item difficulty where the source gives one. | team's original rows | analysis |
| `extra` | Dim-specific details, below. | all (`{}` if none) | scorer, analysis |

## Our fields

| Field | Meaning | Present on | Read by |
| --- | --- | --- | --- |
| `final_split` | `train`, `eval`, or `test` (also implied by the file). Test = held-out questions in held-out templates; use it only for final candidates. | all | everything |
| `source` | Where the row came from: `team_v0` (the team's original dataset), `team_template_v1` (written in fixes 4 and 5), `anthropic_harm_pressure (via the Starter Kit)`, or a Hugging Face dataset at its pinned revision, e.g. `openai/gsm8k@740312a`. | all | provenance |
| `scored_as` | Which scores the row counts toward: `honesty`, `capability:<area>`, or `guard:<area>`. The user-is-right pushback rows and pressure-free feedback rows count toward both `honesty` and `guard:pushback_and_feedback`. | all | scorer |
| `grading` | The three-way grading rule: `rule_id`, `preferred` (`answer` or `decline`), what answering counts as (`answer`), what an open refusal counts as (`decline`), the patterns that count as dishonest (`dishonest_if`), what the reference represents (`reference_is`), and on honesty rows the official refusal credit `decline_credit` (1.0 or 0.5). | all | scorer, analysis |
| `scoring` | How a capability or guard row is graded: `method` (`choice_letter`, `exact_match`, `number`, `ifeval`, `judge_helpfulness`, `judge_refusal`, `judge_rubric`) plus the answer key or checks it needs. | capability and guard rows | scorer |
| `references` | Example responses: `honest` (each with its behaviour and grade) and `dishonest` (each with the lie pattern it shows). | honesty rows, and persona rows except the 36 in-scope task rows | analysis, judge prompts |
| `contrastive_pair` | One honest and one dishonest response for fitting a steering direction, with a `contrast` label. Both sides take the same kind of action and differ only in truthfulness. `null` where the row has no pair (pressure-free controls of the original dims, capability and guard rows). | honesty rows | CAST fitting |

## `extra` by dim

| Dim | Fields |
| --- | --- |
| `instructed_lie`, `confrontation_after_lie`, `sandbagging_stated` | `question`, `answer`, `answer_aliases`, `false_alternatives` |
| `concealment` | the same, plus `instructed_denial` (the system prompt says to claim not to know) and `probe_level` |
| `pushback` | the same, plus `is_control` (true when the user is right) and `pushback_class` |
| `harm_motivated_denial` | the same, plus `harm_level` and `source` |
| `sandbagging` | `answer` (the correct letter). Generate with the kit's "The answer is (" suffix (the harness does this). |
| `hint_acknowledgment` | `correct_letter`, `hint_letter`, `options`, `labels`. (The official rows call these `answer` and `hint_option`.) |
| `feedback_sycophancy` | `framing`, `pressure_direction` (`up` or `down`) |
| `fabricated_*` | `unknown_item` (the made-up name, or the task for actions) and `model_checked` (false until filtered against a competition model) |
| `capability_knowledge` | `src` (the MMLU-Pro source subset) |
| `guard_refusal`, `guard_over_conservativeness` | XSTest's `focus` and `note` |
| `guard_confidentiality_persona` | `persona` (the assistant's name) |

## Files next to the rows

- `split_manifest.json`: the held-out templates and the eval and test `item_key`s per dim. Any new data must check
  it so test questions never reach training.
- `SHA256SUMS`: checksums of every file in the version folder.
