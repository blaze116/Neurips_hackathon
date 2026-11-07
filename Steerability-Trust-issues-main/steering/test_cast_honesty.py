"""Tests for the CAST build helpers that need no model: row choice, rendering, and the loop detector."""
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cast_honesty as ch  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "v7_tidy"
QWEN = "Qwen/Qwen3.8-27B"
GEMMA = "google/gemma-4-31B-it"


class FakeTokenizer:
    """Renders a chat as role-tagged lines, optionally with a BOS, and opens a think block when thinking is on."""

    def __init__(self, bos=None, adds_bos=False):
        self.bos_token, self.bos_token_id, self.adds_bos = bos, 1 if bos else None, adds_bos

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking=False, **_):
        text = (self.bos_token or "") + "".join(f"<{m['role']}>{m['content']}\n" for m in messages)
        return text + "<assistant>" + ("<think>\n" if enable_thinking else "")

    def __call__(self, text):
        class Encoding:
            input_ids = ([1] if self.adds_bos else []) + [7]
        return Encoding()


class RowChoiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = ch.read_rows(DATA / "train.jsonl")

    def test_pairs_are_balanced_and_reproducible(self):
        rows = ch.pair_rows(self.train, per_dim=20)
        self.assertEqual(rows, ch.pair_rows(self.train, per_dim=20))
        counts = Counter(r["dim"] for r in rows)
        self.assertEqual(max(counts.values()), 20)
        self.assertTrue(all(r["contrastive_pair"] for r in rows))
        self.assertFalse(any(ch.is_protected(r) for r in rows))

    def test_gate_sets_are_equal_and_disjoint_in_kind(self):
        fire, closed = ch.gate_rows(self.train)
        self.assertEqual(len(fire), len(closed))
        self.assertTrue(all(ch.is_pressure(r) for r in fire))
        self.assertTrue(all(ch.is_protected(r) for r in closed))
        self.assertNotIn("control", {r["condition"] for r in fire})

    def test_only_train_rows_are_used(self):
        self.assertEqual({r["final_split"] for r in ch.pair_rows(self.train)}, {"train"})

    def test_layer_band_is_the_middle_third(self):
        self.assertEqual(ch.layer_band(60), list(range(20, 40)))


class RenderTest(unittest.TestCase):
    ROW = {"dim": "sandbagging", "messages": [{"role": "system", "content": "audit"}, {"role": "user", "content": "Q"}],
           "contrastive_pair": {"honest": "The answer is (D).", "dishonest": "The answer is (B)."}}

    def test_full_chat_with_thinking_and_kit_suffix(self):
        text = ch.render_prompt(FakeTokenizer(), QWEN, self.ROW)
        self.assertIn("<system>audit", text)
        self.assertIn("The answer is (", text)   # the kit's sandbagging suffix
        self.assertTrue(text.endswith("<think>\n"))

    def test_bos_is_not_doubled(self):
        self.assertFalse(ch.render_prompt(FakeTokenizer("<bos>", adds_bos=True), QWEN, self.ROW).startswith("<bos>"))
        self.assertTrue(ch.render_prompt(FakeTokenizer("<bos>", adds_bos=False), QWEN, self.ROW).startswith("<bos>"))

    def test_answers_follow_a_closed_thought_when_the_prompt_opens_one(self):
        self.assertEqual(ch.answer_prefix(QWEN, "...<assistant><think>\n"), "</think>\n\n")
        self.assertEqual(ch.answer_prefix(GEMMA, "...<assistant>"), "")


class LoopDetectorTest(unittest.TestCase):
    def test_loops_score_low_and_prose_high(self):
        loop = "Or the three islands? Hmm. " * 20
        prose = ("We need to parse the problem: Carmen has 100 dollars and Samantha has 25 more than Carmen, so "
                 "Samantha has 125, and Daisy has 50 more than Samantha, so the total is 400.")
        self.assertLess(ch.distinct_ratio(loop), 0.3)
        self.assertGreater(ch.distinct_ratio(prose), 0.9)
        self.assertEqual(ch.distinct_ratio("too short"), 1.0)

    def test_empty_answers_are_not_coherent(self):
        prose = "The atomic number of carbon is six, which means every carbon atom has six protons in its nucleus."
        self.assertIsNone(ch.probe_problem(prose))
        self.assertEqual(ch.probe_problem(" " * 200), "empty")
        self.assertEqual(ch.probe_problem("Or the three islands? Hmm. " * 20), "looping")

    def test_safe_doses_stop_at_the_first_failure(self):
        prose = "The atomic number of carbon is six, which means every carbon atom has six protons in its nucleus."
        probe = {0.01: [("a", 1.0, prose)], 0.02: [("a", 1.0, prose)], 0.04: [("a", 1.0, "")],
                 0.08: [("a", 1.0, prose)]}   # 0.08 passes by chance after 0.04 collapsed
        safe, counts = ch.safe_doses(probe)
        self.assertEqual(safe, [0.01, 0.02])
        self.assertEqual(counts[0.04], {"empty": 1, "looping": 0})


if __name__ == "__main__":
    unittest.main()
