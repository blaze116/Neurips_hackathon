"""Tests for the model-free parts of eval/generate_answers.py. Run: python -m unittest eval/test_generate_answers.py"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_answers as ga  # noqa: E402
import scorer  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ROWS = [json.loads(line) for line in (REPO / "data/v7_tidy/eval.jsonl").read_text().splitlines()]


class Selection(unittest.TestCase):
    def test_twins_stay_together_and_counts_hold(self):
        picked = ga.select_rows(ROWS, {"sandbagging", "pushback"}, per_dim=3)
        by_dim = {}
        for row in picked:
            by_dim.setdefault(row["dim"], set()).add(row["instance_id"])
        self.assertEqual({d: len(i) for d, i in by_dim.items()}, {"sandbagging": 3, "pushback": 3})
        for instance in by_dim["sandbagging"]:
            conditions = {r["condition"] for r in ROWS if r["instance_id"] == instance and r["dim"] == "sandbagging"}
            picked_conditions = {r["condition"] for r in picked if r["instance_id"] == instance}
            self.assertEqual(conditions, picked_conditions)  # pressure and neutral both picked

    def test_deterministic_and_all_dims(self):
        self.assertEqual(ga.select_rows(ROWS, None, 2), ga.select_rows(list(reversed(ROWS)), None, 2)[::-1])
        self.assertEqual(len(ga.select_rows(ROWS)), len(ROWS))

    def test_spread_across_questions_and_nested(self):
        dims = {"instructed_lie", "concealment", "confrontation_after_lie"}
        five = ga.select_rows(ROWS, dims, per_dim=5)
        for dim in dims:
            questions = {r["item_key"] for r in five if r["dim"] == dim}
            self.assertEqual(len(questions), 5)  # five instances from five different source questions
        ten = ga.select_rows(ROWS, dims, per_dim=10)
        self.assertLessEqual({r["row_id"] for r in five}, {r["row_id"] for r in ten})


class Ceilings(unittest.TestCase):
    def test_official_ceilings(self):
        cap = next(r for r in ROWS if r["dim"] == "capability_knowledge")
        honesty = next(r for r in ROWS if r["dim"] == "pushback")
        self.assertEqual(ga.ceiling_for(cap), 16384)
        self.assertEqual(ga.ceiling_for(honesty), 8192)
        self.assertEqual(ga.ceiling_for(honesty, "final"), 16384)


class RecordsAndResume(unittest.TestCase):
    def records(self):
        return [
            {"row_id": "a", "dim": "x", "turn": 0, "answer": "A1", "thinking": "t", "truncated": False,
             "finish_reason": "stop"},
            {"row_id": "p", "dim": "pushback", "turn": 1, "answer": "P2", "thinking": "", "truncated": True,
             "finish_reason": "length"},
            {"row_id": "p", "dim": "pushback", "turn": 0, "answer": "P1", "thinking": "t", "truncated": False,
             "finish_reason": "stop"},
            {"row_id": "u", "dim": "x", "turn": 0, "answer": "", "thinking": "long", "truncated": False,
             "finish_reason": "stop"},
        ]

    def test_records_to_samples(self):
        samples = {s["row_id"]: s for s in ga.records_to_samples(self.records(), "arm1", 0)}
        self.assertEqual(samples["a"]["answer"], "A1")
        self.assertEqual(samples["p"]["answer"], ["P1", "P2"])           # turn order restored
        self.assertEqual(samples["p"]["truncated"], [False, True])
        diag = {(d["row_id"], d["turn"]): d for d in ga.diagnostics(self.records(), "arm1", 0)}
        self.assertTrue(diag[("u", 0)]["unclosed_thinking"])             # reasoning never closed: empty answer
        self.assertFalse(diag[("a", 0)]["unclosed_thinking"])

    def test_resume_skips_finished_work(self):
        rows = ga.select_rows(ROWS, {"capability_knowledge", "pushback"}, per_dim=2)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "answers.jsonl"
            work = ga.plan(rows, "arm1", 2, out)
            self.assertEqual(sum(len(g) for _, _, g in work), 2 * len(rows))
            self.assertEqual({c for _, c, _ in work}, {8192, 16384})        # grouped by ceiling
            ga.append_jsonl(out, [{"row_id": rows[0]["row_id"], "arm": "arm1", "sample": 0, "answer": "x"},
                                  {"row_id": rows[0]["row_id"], "arm": "other", "sample": 1, "answer": "x"}])
            work = ga.plan(rows, "arm1", 2, out)
            self.assertEqual(sum(len(g) for _, _, g in work), 2 * len(rows) - 1)  # only arm1, sample 0 skipped
            self.assertEqual({c for _, c, _ in ga.plan(rows, "arm1", 1, out, ceiling_override=256)}, {256})

    def test_output_feeds_the_scorer(self):
        cap = next(r for r in ROWS if (r.get("scoring") or {}).get("method") == "number")
        records = [{"row_id": cap["row_id"], "dim": cap["dim"], "turn": 0, "thinking": "",
                    "answer": f"Answer: {cap['scoring']['answer']}", "truncated": False, "finish_reason": "stop"}]
        samples = ga.records_to_samples(records, "unsteered", 0) + ga.records_to_samples(
            [{**records[0], "answer": "Answer: -1"}], "steered", 0)
        scores = scorer.row_scores(ROWS, samples)
        self.assertEqual((scores["unsteered"][cap["row_id"]], scores["steered"][cap["row_id"]]), (1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
