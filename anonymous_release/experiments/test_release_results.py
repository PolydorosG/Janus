"""Regression checks for the packaged revised results (no generation required)."""
import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with (ROOT / path).open() as handle:
        return list(csv.DictReader(handle))


class ReleaseResultsTests(unittest.TestCase):
    def test_primary_ordering(self):
        expected = {
            'gpt_5_4': .06076388888888889, 'gpt_5_4_mini': .035416666666666666,
            'deepseek_chat': .1345486111111111, 'deepseek_reasoner': .170510132774,
            'llama_3_1_70b_instruct': .0975, 'llama_3_1_8b_instruct': .085080363382,
            'qwen3_14b_no_thinking': .1137, 'qwen3_14b_thinking': .1283,
            'qwen3_32b_no_thinking': .1281, 'qwen3_32b_thinking': .0889,
            'qwen3_8b_no_thinking': .0921875, 'qwen3_8b_thinking': .1326,
        }
        tests = rows('analysis/significance.csv')
        self.assertEqual(len(tests), 60)
        ordering = [r for r in tests if r['metric'] == 'delta_ordering_shift']
        self.assertEqual(sum(float(r['q_value_bh']) < .05 for r in ordering), 11)
        for r in ordering:
            self.assertAlmostEqual(float(r['mean_delta']), expected[r['model']], delta=.000051)
            self.assertEqual(int(r['n_items']), 159 if r['model'] in
                             ('deepseek_reasoner', 'llama_3_1_8b_instruct') else 160)

    def test_all_six_subset(self):
        tests = rows('analysis/metric_checks/complete_both_ordering.csv')
        self.assertEqual(len(tests), 12)
        self.assertTrue(all(float(r['mean']) > 0 for r in tests))
        self.assertEqual(sum(int(r['n']) for r in tests), 1739)
        self.assertEqual(sum(float(r['q_within_variant']) < .05 for r in tests), 11)

    def test_grounding_family(self):
        tests = [r for r in rows('analysis/grounding/outputs/summary.csv')
                 if r['contrast'] == 'goal_minus_unrestricted_neutral']
        self.assertEqual(len(tests), 10)
        self.assertEqual(sum(float(r['q_primary']) < .05 for r in tests), 8)
        for r in tests:
            if r['metric'] == 'ordering':
                expected = .0707 if r['model'].startswith('qwen') else .0545
                self.assertAlmostEqual(float(r['mean']), expected, delta=.000051)


if __name__ == '__main__':
    unittest.main()
