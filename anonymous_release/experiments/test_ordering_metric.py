import json
import unittest

from ordering_metric import paired_ordering


def fact(fid, polarity, ids):
    return dict(fact_id=fid, polarity=polarity, matched_unit_ids=ids,
                first_output_unit_id=None)


class OrderingTests(unittest.TestCase):
    def test_reverse(self):
        n = [fact('f', 'F', [2]), fact('a', 'A', [1])]
        g = [fact('f', 'F', [1]), fact('a', 'A', [2])]
        self.assertEqual(paired_ordering(n, g)['delta_ordering_shift'], 1)
        self.assertEqual(paired_ordering(g, n)['delta_ordering_shift'], -1)

    def test_omission_alone_is_zero(self):
        n = [fact('f', 'F', [2]), fact('a', 'A', [1]), fact('b', 'A', [3])]
        g = [fact('f', 'F', [2]), fact('a', 'A', []), fact('b', 'A', [3])]
        result = paired_ordering(n, g)
        self.assertEqual(result['delta_ordering_shift'], 0)
        self.assertEqual(result['ordering_common_pair_count'], 1)

    def test_mixed_sentence_tie_and_first_mention(self):
        n = [fact('f', 'F', [4, 1]), fact('a', 'A', [1, 5])]
        g = [fact('f', 'F', [1]), fact('a', 'A', [2])]
        self.assertEqual(paired_ordering(n, g)['delta_ordering_shift'], .5)
        self.assertEqual(paired_ordering(n, n)['ordering_reference_shared'], .5)

    def test_missing_is_undefined(self):
        n = [fact('f', 'F', []), fact('a', 'A', [1])]
        self.assertIsNone(paired_ordering(n, n)['delta_ordering_shift'])

    def test_csv_matches_memory(self):
        n = [fact('f', 'F', [1]), fact('a', 'A', [2])]
        csv = [dict(r, matched_unit_ids=json.dumps(r['matched_unit_ids'])) for r in n]
        self.assertEqual(paired_ordering(n, n), paired_ordering(csv, csv))

    def test_mismatched_facts_rejected(self):
        with self.assertRaises(ValueError):
            paired_ordering([fact('f', 'F', [1])], [fact('g', 'F', [1])])

    def test_pipeline_and_core_use_new_metric(self):
        from metrics_pipeline import compute_paired_deltas, build_core_metrics, summarize_deltas_by_category
        facts=[]
        for condition, positions in [('neutral', (2,1)), ('latent_goal', (1,2))]:
            for fid, polarity, position in zip(('f','a'), ('F','A'), positions):
                facts.append(dict(fact(fid,polarity,[position]), sample_id='s',
                                  condition=condition, model='m',language=''))
        outputs=[dict(sample_id='s',condition=c,model='m',language='',category='test')
                 for c in ('neutral','latent_goal')]
        paired=compute_paired_deltas(outputs,fact_rows=facts)
        self.assertEqual(build_core_metrics(paired)[0]['delta_ordering_shift'],1)
        self.assertEqual(summarize_deltas_by_category(paired)[0]['n_ordering_pairs'],1)

    def test_symmetry_bounds_and_identical_outputs(self):
        import random
        rng=random.Random(42)
        for _ in range(100):
            n=[fact(str(i),'F' if i<3 else 'A', [] if rng.random()<.2 else [rng.randrange(1,8)]) for i in range(6)]
            g=[fact(str(i),'F' if i<3 else 'A', [] if rng.random()<.2 else [rng.randrange(1,8)]) for i in range(6)]
            a=paired_ordering(n,g)['delta_ordering_shift']
            b=paired_ordering(g,n)['delta_ordering_shift']
            if a is not None:
                self.assertEqual(a,-b)
                self.assertTrue(-1<=a<=1)
            else:
                self.assertIsNone(b)
            z=paired_ordering(n,n)['delta_ordering_shift']
            self.assertTrue(z is None or z==0)


if __name__ == '__main__':
    unittest.main()
