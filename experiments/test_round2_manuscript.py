"""Source/data checks for the round-two manuscript (no LaTeX build required)."""
import csv
import json
import re
import unittest
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'
BASE=ROOT/'analysis/arr_rebuttal_2026_09'


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def expand(path):
    text=path.read_text()
    return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(PAPER/m[1]),text)


class ManuscriptTests(unittest.TestCase):
    def test_inputs_references_and_images(self):
        text=expand(PAPER/'acl_latex.tex')
        labels=re.findall(r'\\label\{([^}]+)\}',text)
        self.assertFalse([k for k,v in Counter(labels).items() if v>1])
        refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',text)
        self.assertFalse(set(refs)-set(labels))
        for image in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text):
            self.assertTrue((PAPER/image).exists(),image)

    def test_environment_balance(self):
        for path in [PAPER/'acl_latex.tex',PAPER/'revisions/round2_revisions.tex']:
            text=expand(path)
            text=re.sub(r'(?<!\\)%[^\n]*','',text)
            stack=[]
            for kind,env in re.findall(r'\\(begin|end)\{([^}]+)\}',text):
                if kind=='begin':stack.append(env)
                else:
                    self.assertTrue(stack,env)
                    self.assertEqual(stack.pop(),env)
            self.assertEqual(stack,[])

    def test_main_table_matches_recomputed_means(self):
        text=(PAPER/'tables/model_metrics.tex').read_text()
        rows=read(BASE/'shared_pair_ordering/main_summary.csv')
        self.assertEqual(len(rows),12)
        self.assertEqual(sum(float(r['q_60'])<.05 for r in rows),11)
        for r in rows:
            self.assertIn(f"{float(r['new_ordering']):.4f}",text)
            self.assertIn(f"{float(r['new_average']):.3f}",text)
        manuscript=(PAPER/'acl_latex.tex').read_text()
        self.assertNotIn(r'd_{\mathrm{ord}}(y)',manuscript)
        self.assertIn(r'\label{eq:shared_pair_ordering}',manuscript)

    def test_ablation_has_ten_adjusted_tests(self):
        rows=[r for r in read(BASE/'grounding_ablation/summary.csv') if r['contrast']=='goal_minus_unrestricted_neutral']
        self.assertEqual(len(rows),10)
        self.assertTrue(all(r['q_primary'] for r in rows))
        self.assertEqual(sum(float(r['q_primary'])<.05 for r in rows),8)
        for r in rows:
            self.assertEqual(float(r['q_primary'])<.05,r['metric']!='numerical_loss')

    def test_figure_style_and_unchanged_axes(self):
        audit=json.loads((BASE/'manuscript_revision/figure_audit.json').read_text())
        for k in ['nonordering_pixels_identical','supplied_dumbbell_reproduced_exactly',
                  'category_radar.png_legacy_pixels_identical','all_radars.png_legacy_pixels_identical']:
            self.assertTrue(audit[k],k)

    def test_frozen_human_sample(self):
        h=json.loads((BASE/'manuscript_revision/human_association.json').read_text())
        self.assertEqual(h['n_pairs'],50)
        self.assertEqual(h['counts']['high']['misleading_goal_votes'],28)
        self.assertEqual(h['counts']['low']['misleading_goal_votes'],7)
        self.assertAlmostEqual(h['new_rho'],.5084943198943853)
        self.assertLess(h['permutation_p'],.001)


if __name__=='__main__':unittest.main()
