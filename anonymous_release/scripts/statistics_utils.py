"""Scenario-level bootstrap/sign-flip statistics and BH adjustment."""
import numpy as np
from scripts.analyze_significance import sign_flip_p_value, benjamini_hochberg


def stats(values, seed=160):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return dict(n=0, mean=None, ci_low=None, ci_high=None, p=None)
    p, low, high = sign_flip_p_value(values, draws=200000, seed=seed)
    return dict(n=len(values), mean=float(values.mean()), ci_low=low, ci_high=high, p=p)


def bh(rows):
    for row, q in zip(rows, benjamini_hochberg([r['p'] for r in rows])):
        row['q_within_variant'] = q
