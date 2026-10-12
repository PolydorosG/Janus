"""Paired ordering over shared fact pairs; all first mentions, ties = 1/2.

Inputs are fact_level_metrics rows (CSV strings or in-memory lists).
All matched sentences are used, including mixed-polarity sentences.
Scores are scenario-level; aggregate nonempty scenarios with equal weight.
"""
import json
from statistics import mean

ORDERING_VERSION = "shared_pairs_all_mentions_ties_half_v1"


def observed_pair_scores(rows):
    positions = {}
    polarities = {}
    for row in rows:
        fid = row["fact_id"]
        if fid in polarities:
            raise ValueError(f"Duplicate fact: {fid}")
        polarities[fid] = row["polarity"]
        ids = row["matched_unit_ids"]
        if isinstance(ids, str):
            ids = json.loads(ids)
        if ids:
            positions[fid] = min(ids)
    return {
        (f, a): 1.0 if positions[f] < positions[a] else
        0.5 if positions[f] == positions[a] else 0.0
        for f in positions if polarities[f] == "F"
        for a in positions if polarities[a] == "A"
    }


def paired_ordering(reference_rows, target_rows):
    """Return target-reference effect, shared-pair count and paired arm scores.

    None means no shared comparable pair, never zero or perfect ordering.
    Arm scores depend on BOTH responses and are not standalone output metrics.
    """
    reference_rows, target_rows = list(reference_rows), list(target_rows)
    rp = {r["fact_id"]: r["polarity"] for r in reference_rows}
    tp = {r["fact_id"]: r["polarity"] for r in target_rows}
    if rp != tp:
        raise ValueError("Paired outputs must use identical fact IDs and polarities")
    ref, target = observed_pair_scores(reference_rows), observed_pair_scores(target_rows)
    common = sorted(ref.keys() & target.keys())
    if not common:
        return dict(delta_ordering_shift=None, ordering_common_pair_count=0,
                    ordering_reference_shared=None, ordering_target_shared=None)
    return dict(
        delta_ordering_shift=mean(target[p] - ref[p] for p in common),
        ordering_common_pair_count=len(common),
        ordering_reference_shared=mean(ref[p] for p in common),
        ordering_target_shared=mean(target[p] for p in common),
    )
