"""Absolute inclusion and all-six-facts Ordering robustness from released metrics."""
import csv
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ordering_metric import paired_ordering
from scripts.analyze_significance import MODELS, read_csv
from scripts.statistics_utils import stats, bh


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    coverage, complete = [], []
    for model in MODELS:
        outputs = read_csv(ROOT / 'metrics' / model / 'output_level_metrics.csv')
        for condition in ('neutral', 'latent_goal'):
            rows = [r for r in outputs if r['condition'] == condition]
            coverage.append(dict(model=model, condition=condition, n=len(rows),
                favorable_coverage=sum(float(r['favorable_inclusion_rate']) for r in rows)/len(rows),
                adverse_coverage=sum(float(r['adverse_inclusion_rate']) for r in rows)/len(rows),
                complete_outputs=sum(int(r['included_favorable_facts']) + int(r['included_adverse_facts']) == 6 for r in rows)))
        grouped = defaultdict(list)
        for row in read_csv(ROOT / 'metrics' / model / 'fact_level_metrics.csv'):
            grouped[row['sample_id'], row['condition']].append(row)
        values = []
        for sid in sorted({sid for sid, condition in grouped}):
            result = paired_ordering(grouped[sid, 'neutral'], grouped[sid, 'latent_goal'])
            if result['ordering_common_pair_count'] == 9:
                values.append(result['delta_ordering_shift'])
        complete.append(dict(model=model, **stats(values)))
    bh(complete)
    write(ROOT / 'analysis/metric_checks/absolute_coverage.csv', coverage)
    write(ROOT / 'analysis/metric_checks/complete_both_ordering.csv', complete)
    print('Wrote absolute inclusion and all-six-facts Ordering results (BH across 12 tests).')


if __name__ == '__main__':
    main()
