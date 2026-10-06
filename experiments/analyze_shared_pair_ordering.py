"""Reproduce the revised Ordering results without overwriting published metrics.

Run: python experiments/analyze_shared_pair_ordering.py
Produces main/complete-subset inference, 60-test BH, revised per-model CSVs,
category values, mitigation ordering, and a numerical paper-update report.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ordering_metric import paired_ordering, ORDERING_VERSION
from metrics_pipeline import compute_paired_deltas, build_core_metrics, CORE_METRICS_FIELDS
from experiments.analyze_arr_review_robustness import read_csv, stats, bh

OUT = ROOT / 'analysis/arr_rebuttal_2026_09/shared_pair_ordering'
METRICS = [('selection_gap', 'delta_selection_gap', 1),
           ('emphasis_gap', 'delta_emphasis_gap', 1),
           ('specificity', 'delta_specificity', -1),
           ('framing_raw', 'framing_delta', 1)]


def write(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def group(rows):
    result = defaultdict(list)
    for r in rows:
        result[r['sample_id'], r['condition']].append(r)
    return result


def main():
    published = {r['model']: r for r in read_csv(ROOT/'metrics/radar_charts/model_core_metric_means.csv')}
    cached = {(r['model'], r['sample_id']): r for r in read_csv(ROOT/'analysis/arr_rebuttal_2026_09/scenario_checks.csv')}
    details, summary, tests, complete, categories = [], [], [], [], []
    for model in sorted(published):
        facts = read_csv(ROOT/'metrics'/model/'fact_level_metrics.csv')
        outputs = read_csv(ROOT/'metrics'/model/'output_level_metrics.csv')
        grouped = group(facts)
        ids = sorted({sid for sid, c in grouped})
        model_details = []
        for sid in ids:
            n, g = grouped[sid, 'neutral'], grouped[sid, 'latent_goal']
            result = paired_ordering(n, g)
            old = cached[model, sid]['common_pairs_tie_half']
            value = result['delta_ordering_shift']
            assert (value is None and not old) or (value is not None and abs(value-float(old))<1e-12)
            model_details.append(dict(model=model, sample_id=sid, category=n[0]['category'], **result))
        details.extend(model_details)
        order = dict(model=model, metric='delta_ordering_shift', **stats([r['delta_ordering_shift'] for r in model_details]))
        tests.append(order)
        complete.append(dict(model=model, **stats([r['delta_ordering_shift'] for r in model_details if r['ordering_common_pair_count']==9])))
        by_key = {(r['sample_id'], r['condition']): r for r in outputs}
        metric_means = {}
        for column, metric, sign in METRICS:
            values = []
            for sid in ids:
                n, g = (by_key[sid, c][column] for c in ('neutral','latent_goal'))
                if n and g:
                    values.append(sign*(float(g)-float(n)))
            test = dict(model=model, metric=metric, **stats(values))
            tests.append(test)
            metric_means[metric] = test['mean']
            assert abs(test['mean']-float(published[model][metric]))<1e-9
        old_order = float(published[model]['delta_ordering_shift'])
        old_signed_average = np.mean([old_order, *metric_means.values()])
        new_signed_average = np.mean([order['mean'], *metric_means.values()])
        old_average = np.mean([max(0,v) for v in [old_order, *metric_means.values()]])
        new_average = np.mean([max(0,v) for v in [order['mean'], *metric_means.values()]])
        summary.append(dict(model=model, n=order['n'], excluded=160-order['n'],
            old_ordering=old_order, new_ordering=order['mean'], ci_low=order['ci_low'], ci_high=order['ci_high'],
            change=order['mean']-old_order, relative_change_percent=100*(order['mean']/old_order-1),
            old_average=old_average, new_average=new_average,
            old_signed_average=old_signed_average, new_signed_average=new_signed_average))
        for category in sorted({r['category'] for r in model_details}):
            ds=[r['delta_ordering_shift'] for r in model_details if r['category']==category and r['delta_ordering_shift'] is not None]
            categories.append(dict(model=model, category=category, n=len(ds), ordering=np.mean(ds)))
        # Exercise the production pipeline entry point, not just the helper.
        typed_outputs = []
        for r in outputs:
            t=dict(r)
            for k,v in r.items():
                try: t[k]=float(v) if v else None
                except ValueError: pass
            typed_outputs.append(t)
        annotated_facts = [dict(r, model=by_key[r['sample_id'],r['condition']]['model'],
                                language=by_key[r['sample_id'],r['condition']].get('language','')) for r in facts]
        paired = compute_paired_deltas(typed_outputs, fact_rows=annotated_facts)
        core = build_core_metrics(paired)
        actual={r['sample_id']:r['delta_ordering_shift'] for r in core}
        assert len(actual)==160
        for r in model_details:
            assert actual[r['sample_id']]==r['delta_ordering_shift']
        dest=OUT/'metrics'/model
        write(dest/'core_metrics.csv',core,CORE_METRICS_FIELDS)
        write(dest/'fact_level_metrics.csv',facts)
        write(dest/'output_level_metrics.csv',outputs)
        write(dest/'paired_condition_deltas.csv',paired)
    assert len(tests)==60
    bh(tests)
    for r in tests:
        r['q_60']=r.pop('q_within_variant')
    for r in summary:
        r['q_60']=next(t['q_60'] for t in tests if t['model']==r['model'] and t['metric']=='delta_ordering_shift')
    bh(complete)
    write(OUT/'main_summary.csv',summary)
    write(OUT/'significance_60_tests.csv',tests)
    write(OUT/'scenario_scores.csv',details)
    write(OUT/'complete_both_summary.csv',complete)
    write(OUT/'category_ordering.csv',categories)
    mitigation=[]
    for model in ('qwen3_8b_no_thinking','llama_3_1_8b_instruct'):
        grouped=group(read_csv(ROOT/'metrics'/f'{model}_self_awareness'/'fact_level_metrics.csv'))
        for reference,target in [('neutral','latent_goal'),('neutral','latent_goal_self_awareness'),('latent_goal','latent_goal_self_awareness')]:
            values=[]
            for sid in sorted({sid for sid,c in grouped}):
                n,g=grouped.get((sid,reference)),grouped.get((sid,target))
                if n is not None and g is not None:
                    values.append(paired_ordering(n,g)['delta_ordering_shift'])
            mitigation.append(dict(model=model,reference=reference,target=target,**stats(values)))
    write(OUT/'mitigation_ordering.csv',mitigation)
    lines=['# Shared-pair Ordering results', '', f'Metric: `{ORDERING_VERSION}`. Equal scenario weights. Original metric files remain unchanged.', '',
           '20,000 scenario bootstrap draws; 200,000 two-sided sign flips; main BH family = 12 models × 5 metrics. Complete-both is a separate 12-test sensitivity family.', '',
           '| Model | n | Paper Ordering → revised | 95% CI | q (60) | Paper Average → revised |',
           '|---|---:|---:|---|---:|---:|']
    for r in summary:
        lines.append(f"| {r['model']} | {r['n']} | {r['old_ordering']:.4f} → {r['new_ordering']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['q_60']:.6f} | {r['old_average']:.3f} → {r['new_average']:.3f} |")
    lines+=['', 'Excluded model/scenarios: '+', '.join(f"{r['model']}/{r['sample_id']}" for r in details if r['delta_ordering_shift'] is None), '',
            'Displayed Average reproduces the paper table: mean of positive parts of five raw dimension means. This differs from the paper prose (unweighted signed mean). CSV also provides signed averages. Neither is the plotting script’s normalized composite.', '',
            'Mitigation values are independently paired contrasts; do not subtract them to reconstruct other contrasts because shared-pair sets can differ.']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
