"""Generate manuscript tables/figures and audit the frozen human sample.

Run the shared-pair and grounding analyses first. Historical result caches and
the human sample/annotations are never modified. Only generated paper assets
and analysis/arr_rebuttal_2026_09/manuscript_revision outputs are written.
"""
import csv
import json
import sys
import shutil
import importlib.util
import copy
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('janus_charts', ROOT/'scripts/make_model_radar_charts.py')
charts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(charts)
from experiments.analyze_shared_pair_ordering import read_csv, write
from experiments.persuasiveness_review import distortion_score, NATURAL_SCALES

BASE = ROOT/'analysis/arr_rebuttal_2026_09'
DATA = BASE/'shared_pair_ordering'
OUT = BASE/'manuscript_revision'
TABLES = ROOT/'paper/tables'
MODELS = charts.sorted_models({r['model']: None for r in read_csv(DATA/'main_summary.csv')})
NAMES = dict(zip(MODELS, ['GPT-5.4','GPT-5.4 Mini','DeepSeek v4','DeepSeek v4 (Thinking)',
    'Llama 3.1 70B','Llama 3.1 8B','Qwen3 14B','Qwen3 14B (Thinking)',
    'Qwen3 32B','Qwen3 32B (Thinking)','Qwen3 8B','Qwen3 8B (Thinking)']))


def table(name, spec, head, rows, caption, label, wide=True):
    env='table*' if wide else 'table'
    size=r'\scriptsize' if name=='ordering_robustness.tex' else r'\small'
    text=[f'\\begin{{{env}}}[t]',r'\centering',size,r'\setlength{\tabcolsep}{3pt}',
          f'\\begin{{tabular}}{{{spec}}}',r'\toprule',head+r' \\',r'\midrule',
          *[r+r' \\' for r in rows],r'\bottomrule',r'\end{tabular}',
          f'\\caption{{{caption}}}',f'\\label{{{label}}}',f'\\end{{{env}}}']
    (TABLES/name).write_text('\n'.join(text)+'\n')


def generate_tables():
    summary={r['model']:r for r in read_csv(DATA/'main_summary.csv')}
    tests={(r['model'],r['metric']):r for r in read_csv(DATA/'significance_60_tests.csv')}
    metrics=[m for m,_ in charts.METRICS]
    values={m:[float(tests[m,k]['mean']) for k in metrics]+[float(summary[m]['new_average'])] for m in MODELS}
    best=[sorted({v[i] for v in values.values()},reverse=True)[:2] for i in range(6)]
    rows=[]
    for m in MODELS:
        cells=[]
        for i,v in enumerate(values[m]):
            val=f'{v:.4f}' if i<5 else f'{v:.3f}'
            if v==best[i][0]: val=r'\textbf{'+val+'}'
            elif v==best[i][1]: val=r'\underline{'+val+'}'
            if i<5:
                q=float(tests[m,metrics[i]]['q_60'])
                if q<.05:
                    shade=round(10+35*min(1,(.05-q)/.049))
                    val=f'\\cellcolor{{janusTeal!{shade}}}'+val
            cells.append(val)
        rows.append(NAMES[m]+' & '+' & '.join(cells))
    table('model_metrics.tex','lcccccc',r'Models & Selection & Emphasis & Ordering & \makecell{Numerical\\specificity loss} & Framing & Average',rows,
          r'Mean paired JANUS effects. Numerical specificity loss is a nondirectional fidelity diagnostic. Average is the mean of the positive parts of the five raw dimension means, retained as a descriptive summary rather than a calibrated misleadingness score. Bold and underline mark the largest and second-largest values. Shading indicates $q<.05$ across 60 tests (Appendix~\ref{app:stat_sig}).',
          'tab:janus-model-core-metrics-absolute')
    coverage={(r['model'],r['condition']):r for r in read_csv(BASE/'coverage_by_model.csv')}
    rows=[]
    for m in MODELS:
        n,g=(coverage[m,c] for c in ('neutral','latent_goal'))
        rows.append(NAMES[m]+' & '+' & '.join(f'{100*float(r[k]):.2f}' for k in ('favorable_coverage','adverse_coverage') for r in (n,g))+' & '+n['complete_outputs']+' & '+g['complete_outputs'])
    rows.append(r'All models & 99.31 & 98.85 & 99.60 & 98.52 & 1,858 & 1,783')
    table('absolute_coverage.tex','lrrrrrr',r'Model & $F^+$ N (\%) & $F^+$ G (\%) & $F^-$ N (\%) & $F^-$ G (\%) & All six N & All six G',rows,
          r'Absolute inclusion of supplied facts from sentence--fact matches. Each model has 160 responses per condition and 480 supplied facts of each polarity per condition. Aggregate fact percentages use 5,760 facts per polarity and condition; all-six counts use 1,920 responses per condition.', 'tab:absolute_coverage')
    complete={r['model']:r for r in read_csv(DATA/'complete_both_summary.csv')}
    rows=[]
    for m in MODELS:
        r,c=summary[m],complete[m]
        interval=lambda a:f"[{float(a['ci_low']):.4f}, {float(a['ci_high']):.4f}]"
        rows.append(f"{NAMES[m]} & {r['n']} & {float(r['new_ordering']):.4f} & {interval(r)} & {c['n']} & {float(c['mean']):.4f} & {interval(c)}")
    table('ordering_robustness.tex','lrrlrrl',r'Model & $n$ & Shared-pair & 95\% CI & $n_6$ & All-six & 95\% CI',rows,
          r'Shared-pair ordering and the subset retaining all six facts in both responses. Intervals are individual 95\% percentile bootstrap intervals from 20,000 scenario resamples. Main effects use the 60-test adjustment; the all-six check uses a separate 12-test adjustment. Eleven models are significant in each analysis; GPT-5.4 Mini is the exception.', 'tab:ordering_robustness')
    ablation=read_csv(BASE/'grounding_ablation/summary.csv')
    lookup={(r['model'],r['contrast'],r['metric']):r for r in ablation}
    short={'qwen3_8b_no_thinking':'Qwen3-8B','llama_3_1_8b_instruct':'Llama-3.1-8B'}
    labels={'selection':'Selection','emphasis':'Emphasis','ordering':'Ordering','numerical_loss':'Numerical specificity loss','framing':'Framing'}
    rows=[]
    for m in short:
        for metric in labels:
            old=lookup[m,'original_goal_minus_neutral',metric]
            r=lookup[m,'goal_minus_unrestricted_neutral',metric]
            q=float(r['q_primary']);qtext=r'$<0.0001$' if q<.0001 else f'{q:.4f}'
            rows.append(f"{short[m]} & {labels[metric]} & {float(old['mean']):.4f} & {float(r['mean']):.4f} & [{float(r['ci_low']):.4f}, {float(r['ci_high']):.4f}] & {r['n']} & {qtext}")
    table('grounding_ablation.tex','llrrlrr',r'Model & Dimension & Original & Sentence removed & 95\% CI (removed) & $n$ & BH $q$',rows,
          r'Grounding ablation: goal-minus-neutral effects before and after deleting only the neutral grounding sentence. Both ordering columns use the shared-pair metric. The original goal responses are reused. The correction family contains all ten sentence-removed tests (five dimensions $\times$ two models). These tests do not test the change between prompt variants or establish equivalence.', 'tab:grounding_ablation')


def figures():
    """Use the existing plotting functions; change only ordering input data."""
    original=ROOT/'metrics'
    means={m:r for m,r in charts.load_model_means(original).items() if m in MODELS}
    categories={m:r for m,r in charts.load_category_means(original).items() if m in MODELS}
    # Archived category inputs are retained for ALL other dimensions.
    charts.apply_pairwise_ordering_metrics(original,means,categories)
    arms={m:r for m,r in charts.load_condition_absolute_means(original).items() if m in MODELS}
    old_arms={m:{c:dict(v) for c,v in r.items()} for m,r in arms.items()}
    for m in MODELS:
        by_c=defaultdict(list)
        for (sid,c),rows in charts.grouped_fact_rows(original/m/'fact_level_metrics.csv').items():
            by_c[c].append(1-charts.adverse_first_pair_rate(rows))
        for c in ('neutral','latent_goal'):
            old_arms[m][c]['ordering_bias']=sum(by_c[c])/len(by_c[c])
    old_categories=copy.deepcopy(categories)
    for m in MODELS:
        grouped=charts.grouped_fact_rows(original/m/'fact_level_metrics.csv')
        ds=defaultdict(list)
        for sid in sorted({sid for sid,c in grouped}):
            n,g=grouped[sid,'neutral'],grouped[sid,'latent_goal']
            ds[n[0]['category']].append(charts.adverse_first_pair_rate(n)-charts.adverse_first_pair_rate(g))
        for c,v in ds.items():old_categories[m][c]['delta_ordering_shift']=sum(v)/len(v)
    # Preserve exact legacy style; do not relabel any other axes.
    dest=ROOT/'paper/images'
    archive=OUT/'original_figures'
    archive.mkdir(exist_ok=True)
    for name in ('category_radar.png','image.png','all_radars.png'):
        if not (archive/name).exists():
            shutil.copy2(dest/name,archive/name)
    charts.save_category_composite_radar_chart(dest/'category_radar.png',category_means=categories,scales=charts.metric_scales(means.values()))
    charts.save_condition_metric_dumbbell_chart(dest/'image.png',condition_means=arms)
    charts.save_condition_overlay_radar_grid(dest/'all_radars.png',condition_means=arms,axis_normalized=True)
    # Test non-ordering image columns against a same-style legacy rendering.
    baseline=OUT/'legacy_dumbbell_reproduction.png'
    charts.save_condition_metric_dumbbell_chart(baseline,condition_means=old_arms)
    a,b=np.asarray(Image.open(baseline)),np.asarray(Image.open(dest/'image.png'))
    assert a.shape==b.shape
    # Ordering lane occupies x=612..765 in the fixed 1116px rendering.
    mask=np.any(a!=b,axis=2)
    xs=np.where(mask)[1]
    audit={'changed_pixel_x_min':int(xs.min()),'changed_pixel_x_max':int(xs.max()),
           'nonordering_pixels_identical':bool(not mask[:,:612].any() and not mask[:,765:].any()),
           'note':'Comparison to same-style legacy rendering; only Ordering lane should change.'}
    assert audit['nonordering_pixels_identical'],audit
    audit['supplied_dumbbell_reproduced_exactly']=bool(np.array_equal(
        np.asarray(Image.open(archive/'image.png').convert('RGB')),
        np.asarray(Image.open(baseline).convert('RGB'))))
    charts.save_category_composite_radar_chart(OUT/'legacy_category_reproduction.png',category_means=old_categories,scales=charts.metric_scales(means.values()))
    charts.save_condition_overlay_radar_grid(OUT/'legacy_radars_reproduction.png',condition_means=old_arms,axis_normalized=True)
    for source,render in [('category_radar.png','legacy_category_reproduction.png'),('all_radars.png','legacy_radars_reproduction.png')]:
        audit[source+'_legacy_pixels_identical']=bool(np.array_equal(
            np.asarray(Image.open(archive/source).convert('RGB')),
            np.asarray(Image.open(OUT/render).convert('RGB'))))
    (OUT/'figure_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    charts.write_category_composite_summary(OUT/'category_composite_scores.csv',category_means=categories,scales=charts.metric_scales(means.values()))


def human():
    annotations=read_csv(ROOT/'analysis/human_ratings_minimal/derived_pairwise_annotations.csv')
    grouped=defaultdict(list)
    for r in annotations:grouped[r['pair_id']].append(r)
    assert len(grouped)==50 and len(annotations)==100
    old_cores={m:{r['sample_id']:r for r in read_csv(ROOT/'metrics'/m/'core_metrics.csv')} for m in MODELS}
    new_orders={(r['model'],r['sample_id']):r['delta_ordering_shift'] for r in read_csv(DATA/'scenario_scores.csv')}
    rows=[]
    for pid,rs in sorted(grouped.items()):
        r=rs[0];m,sid=r['model'],r['sample_id'];core=old_cores[m][sid]
        assert abs(distortion_score(core)-float(r['distortion_score']))<1e-8
        assert new_orders[m,sid]!='', (m,sid)
        new=dict(core,delta_ordering_shift=new_orders[m,sid])
        rows.append(dict(pair_id=pid,model=m,sample_id=sid,stratum=r['stratum'],
            old_score=float(r['distortion_score']),new_score=distortion_score(new),
            misleading_goal_votes=sum(int(a['misleading_goal_conditioned_choice'])==1 for a in rs),
            support_goal_votes=sum(int(a['support_goal_conditioned_choice'])==1 for a in rs)))
    votes=np.array([r['misleading_goal_votes'] for r in rows])
    scores=np.array([r['new_score'] for r in rows])
    oldrho=float(spearmanr([r['old_score'] for r in rows],votes).statistic)
    rho=float(spearmanr(scores,votes).statistic)
    x=rankdata(scores);x-=x.mean();y=rankdata(votes);y-=y.mean()
    den=np.linalg.norm(x)*np.linalg.norm(y);rng=np.random.default_rng(160);exceed=0
    for _ in range(200):
        perms=np.array([rng.permutation(y) for _ in range(1000)])
        exceed+=int((np.abs(perms@x/den)>=abs(rho)-1e-12).sum())
    p=(exceed+1)/200001
    write(OUT/'human_rescored_pairs.csv',rows)
    stats={'n_pairs':50,'old_rho':oldrho,'new_rho':rho,'permutation_p':p,'draws':200000,
           'unit':'one response pair, with its two annotator votes aggregated',
           'score':'historical clipped positive-part normalized five-dimension composite; replace only Ordering',
           'selection':'frozen original high/low quartile-stratified sample; no reselection',
           'counts':{s:{k:sum(r[k] for r in rows if r['stratum']==s) for k in ('support_goal_votes','misleading_goal_votes')} for s in ('high','low')}}
    (OUT/'human_association.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(stats)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);TABLES.mkdir(parents=True,exist_ok=True)
    generate_tables();figures();human()
