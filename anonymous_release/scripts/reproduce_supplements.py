"""Recompute the packaged mitigation and valence summaries, without model calls."""
import sys
import contextlib
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from metrics_pipeline import run_pipeline
from analyze_mitigation import analyze
from analyze_valence_sensitivity import analyze as analyze_valence
from analyze_grounding_ablation import analyze as analyze_grounding
from scripts.analyze_metric_checks import main as analyze_checks


def main():
    runs = []
    for model, label in [('qwen3_8b_no_thinking', 'Qwen3-8B'),
                         ('llama_3_1_8b_instruct', 'Llama-3.1-8B')]:
        stem = model + '_self_awareness'
        source = ROOT / 'analysis/mitigation/results'
        paths = []
        for suffix, reference in [('_direct', 'latent_goal'), ('', 'neutral')]:
            dest = ROOT / 'analysis/mitigation/metrics' / (stem + suffix)
            with contextlib.redirect_stdout(io.StringIO()):
                run_pipeline(raw_jsonl=source / (stem + '.jsonl'),
                         matched_jsonl=source / (stem + '_matched.jsonl'),
                         framing_jsonl=source / (stem + '_framing.jsonl'),
                         output_dir=dest, reference_condition=reference,
                         target_condition='latent_goal_self_awareness', test_sample_id='')
            paths.append(str(dest / 'core_metrics.csv'))
        runs.append([label, *paths])
    analyze(runs, ROOT / 'analysis/mitigation/outputs', 160)
    analyze_valence(ROOT / 'analysis/valence/meta_llama_llama_3_1_70b_instruct.jsonl',
                    ROOT / 'metrics', ROOT / 'analysis/valence')
    for model in ('qwen3_8b_no_thinking', 'llama_3_1_8b_instruct'):
        source = ROOT / 'analysis/grounding/results'
        with contextlib.redirect_stdout(io.StringIO()):
            run_pipeline(raw_jsonl=source / (model + '.jsonl'),
                         matched_jsonl=source / (model + '_matched.jsonl'),
                         framing_jsonl=source / (model + '_framing.jsonl'),
                         output_dir=ROOT / 'analysis/grounding/metrics' / model,
                         reference_condition='neutral_no_grounding',
                         target_condition='latent_goal', test_sample_id='')
    analyze_grounding()
    analyze_checks()


if __name__ == '__main__':
    main()
