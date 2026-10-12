"""Normalize grounding-ablation artifacts to the anonymous release schema."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    dest = ROOT / 'anonymous_release/analysis/grounding/results'
    dest.mkdir(parents=True, exist_ok=True)
    for model in ('qwen3_8b_no_thinking', 'llama_3_1_8b_instruct'):
        for suffix in ('', '_matched', '_framing'):
            name = model + suffix + '.jsonl'
            template = ROOT / 'anonymous_release/results' / name
            with template.open() as handle:
                keys = set(json.loads(next(handle)))
            with (ROOT / 'results/arr_grounding_ablation_2026_09' / name).open() as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            with (dest / name).open('w') as handle:
                for row in rows:
                    normalized = {k: v for k, v in row.items() if k in keys}
                    if not suffix and 'hit_token_limit' in row:
                        normalized['hit_token_limit'] = row['hit_token_limit']
                    handle.write(json.dumps(normalized, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
