"""Delete only the neutral grounding sentence; reuse original goal outputs/judgments."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from janus_payloads import PresentedFact, load_samples, render_user_content
from judge_fact_units import append_jsonl, output_file_lock

MODELS = {
    "qwen3_8b_no_thinking": ("Qwen/Qwen3-8B", "qwen_local"),
    "llama_3_1_8b_instruct": ("meta-llama/Llama-3.1-8B-Instruct", "llama_local"),
}
DEFAULT_OUTPUT = ROOT / "results/arr_grounding_ablation_2026_09"
GROUNDING_SENTENCE = "Use only the provided facts and do not add outside information."


def jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_manifest(model_key, dataset=ROOT / "data/janus.csv"):
    samples = load_samples(dataset)
    reference = ROOT / "results" / f"{model_key}.jsonl"
    raw = jsonl(reference)
    archived = {(r["sample_id"], r["condition"]): r for r in raw if "response_text" in r}
    if len(samples) != 160 or len(archived) != 320:
        raise ValueError("Expected the complete 160-scenario primary reference")
    payloads = []
    for index, sample in enumerate(samples):
        n, g = (archived[sample.sample_id, c] for c in ("neutral", "latent_goal"))
        if n["presented_facts"] != g["presented_facts"] or n["scenario"] != g["scenario"]:
            raise ValueError(f"Unpaired archived evidence: {sample.sample_id}")
        if g["system_prompt"] != sample.latent_goal_prompt or sample.scenario != g["scenario"]:
            raise ValueError(f"Dataset differs from archive: {sample.sample_id}")
        if n["system_prompt"] != sample.neutral_prompt or sample.neutral_prompt.count(GROUNDING_SENTENCE) != 1:
            raise ValueError(f"Unexpected original neutral prompt: {sample.sample_id}")
        neutral = sample.neutral_prompt.replace(GROUNDING_SENTENCE, "").strip()
        facts = [PresentedFact(**f) for f in n["presented_facts"]]
        expected = {f"favorable_fact_{i}": t for i, t in enumerate(sample.favorable_facts, 1)}
        expected.update({f"adverse_material_fact_{i}": t for i, t in enumerate(sample.adverse_material_facts, 1)})
        if {f.fact_id: f.text for f in facts} != expected:
            raise ValueError("Dataset facts differ from archive")
        user = render_user_content(sample, facts)
        conditions = ("neutral_no_grounding", "latent_goal")
        for condition in conditions:
            payloads.append({
                "messages": [{"role": "system", "content": neutral if condition == "neutral_no_grounding" else sample.latent_goal_prompt}, {"role": "user", "content": user}],
                "metadata": {"sample_id": sample.sample_id, "category": sample.category,
                    "item": sample.item, "condition": condition, "presented_fact_order": n["presented_facts"]},
            })
    return {"experiment": "arr_grounding_ablation_2026_09", "model_key": model_key,
        "model_id": MODELS[model_key][0], "dataset_sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "design": "delete only original neutral grounding sentence; reuse original latent_goal; role/audience remain different",
        "payloads": payloads}


def validate_existing(rows, payloads, run_id):
    expected = {(p["metadata"]["sample_id"], p["metadata"]["condition"]): p for p in payloads}
    completed = set()
    for row in rows:
        key = row["sample_id"], row["condition"]
        if key in completed or key not in expected or row.get("run_id") != run_id:
            raise ValueError("Duplicate, unexpected, or incompatible resumed output")
        if row.get("messages") != expected[key]["messages"] or not row.get("response_text", "").strip():
            raise ValueError("Prompt mismatch or empty resumed output")
        completed.add(key)
    return completed


def copy_goal_artifacts(model_key, path, manifest, completed):
    """Copy exact historical text and cached judgments, recording source paths."""
    from llm_clients import record_input_fields
    source = ROOT / "results" / f"{model_key}.jsonl"
    goal = {r["sample_id"]: r for r in jsonl(source) if r["condition"] == "latent_goal"}
    for payload in manifest["payloads"]:
        meta = payload["metadata"]
        key = meta["sample_id"], meta["condition"]
        if meta["condition"] != "latent_goal" or key in completed:
            continue
        old = goal[meta["sample_id"]]
        append_jsonl(path, {**old, **record_input_fields(payload), "messages": payload["messages"],
            "run_id": manifest["run_id"], "generation_origin": "archived_goal",
            "reference_source": str(source)})
        completed.add(key)
    for suffix in ("_matched", "_framing"):
        source = ROOT / "results" / f"{model_key}{suffix}.jsonl"
        target = path.with_name(path.stem + suffix + ".jsonl")
        def key(r):
            return (r["sample_id"], r["condition"], r.get("unit_id"), r.get("fact_id"))
        with output_file_lock(target):
            existing = {key(r) for r in jsonl(target)} if target.exists() else set()
            for row in jsonl(source):
                if row["condition"] == "latent_goal" and key(row) not in existing:
                    append_jsonl(target, {**row, "reference_source": str(source)})
                    existing.add(key(row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=MODELS, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    manifest = build_manifest(args.model_key)
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(manifest["model_id"], local_files_only=True)
    manifest["settings"] = {"snapshot": snapshot, "batch_size": args.batch_size,
        "max_new_tokens": args.max_tokens, "temperature_override": None,
        "decoding": "checkpoint generation defaults, matching original launch scripts",
        "dtype": "bfloat16", "qwen_thinking": False, "seed": 20260916,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}}
    manifest["checkpoint_generation_config"] = json.loads((Path(snapshot) / "generation_config.json").read_text())
    manifest["reference_judgment_hashes"] = {suffix: hashlib.sha256((ROOT / "results" / f"{args.model_key}_{suffix}.jsonl").read_bytes()).hexdigest() for suffix in ("matched", "framing")}
    manifest["run_id"] = digest(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{args.model_key}.jsonl"
    manifest_path = path.with_suffix(".manifest.json")
    with output_file_lock(path):
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Existing manifest differs; use a new output directory")
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        completed = validate_existing(jsonl(path) if path.exists() else [], manifest["payloads"], manifest["run_id"])
        if args.dry_run:
            print(f"Validated 160 new neutral-without-grounding requests + 160 archived goal outputs; {len(completed)} staged/complete; manifest: {manifest_path}")
            return
        import torch
        from transformers import set_seed
        from llm_clients import build_provider_request, call_local_transformers_batch, record_input_fields, _load_local_model
        if not torch.cuda.is_available():
            raise RuntimeError("This experiment must run on a GPU allocation")
        set_seed(manifest["settings"]["seed"])
        copy_goal_artifacts(args.model_key, path, manifest, completed)
        tokenizer, model = _load_local_model(snapshot, device_map="auto", dtype="bfloat16", trust_remote_code=False, load_in_4bit=False, load_in_8bit=False)
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        runtime = {"gpu": torch.cuda.get_device_name(), "generation_config": model.generation_config.to_dict(),
            "tokenizer_class": type(tokenizer).__name__, "model_class": type(model).__name__}
        new_payloads = [p for p in manifest["payloads"] if p["metadata"]["condition"] == "neutral_no_grounding"]
        for start in range(0, len(new_payloads), args.batch_size):
            batch = new_payloads[start:start + args.batch_size]
            if all((p["metadata"]["sample_id"], p["metadata"]["condition"]) in completed for p in batch):
                continue
            # Regenerate a partially saved batch intact, retaining identical batch composition.
            # Per-batch seeds keep sampling reproducible across resumptions.
            set_seed(manifest["settings"]["seed"] + start)
            requests = [build_provider_request(p, MODELS[args.model_key][1], model=snapshot, max_tokens=args.max_tokens, temperature=None) for p in batch]
            for request in requests:
                if args.model_key.startswith("qwen"):
                    request["chat_template_kwargs"] = {"enable_thinking": False}
            responses = call_local_transformers_batch(requests, dtype="bfloat16")
            if len(responses) != len(batch) or any(not r.get("text", "").strip() for r in responses):
                raise RuntimeError("Incomplete or empty generation batch")
            for payload, response in zip(batch, responses):
                key = payload["metadata"]["sample_id"], payload["metadata"]["condition"]
                if key in completed:
                    continue
                ids = response["generated_token_ids"]
                stop = next((i for i, token in enumerate(ids) if token in eos), None)
                append_jsonl(path, {**record_input_fields(payload), "model": manifest["model_id"],
                    "messages": payload["messages"], "response_text": response["text"],
                    "generation_origin": "new_neutral_no_grounding",
                    "run_id": manifest["run_id"], "runtime": runtime,
                    "generated_token_count": len(ids) if stop is None else stop + 1,
                    "hit_token_limit": stop is None and len(ids) >= args.max_tokens})
                completed.add(key)
            print(f"{args.model_key}: {len(completed)}/320", flush=True)
        if len(validate_existing(jsonl(path), manifest["payloads"], manifest["run_id"])) != 320:
            raise RuntimeError("Incomplete generation set")


if __name__ == "__main__":
    main()
