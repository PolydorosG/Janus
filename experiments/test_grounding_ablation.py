"""Offline checks for the requested deletion, pairing, reuse, and resume safety."""
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_grounding_ablation import (
    ROOT, MODELS, GROUNDING_SENTENCE, build_manifest, copy_goal_artifacts,
    digest, jsonl, validate_existing,
)


class GroundingAblationTests(unittest.TestCase):
    def test_only_requested_sentence_changes_all_scenarios(self):
        for model in MODELS:
            manifest = build_manifest(model)
            old = {(r["sample_id"], r["condition"]): r for r in jsonl(ROOT / "results" / f"{model}.jsonl")}
            self.assertEqual(len(manifest["payloads"]), 320)
            for offset in range(0, 320, 2):
                neutral, goal = manifest["payloads"][offset:offset + 2]
                sid = neutral["metadata"]["sample_id"]
                self.assertEqual(neutral["messages"][0]["content"], old[sid, "neutral"]["system_prompt"].replace(GROUNDING_SENTENCE, "").strip())
                self.assertEqual(goal["messages"][0]["content"], old[sid, "latent_goal"]["system_prompt"])
                self.assertEqual(neutral["messages"][1], goal["messages"][1])
                self.assertEqual(neutral["metadata"]["presented_fact_order"], old[sid, "neutral"]["presented_facts"])

    def test_exact_reuse_and_idempotence(self):
        model = "qwen3_8b_no_thinking"
        manifest = build_manifest(model)
        manifest["run_id"] = digest(manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{model}.jsonl"
            complete = set()
            copy_goal_artifacts(model, path, manifest, complete)
            self.assertEqual(len(complete), 160)
            sizes = {p.name: p.stat().st_size for p in Path(directory).glob("*.jsonl")}
            copy_goal_artifacts(model, path, manifest, complete)
            self.assertEqual(sizes, {p.name: p.stat().st_size for p in Path(directory).glob("*.jsonl")})
            rows = jsonl(path)
            self.assertEqual(validate_existing(rows, manifest["payloads"], manifest["run_id"]), complete)
            old = {r["sample_id"]: r for r in jsonl(ROOT / "results" / f"{model}.jsonl") if r["condition"] == "latent_goal"}
            for row in rows:
                self.assertEqual(row["response_text"], old[row["sample_id"]]["response_text"])
            with self.assertRaises(ValueError):
                validate_existing(rows + rows[:1], manifest["payloads"], manifest["run_id"])
            with self.assertRaises(ValueError):
                validate_existing(rows, manifest["payloads"], "different-run")
            rows[0]["messages"][0]["content"] += " extra instruction"
            with self.assertRaises(ValueError):
                validate_existing(rows, manifest["payloads"], manifest["run_id"])

    def test_full_judgment_validation_rejects_missing_score(self):
        from experiments import analyze_grounding_ablation as analysis
        from llm_clients import record_input_fields
        model = "qwen3_8b_no_thinking"
        manifest = build_manifest(model)
        manifest["run_id"] = digest(manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / f"{model}.jsonl"
            path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
            copy_goal_artifacts(model, path, manifest, set())
            archived = {r["sample_id"]: r for r in jsonl(ROOT / "results" / f"{model}.jsonl") if r["condition"] == "neutral"}
            rows = jsonl(path)
            # Historical neutral text is a synthetic fixture, never experiment data.
            for payload in manifest["payloads"]:
                if payload["metadata"]["condition"] == "neutral_no_grounding":
                    rows.append({**archived[payload["metadata"]["sample_id"]], **record_input_fields(payload),
                        "messages": payload["messages"], "run_id": manifest["run_id"]})
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            for suffix in ("_matched", "_framing"):
                target = path.with_name(path.stem + suffix + ".jsonl")
                rows = jsonl(target)
                rows += [{**r, "condition": "neutral_no_grounding"} for r in jsonl(ROOT / "results" / f"{model}{suffix}.jsonl") if r["condition"] == "neutral"]
                target.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with patch.object(analysis, "DEFAULT_OUTPUT", Path(directory)):
                analysis.validate(model, "frame")
                target.write_text("".join(json.dumps(r) + "\n" for r in rows[:-1]))
                with self.assertRaises(ValueError):
                    analysis.validate(model, "frame")


if __name__ == "__main__":
    unittest.main()
