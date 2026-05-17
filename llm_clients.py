"""Provider adapters and callers for Janus LLM runs.

Optional dependencies:
  - OpenAI and DeepSeek API calls: pip install openai
  - Claude API calls: pip install anthropic
  - Local Qwen/Llama calls: pip install transformers torch accelerate
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from janus_payloads import Condition, build_payload, load_samples


Provider = Literal["openai", "deepseek", "claude", "qwen_local", "llama_local"]

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OPENAI_BATCH_ENDPOINT = "/v1/responses"
OPENAI_BATCH_COMPLETION_WINDOW = "24h"
DEFAULT_RESULTS_PATH = Path("results") / "janus_results.jsonl"
DEFAULT_FACT_SHUFFLE_SEED = 1234
DEFAULT_DEEPSEEK_TIMEOUT_SECONDS = 120.0
DEFAULT_DEEPSEEK_MAX_RETRIES = 0


@dataclass(frozen=True)
class LLMResult:
    """Normalized text plus provider-specific details."""

    provider: Provider
    model: str
    text: str
    request: dict[str, Any]
    metadata: dict[str, Any]
    raw_response: Any
    reasoning_text: str | None = None
    token_usage: dict[str, Any] | None = None


def build_provider_request(
    payload: dict[str, Any],
    provider: Provider,
    *,
    model: str,
    max_tokens: int | None = 512,
    temperature: float | None = None,
    openai_reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Convert a Janus payload into the request shape a provider expects."""

    messages = _payload_messages(payload)
    system_prompt, conversation = _split_system_message(messages)

    if provider == "openai":
        request: dict[str, Any] = {
            "model": model,
            "input": conversation,
        }
        if max_tokens is not None:
            request["max_output_tokens"] = max_tokens
        if system_prompt:
            request["instructions"] = system_prompt
        if temperature is not None:
            request["temperature"] = temperature
        if openai_reasoning_effort is not None:
            request["reasoning"] = {"effort": openai_reasoning_effort}
        return request

    if provider == "deepseek":
        request = {
            "model": model,
            "messages": messages,
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if temperature is not None:
            request["temperature"] = temperature
        return request

    if provider == "claude":
        request = {
            "model": model,
            "messages": conversation,
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if system_prompt:
            request["system"] = system_prompt
        if temperature is not None:
            request["temperature"] = temperature
        return request

    if provider in {"qwen_local", "llama_local"}:
        request = {
            "model": model,
            "messages": messages,
        }
        if max_tokens is not None:
            request["max_new_tokens"] = max_tokens
        if temperature is not None:
            request["temperature"] = temperature
        if provider == "qwen_local":
            request["chat_template_kwargs"] = {"enable_thinking": False}
        return request

    raise ValueError(f"Unsupported provider: {provider!r}")


def call_llm(
    payload: dict[str, Any],
    provider: Provider,
    *,
    model: str,
    max_tokens: int | None = 512,
    temperature: float | None = None,
    **kwargs: Any,
) -> LLMResult:
    """Call one supported provider with a Janus payload."""

    request = build_provider_request(
        payload,
        provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        openai_reasoning_effort=kwargs.get("openai_reasoning_effort"),
    )

    if provider == "openai":
        raw_response = call_openai(request, api_key=kwargs.get("api_key"))
        text = _extract_openai_text(raw_response)
        reasoning_text = None
        token_usage = _extract_token_usage(raw_response)
    elif provider == "deepseek":
        raw_response = call_deepseek(
            request,
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url", DEEPSEEK_BASE_URL),
            timeout=kwargs.get("deepseek_timeout", DEFAULT_DEEPSEEK_TIMEOUT_SECONDS),
            max_retries=kwargs.get("deepseek_max_retries", DEFAULT_DEEPSEEK_MAX_RETRIES),
        )
        text = _extract_chat_completion_text(raw_response)
        reasoning_text = _extract_chat_completion_reasoning_text(raw_response)
        token_usage = _extract_token_usage(raw_response)
    elif provider == "claude":
        raw_response = call_claude(request, api_key=kwargs.get("api_key"))
        text = _extract_claude_text(raw_response)
        reasoning_text = None
        token_usage = _extract_token_usage(raw_response)
    elif provider in {"qwen_local", "llama_local"}:
        if provider == "qwen_local" and "qwen_thinking" in kwargs:
            request["chat_template_kwargs"] = {
                "enable_thinking": bool(kwargs["qwen_thinking"])
            }
        raw_response = call_local_transformers(
            request,
            device_map=kwargs.get("device_map", "auto"),
            dtype=kwargs.get("dtype", "auto"),
            trust_remote_code=kwargs.get("trust_remote_code", False),
            load_in_4bit=kwargs.get("load_in_4bit", False),
            load_in_8bit=kwargs.get("load_in_8bit", False),
        )
        text = raw_response["text"]
        reasoning_text = raw_response.get("reasoning_text")
        token_usage = _extract_token_usage(raw_response)
    else:
        raise ValueError(f"Unsupported provider: {provider!r}")

    return LLMResult(
        provider=provider,
        model=model,
        text=text,
        request=request,
        metadata=dict(payload.get("metadata", {})),
        raw_response=raw_response,
        reasoning_text=reasoning_text,
        token_usage=token_usage,
    )


def run_benchmark(
    *,
    provider: Provider,
    model: str,
    output_path: str | Path = DEFAULT_RESULTS_PATH,
    conditions: tuple[Condition, ...] = ("neutral", "latent_goal"),
    max_tokens: int | None = 512,
    temperature: float | None = None,
    api_key: str | None = None,
    base_url: str = DEEPSEEK_BASE_URL,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    qwen_thinking: bool = False,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    batch_size: int = 1,
    dataset_path: str | Path | None = None,
    limit: int | None = None,
    seed: int | None = DEFAULT_FACT_SHUFFLE_SEED,
    retry_empty_responses: bool = False,
    use_openai_batch: bool = True,
    openai_batch_id: str | None = None,
    openai_reasoning_effort: str | None = None,
    openai_batch_poll_seconds: float = 60.0,
    openai_batch_timeout_seconds: float | None = None,
    openai_batch_dir: str | Path | None = None,
    deepseek_timeout: float = DEFAULT_DEEPSEEK_TIMEOUT_SECONDS,
    deepseek_max_retries: int = DEFAULT_DEEPSEEK_MAX_RETRIES,
) -> dict[str, int]:
    """Run a resumable benchmark and append each result immediately."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    if provider in {"qwen_local", "llama_local"}:
        _load_local_model(
            model,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )

    samples = load_samples(dataset_path) if dataset_path is not None else load_samples()
    if limit is not None:
        samples = samples[:limit]

    completed = load_completed_items(
        output_path,
        retry_empty_responses=retry_empty_responses,
    )
    stats = {"dataset_entries": len(samples), "completed": 0, "skipped": 0, "errors": 0}

    if provider == "openai" and use_openai_batch:
        pending_openai_payloads: list[dict[str, Any]] = []
        for sample in samples:
            fact_order_seed = f"{seed}|{sample.sample_id}"
            for condition in conditions:
                item = (sample.sample_id, condition)
                if item in completed:
                    stats["skipped"] += 1
                    continue
                payload_rng = random.Random(fact_order_seed)
                pending_openai_payloads.append(
                    build_payload(sample, condition, rng=payload_rng)
                )

        if pending_openai_payloads or openai_batch_id:
            run_openai_batch(
                pending_openai_payloads,
                model=model,
                output_path=output_path,
                completed=completed,
                stats=stats,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=api_key,
                replace_empty_responses=retry_empty_responses,
                batch_id=openai_batch_id,
                reasoning_effort=openai_reasoning_effort,
                poll_seconds=openai_batch_poll_seconds,
                timeout_seconds=openai_batch_timeout_seconds,
                batch_dir=openai_batch_dir,
            )
        return stats

    pending_payloads: list[dict[str, Any]] = []
    for sample in samples:
        fact_order_seed = f"{seed}|{sample.sample_id}"

        for condition in conditions:
            item = (sample.sample_id, condition)
            if item in completed:
                stats["skipped"] += 1
                continue

            payload_rng = random.Random(fact_order_seed)
            payload = build_payload(sample, condition, rng=payload_rng)
            if provider in {"qwen_local", "llama_local"} and batch_size > 1:
                pending_payloads.append(payload)
                if len(pending_payloads) >= batch_size:
                    flush_local_batch(
                        pending_payloads,
                        provider=provider,
                        model=model,
                        output_path=output_path,
                        completed=completed,
                        stats=stats,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        device_map=device_map,
                        dtype=dtype,
                        trust_remote_code=trust_remote_code,
                        qwen_thinking=qwen_thinking,
                        load_in_4bit=load_in_4bit,
                        load_in_8bit=load_in_8bit,
                        replace_empty_responses=retry_empty_responses,
                    )
                    pending_payloads = []
                continue

            try:
                result = call_llm(
                    payload,
                    provider,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    openai_reasoning_effort=openai_reasoning_effort,
                    api_key=api_key,
                    base_url=base_url,
                    deepseek_timeout=deepseek_timeout,
                    deepseek_max_retries=deepseek_max_retries,
                    device_map=device_map,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                    qwen_thinking=qwen_thinking,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                )
            except Exception as exc:
                append_jsonl(
                    output_path,
                    build_error_record(
                        payload=payload,
                        error=exc,
                    ),
                )
                stats["errors"] += 1
                continue

            if not result.text.strip():
                print(
                    "Blank model generation: "
                    f"provider={provider} model={model} "
                    f"sample_id={sample.sample_id} condition={condition}",
                    flush=True,
                )
            write_result_record(
                output_path,
                build_result_record(payload=payload, result=result),
                replace_empty_response=retry_empty_responses,
            )
            completed.add(item)
            stats["completed"] += 1

    if pending_payloads:
        flush_local_batch(
            pending_payloads,
            provider=provider,
            model=model,
            output_path=output_path,
            completed=completed,
            stats=stats,
            max_tokens=max_tokens,
            temperature=temperature,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            qwen_thinking=qwen_thinking,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            replace_empty_responses=retry_empty_responses,
        )

    return stats


def run_openai_batch(
    payloads: list[dict[str, Any]],
    *,
    model: str,
    output_path: str | Path,
    completed: set[tuple[str, str]],
    stats: dict[str, int],
    max_tokens: int | None,
    temperature: float | None,
    api_key: str | None,
    replace_empty_responses: bool,
    batch_id: str | None = None,
    reasoning_effort: str | None = None,
    poll_seconds: float = 60.0,
    timeout_seconds: float | None = None,
    batch_dir: str | Path | None = None,
) -> None:
    """Submit or resume an OpenAI Batch API run and write Janus result rows."""

    if poll_seconds <= 0:
        raise ValueError("openai batch poll seconds must be positive")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI SDK with: pip install openai") from exc

    client = OpenAI(api_key=api_key or os.environ.get(OPENAI_API_KEY_ENV))
    manifest_path: Path | None = None

    if batch_id is None:
        if not payloads:
            return
        batch_root = Path(batch_dir) if batch_dir is not None else batch_artifact_dir(output_path)
        batch_root.mkdir(parents=True, exist_ok=True)
        input_path, manifest_path = write_openai_batch_input(
            payloads,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            batch_dir=batch_root,
            output_path=output_path,
        )
        with input_path.open("rb") as batch_file:
            batch_input_file = client.files.create(file=batch_file, purpose="batch")
        batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint=OPENAI_BATCH_ENDPOINT,
            completion_window=OPENAI_BATCH_COMPLETION_WINDOW,
            metadata={"output": str(output_path), "model": model},
        )
        batch_id = batch.id
        update_openai_batch_manifest(
            manifest_path,
            batch=batch,
            input_file_id=batch_input_file.id,
        )
        print(
            f"Submitted OpenAI batch {batch_id} with {len(payloads)} request(s).",
            flush=True,
        )
    else:
        if payloads:
            batch_root = Path(batch_dir) if batch_dir is not None else batch_artifact_dir(output_path)
            batch_root.mkdir(parents=True, exist_ok=True)
            _, manifest_path = write_openai_batch_input(
                payloads,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                batch_dir=batch_root,
                output_path=output_path,
                batch_id=batch_id,
            )
        else:
            raise ValueError("--openai-batch-id requires pending payloads to map custom_id values")

    batch = wait_for_openai_batch(
        client,
        batch_id,
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
    )
    status = object_field(batch, "status")
    if status != "completed":
        for payload in payloads:
            append_jsonl(
                output_path,
                build_error_record(
                    payload=payload,
                    error=RuntimeError(f"OpenAI batch {batch_id} ended with status {status}"),
                ),
            )
            stats["errors"] += 1
        return

    output_file_id = object_field(batch, "output_file_id")
    if not isinstance(output_file_id, str) or not output_file_id:
        raise RuntimeError(f"OpenAI batch {batch_id} completed without output_file_id")

    output_text = download_openai_file_text(client, output_file_id)
    artifact_root = Path(batch_dir) if batch_dir is not None else batch_artifact_dir(output_path)
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_artifact_path = artifact_root / f"{sanitize_filename(batch_id)}_output.jsonl"
    output_artifact_path.write_text(output_text, encoding="utf-8")

    request_by_custom_id = {
        openai_batch_custom_id(payload): build_provider_request(
            payload,
            "openai",
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            openai_reasoning_effort=reasoning_effort,
        )
        for payload in payloads
    }
    payload_by_custom_id = {openai_batch_custom_id(payload): payload for payload in payloads}
    returned: set[str] = set()

    for line in output_text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or custom_id not in payload_by_custom_id:
            continue
        returned.add(custom_id)
        payload = payload_by_custom_id[custom_id]
        response = row.get("response") or {}
        status_code = response.get("status_code")
        body = response.get("body")
        if status_code != 200 or not isinstance(body, dict):
            append_jsonl(
                output_path,
                build_error_record(
                    payload=payload,
                    error=RuntimeError(json.dumps(row.get("error") or response)),
                ),
            )
            stats["errors"] += 1
            continue

        text = _extract_openai_text(body)
        metadata = dict(payload.get("metadata", {}))
        if not text.strip():
            print(
                "Blank model generation: "
                f"provider=openai model={model} "
                f"sample_id={metadata.get('sample_id')} "
                f"condition={metadata.get('condition')}",
                flush=True,
            )
        result = LLMResult(
            provider="openai",
            model=model,
            text=text,
            request=request_by_custom_id[custom_id],
            metadata=metadata,
            raw_response=body,
            reasoning_text=None,
            token_usage=_extract_token_usage(body),
        )
        write_result_record(
            output_path,
            build_result_record(payload=payload, result=result),
            replace_empty_response=replace_empty_responses,
        )
        sample_id = metadata.get("sample_id")
        condition = metadata.get("condition")
        if isinstance(sample_id, str) and isinstance(condition, str):
            completed.add((sample_id, condition))
        stats["completed"] += 1

    error_file_id = object_field(batch, "error_file_id")
    if isinstance(error_file_id, str) and error_file_id:
        write_openai_batch_error_records(
            client,
            error_file_id=error_file_id,
            output_path=output_path,
            payload_by_custom_id=payload_by_custom_id,
            stats=stats,
            batch_dir=artifact_root,
        )

    for custom_id, payload in payload_by_custom_id.items():
        if custom_id in returned:
            continue
        append_jsonl(
            output_path,
            build_error_record(
                payload=payload,
                error=RuntimeError(f"OpenAI batch {batch_id} did not return {custom_id}"),
            ),
        )
        stats["errors"] += 1


def batch_artifact_dir(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.parent / "openai_batches" / path.stem


def write_openai_batch_input(
    payloads: list[dict[str, Any]],
    *,
    model: str,
    max_tokens: int | None,
    temperature: float | None,
    reasoning_effort: str | None,
    batch_dir: Path,
    output_path: str | Path,
    batch_id: str | None = None,
) -> tuple[Path, Path]:
    """Write the Batch API JSONL and a local manifest for inspection."""

    suffix = sanitize_filename(batch_id) if batch_id else str(int(time.time()))
    input_path = batch_dir / f"{sanitize_filename(model)}_{suffix}_input.jsonl"
    manifest_path = input_path.with_suffix(".manifest.json")

    manifest_requests: list[dict[str, Any]] = []
    with input_path.open("w", encoding="utf-8") as input_file:
        for payload in payloads:
            request = build_provider_request(
                payload,
                "openai",
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                openai_reasoning_effort=reasoning_effort,
            )
            custom_id = openai_batch_custom_id(payload)
            input_file.write(
                json.dumps(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": OPENAI_BATCH_ENDPOINT,
                        "body": request,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            manifest_requests.append(
                {
                    "custom_id": custom_id,
                    "metadata": dict(payload.get("metadata", {})),
                    "request": request,
                }
            )

    manifest = {
        "model": model,
        "endpoint": OPENAI_BATCH_ENDPOINT,
        "completion_window": OPENAI_BATCH_COMPLETION_WINDOW,
        "output_path": str(output_path),
        "input_path": str(input_path),
        "requests": manifest_requests,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return input_path, manifest_path


def update_openai_batch_manifest(
    manifest_path: Path | None,
    *,
    batch: Any,
    input_file_id: str,
) -> None:
    if manifest_path is None:
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["batch_id"] = object_field(batch, "id")
    manifest["input_file_id"] = input_file_id
    manifest["batch"] = to_jsonable(batch)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def wait_for_openai_batch(
    client: Any,
    batch_id: str,
    *,
    poll_seconds: float,
    timeout_seconds: float | None,
) -> Any:
    start = time.monotonic()
    terminal_statuses = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch_id)
        status = object_field(batch, "status")
        request_counts = object_field(batch, "request_counts")
        print(
            f"OpenAI batch {batch_id} status={status} request_counts={to_jsonable(request_counts)}",
            flush=True,
        )
        if status in terminal_statuses:
            return batch
        if timeout_seconds is not None and time.monotonic() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for OpenAI batch {batch_id}")
        time.sleep(poll_seconds)


def download_openai_file_text(client: Any, file_id: str) -> str:
    file_response = client.files.content(file_id)
    text = getattr(file_response, "text", None)
    if isinstance(text, str):
        return text
    read = getattr(file_response, "read", None)
    if callable(read):
        data = read()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        if isinstance(data, str):
            return data
    content = getattr(file_response, "content", None)
    if isinstance(content, bytes):
        return content.decode("utf-8")
    if isinstance(content, str):
        return content
    return str(file_response)


def write_openai_batch_error_records(
    client: Any,
    *,
    error_file_id: str,
    output_path: str | Path,
    payload_by_custom_id: dict[str, dict[str, Any]],
    stats: dict[str, int],
    batch_dir: Path,
) -> None:
    error_text = download_openai_file_text(client, error_file_id)
    error_artifact_path = batch_dir / f"{sanitize_filename(error_file_id)}_errors.jsonl"
    error_artifact_path.write_text(error_text, encoding="utf-8")
    for line in error_text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or custom_id not in payload_by_custom_id:
            continue
        append_jsonl(
            output_path,
            build_error_record(
                payload=payload_by_custom_id[custom_id],
                error=RuntimeError(json.dumps(row.get("error") or row)),
            ),
        )
        stats["errors"] += 1


def openai_batch_custom_id(payload: dict[str, Any]) -> str:
    metadata = dict(payload.get("metadata", {}))
    sample_id = sanitize_filename(str(metadata.get("sample_id", "sample")))
    condition = sanitize_filename(str(metadata.get("condition", "condition")))
    return f"janus-{sample_id}-{condition}"


def flush_local_batch(
    payloads: list[dict[str, Any]],
    *,
    provider: Provider,
    model: str,
    output_path: str | Path,
    completed: set[tuple[str, str]],
    stats: dict[str, int],
    max_tokens: int | None,
    temperature: float | None,
    device_map: str,
    dtype: str,
    trust_remote_code: bool,
    qwen_thinking: bool,
    load_in_4bit: bool,
    load_in_8bit: bool,
    replace_empty_responses: bool = False,
) -> None:
    """Run and append one local inference batch."""

    requests: list[dict[str, Any]] = []
    for payload in payloads:
        request = build_provider_request(
            payload,
            provider,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if provider == "qwen_local":
            request["chat_template_kwargs"] = {"enable_thinking": qwen_thinking}
        requests.append(request)

    try:
        raw_responses = call_local_transformers_batch(
            requests,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
    except Exception as exc:
        for payload in payloads:
            append_jsonl(output_path, build_error_record(payload=payload, error=exc))
            stats["errors"] += 1
        return

    for payload, raw_response, request in zip(payloads, raw_responses, requests):
        metadata = payload.get("metadata", {})
        sample_id = metadata.get("sample_id")
        condition = metadata.get("condition")
        response_text = raw_response["text"]
        if not response_text.strip():
            print(
                "Blank model generation: "
                f"provider={provider} model={model} "
                f"sample_id={sample_id} condition={condition} "
                f"generated_tokens={len(raw_response.get('generated_token_ids', []))}",
                flush=True,
            )
        result = LLMResult(
            provider=provider,
            model=model,
            text=response_text,
            request=request,
            metadata=dict(metadata),
            raw_response=raw_response,
            reasoning_text=raw_response.get("reasoning_text"),
            token_usage=_extract_token_usage(raw_response),
        )
        write_result_record(
            output_path,
            build_result_record(payload=payload, result=result),
            replace_empty_response=replace_empty_responses,
        )
        if isinstance(sample_id, str) and isinstance(condition, str):
            completed.add((sample_id, condition))
        stats["completed"] += 1


def load_completed_items(
    path: str | Path,
    *,
    retry_empty_responses: bool = False,
) -> set[tuple[str, str]]:
    """Read existing JSONL results and return completed sample/condition pairs."""

    results_path = Path(path)
    if not results_path.exists():
        return set()

    completed: set[tuple[str, str]] = set()
    with results_path.open(encoding="utf-8") as results_file:
        for line_number, line in enumerate(results_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {results_path} at line {line_number}"
                ) from exc
            sample_id = record.get("sample_id")
            condition = record.get("condition")
            if (
                isinstance(sample_id, str)
                and isinstance(condition, str)
                and isinstance(record.get("response_text"), str)
            ):
                if retry_empty_responses and not record["response_text"].strip():
                    continue
                completed.add((sample_id, condition))
    return completed


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON record and fsync so progress survives interruption."""

    results_path = Path(path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as results_file:
        results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        results_file.flush()
        os.fsync(results_file.fileno())


def write_result_record(
    path: str | Path,
    record: dict[str, Any],
    *,
    replace_empty_response: bool = False,
) -> None:
    """Write a result, replacing an existing blank response for the same key."""

    if replace_empty_response and replace_empty_response_record(path, record):
        return
    append_jsonl(path, record)


def replace_empty_response_record(path: str | Path, record: dict[str, Any]) -> bool:
    """Replace the latest matching empty-response JSONL row, if one exists."""

    results_path = Path(path)
    if not results_path.exists():
        return False

    target_key = (record.get("sample_id"), record.get("condition"))
    lines = results_path.read_text(encoding="utf-8").splitlines()
    replace_index: int | None = None

    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            existing = json.loads(line)
        except json.JSONDecodeError:
            continue
        existing_key = (existing.get("sample_id"), existing.get("condition"))
        response_text = existing.get("response_text")
        if (
            existing_key == target_key
            and isinstance(response_text, str)
            and not response_text.strip()
        ):
            replace_index = index

    if replace_index is None:
        return False

    lines[replace_index] = json.dumps(record, ensure_ascii=False)
    tmp_path = results_path.with_suffix(results_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as tmp_file:
        for line in lines:
            tmp_file.write(line + "\n")
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
    tmp_path.replace(results_path)
    return True


def default_output_path_for_model(model: str) -> Path:
    """Return the default JSONL path when the caller does not provide one."""

    return Path("results") / f"{sanitize_filename(model)}.jsonl"


def sanitize_filename(value: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in value)
    return "_".join(part for part in safe.split("_") if part).lower()


def parse_bool(value: str | bool) -> bool:
    """Parse shell-friendly boolean values."""

    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def build_result_record(
    *,
    payload: dict[str, Any],
    result: LLMResult,
) -> dict[str, Any]:
    """Convert a successful LLM call into a compact result record."""

    record = {
        **record_input_fields(payload),
        "response_text": result.text,
    }
    if result.reasoning_text:
        record["reasoning_text"] = result.reasoning_text
    if result.token_usage:
        record["token_usage"] = result.token_usage
    return record


def build_error_record(
    *,
    payload: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    """Convert a failed LLM call into a compact error record."""

    return {
        **record_input_fields(payload),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def record_input_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the non-duplicated prompt fields saved with each result."""

    metadata = dict(payload.get("metadata", {}))
    messages = _payload_messages(payload)
    system_prompt, conversation = _split_system_message(messages)
    user_prompt = conversation[0]["content"] if conversation else ""
    scenario = _extract_section(user_prompt, "Scenario:", "Facts:")

    return {
        "sample_id": metadata.get("sample_id"),
        "condition": metadata.get("condition"),
        "category": metadata.get("category"),
        "item": metadata.get("item"),
        "system_prompt": system_prompt,
        "scenario": scenario,
        "presented_facts": metadata.get("presented_fact_order", []),
    }


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def to_jsonable(value: Any) -> Any:
    """Best-effort conversion of SDK response objects into JSON values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return to_jsonable(model_dump())
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_jsonable(to_dict())
    return repr(value)


def object_field(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _extract_token_usage(response: Any) -> dict[str, Any] | None:
    """Return provider token accounting when available."""

    usage = object_field(response, "usage")
    if usage is not None:
        converted = to_jsonable(usage)
        if isinstance(converted, dict):
            return converted
        return {"value": converted}

    generated_token_ids = object_field(response, "generated_token_ids")
    if isinstance(generated_token_ids, list):
        return {"completion_tokens": len(generated_token_ids)}
    return None


def call_openai(request: dict[str, Any], *, api_key: str | None = None) -> Any:
    """Call OpenAI's Responses API."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI SDK with: pip install openai") from exc

    client = OpenAI(api_key=api_key or os.environ.get(OPENAI_API_KEY_ENV))
    return client.responses.create(**request)


def call_deepseek(
    request: dict[str, Any],
    *,
    api_key: str | None = None,
    base_url: str = DEEPSEEK_BASE_URL,
    timeout: float = DEFAULT_DEEPSEEK_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_DEEPSEEK_MAX_RETRIES,
) -> Any:
    """Call DeepSeek's OpenAI-compatible chat completions API."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI SDK with: pip install openai") from exc

    client = OpenAI(
        api_key=api_key or os.environ.get(DEEPSEEK_API_KEY_ENV),
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    return client.chat.completions.create(**request)


def call_claude(request: dict[str, Any], *, api_key: str | None = None) -> Any:
    """Call Anthropic Claude's Messages API."""

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("Install the Anthropic SDK with: pip install anthropic") from exc

    client = Anthropic(api_key=api_key or os.environ.get(ANTHROPIC_API_KEY_ENV))
    return client.messages.create(**request)


def call_local_transformers(
    request: dict[str, Any],
    *,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> dict[str, Any]:
    """Call a local chat model through Hugging Face Transformers."""

    return call_local_transformers_batch(
        [request],
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )[0]


def call_local_transformers_batch(
    requests: list[dict[str, Any]],
    *,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> list[dict[str, Any]]:
    """Call a local chat model for a batch of requests."""

    if not requests:
        return []

    model_ids = {request["model"] for request in requests}
    if len(model_ids) != 1:
        raise ValueError("Local batch requests must all use the same model")

    tokenizer, model = _load_local_model(
        requests[0]["model"],
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    _prepare_tokenizer_for_batching(tokenizer)
    input_device = _model_input_device(model)
    prompts = [
        tokenizer.apply_chat_template(
            request["messages"],
            add_generation_prompt=True,
            tokenize=False,
            **request.get("chat_template_kwargs", {}),
        )
        for request in requests
    ]
    previous_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    tokenizer.padding_side = previous_padding_side
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    generation_options: dict[str, Any] = {
        "max_new_tokens": max(request.get("max_new_tokens", 512) for request in requests),
        "pad_token_id": tokenizer.eos_token_id,
    }
    temperatures = [
        request.get("temperature")
        for request in requests
        if request.get("temperature") is not None
    ]
    if temperatures:
        if len(set(temperatures)) > 1:
            raise ValueError("Local batch requests must use the same temperature")
        generation_options["temperature"] = temperatures[0]
        generation_options["do_sample"] = bool(temperatures[0])

    output_ids = model.generate(**inputs, **generation_options)
    input_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[:, input_length:]
    responses: list[dict[str, Any]] = []
    for generated_row in generated_ids:
        text = tokenizer.decode(generated_row, skip_special_tokens=True).strip()
        response_text, reasoning_text = _split_qwen_thinking_content(text)
        response: dict[str, Any] = {
            "text": response_text,
            "generated_token_ids": generated_row.tolist(),
        }
        if reasoning_text:
            response["reasoning_text"] = reasoning_text
        responses.append(response)
    return responses


@lru_cache(maxsize=4)
def _load_local_model(
    model_id: str,
    *,
    device_map: str,
    dtype: str,
    trust_remote_code: bool,
    load_in_4bit: bool,
    load_in_8bit: bool,
) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Install local model dependencies with: pip install transformers torch accelerate"
        ) from exc

    local_device, load_device_map = _resolve_local_device(device_map, torch)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if load_device_map is not None:
        model_kwargs["device_map"] = load_device_map
    torch_dtype = _resolve_torch_dtype(dtype, torch, local_device)
    if torch_dtype is not None and not (load_in_4bit or load_in_8bit):
        model_kwargs["torch_dtype"] = torch_dtype
    if load_in_4bit and load_in_8bit:
        raise ValueError("Use only one of load_in_4bit or load_in_8bit")
    if load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        if torch_dtype is not None:
            model_kwargs["bnb_4bit_compute_dtype"] = torch_dtype
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if load_device_map is None:
        model.to(local_device)
    model.eval()
    return tokenizer, model


def _prepare_tokenizer_for_batching(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def _resolve_local_device(device_map: str, torch: Any) -> tuple[str, str | None]:
    """Choose a local execution device, preferring MPS on macOS."""

    if device_map == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", None
        if torch.cuda.is_available():
            return "cuda", "auto"
        return "cpu", None

    if device_map in {"mps", "cpu", "cuda"} or device_map.startswith("cuda:"):
        return device_map, None

    return "cpu", device_map


def _resolve_torch_dtype(dtype: str, torch: Any, local_device: str) -> Any:
    if dtype == "auto":
        if local_device == "mps":
            return torch.float16
        return None
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype!r}")


def _model_input_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return getattr(model, "device", "cpu")


def _strip_qwen_thinking_content(text: str) -> str:
    """Return the final answer if a Qwen3 thinking block was generated."""

    return _split_qwen_thinking_content(text)[0]


def _split_qwen_thinking_content(text: str) -> tuple[str, str | None]:
    """Return final answer text plus optional Qwen3 thinking content."""

    start_tag = "<think>"
    end_tag = "</think>"
    if end_tag in text:
        prefix, response_text = text.rsplit(end_tag, maxsplit=1)
        if start_tag in prefix:
            reasoning_text = prefix.split(start_tag, maxsplit=1)[-1].strip()
        else:
            reasoning_text = prefix.strip()
        return response_text.strip(), reasoning_text or None
    if start_tag in text:
        print(
            "Unclosed Qwen thinking block; preserving generated text instead of "
            "stripping it to blank.",
            flush=True,
        )
        return text.replace(start_tag, "", 1).strip(), None
    return text, None


def _payload_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Payload must contain a messages list")
    return [_normalize_message(message) for message in messages]


def _normalize_message(message: dict[str, Any]) -> dict[str, str]:
    role = message.get("role")
    content = message.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        raise ValueError(f"Invalid message shape: {message!r}")
    return {"role": role, "content": content}


def _split_system_message(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[dict[str, str]]]:
    system_parts = [
        message["content"] for message in messages if message["role"] == "system"
    ]
    conversation = [
        message for message in messages if message["role"] != "system"
    ]
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, conversation


def _extract_openai_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text
    if isinstance(response, dict) and isinstance(response.get("output_text"), str):
        return response["output_text"]
    output = object_field(response, "output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            content = object_field(item, "content")
            if not isinstance(content, list):
                continue
            for block in content:
                text = object_field(block, "text")
                block_type = object_field(block, "type")
                if isinstance(text, str) and block_type in {None, "output_text", "text"}:
                    parts.append(text)
        if parts:
            return "".join(parts)
    return str(response)


def _extract_chat_completion_text(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError):
        pass
    if isinstance(response, dict):
        return response["choices"][0]["message"].get("content", "")
    return str(response)


def _extract_chat_completion_reasoning_text(response: Any) -> str | None:
    """Extract DeepSeek-style reasoning_content from chat completions."""

    try:
        reasoning_text = response.choices[0].message.reasoning_content
        if isinstance(reasoning_text, str):
            return reasoning_text
    except (AttributeError, IndexError):
        pass
    if isinstance(response, dict):
        try:
            reasoning_text = response["choices"][0]["message"].get("reasoning_content")
        except (KeyError, IndexError, TypeError, AttributeError):
            reasoning_text = None
        if isinstance(reasoning_text, str):
            return reasoning_text
    return None


def _extract_claude_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return str(response)


def main() -> None:
    """Run a resumable Janus benchmark and save JSONL results."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=["openai", "deepseek", "claude", "qwen_local", "llama_local"],
        default="openai",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--condition", choices=["neutral", "latent_goal"])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_FACT_SHUFFLE_SEED,
        help="Dataset fact-shuffle seed, keyed only by sample_id for cross-model comparability.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum generated tokens. Omit to use the provider default.",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--api-key")
    parser.add_argument(
        "--deepseek-timeout",
        type=float,
        default=float(
            os.environ.get("DEEPSEEK_TIMEOUT", DEFAULT_DEEPSEEK_TIMEOUT_SECONDS)
        ),
        help="DeepSeek OpenAI SDK request timeout in seconds.",
    )
    parser.add_argument(
        "--deepseek-max-retries",
        type=int,
        default=int(
            os.environ.get("DEEPSEEK_MAX_RETRIES", DEFAULT_DEEPSEEK_MAX_RETRIES)
        ),
        help="DeepSeek OpenAI SDK retry count.",
    )
    parser.add_argument(
        "--openai-reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=os.environ.get("OPENAI_REASONING_EFFORT"),
        help="Responses API reasoning.effort for OpenAI models that support it.",
    )
    parser.add_argument("--base-url", default=DEEPSEEK_BASE_URL)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument(
        "--no-openai-batch",
        action="store_true",
        help="Use synchronous OpenAI calls instead of the Batch API.",
    )
    parser.add_argument(
        "--openai-batch-id",
        help="Poll and download an existing OpenAI batch instead of submitting a new one.",
    )
    parser.add_argument(
        "--openai-batch-poll-seconds",
        type=float,
        default=float(os.environ.get("OPENAI_BATCH_POLL_SECONDS", "60")),
    )
    parser.add_argument(
        "--openai-batch-timeout-seconds",
        type=float,
        default=(
            float(os.environ["OPENAI_BATCH_TIMEOUT_SECONDS"])
            if os.environ.get("OPENAI_BATCH_TIMEOUT_SECONDS")
            else None
        ),
    )
    parser.add_argument("--openai-batch-dir")
    parser.add_argument(
        "--retry-empty-responses",
        action="store_true",
        help="Treat existing blank response_text values as incomplete and rerun them.",
    )
    parser.add_argument(
        "--qwen-thinking",
        type=parse_bool,
        default=False,
        help="Enable Qwen3 thinking mode when using --provider qwen_local.",
    )
    args = parser.parse_args()

    conditions: tuple[Condition, ...]
    if args.condition:
        conditions = (args.condition,)
    else:
        conditions = ("neutral", "latent_goal")

    output_path = args.output or str(default_output_path_for_model(args.model))

    stats = run_benchmark(
        provider=args.provider,
        model=args.model,
        output_path=output_path,
        conditions=conditions,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        api_key=args.api_key,
        base_url=args.base_url,
        device_map=args.device_map,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        batch_size=args.batch_size,
        qwen_thinking=args.qwen_thinking,
        dataset_path=args.dataset,
        limit=args.limit,
        seed=args.seed,
        retry_empty_responses=args.retry_empty_responses,
        use_openai_batch=not args.no_openai_batch,
        openai_batch_id=args.openai_batch_id,
        openai_reasoning_effort=args.openai_reasoning_effort,
        openai_batch_poll_seconds=args.openai_batch_poll_seconds,
        openai_batch_timeout_seconds=args.openai_batch_timeout_seconds,
        openai_batch_dir=args.openai_batch_dir,
        deepseek_timeout=args.deepseek_timeout,
        deepseek_max_retries=args.deepseek_max_retries,
    )
    print(
        "Benchmark complete: "
        f"{stats['completed']} completed, "
        f"{stats['skipped']} skipped, "
        f"{stats['errors']} errors. "
        f"Results: {output_path}"
    )
    if args.fail_on_error and stats["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
