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
DEFAULT_RESULTS_PATH = Path("results") / "janus_results.jsonl"
DEFAULT_FACT_SHUFFLE_SEED = 1234


@dataclass(frozen=True)
class LLMResult:
    """Normalized text plus provider-specific details."""

    provider: Provider
    model: str
    text: str
    request: dict[str, Any]
    metadata: dict[str, Any]
    raw_response: Any


def build_provider_request(
    payload: dict[str, Any],
    provider: Provider,
    *,
    model: str,
    max_tokens: int = 512,
    temperature: float | None = 0,
) -> dict[str, Any]:
    """Convert a Janus payload into the request shape a provider expects."""

    messages = _payload_messages(payload)
    system_prompt, conversation = _split_system_message(messages)

    if provider == "openai":
        request: dict[str, Any] = {
            "model": model,
            "input": conversation,
            "max_output_tokens": max_tokens,
        }
        if system_prompt:
            request["instructions"] = system_prompt
        if temperature is not None:
            request["temperature"] = temperature
        return request

    if provider == "deepseek":
        request = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            request["temperature"] = temperature
        return request

    if provider == "claude":
        request = {
            "model": model,
            "messages": conversation,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            request["system"] = system_prompt
        if temperature is not None:
            request["temperature"] = temperature
        return request

    if provider in {"qwen_local", "llama_local"}:
        request = {
            "model": model,
            "messages": messages,
            "max_new_tokens": max_tokens,
        }
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
    max_tokens: int = 512,
    temperature: float | None = 0,
    **kwargs: Any,
) -> LLMResult:
    """Call one supported provider with a Janus payload."""

    request = build_provider_request(
        payload,
        provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if provider == "openai":
        raw_response = call_openai(request, api_key=kwargs.get("api_key"))
        text = _extract_openai_text(raw_response)
    elif provider == "deepseek":
        raw_response = call_deepseek(
            request,
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url", DEEPSEEK_BASE_URL),
        )
        text = _extract_chat_completion_text(raw_response)
    elif provider == "claude":
        raw_response = call_claude(request, api_key=kwargs.get("api_key"))
        text = _extract_claude_text(raw_response)
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
        )
        text = raw_response["text"]
    else:
        raise ValueError(f"Unsupported provider: {provider!r}")

    return LLMResult(
        provider=provider,
        model=model,
        text=text,
        request=request,
        metadata=dict(payload.get("metadata", {})),
        raw_response=raw_response,
    )


def run_benchmark(
    *,
    provider: Provider,
    model: str,
    output_path: str | Path = DEFAULT_RESULTS_PATH,
    conditions: tuple[Condition, ...] = ("neutral", "latent_goal"),
    max_tokens: int = 512,
    temperature: float | None = 0,
    api_key: str | None = None,
    base_url: str = DEEPSEEK_BASE_URL,
    device_map: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    qwen_thinking: bool = False,
    dataset_path: str | Path | None = None,
    limit: int | None = None,
    seed: int | None = DEFAULT_FACT_SHUFFLE_SEED,
) -> dict[str, int]:
    """Run a resumable benchmark and append each result immediately."""

    if provider in {"qwen_local", "llama_local"}:
        _load_local_model(
            model,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    samples = load_samples(dataset_path) if dataset_path is not None else load_samples()
    if limit is not None:
        samples = samples[:limit]

    completed = load_completed_items(output_path)
    stats = {"dataset_entries": len(samples), "completed": 0, "skipped": 0, "errors": 0}

    for sample in samples:
        fact_order_seed = f"{seed}|{sample.sample_id}"

        for condition in conditions:
            item = (sample.sample_id, condition)
            if item in completed:
                stats["skipped"] += 1
                continue

            payload_rng = random.Random(fact_order_seed)
            payload = build_payload(sample, condition, rng=payload_rng)
            try:
                result = call_llm(
                    payload,
                    provider,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    api_key=api_key,
                    base_url=base_url,
                    device_map=device_map,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                    qwen_thinking=qwen_thinking,
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

            append_jsonl(
                output_path,
                build_result_record(
                    payload=payload,
                    result=result,
                ),
            )
            completed.add(item)
            stats["completed"] += 1

    return stats


def load_completed_items(path: str | Path) -> set[tuple[str, str]]:
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

    return {
        **record_input_fields(payload),
        "response_text": result.text,
    }


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
) -> Any:
    """Call DeepSeek's OpenAI-compatible chat completions API."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI SDK with: pip install openai") from exc

    client = OpenAI(
        api_key=api_key or os.environ.get(DEEPSEEK_API_KEY_ENV),
        base_url=base_url,
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
) -> dict[str, Any]:
    """Call a local chat model through Hugging Face Transformers."""

    tokenizer, model = _load_local_model(
        request["model"],
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    input_device = _model_input_device(model)
    messages = request["messages"]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
        **request.get("chat_template_kwargs", {}),
    )
    inputs = {key: value.to(input_device) for key, value in inputs.items()}

    generation_options: dict[str, Any] = {
        "max_new_tokens": request.get("max_new_tokens", 512),
        "do_sample": bool(request.get("temperature", 0)),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if request.get("temperature"):
        generation_options["temperature"] = request["temperature"]

    output_ids = model.generate(**inputs, **generation_options)
    input_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[:, input_length:]
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    text = _strip_qwen_thinking_content(text)
    return {"text": text, "generated_token_ids": generated_ids[0].tolist()}


@lru_cache(maxsize=4)
def _load_local_model(
    model_id: str,
    *,
    device_map: str,
    dtype: str,
    trust_remote_code: bool,
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
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if load_device_map is None:
        model.to(local_device)
    model.eval()
    return tokenizer, model


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

    end_tag = "</think>"
    if end_tag not in text:
        return text
    return text.rsplit(end_tag, maxsplit=1)[-1].strip()


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
    return str(response)


def _extract_chat_completion_text(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError):
        pass
    if isinstance(response, dict):
        return response["choices"][0]["message"].get("content", "")
    return str(response)


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
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--api-key")
    parser.add_argument("--base-url", default=DEEPSEEK_BASE_URL)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
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
        qwen_thinking=args.qwen_thinking,
        dataset_path=args.dataset,
        limit=args.limit,
        seed=args.seed,
    )
    print(
        "Benchmark complete: "
        f"{stats['completed']} completed, "
        f"{stats['skipped']} skipped, "
        f"{stats['errors']} errors. "
        f"Results: {output_path}"
    )


if __name__ == "__main__":
    main()
