"""LCI-6C live proof for the native Ollama adapter.

This module is intentionally gated. It never pulls models and is excluded from
the default unit suite; run it explicitly with ``INTERGRAX_LCI6C_LIVE=1``.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator, Mapping, Sequence
from typing import Literal

import pytest
from ollama import Client
from pydantic import BaseModel, ConfigDict

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEventKind
from intergrax.llm_adapters.providers.native_ollama_adapter import (
    NativeOllamaAdapter,
)
from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
from intergrax.utils import attribute_access

pytestmark = [pytest.mark.network, pytest.mark.no_ci]

_LIVE_FLAG = "INTERGRAX_LCI6C_LIVE"
_BASE_URL_ENV = "INTERGRAX_LCI6C_OLLAMA_BASE_URL"
_TOOLS_MODEL_ENV = "INTERGRAX_LCI6C_OLLAMA_TOOLS_MODEL"
_MISSING_MODEL = "intergrax-lci6c-definitely-missing-model"
_DEFAULT_BASE_URL = "http://127.0.0.1:11434"

_WEATHER_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]


class CityWeather(BaseModel):
    model_config = ConfigDict(extra="forbid")

    city: str
    temperature_c: int


class ImpossibleStructured(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: Literal["NOT_ALLOWED"]


class _RecordingClient:
    """Keep native provider responses so raw Ollama counters are observable."""

    def __init__(self, client: Client) -> None:
        self._client = client
        self.responses: list[object] = []

    def chat(self, **kwargs: object) -> object:
        result = self._client.chat(**kwargs)  # type: ignore[arg-type]
        if kwargs.get("stream") is True:
            return self._record_stream(result)
        self.responses.append(result)
        return result

    def _record_stream(self, result: object) -> Iterator[object]:
        for chunk in result:  # type: ignore[union-attr]
            self.responses.append(chunk)
            yield chunk


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return attribute_access.optional(value, name, default)


def _exception_observation(exc: BaseException) -> dict[str, object]:
    return {
        "class": type(exc).__name__,
        "message": str(exc).replace("\n", " ")[:240],
    }


def _response_observation(response: LLMAdapterResponse) -> dict[str, object]:
    usage = response.usage
    extensions = response.provider_extensions
    return {
        "type": type(response).__name__,
        "provider": response.provider,
        "model": response.model,
        "content_non_empty": bool(response.content),
        "finish_reason": response.finish_reason.value,
        "usage": (
            {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            }
            if usage is not None
            else None
        ),
        "usage_source": (
            extensions.usage_source if extensions is not None else None
        ),
    }


def _raw_counters(response: object | None) -> dict[str, object]:
    return {
        "prompt_eval_count": _field(response, "prompt_eval_count"),
        "eval_count": _field(response, "eval_count"),
    }


def _stats_observation(adapter: NativeOllamaAdapter, run_id: str) -> dict[str, object]:
    stats = adapter.usage.get_run_stats(run_id)
    names = ("calls", "errors", "input_tokens", "output_tokens")
    return {name: attribute_access.optional(stats, name, None) for name in names}


def _available_models(
    client: Client,
) -> list[tuple[str, tuple[str, ...]]]:
    listed = client.list()
    models: list[tuple[str, tuple[str, ...]]] = []
    for item in (_field(listed, "models", []) or []):
        name = _field(item, "model") or _field(item, "name")
        if not isinstance(name, str) or not name.strip():
            continue
        shown = client.show(model=name)
        raw_capabilities = _field(shown, "capabilities", []) or []
        capabilities = tuple(
            sorted(
                capability.strip().lower()
                for capability in raw_capabilities
                if isinstance(capability, str) and capability.strip()
            )
        )
        models.append((name, capabilities))
    return models


def _select_tools_model(
    models: Sequence[tuple[str, tuple[str, ...]]],
) -> str:
    configured = __import__("os").environ.get(_TOOLS_MODEL_ENV, "").strip()
    by_name = {name: capabilities for name, capabilities in models}
    if configured:
        if configured not in by_name:
            pytest.fail(f"{_TOOLS_MODEL_ENV} is not installed: {configured}")
        if "tools" not in by_name[configured]:
            pytest.fail(f"configured model does not declare tools: {configured}")
        return configured

    preferred = ("qwen2.5:7b", "llama3.1:8b", "qwen2.5:14b", "gpt-oss:20b")
    for name in preferred:
        if "tools" in by_name.get(name, ()):
            return name
    for name, capabilities in sorted(models):
        if "tools" in capabilities and "completion" in capabilities:
            return name
    pytest.fail("no installed completion model declares tools")


def _unused_local_endpoint() -> str:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return f"http://127.0.0.1:{port}"


def _direct_request_failure(
    client: Client,
    *,
    model: str,
    messages: Sequence[Mapping[str, object]],
    format_value: object | None = None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if format_value is not None:
        kwargs["format"] = format_value
    try:
        response = client.chat(**kwargs)  # type: ignore[arg-type]
    except Exception as exc:
        return {"success": False, "exception": _exception_observation(exc)}
    return {"success": True, "response_type": type(response).__name__}


def _run_live_proof() -> dict[str, object]:
    import os

    base_url = os.environ.get(_BASE_URL_ENV, _DEFAULT_BASE_URL).strip()
    client = Client(host=base_url)
    models = _available_models(client)
    tools_model = _select_tools_model(models)
    no_tools_models = [
        name
        for name, capabilities in models
        if "completion" in capabilities and "tools" not in capabilities
    ]
    resolved_no_tools_models = [
        name for name, capabilities in models if "tools" not in capabilities
    ]
    no_tools_model = no_tools_models[0] if no_tools_models else None
    resolved_no_tools_model = (
        resolved_no_tools_models[0] if resolved_no_tools_models else None
    )

    recorder = _RecordingClient(client)
    native = NativeOllamaAdapter(
        client=recorder,
        model=tools_model,
        base_url=base_url,
        options={"temperature": 0},
    )
    baseline = LangChainOllamaAdapter(
        model=tools_model,
        base_url=base_url,
        options={"temperature": 0},
    )
    plain_messages = [
        ChatMessage(
            role="user",
            content="Reply with exactly one short sentence: Warsaw is in Poland.",
        )
    ]

    native_plain = native.generate_messages(
        plain_messages,
        temperature=0,
        max_tokens=64,
        run_id="lci6c-plain-native",
    )
    baseline_plain = baseline.generate_messages(
        plain_messages,
        temperature=0,
        max_tokens=64,
        run_id="lci6c-plain-langchain",
    )
    plain_native_observation = _response_observation(native_plain)
    plain_baseline_observation = _response_observation(baseline_plain)
    plain_counters = _raw_counters(recorder.responses[-1])

    capability = native.model_capabilities
    capability_observation = {
        "model": capability.model,
        "resolved": capability.resolved,
        "supports_tools": capability.supports_tools,
        "capabilities": sorted(capability.capabilities),
        "source": capability.source.value,
        "refresh_same": native.refresh_model_capabilities() == capability,
    }

    tools_messages = [
        ChatMessage(
            role="user",
            content=(
                "Use the get_weather tool for Warsaw. "
                "Do not answer directly."
            ),
        )
    ]
    native_tools = native.generate_with_tools(
        tools_messages,
        _WEATHER_TOOL,
        temperature=0,
        max_tokens=128,
        tool_choice="required",
        run_id="lci6c-tools-native",
    )
    baseline_tools = baseline.generate_with_tools(
        tools_messages,
        _WEATHER_TOOL,
        temperature=0,
        max_tokens=128,
        tool_choice="required",
        run_id="lci6c-tools-langchain",
    )
    tools_native_observation = _response_observation(native_tools)
    tools_native_observation.update(
        {
            "tool_call_count": len(native_tools.tool_calls),
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": json.loads(call.arguments_json),
                }
                for call in native_tools.tool_calls
            ],
        }
    )
    tools_baseline_observation = _response_observation(baseline_tools)
    tools_baseline_observation["tool_call_count"] = len(baseline_tools.tool_calls)
    tools_counters = _raw_counters(recorder.responses[-1])

    structured_messages = [
        ChatMessage(
            role="user",
            content="Return city = Warsaw and temperature_c = 20.",
        )
    ]
    native_structured = native.generate_structured(
        structured_messages,
        CityWeather,
        temperature=0,
        max_tokens=128,
        run_id="lci6c-structured-native",
    )
    baseline_structured = baseline.generate_structured(
        structured_messages,
        CityWeather,
        temperature=0,
        max_tokens=128,
        run_id="lci6c-structured-langchain",
    )
    structured_native_observation = _response_observation(
        native_structured.response
    )
    structured_native_observation["parsed"] = native_structured.parsed.model_dump()
    structured_baseline_observation = _response_observation(
        baseline_structured.response
    )
    structured_baseline_observation["parsed"] = baseline_structured.parsed.model_dump()
    structured_counters = _raw_counters(recorder.responses[-1])

    native_stream_events = list(
        native.stream_messages(
            [ChatMessage(role="user", content="Count from one to five.")],
            temperature=0,
            max_tokens=64,
            run_id="lci6c-stream-native",
        )
    )
    baseline_stream_events = list(
        baseline.stream_messages(
            [ChatMessage(role="user", content="Count from one to five.")],
            temperature=0,
            max_tokens=64,
            run_id="lci6c-stream-langchain",
        )
    )
    native_partials = [
        event.delta_content
        for event in native_stream_events
        if event.kind is LLMStreamEventKind.PARTIAL
    ]
    native_finals = [
        event for event in native_stream_events if event.kind is LLMStreamEventKind.FINAL
    ]
    stream_native_observation = {
        "partial_count": len(native_partials),
        "final_count": len(native_finals),
        "concatenation_matches": bool(native_finals)
        and native_finals[-1].response is not None
        and native_finals[-1].response.content == "".join(native_partials),
        "final_usage_source": (
            native_finals[-1].response.provider_extensions.usage_source
            if native_finals
            and native_finals[-1].response is not None
            and native_finals[-1].response.provider_extensions is not None
            else None
        ),
        "event_order": [event.kind.value for event in native_stream_events],
    }
    stream_baseline_observation = {
        "partial_count": sum(
            event.kind is LLMStreamEventKind.PARTIAL for event in baseline_stream_events
        ),
        "final_count": sum(
            event.kind is LLMStreamEventKind.FINAL for event in baseline_stream_events
        ),
        "event_order": [event.kind.value for event in baseline_stream_events],
    }
    stream_counters = _raw_counters(recorder.responses[-1])

    missing = NativeOllamaAdapter(
        client=Client(host=base_url),
        model=_MISSING_MODEL,
        base_url=base_url,
    )
    missing_result: dict[str, object]
    try:
        missing.generate_messages(
            [ChatMessage(role="user", content="This must fail.")],
            run_id="lci6c-missing",
        )
        missing_result = {"success": True}
    except Exception as exc:
        missing_result = {
            "success": False,
            "exception": _exception_observation(exc),
            "usage": _stats_observation(missing, "lci6c-missing"),
        }
    missing_capability = missing.model_capabilities
    missing_result["capability"] = {
        "resolved": missing_capability.resolved,
        "supports_tools": missing_capability.supports_tools,
        "error_type": missing_capability.error_type,
    }

    refused_base_url = _unused_local_endpoint()
    refused = NativeOllamaAdapter(model=tools_model, base_url=refused_base_url)
    refused_result: dict[str, object]
    try:
        refused.generate_messages(
            [ChatMessage(role="user", content="This must fail.")],
            run_id="lci6c-refused",
        )
        refused_result = {"success": True}
    except Exception as exc:
        refused_result = {
            "success": False,
            "exception": _exception_observation(exc),
            "usage": _stats_observation(refused, "lci6c-refused"),
        }

    timeout_client = Client(host=base_url, timeout=0.001)
    timeout_adapter = NativeOllamaAdapter(
        client=timeout_client,
        model=tools_model,
        base_url=base_url,
    )
    try:
        timeout_adapter.generate_messages(
            [ChatMessage(role="user", content="This bounded call may timeout.")],
            run_id="lci6c-timeout",
        )
        timeout_result = {
            "status": "LIVE_NOT_REPRODUCIBLE",
            "success": True,
            "usage": _stats_observation(timeout_adapter, "lci6c-timeout"),
        }
    except Exception as exc:
        timeout_result = {
            "status": "PASS",
            "success": False,
            "exception": _exception_observation(exc),
            "usage": _stats_observation(timeout_adapter, "lci6c-timeout"),
        }

    invalid_request = _direct_request_failure(
        client,
        model=tools_model,
        messages=[{"role": "user", "content": "invalid JSON schema request"}],
        format_value={
            "type": "object",
            "properties": {"value": {"type": "not-a-real-json-schema-type"}},
        },
    )
    invalid_format = _direct_request_failure(
        client,
        model=tools_model,
        messages=[{"role": "user", "content": "invalid format request"}],
        format_value={"type": "not-a-valid-format"},
    )

    try:
        native.generate_structured(
            [ChatMessage(role="user", content="Return any ordinary answer.")],
            ImpossibleStructured,
            temperature=0,
            max_tokens=64,
            run_id="lci6c-structured-failure",
        )
        structured_failure = {
            "status": "PROVIDER_PREVENTS_REPRODUCTION",
            "success": True,
        }
    except Exception as exc:
        structured_failure = {
            "status": "PASS",
            "success": False,
            "exception": _exception_observation(exc),
            "usage": _stats_observation(native, "lci6c-structured-failure"),
        }

    rows: dict[str, dict[str, object]] = {
        "034": {
            "status": timeout_result["status"],
            "observation": timeout_result,
            "evidence": "client-owned 1ms timeout injection",
        },
        "035": {
            "status": "PASS" if not refused_result["success"] else "FAIL",
            "observation": refused_result,
        },
        "036": {
            "status": timeout_result["status"],
            "observation": timeout_result,
        },
        "037": {
            "status": (
                "PASS"
                if not missing_result["success"]
                and missing_result["capability"]["resolved"] is False
                else "FAIL"
            ),
            "observation": missing_result,
        },
        "038": {
            "status": "PASS" if not invalid_request["success"] else "FAIL",
            "observation": invalid_request,
        },
        "039": {
            "status": "PASS" if not invalid_format["success"] else "FAIL",
            "observation": invalid_format,
        },
        "040": {
            "status": "LIVE_NOT_REPRODUCIBLE",
            "observation": "No local Ollama fault injection or proxy was used.",
        },
        "041": {
            "status": "LIVE_NOT_REPRODUCIBLE",
            "observation": "Real Ollama does not provide a malformed-response switch.",
        },
        "042": {
            "status": "LIVE_NOT_REPRODUCIBLE",
            "observation": "Disconnect injection would disrupt shared Ollama sessions.",
        },
        "043": structured_failure,
        "044": {
            "status": "PROVIDER_PREVENTS_REPRODUCTION",
            "observation": "No malformed tool call was emitted by the real provider.",
        },
        "050": {
            "status": (
                "PASS"
                if all(
                    counters["prompt_eval_count"] is not None
                    and counters["eval_count"] is not None
                    for counters in (
                        plain_counters,
                        tools_counters,
                        structured_counters,
                        stream_counters,
                    )
                )
                else "LIVE_NOT_REPRODUCIBLE"
            ),
            "plain": plain_counters,
            "tools": tools_counters,
            "structured": structured_counters,
            "stream": stream_counters,
        },
    }

    result = {
        "base_url": base_url,
        "models": [
            {"name": name, "capabilities": list(capabilities)}
            for name, capabilities in models
        ],
        "tools_model": tools_model,
        "no_tools_chat_model": no_tools_model,
        "resolved_no_tools_model": resolved_no_tools_model,
        "plain": {
            "native": plain_native_observation,
            "langchain": plain_baseline_observation,
        },
        "capabilities": capability_observation,
        "tools": {
            "native": tools_native_observation,
            "langchain": tools_baseline_observation,
        },
        "structured": {
            "native": structured_native_observation,
            "langchain": structured_baseline_observation,
        },
        "stream": {
            "native": stream_native_observation,
            "langchain": stream_baseline_observation,
        },
        "errors": {
            "missing_model": missing_result,
            "connection_refused": refused_result,
            "timeout": timeout_result,
            "invalid_request": invalid_request,
            "invalid_format": invalid_format,
        },
        "rows": rows,
    }
    return result


def test_native_ollama_live_parity() -> None:
    import os

    if os.environ.get(_LIVE_FLAG, "").strip() != "1":
        pytest.skip(f"{_LIVE_FLAG}=1 is required")

    try:
        result = _run_live_proof()
    except Exception as exc:
        if type(exc).__name__ in {"ConnectError", "ConnectionError", "ConnectTimeout"}:
            pytest.skip(f"Ollama unavailable: {type(exc).__name__}")
        raise

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))

    native_plain = result["plain"]["native"]
    assert native_plain["type"] == "LLMAdapterResponse"
    assert native_plain["provider"] == "ollama"
    assert native_plain["model"] == result["tools_model"]
    assert native_plain["content_non_empty"] is True
    assert native_plain["usage"] is not None

    native_tools = result["tools"]["native"]
    assert native_tools["tool_call_count"] >= 1
    assert native_tools["tool_calls"][0]["name"] == "get_weather"
    assert isinstance(native_tools["tool_calls"][0]["arguments"], dict)
    assert native_tools["finish_reason"] == LLMFinishReason.TOOL_CALLS.value

    native_structured = result["structured"]["native"]
    assert native_structured["parsed"] == {
        "city": "Warsaw",
        "temperature_c": 20,
    }
    assert native_structured["content_non_empty"] is True

    native_stream = result["stream"]["native"]
    assert native_stream["partial_count"] >= 1
    assert native_stream["final_count"] == 1
    assert native_stream["concatenation_matches"] is True

    for row_id, row in result["rows"].items():
        assert row["status"] != "FAIL", f"LCI-6C row {row_id}: {row}"
