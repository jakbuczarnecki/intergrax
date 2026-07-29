# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import traceback
from unittest.mock import MagicMock

import httpx
import pytest

from intergrax.llm_adapters._shared.openai_completion_mapping import (
    adapter_response_from_openai_chat_completion,
    usage_from_openai_chat_completion,
    vllm_usage_from_openai_chat_completion,
)
from intergrax.llm_adapters.contracts.provider_extensions import VllmProviderExtensions
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsageValidationError
from intergrax.llm_adapters.providers.vllm_diagnostics import (
    VllmDiagnosticsError,
    aggregate_metric_values,
    derive_vllm_server_root,
    fetch_vllm_health,
    fetch_vllm_metrics,
    fetch_vllm_version,
    parse_prometheus_metric_series,
    parse_vllm_metrics_snapshot,
)


def _usage_mock(*, prompt_tokens: int, cached_tokens: int | None) -> MagicMock:
    usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=1)
    if cached_tokens is None:
        usage.prompt_tokens_details = None
    else:
        usage.prompt_tokens_details = MagicMock(cached_tokens=cached_tokens)
    usage.completion_tokens_details = None
    return usage


def test_cached_tokens_present_and_positive() -> None:
    usage = _usage_mock(prompt_tokens=120, cached_tokens=80)
    mapped = usage_from_openai_chat_completion(usage)
    assert mapped.cached_input_tokens == 80
    assert mapped.uncached_input_tokens == 40


def test_cached_tokens_explicit_zero() -> None:
    usage = _usage_mock(prompt_tokens=50, cached_tokens=0)
    mapped, extensions = vllm_usage_from_openai_chat_completion(usage)
    assert mapped.cached_input_tokens == 0
    assert mapped.uncached_input_tokens == 50
    assert extensions == VllmProviderExtensions(prompt_tokens_details_reported=True)


def test_prompt_tokens_details_absent() -> None:
    usage = _usage_mock(prompt_tokens=50, cached_tokens=None)
    mapped, extensions = vllm_usage_from_openai_chat_completion(usage)
    assert mapped.cached_input_tokens == 0
    assert extensions.prompt_tokens_details_reported is False


def test_invalid_cached_count_greater_than_input() -> None:
    usage = _usage_mock(prompt_tokens=10, cached_tokens=20)
    with pytest.raises(LLMTokenUsageValidationError):
        vllm_usage_from_openai_chat_completion(usage)


def test_vllm_provider_extension_on_adapter_response() -> None:
    usage = _usage_mock(prompt_tokens=30, cached_tokens=12)
    msg = MagicMock(content="ok", tool_calls=None)
    choice = MagicMock(message=msg, finish_reason="stop")
    res = MagicMock(usage=usage, choices=[choice], id="resp-1", system_fingerprint=None)
    response = adapter_response_from_openai_chat_completion(
        res,
        model="Qwen/Qwen2.5-7B-Instruct",
        provider="vllm",
    )
    assert response.provider_extensions is not None
    assert response.provider_extensions.vllm is not None
    assert response.provider_extensions.vllm.prompt_tokens_details_reported is True
    assert response.usage is not None
    assert response.usage.cached_input_tokens == 12


def test_derive_v1_base_url_to_diagnostics_root() -> None:
    assert (
        derive_vllm_server_root("http://127.0.0.1:8100/v1")
        == "http://127.0.0.1:8100"
    )
    assert (
        derive_vllm_server_root("http://127.0.0.1:8100/v1/")
        == "http://127.0.0.1:8100"
    )


def test_health_success_and_failure() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200 if request.url.path == "/health" else 404)
    )
    client = httpx.Client(base_url="http://127.0.0.1:8100", transport=transport)
    healthy = fetch_vllm_health(client, server_root="http://127.0.0.1:8100")
    assert healthy.healthy is True

    failing = httpx.Client(
        base_url="http://127.0.0.1:8100",
        transport=httpx.MockTransport(lambda request: httpx.Response(503)),
    )
    status = fetch_vllm_health(failing, server_root="http://127.0.0.1:8100")
    assert status.healthy is False


def test_required_metric_parsing_and_label_aggregation() -> None:
    text = """
# HELP vllm:prefix_cache_queries
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries{engine="0"} 3.0
vllm:prefix_cache_queries{engine="1"} 2.0
vllm:prefix_cache_hits{engine="0"} 1.0
vllm:prompt_tokens_cached 12.0
vllm:kv_cache_usage_perc 0.25
"""
    snapshot = parse_vllm_metrics_snapshot(text)
    assert snapshot.prefix_cache_queries == 5.0
    assert snapshot.prefix_cache_hits == 1.0
    assert snapshot.prompt_tokens_cached == 12.0
    assert snapshot.kv_cache_usage_perc == 0.25


def test_missing_required_metric_fails_closed() -> None:
    with pytest.raises(VllmDiagnosticsError, match="required vLLM metrics missing"):
        parse_vllm_metrics_snapshot("vllm:prefix_cache_queries 1.0\n")


def test_malformed_metrics_fail_without_body_leak() -> None:
    with pytest.raises(VllmDiagnosticsError, match="malformed Prometheus metrics payload"):
        parse_prometheus_metric_series("not a metric line with secret-body")


def test_fetch_metrics_non_200_fails_closed() -> None:
    client = httpx.Client(
        base_url="http://127.0.0.1:8100",
        transport=httpx.MockTransport(lambda request: httpx.Response(500, text="secret")),
    )
    with pytest.raises(VllmDiagnosticsError, match="/metrics request failed"):
        fetch_vllm_metrics(client, server_root="http://127.0.0.1:8100")


def test_aggregate_metric_values() -> None:
    series = parse_prometheus_metric_series(
        "vllm:prefix_cache_hits{a=\"1\"} 1.0\nvllm:prefix_cache_hits{b=\"2\"} 2.5\n"
    )
    aggregated = aggregate_metric_values(series)
    assert aggregated["vllm:prefix_cache_hits"] == 3.5


def test_fetch_vllm_version_success() -> None:
    client = httpx.Client(
        base_url="http://127.0.0.1:8100",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text='{"version":"0.23.0"}')
        ),
    )
    assert fetch_vllm_version(client, server_root="http://127.0.0.1:8100") == "0.23.0"


def test_fetch_vllm_version_non_200_fails_closed() -> None:
    client = httpx.Client(
        base_url="http://127.0.0.1:8100",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(500, text="SYNTHETIC-SECRET-VERSION-BODY")
        ),
    )
    with pytest.raises(VllmDiagnosticsError, match="/version request failed"):
        fetch_vllm_version(client, server_root="http://127.0.0.1:8100")


def test_fetch_vllm_version_malformed_json_does_not_leak_body() -> None:
    secret = "SYNTHETIC-SECRET-VERSION-BODY"
    client = httpx.Client(
        base_url="http://127.0.0.1:8100",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=secret)
        ),
    )
    with pytest.raises(VllmDiagnosticsError, match="invalid JSON") as exc_info:
        fetch_vllm_version(client, server_root="http://127.0.0.1:8100")

    error = exc_info.value
    assert error.__cause__ is None
    assert secret not in str(error)
    assert secret not in repr(error)
    rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    assert secret not in rendered


def test_fetch_vllm_version_missing_version_field() -> None:
    client = httpx.Client(
        base_url="http://127.0.0.1:8100",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text='{"build":"abc"}')
        ),
    )
    with pytest.raises(VllmDiagnosticsError, match="missing version field"):
        fetch_vllm_version(client, server_root="http://127.0.0.1:8100")


def test_fetch_vllm_version_empty_version_field() -> None:
    client = httpx.Client(
        base_url="http://127.0.0.1:8100",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text='{"version":"   "}')
        ),
    )
    with pytest.raises(VllmDiagnosticsError, match="missing version field"):
        fetch_vllm_version(client, server_root="http://127.0.0.1:8100")
