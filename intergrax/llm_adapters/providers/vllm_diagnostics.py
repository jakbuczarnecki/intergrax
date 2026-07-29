# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Read-only vLLM server diagnostics for health, version, and cache metrics."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import httpx

VLLM_PINNED_VERSION = "0.23.0"

REQUIRED_METRIC_NAMES: tuple[str, ...] = (
    "vllm:prefix_cache_queries",
    "vllm:prefix_cache_hits",
    "vllm:prompt_tokens_cached",
    "vllm:kv_cache_usage_perc",
)

OPTIONAL_METRIC_NAMES: tuple[str, ...] = (
    "vllm:request_prefill_kv_computed_tokens",
    "vllm:request_prefill_time_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:e2e_request_latency_seconds",
)

_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)(?:\s+\d+)?$"
)


class VllmDiagnosticsError(RuntimeError):
    """Raised when required vLLM diagnostics are unavailable or invalid."""


@dataclass(frozen=True, slots=True)
class VllmHealthStatus:
    healthy: bool
    status_code: int | None = None


@dataclass(frozen=True, slots=True)
class VllmMetricsSnapshot:
    prefix_cache_queries: float
    prefix_cache_hits: float
    prompt_tokens_cached: float
    kv_cache_usage_perc: float
    request_prefill_kv_computed_tokens: float | None = None
    request_prefill_time_seconds: float | None = None
    time_to_first_token_seconds: float | None = None
    e2e_request_latency_seconds: float | None = None

    def metric_delta(self, before: VllmMetricsSnapshot) -> VllmMetricDeltas:
        return VllmMetricDeltas(
            prefix_cache_queries=self.prefix_cache_queries - before.prefix_cache_queries,
            prefix_cache_hits=self.prefix_cache_hits - before.prefix_cache_hits,
            prompt_tokens_cached=self.prompt_tokens_cached - before.prompt_tokens_cached,
            kv_cache_usage_perc=self.kv_cache_usage_perc - before.kv_cache_usage_perc,
            request_prefill_kv_computed_tokens=_optional_delta(
                before.request_prefill_kv_computed_tokens,
                self.request_prefill_kv_computed_tokens,
            ),
            request_prefill_time_seconds=_optional_delta(
                before.request_prefill_time_seconds,
                self.request_prefill_time_seconds,
            ),
            time_to_first_token_seconds=_optional_delta(
                before.time_to_first_token_seconds,
                self.time_to_first_token_seconds,
            ),
            e2e_request_latency_seconds=_optional_delta(
                before.e2e_request_latency_seconds,
                self.e2e_request_latency_seconds,
            ),
        )


@dataclass(frozen=True, slots=True)
class VllmMetricDeltas:
    prefix_cache_queries: float
    prefix_cache_hits: float
    prompt_tokens_cached: float
    kv_cache_usage_perc: float
    request_prefill_kv_computed_tokens: float | None = None
    request_prefill_time_seconds: float | None = None
    time_to_first_token_seconds: float | None = None
    e2e_request_latency_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class VllmDiagnosticsSnapshot:
    health: VllmHealthStatus
    server_version: str | None
    metrics: VllmMetricsSnapshot


def _optional_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return after - before


def derive_vllm_server_root(base_url: str) -> str:
    """Convert an OpenAI-compatible ``/v1`` base URL to the vLLM server root."""
    trimmed = base_url.strip()
    if not trimmed:
        raise VllmDiagnosticsError("base_url must be non-empty")
    parsed = urlparse(trimmed)
    if parsed.scheme not in {"http", "https"}:
        raise VllmDiagnosticsError("base_url must use http or https")
    if not parsed.netloc:
        raise VllmDiagnosticsError("base_url must include host")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    root_path = path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{root_path}"


def _validate_target_host(client: httpx.Client, server_root: str) -> None:
    configured = urlparse(str(client.base_url))
    target = urlparse(server_root)
    if configured.scheme != target.scheme or configured.netloc != target.netloc:
        raise VllmDiagnosticsError("configured client host does not match server root")


def _request_text(
    client: httpx.Client,
    *,
    server_root: str,
    path: str,
    connect_timeout: float,
    read_timeout: float,
) -> tuple[int, str]:
    _validate_target_host(client, server_root)
    timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=read_timeout, pool=connect_timeout)
    response = client.get(
        path,
        timeout=timeout,
        follow_redirects=False,
    )
    return response.status_code, response.text


def fetch_vllm_health(
    client: httpx.Client,
    *,
    server_root: str,
    connect_timeout: float = 5.0,
    read_timeout: float = 10.0,
) -> VllmHealthStatus:
    status_code, _ = _request_text(
        client,
        server_root=server_root,
        path="/health",
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
    )
    return VllmHealthStatus(healthy=status_code == 200, status_code=status_code)


def fetch_vllm_version(
    client: httpx.Client,
    *,
    server_root: str,
    connect_timeout: float = 5.0,
    read_timeout: float = 10.0,
) -> str:
    status_code, body = _request_text(
        client,
        server_root=server_root,
        path="/version",
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
    )
    if status_code != 200:
        raise VllmDiagnosticsError("vLLM /version request failed")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise VllmDiagnosticsError("vLLM /version returned invalid JSON") from None
    version = payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise VllmDiagnosticsError("vLLM /version missing version field")
    return version.strip()


def aggregate_metric_values(series: Mapping[str, Sequence[float]]) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    for name, values in series.items():
        aggregated[name] = float(sum(values))
    return aggregated


def parse_prometheus_metric_series(text: str) -> dict[str, tuple[float, ...]]:
    series: dict[str, list[float]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE.match(line)
        if match is None:
            raise VllmDiagnosticsError("malformed Prometheus metrics payload")
        name = match.group("name")
        value = float(match.group("value"))
        series.setdefault(name, []).append(value)
    return {name: tuple(values) for name, values in series.items()}


def parse_vllm_metrics_snapshot(text: str) -> VllmMetricsSnapshot:
    series = parse_prometheus_metric_series(text)
    aggregated = aggregate_metric_values(series)
    missing = [name for name in REQUIRED_METRIC_NAMES if name not in aggregated]
    if missing:
        raise VllmDiagnosticsError(
            f"required vLLM metrics missing: {', '.join(sorted(missing))}"
        )
    optional_values = {
        name: aggregated.get(name)
        for name in OPTIONAL_METRIC_NAMES
    }
    return VllmMetricsSnapshot(
        prefix_cache_queries=aggregated["vllm:prefix_cache_queries"],
        prefix_cache_hits=aggregated["vllm:prefix_cache_hits"],
        prompt_tokens_cached=aggregated["vllm:prompt_tokens_cached"],
        kv_cache_usage_perc=aggregated["vllm:kv_cache_usage_perc"],
        request_prefill_kv_computed_tokens=optional_values[
            "vllm:request_prefill_kv_computed_tokens"
        ],
        request_prefill_time_seconds=optional_values["vllm:request_prefill_time_seconds"],
        time_to_first_token_seconds=optional_values["vllm:time_to_first_token_seconds"],
        e2e_request_latency_seconds=optional_values["vllm:e2e_request_latency_seconds"],
    )


def fetch_vllm_metrics(
    client: httpx.Client,
    *,
    server_root: str,
    connect_timeout: float = 5.0,
    read_timeout: float = 30.0,
) -> VllmMetricsSnapshot:
    status_code, body = _request_text(
        client,
        server_root=server_root,
        path="/metrics",
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
    )
    if status_code != 200:
        raise VllmDiagnosticsError("vLLM /metrics request failed")
    try:
        return parse_vllm_metrics_snapshot(body)
    except VllmDiagnosticsError:
        raise
    except Exception as exc:
        raise VllmDiagnosticsError("failed to parse vLLM metrics") from exc


def collect_vllm_diagnostics(
    base_url: str,
    *,
    connect_timeout: float = 5.0,
    read_timeout: float = 30.0,
    http_client: httpx.Client | None = None,
) -> VllmDiagnosticsSnapshot:
    server_root = derive_vllm_server_root(base_url)
    owns_client = http_client is None
    client = http_client or httpx.Client(base_url=server_root)
    try:
        health = fetch_vllm_health(
            client,
            server_root=server_root,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )
        if not health.healthy:
            raise VllmDiagnosticsError("vLLM /health reported unhealthy")
        server_version = fetch_vllm_version(
            client,
            server_root=server_root,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )
        metrics = fetch_vllm_metrics(
            client,
            server_root=server_root,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )
        return VllmDiagnosticsSnapshot(
            health=health,
            server_version=server_version,
            metrics=metrics,
        )
    finally:
        if owns_client:
            client.close()


def diagnostics_snapshot_to_safe_dict(snapshot: VllmDiagnosticsSnapshot) -> dict[str, Any]:
    return {
        "healthy": snapshot.health.healthy,
        "health_status_code": snapshot.health.status_code,
        "server_version": snapshot.server_version,
        "metrics": {
            "prefix_cache_queries": snapshot.metrics.prefix_cache_queries,
            "prefix_cache_hits": snapshot.metrics.prefix_cache_hits,
            "prompt_tokens_cached": snapshot.metrics.prompt_tokens_cached,
            "kv_cache_usage_perc": snapshot.metrics.kv_cache_usage_perc,
            "request_prefill_kv_computed_tokens": snapshot.metrics.request_prefill_kv_computed_tokens,
            "request_prefill_time_seconds": snapshot.metrics.request_prefill_time_seconds,
            "time_to_first_token_seconds": snapshot.metrics.time_to_first_token_seconds,
            "e2e_request_latency_seconds": snapshot.metrics.e2e_request_latency_seconds,
        },
    }
