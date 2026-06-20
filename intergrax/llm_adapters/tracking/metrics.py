# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""
In-process LLM call metrics (Prometheus text + OTLP-style JSON snapshot).

Enable with ``INTERGRAX_LLM_METRICS_ENABLED=true`` or :func:`set_metrics_enabled`.
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from intergrax.llm_adapters.tracking.context import get_llm_tenant_id

_metrics_enabled_override: Optional[bool] = None


def _metrics_enabled() -> bool:
    return os.getenv("INTERGRAX_LLM_METRICS_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def set_metrics_enabled(enabled: bool) -> None:
    global _metrics_enabled_override
    _metrics_enabled_override = enabled


def is_metrics_enabled() -> bool:
    if _metrics_enabled_override is not None:
        return _metrics_enabled_override
    return _metrics_enabled()


@dataclass
class _Counter:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    errors: int = 0


@dataclass
class _CatalogMissCounter:
    count: int = 0


class LLMMetricsCollector:
    """Thread-safe aggregator keyed by (tenant_id, provider, model)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_key: DefaultDict[Tuple[str, str, str], _Counter] = defaultdict(_Counter)
        self._catalog_miss_by_key: DefaultDict[Tuple[str, str, str, str], _CatalogMissCounter] = (
            defaultdict(_CatalogMissCounter)
        )

    def record(
        self,
        *,
        provider: str,
        model: str,
        run_id: str,
        input_tokens: int,
        output_tokens: int,
        duration_ms: int,
        success: bool,
        error_type: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        del run_id, error_type
        tenant = (tenant_id or get_llm_tenant_id() or "_platform").strip() or "_platform"
        key = (tenant, provider or "unknown", model or "unknown")
        with self._lock:
            c = self._by_key[key]
            c.calls += 1
            c.input_tokens += int(input_tokens or 0)
            c.output_tokens += int(output_tokens or 0)
            c.duration_ms += int(duration_ms or 0)
            if not success:
                c.errors += 1

    def record_catalog_miss(
        self,
        *,
        provider: str,
        model: str,
        resolution_tier: str,
        tenant_id: Optional[str] = None,
    ) -> None:
        tenant = (tenant_id or get_llm_tenant_id() or "_platform").strip() or "_platform"
        key = (
            tenant,
            provider or "unknown",
            model or "unknown",
            resolution_tier or "unknown",
        )
        with self._lock:
            self._catalog_miss_by_key[key].count += 1

    def tenant_total_tokens(self, tenant_id: str) -> int:
        """Sum input+output tokens for one tenant across all providers/models."""
        tenant = (tenant_id or "_platform").strip() or "_platform"
        with self._lock:
            total = 0
            for (t, _provider, _model), c in self._by_key.items():
                if t == tenant:
                    total += c.input_tokens + c.output_tokens
            return total

    def snapshot_for_tenant(self, tenant_id: str) -> Dict[str, Dict[str, int]]:
        """Snapshot limited to one tenant (for governance / observability export)."""
        tenant = (tenant_id or "_platform").strip() or "_platform"
        with self._lock:
            out: Dict[str, Dict[str, int]] = {}
            for (t, provider, model), c in self._by_key.items():
                if t != tenant:
                    continue
                out[f"{provider}:{model}"] = {
                    "calls": c.calls,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "duration_ms": c.duration_ms,
                    "errors": c.errors,
                }
            return out

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            out: Dict[str, Dict[str, int]] = {}
            for (tenant, provider, model), c in self._by_key.items():
                out[f"{tenant}:{provider}:{model}"] = {
                    "calls": c.calls,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "duration_ms": c.duration_ms,
                    "errors": c.errors,
                }
            return out

    def prometheus_lines(self) -> List[str]:
        lines: List[str] = []
        with self._lock:
            for (tenant, provider, model), c in self._by_key.items():
                labels = (
                    f'tenant_id="{tenant}",provider="{provider}",model="{model}"'
                )
                lines.append(f"intergrax_llm_calls_total{{{labels}}} {c.calls}")
                lines.append(f"intergrax_llm_input_tokens_total{{{labels}}} {c.input_tokens}")
                lines.append(f"intergrax_llm_output_tokens_total{{{labels}}} {c.output_tokens}")
                lines.append(f"intergrax_llm_duration_ms_total{{{labels}}} {c.duration_ms}")
                lines.append(f"intergrax_llm_errors_total{{{labels}}} {c.errors}")
            for (tenant, provider, model, tier), c in self._catalog_miss_by_key.items():
                labels = (
                    f'tenant_id="{tenant}",provider="{provider}",model="{model}",'
                    f'resolution_tier="{tier}"'
                )
                lines.append(f"intergrax_llm_catalog_miss_total{{{labels}}} {c.count}")
        return lines

    def otlp_resource_metrics(self) -> Dict[str, Any]:
        """
        OTLP-inspired JSON snapshot for observability backends / debug export.

        Not a full OTLP protobuf encoder — stable JSON for HTTP ``/metrics/llm`` routes.
        """
        metrics: List[Dict[str, Any]] = []
        with self._lock:
            for (tenant, provider, model), c in self._by_key.items():
                attrs = {
                    "tenant_id": tenant,
                    "provider": provider,
                    "model": model,
                }
                for name, value in (
                    ("llm.calls", c.calls),
                    ("llm.input_tokens", c.input_tokens),
                    ("llm.output_tokens", c.output_tokens),
                    ("llm.duration_ms", c.duration_ms),
                    ("llm.errors", c.errors),
                ):
                    metrics.append(
                        {
                            "name": name,
                            "sum": {"asInt": int(value)},
                            "attributes": attrs,
                        }
                    )
            for (tenant, provider, model, tier), c in self._catalog_miss_by_key.items():
                metrics.append(
                    {
                        "name": "llm.catalog_miss",
                        "sum": {"asInt": int(c.count)},
                        "attributes": {
                            "tenant_id": tenant,
                            "provider": provider,
                            "model": model,
                            "resolution_tier": tier,
                        },
                    }
                )
        return {
            "resourceMetrics": [
                {
                    "scopeMetrics": [
                        {
                            "scope": {"name": "intergrax.llm_adapters"},
                            "metrics": metrics,
                        }
                    ]
                }
            ]
        }

    def reset(self) -> None:
        with self._lock:
            self._by_key.clear()
            self._catalog_miss_by_key.clear()


_collector = LLMMetricsCollector()


def get_llm_metrics_collector() -> LLMMetricsCollector:
    return _collector


def record_llm_call(
    *,
    provider: str,
    model: str,
    run_id: str,
    input_tokens: int,
    output_tokens: int,
    duration_ms: int,
    success: bool,
    error_type: Optional[str] = None,
) -> None:
    if not is_metrics_enabled():
        return
    _collector.record(
        provider=provider,
        model=model,
        run_id=run_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        duration_ms=duration_ms,
        success=success,
        error_type=error_type,
    )


def record_catalog_miss(
    *,
    provider: str,
    model: str,
    resolution_tier: str,
    tenant_id: Optional[str] = None,
) -> None:
    if not is_metrics_enabled():
        return
    _collector.record_catalog_miss(
        provider=provider,
        model=model,
        resolution_tier=resolution_tier,
        tenant_id=tenant_id,
    )
