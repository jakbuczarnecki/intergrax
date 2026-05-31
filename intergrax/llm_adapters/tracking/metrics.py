# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""
In-process LLM call metrics (Prometheus text exposition + JSON snapshot).

Enable globally with ``INTERGRAX_LLM_METRICS_ENABLED=true`` or call
:func:`set_metrics_enabled` from application startup.
"""

from __future__ import annotations

import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List, Optional, Tuple


def _metrics_enabled() -> bool:
    return os.getenv("INTERGRAX_LLM_METRICS_ENABLED", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


_metrics_enabled_override: Optional[bool] = None


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


class LLMMetricsCollector:
    """Thread-safe aggregator keyed by (provider, model)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_key: DefaultDict[Tuple[str, str], _Counter] = defaultdict(_Counter)

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
    ) -> None:
        del run_id, error_type
        key = (provider or "unknown", model or "unknown")
        with self._lock:
            c = self._by_key[key]
            c.calls += 1
            c.input_tokens += int(input_tokens or 0)
            c.output_tokens += int(output_tokens or 0)
            c.duration_ms += int(duration_ms or 0)
            if not success:
                c.errors += 1

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            out: Dict[str, Dict[str, int]] = {}
            for (provider, model), c in self._by_key.items():
                out[f"{provider}:{model}"] = {
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
            for (provider, model), c in self._by_key.items():
                labels = f'provider="{provider}",model="{model}"'
                lines.append(f"intergrax_llm_calls_total{{{labels}}} {c.calls}")
                lines.append(f"intergrax_llm_input_tokens_total{{{labels}}} {c.input_tokens}")
                lines.append(f"intergrax_llm_output_tokens_total{{{labels}}} {c.output_tokens}")
                lines.append(f"intergrax_llm_duration_ms_total{{{labels}}} {c.duration_ms}")
                lines.append(f"intergrax_llm_errors_total{{{labels}}} {c.errors}")
        return lines

    def reset(self) -> None:
        with self._lock:
            self._by_key.clear()


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
