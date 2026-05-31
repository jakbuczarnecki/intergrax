# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class LLMCallConfig:
    """
    Per-adapter call policy (timeouts, retries).

    Pass via adapter ``**defaults`` or set ``adapter.call_config`` after construction.
    """

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout_sec: Optional[float] = None
    max_retries: int = 0
    retry_backoff_sec: float = 0.5
    retry_on_status: Tuple[int, ...] = (429, 500, 502, 503, 504)
    calls_per_minute: Optional[int] = None
    circuit_breaker_threshold: int = 0
    circuit_breaker_cooldown_sec: float = 30.0


def parse_call_config(defaults: dict) -> LLMCallConfig:
    """Extract ``LLMCallConfig`` fields from adapter constructor kwargs."""
    known = {
        "temperature",
        "max_tokens",
        "timeout_sec",
        "max_retries",
        "retry_backoff_sec",
        "retry_on_status",
        "calls_per_minute",
        "circuit_breaker_threshold",
        "circuit_breaker_cooldown_sec",
    }
    cfg_kwargs = {k: defaults[k] for k in known if k in defaults}
    return LLMCallConfig(**cfg_kwargs)
