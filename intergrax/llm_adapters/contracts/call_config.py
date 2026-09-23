# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Canonical per-adapter call policy configuration (timeouts, retries, resilience)."""

from __future__ import annotations

from dataclasses import dataclass
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
    max_retry_after_sec: float = 30.0
    retry_on_status: Tuple[int, ...] = (429, 500, 502, 503, 504)
    rate_limit_wait_timeout_sec: float = 5.0
    calls_per_minute: Optional[int] = None
    circuit_breaker_threshold: int = 0
    circuit_breaker_cooldown_sec: float = 30.0
    use_distributed_rate_limit: bool = False


__all__ = ["LLMCallConfig"]
