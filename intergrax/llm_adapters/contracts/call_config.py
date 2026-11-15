# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Canonical per-adapter call policy configuration (timeouts, retries, resilience)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMCallConfig:
    """
    Per-adapter call policy (retries, resilience and an admission-time call budget).

    Pass via adapter ``**defaults`` or set ``adapter.call_config`` after construction.

    ``timeout_sec`` is an admission budget: it is clamped to the active execution deadline and a call
    whose effective budget is not positive is rejected before the provider is touched. It is not
    forwarded to the provider SDK and does not interrupt a call that is already running; a hung call is
    bounded only by the SDK default or by an outer execution deadline.
    """

    temperature: float | None = None
    max_tokens: int | None = None
    timeout_sec: float | None = None
    max_retries: int = 0
    retry_backoff_sec: float = 0.5
    max_retry_after_sec: float = 30.0
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504)
    rate_limit_wait_timeout_sec: float = 5.0
    calls_per_minute: int | None = None
    circuit_breaker_threshold: int = 0
    circuit_breaker_cooldown_sec: float = 30.0
    use_distributed_rate_limit: bool = False


__all__ = ["LLMCallConfig"]
