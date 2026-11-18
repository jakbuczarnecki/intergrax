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

    ``timeout_sec`` is an admission budget and the per-call SDK timeout: it is clamped to the active
    execution deadline, and a call whose effective budget is not positive is rejected before the provider
    is touched. The effective value is published for one physical attempt
    (``BaseLLMAdapter._effective_call_timeout_sec``); the callable passed to ``_execute`` receives no
    argument, so the value is not forwarded by the base class itself. Each shipped adapter passes it to
    its SDK call through the mechanism that SDK offers (a per-request ``timeout``, ``timeout_ms``,
    request options, HTTP options, or a per-call client); an adapter that cannot do so declares
    ``supports_call_timeout() -> False`` and a call with an effective budget is refused with
    ``LLMCallBudgetNotForwardableError`` instead of running without it. No generic wall-clock wrapper is
    added: the SDK enforces the timeout on its own transport.
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
