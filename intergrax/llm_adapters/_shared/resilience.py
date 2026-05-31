# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Per-provider rate limiting and circuit breaker for LLM SDK calls."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, DefaultDict, Dict, TypeVar

from intergrax.llm_adapters._shared.call_config import LLMCallConfig

T = TypeVar("T")


class LLMRateLimitError(RuntimeError):
    """Raised when per-provider calls-per-minute budget is exceeded."""


class LLMCircuitOpenError(RuntimeError):
    """Raised when the provider circuit breaker is open after repeated failures."""


@dataclass
class _ProviderResilienceState:
    call_timestamps: deque[float] = field(default_factory=deque)
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0


_lock = threading.Lock()
_states: DefaultDict[str, _ProviderResilienceState] = defaultdict(_ProviderResilienceState)


def _state_key(provider: str) -> str:
    return (provider or "unknown").strip().lower()


def _check_rate_limit(provider: str, config: LLMCallConfig) -> None:
    limit = config.calls_per_minute
    if not limit or limit <= 0:
        return
    now = time.monotonic()
    key = _state_key(provider)
    with _lock:
        st = _states[key]
        window_start = now - 60.0
        while st.call_timestamps and st.call_timestamps[0] < window_start:
            st.call_timestamps.popleft()
        if len(st.call_timestamps) >= limit:
            raise LLMRateLimitError(
                f"LLM rate limit exceeded for provider='{provider}' ({limit} calls/min)."
            )
        st.call_timestamps.append(now)


def _check_circuit(provider: str, config: LLMCallConfig) -> None:
    if config.circuit_breaker_threshold <= 0:
        return
    now = time.monotonic()
    key = _state_key(provider)
    with _lock:
        st = _states[key]
        if st.circuit_open_until > now:
            raise LLMCircuitOpenError(
                f"LLM circuit open for provider='{provider}' until cooldown elapses."
            )


def _record_success(provider: str) -> None:
    key = _state_key(provider)
    with _lock:
        _states[key].consecutive_failures = 0
        _states[key].circuit_open_until = 0.0


def _record_failure(provider: str, config: LLMCallConfig) -> None:
    if config.circuit_breaker_threshold <= 0:
        return
    key = _state_key(provider)
    with _lock:
        st = _states[key]
        st.consecutive_failures += 1
        if st.consecutive_failures >= config.circuit_breaker_threshold:
            st.circuit_open_until = time.monotonic() + float(config.circuit_breaker_cooldown_sec)
            st.consecutive_failures = 0


def reset_provider_resilience(provider: str | None = None) -> None:
    """Test helper — clear rate-limit and circuit state."""
    with _lock:
        if provider is None:
            _states.clear()
            return
        _states.pop(_state_key(provider), None)


def execute_with_resilience(
    fn: Callable[[], T],
    *,
    provider: str,
    config: LLMCallConfig,
    retry_fn: Callable[[Callable[[], T]], T],
) -> T:
    """Apply rate limit + circuit breaker, then optional retry wrapper."""
    _check_circuit(provider, config)
    _check_rate_limit(provider, config)
    try:
        if config.max_retries > 0:
            result = retry_fn(fn)
        else:
            result = fn()
        _record_success(provider)
        return result
    except BaseException:
        _record_failure(provider, config)
        raise
