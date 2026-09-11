# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Per-provider rate limiting and circuit breaker for LLM SDK calls."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, DefaultDict, Optional, TypeVar

from intergrax.contracts.provider_rate_limit import (
    ProviderRateLimitExceededError,
    ProviderRateLimitIdentity,
    ProviderRateLimitOverloadMode,
    ProviderRateLimitPolicy,
    ProviderRateLimitWaitTimeoutError,
)
from intergrax.contracts.retry_budget import (
    RetryBudgetExhaustedError,
    RetryBudgetIdentity,
    RetryBudgetKind,
    RetryBudgetLogicalCall,
    RetryBudgetPolicy,
)
from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters._shared.dependency_admission import (
    is_non_retriable_dependency_admission_failure,
)
from intergrax.llm_adapters._shared.provider_rate_limit_port import (
    get_default_local_provider_rate_limit,
    get_llm_provider_rate_limit_port,
)
from intergrax.llm_adapters._shared.provider_retry_budget import (
    get_llm_provider_retry_budget_port,
)
from intergrax.llm_adapters._shared.retry import (
    compute_provider_retry_delay,
    is_retriable_provider_error,
)

T = TypeVar("T")


class LLMRateLimitError(RuntimeError):
    """Raised when per-provider calls-per-minute budget is exceeded."""


class LLMCircuitOpenError(RuntimeError):
    """Raised when the provider circuit breaker is open after repeated failures."""


@dataclass
class _ProviderCircuitState:
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0


_lock = threading.Lock()
_circuit_states: DefaultDict[str, _ProviderCircuitState] = defaultdict(_ProviderCircuitState)
_distributed_limiter: object | None = None


def set_llm_distributed_rate_limiter(limiter: object | None) -> None:
    """Tier-3 bootstrap: Redis-backed limiter from ``create_redis_rate_limiter()``."""
    global _distributed_limiter
    _distributed_limiter = limiter


def _state_key(provider: str) -> str:
    return (provider or "unknown").strip().lower()


def _retry_budget_identity(provider: str) -> RetryBudgetIdentity:
    return RetryBudgetIdentity(
        kind=RetryBudgetKind.LLM_PROVIDER,
        value=_state_key(provider),
    )


def _rate_limit_identity(provider: str) -> ProviderRateLimitIdentity:
    return ProviderRateLimitIdentity(
        kind=RetryBudgetKind.LLM_PROVIDER,
        value=_state_key(provider),
    )


class _InlineRetryBudgetLogicalCall:
    __slots__ = ("_attempts_used", "_max_attempts")

    def __init__(self, max_attempts: int) -> None:
        self._max_attempts = max_attempts
        self._attempts_used = 0

    def begin_physical_attempt(self) -> None:
        if self._attempts_used >= self._max_attempts:
            raise RetryBudgetExhaustedError("retry budget exhausted")
        self._attempts_used += 1

    def complete_physical_attempt(self) -> None:
        return


def _open_retry_budget(
    provider: str,
    config: LLMCallConfig,
) -> RetryBudgetLogicalCall:
    policy = RetryBudgetPolicy(
        max_attempts_per_logical_call=max(1, int(config.max_retries) + 1),
    )
    port = get_llm_provider_retry_budget_port()
    if port is None:
        return _InlineRetryBudgetLogicalCall(policy.max_attempts_per_logical_call)
    return port.open_logical_call(_retry_budget_identity(provider), policy)


def _rate_limit_policy(config: LLMCallConfig) -> ProviderRateLimitPolicy | None:
    limit = config.calls_per_minute
    if not limit or limit <= 0:
        return None
    return ProviderRateLimitPolicy(
        calls_per_minute=int(limit),
        overload_mode=ProviderRateLimitOverloadMode.REJECT,
        wait_timeout_seconds=None,
    )


def _check_circuit(provider: str, config: LLMCallConfig) -> None:
    if config.circuit_breaker_threshold <= 0:
        return
    now = time.monotonic()
    key = _state_key(provider)
    with _lock:
        st = _circuit_states[key]
        if st.circuit_open_until > now:
            raise LLMCircuitOpenError(
                f"LLM circuit open for provider='{provider}' until cooldown elapses."
            )


def _record_success(provider: str) -> None:
    key = _state_key(provider)
    with _lock:
        _circuit_states[key].consecutive_failures = 0
        _circuit_states[key].circuit_open_until = 0.0


def _record_failure(provider: str, config: LLMCallConfig) -> None:
    if config.circuit_breaker_threshold <= 0:
        return
    key = _state_key(provider)
    with _lock:
        st = _circuit_states[key]
        st.consecutive_failures += 1
        if st.consecutive_failures >= config.circuit_breaker_threshold:
            st.circuit_open_until = time.monotonic() + float(config.circuit_breaker_cooldown_sec)
            st.consecutive_failures = 0


def _should_record_circuit_failure(exc: BaseException) -> bool:
    if is_non_retriable_dependency_admission_failure(exc):
        return False
    if isinstance(
        exc,
        (
            LLMRateLimitError,
            ProviderRateLimitExceededError,
            ProviderRateLimitWaitTimeoutError,
            RetryBudgetExhaustedError,
        ),
    ):
        return False
    return True


def reset_provider_resilience(provider: str | None = None) -> None:
    """Test helper — clear rate-limit and circuit state."""
    get_default_local_provider_rate_limit().reset(
        None if provider is None else _rate_limit_identity(provider)
    )
    with _lock:
        if provider is None:
            _circuit_states.clear()
            return
        _circuit_states.pop(_state_key(provider), None)


def _check_distributed_rate_limit(
    provider: str,
    config: LLMCallConfig,
    *,
    tenant_id: Optional[str],
) -> None:
    if not config.use_distributed_rate_limit or _distributed_limiter is None:
        return
    limit = config.calls_per_minute
    if not limit or limit <= 0:
        return
    tenant = (tenant_id or "_platform").strip() or "_platform"
    refill = float(limit) / 60.0
    from intergrax.distributed.contracts.rate_limiter import DistributedRateLimiter

    if not isinstance(_distributed_limiter, DistributedRateLimiter):
        return
    result = _distributed_limiter.acquire(
        tenant_id=tenant,
        key=f"llm:{_state_key(provider)}",
        capacity=int(limit),
        refill_rate_per_second=refill,
    )
    if not result.allowed:
        raise LLMRateLimitError(
            f"Distributed LLM rate limit exceeded for tenant='{tenant}' provider='{provider}'."
        )


def _acquire_local_rate_limit(provider: str, config: LLMCallConfig) -> None:
    policy = _rate_limit_policy(config)
    if policy is None:
        return
    port = get_llm_provider_rate_limit_port()
    try:
        port.acquire_for_physical_attempt(_rate_limit_identity(provider), policy)
    except ProviderRateLimitExceededError as exc:
        raise LLMRateLimitError(str(exc)) from exc
    except ProviderRateLimitWaitTimeoutError as exc:
        raise LLMRateLimitError(str(exc)) from exc


def execute_with_resilience(
    fn: Callable[[], T],
    *,
    provider: str,
    config: LLMCallConfig,
    retry_fn: Callable[[Callable[[], T]], T],
    tenant_id: Optional[str] = None,
) -> T:
    """Retry loop with per-attempt budget, rate limit, and circuit breaker."""
    del retry_fn
    budget = _open_retry_budget(provider, config)
    attempt_index = 0
    last_exc: BaseException | None = None
    while True:
        try:
            budget.begin_physical_attempt()
        except RetryBudgetExhaustedError:
            if last_exc is not None:
                raise last_exc
            raise
        try:
            _check_distributed_rate_limit(provider, config, tenant_id=tenant_id)
            _acquire_local_rate_limit(provider, config)
            _check_circuit(provider, config)
            try:
                result = fn()
            except BaseException as exc:
                if _should_record_circuit_failure(exc):
                    _record_failure(provider, config)
                raise
            _record_success(provider)
            return result
        except BaseException as exc:
            last_exc = exc
            if not is_retriable_provider_error(exc, config):
                raise
            delay = compute_provider_retry_delay(
                attempt_index=attempt_index,
                config=config,
                exc=exc,
            )
            attempt_index += 1
            if delay > 0:
                time.sleep(delay)
        finally:
            budget.complete_physical_attempt()
