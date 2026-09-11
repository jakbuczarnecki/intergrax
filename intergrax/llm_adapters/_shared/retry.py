# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

from intergrax.contracts.provider_rate_limit import (
    ProviderRateLimitExceededError,
    ProviderRateLimitWaitTimeoutError,
)
from intergrax.contracts.retry_budget import RetryBudgetExhaustedError
from intergrax.llm_adapters._shared.call_config import LLMCallConfig
from intergrax.llm_adapters._shared.dependency_admission import (
    is_non_retriable_dependency_admission_failure,
)
from intergrax.runtime.cancellation.coordinator import (
    CooperativeCancellationAbort,
    cooperative_delay_seconds,
)
from intergrax.utils import attribute_access

T = TypeVar("T")


def is_retriable_provider_error(exc: BaseException, config: LLMCallConfig) -> bool:
    """Return True when ``exc`` warrants retry or profile failover."""
    return _is_retryable(exc, config)


def _http_status_code(exc: BaseException) -> int | None:
    status = attribute_access.optional(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = attribute_access.optional(exc, "response", None)
    if response is not None:
        nested = attribute_access.optional(response, "status_code", None)
        if isinstance(nested, int):
            return nested
    return None


def extract_retry_after_seconds(exc: BaseException) -> float | None:
    """Read Retry-After from provider error/response when present."""
    direct = attribute_access.optional(exc, "retry_after_seconds", None)
    if isinstance(direct, (int, float)) and not isinstance(direct, bool):
        return float(direct)
    response = attribute_access.optional(exc, "response", None)
    if response is None:
        return None
    headers = attribute_access.optional(response, "headers", None)
    if headers is None:
        return None
    raw = None
    if isinstance(headers, Mapping):
        raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.isdigit():
            return float(stripped)
    return None


def compute_provider_retry_delay(
    *,
    attempt_index: int,
    config: LLMCallConfig,
    exc: BaseException | None,
) -> float:
    retry_after = extract_retry_after_seconds(exc) if exc is not None else None
    if retry_after is not None:
        return min(max(retry_after, 0.0), float(config.max_retry_after_sec))
    return float(config.retry_backoff_sec) * (2**attempt_index)


def _is_retryable(exc: BaseException, config: LLMCallConfig) -> bool:
    if is_non_retriable_dependency_admission_failure(exc):
        return False
    if isinstance(
        exc,
        (
            ProviderRateLimitExceededError,
            ProviderRateLimitWaitTimeoutError,
            RetryBudgetExhaustedError,
        ),
    ):
        return False
    if type(exc).__name__ == "LLMRateLimitError":
        return False
    status = _http_status_code(exc)
    if status is not None and status in config.retry_on_status:
        return True
    name = type(exc).__name__.lower()
    return any(token in name for token in ("timeout", "connection", "rate", "overloaded"))


def call_with_retry(
    fn: Callable[[], T],
    *,
    config: LLMCallConfig,
    should_abort: Callable[[], bool] | None = None,
) -> T:
    """Invoke ``fn`` with bounded retries for transient provider errors."""
    attempts = max(1, int(config.max_retries) + 1)
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except CooperativeCancellationAbort:
            raise
        except BaseException as exc:
            last_exc = exc
            if attempt >= attempts - 1 or not _is_retryable(exc, config):
                raise
            cooperative_delay_seconds(
                compute_provider_retry_delay(
                    attempt_index=attempt,
                    config=config,
                    exc=exc,
                ),
                should_abort=should_abort,
            )
    assert last_exc is not None
    raise last_exc
