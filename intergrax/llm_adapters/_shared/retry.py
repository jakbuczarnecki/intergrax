# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations
from intergrax.utils import attribute_access

import time
from typing import Callable, TypeVar

from intergrax.llm_adapters._shared.call_config import LLMCallConfig

T = TypeVar("T")


def is_retriable_provider_error(exc: BaseException, config: LLMCallConfig) -> bool:
    """Return True when ``exc`` warrants retry or profile failover."""
    return _is_retryable(exc, config)


def _is_retryable(exc: BaseException, config: LLMCallConfig) -> bool:
    status = attribute_access.optional(exc, "status_code", None)
    if status is None:
        response = attribute_access.optional(exc, "response", None)
        if response is not None:
            status = attribute_access.optional(response, "status_code", None)
    if isinstance(status, int) and status in config.retry_on_status:
        return True
    name = type(exc).__name__.lower()
    return any(token in name for token in ("timeout", "connection", "rate", "overloaded"))


def call_with_retry(fn: Callable[[], T], *, config: LLMCallConfig) -> T:
    """Invoke ``fn`` with bounded retries for transient provider errors."""
    attempts = max(1, int(config.max_retries) + 1)
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except BaseException as exc:
            last_exc = exc
            if attempt >= attempts - 1 or not _is_retryable(exc, config):
                raise
            time.sleep(config.retry_backoff_sec * (2**attempt))
    assert last_exc is not None
    raise last_exc
