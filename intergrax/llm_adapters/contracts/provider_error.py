# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Typed provider-error contract, version 1 (EX-D3-03).

An error raised by a provider SDK call is converted at the adapter boundary
(``BaseLLMAdapter._execute`` / ``_execute_streaming``) into one ``LLMProviderError`` subclass per
``LLMProviderErrorCategory``. The typed error carries the category, the provider slug, the requested
model, the HTTP status (when the SDK reported one), the ``Retry-After`` hint and whether the platform
retry classification treats it as transient. The SDK exception stays the ``__cause__``; its message
text is not copied into the typed message and is not serialized by this contract.

Members of v1 by reference (pre-existing typed errors that pass through the boundary unchanged, so
their identity is preserved): ``LLMRateLimitError`` and ``LLMCircuitOpenError``
(``llm_adapters._shared.resilience``), ``ProviderRateLimitExceededError`` and
``ProviderRateLimitWaitTimeoutError`` (``contracts.provider_rate_limit``), ``RetryBudgetExhaustedError``
(``contracts.retry_budget``), ``LLMQuotaExceeded`` (``llm_adapters.governance.quota``),
``ExecutionProtectedWorkDeniedError`` (``runtime.execution.deadline_provider_guard``),
``CooperativeCancellationAbort`` (``runtime.cancellation.coordinator``),
``LLMCapabilityMismatchError`` (``llm_adapters.contracts.capabilities``) and the dependency-admission
failures. Every ``BaseException`` that is not an ``Exception`` also passes through unchanged.

Not converted in v1 (deviation from plan T-IGX-051-D section 3.3, which converts every non-typed
``Exception``): an ``Exception`` defined in ``intergrax``, and an exception from outside the provider SDK and
transport packages (``BaseLLMAdapter._PROVIDER_SDK_ROOTS``) that carries no provider signal, that is no
provider-hook category, no HTTP status code and no transient class-name word (for example a ``KeyError``,
``ValueError`` or ``json.JSONDecodeError`` raised while a response is handled). Such an error reaches the
caller unchanged, so a consumer that catches only ``LLMProviderError`` does not see it. Reason: existing tests
outside the ticket's write set raise plain local errors through the adapters and assert their type and
message. An exception from a provider SDK or transport package without a signal is typed ``UNKNOWN``.

Class names keep the transient-error words that ``llm_adapters._shared.retry`` reads (``Timeout``,
``Connection``, ``Overloaded``, ``RateLimit``) and the typed error keeps the status code and the
``Retry-After`` hint, so retry and failover classify a typed error as they classified the SDK error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from intergrax.llm_adapters.contracts.capabilities import LLMCapabilityMismatchError

LLM_PROVIDER_ERROR_CONTRACT_VERSION = 1


class LLMProviderErrorCategory(StrEnum):
    """Closed set of provider failure categories (contract v1)."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    INVALID_REQUEST = "invalid_request"
    MODEL_NOT_FOUND = "model_not_found"
    QUOTA_EXHAUSTED = "quota_exhausted"
    OVERLOADED = "overloaded"
    SERVER_ERROR = "server_error"
    CONNECTION = "connection"
    UNKNOWN = "unknown"


class LLMProviderError(RuntimeError):
    """Base of the typed provider errors; one subclass per category."""

    category: ClassVar[LLMProviderErrorCategory] = LLMProviderErrorCategory.UNKNOWN

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
        retriable: bool = False,
    ) -> None:
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.retriable = retriable
        status = "none" if status_code is None else str(status_code)
        super().__init__(
            f"LLM provider error category={self.category.value} provider={provider!r} model={model!r} status={status}"
        )


class LLMProviderTimeoutError(LLMProviderError):
    category: ClassVar[LLMProviderErrorCategory] = LLMProviderErrorCategory.TIMEOUT


class LLMProviderRateLimitError(LLMProviderError):
    category: ClassVar[LLMProviderErrorCategory] = LLMProviderErrorCategory.RATE_LIMIT


class LLMProviderAuthenticationError(LLMProviderError):
    category: ClassVar[LLMProviderErrorCategory] = LLMProviderErrorCategory.AUTHENTICATION


class LLMProviderPermissionError(LLMProviderError):
    category: ClassVar[LLMProviderErrorCategory] = LLMProviderErrorCategory.PERMISSION


class LLMProviderInvalidRequestError(LLMProviderError):
    category: ClassVar[LLMProviderErrorCategory] = LLMProviderErrorCategory.INVALID_REQUEST


class LLMProviderModelNotFoundError(LLMProviderError):
    category: ClassVar[LLMProviderErrorCategory] = LLMProviderErrorCategory.MODEL_NOT_FOUND


class LLMProviderQuotaExhaustedError(LLMProviderError):
    category: ClassVar[LLMProviderErrorCategory] = LLMProviderErrorCategory.QUOTA_EXHAUSTED


class LLMProviderOverloadedError(LLMProviderError):
    """Overloaded or unavailable (503, 504, an SDK ``Overloaded`` error)."""

    category: ClassVar[LLMProviderErrorCategory] = LLMProviderErrorCategory.OVERLOADED


class LLMProviderServerError(LLMProviderError):
    category: ClassVar[LLMProviderErrorCategory] = LLMProviderErrorCategory.SERVER_ERROR


class LLMProviderConnectionError(LLMProviderError):
    category: ClassVar[LLMProviderErrorCategory] = LLMProviderErrorCategory.CONNECTION


class LLMProviderUnknownError(LLMProviderError):
    category: ClassVar[LLMProviderErrorCategory] = LLMProviderErrorCategory.UNKNOWN


def provider_error_class_for(category: LLMProviderErrorCategory) -> type[LLMProviderError]:
    """The typed class of ``category`` (explicit table, no name lookup)."""
    match category:
        case LLMProviderErrorCategory.TIMEOUT:
            return LLMProviderTimeoutError
        case LLMProviderErrorCategory.RATE_LIMIT:
            return LLMProviderRateLimitError
        case LLMProviderErrorCategory.AUTHENTICATION:
            return LLMProviderAuthenticationError
        case LLMProviderErrorCategory.PERMISSION:
            return LLMProviderPermissionError
        case LLMProviderErrorCategory.INVALID_REQUEST:
            return LLMProviderInvalidRequestError
        case LLMProviderErrorCategory.MODEL_NOT_FOUND:
            return LLMProviderModelNotFoundError
        case LLMProviderErrorCategory.QUOTA_EXHAUSTED:
            return LLMProviderQuotaExhaustedError
        case LLMProviderErrorCategory.OVERLOADED:
            return LLMProviderOverloadedError
        case LLMProviderErrorCategory.SERVER_ERROR:
            return LLMProviderServerError
        case LLMProviderErrorCategory.CONNECTION:
            return LLMProviderConnectionError
        case LLMProviderErrorCategory.UNKNOWN:
            return LLMProviderUnknownError


@dataclass(frozen=True, slots=True)
class LLMProviderErrorSignal:
    """What a provider hook read from an SDK error: the category and, when known, the HTTP status."""

    category: LLMProviderErrorCategory
    status_code: int | None = None


#: HTTP status -> category. 408 and 529 are deliberately absent: a generic status error with those codes
#: is not retried by the default retry policy, so mapping them to a transient-named class would change
#: retry and failover behaviour (529 still maps through the SDK class name ``OverloadedError``).
_STATUS_CATEGORIES: dict[int, LLMProviderErrorCategory] = {
    400: LLMProviderErrorCategory.INVALID_REQUEST,
    401: LLMProviderErrorCategory.AUTHENTICATION,
    402: LLMProviderErrorCategory.QUOTA_EXHAUSTED,
    403: LLMProviderErrorCategory.PERMISSION,
    404: LLMProviderErrorCategory.MODEL_NOT_FOUND,
    409: LLMProviderErrorCategory.INVALID_REQUEST,
    413: LLMProviderErrorCategory.INVALID_REQUEST,
    422: LLMProviderErrorCategory.INVALID_REQUEST,
    429: LLMProviderErrorCategory.RATE_LIMIT,
    500: LLMProviderErrorCategory.SERVER_ERROR,
    502: LLMProviderErrorCategory.SERVER_ERROR,
    503: LLMProviderErrorCategory.OVERLOADED,
    504: LLMProviderErrorCategory.OVERLOADED,
}

_NAME_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")

#: Transient class-name words (the same set the retry module reads) -> category, in priority order.
_NAME_CATEGORIES: tuple[tuple[frozenset[str], LLMProviderErrorCategory], ...] = (
    (frozenset({"timeout"}), LLMProviderErrorCategory.TIMEOUT),
    (frozenset({"rate", "ratelimit"}), LLMProviderErrorCategory.RATE_LIMIT),
    (frozenset({"overloaded"}), LLMProviderErrorCategory.OVERLOADED),
    (frozenset({"connection"}), LLMProviderErrorCategory.CONNECTION),
)


def category_from_class_name(class_name: str) -> LLMProviderErrorCategory | None:
    """Category from the transient words of an SDK error class name, or ``None``."""
    words = {word.lower() for word in _NAME_WORD.findall(class_name)}
    for markers, category in _NAME_CATEGORIES:
        if not words.isdisjoint(markers):
            return category
    return None


def category_from_status(status_code: int) -> LLMProviderErrorCategory | None:
    """Category of an HTTP status, or ``None`` when the status carries no category."""
    known = _STATUS_CATEGORIES.get(status_code)
    if known is not None:
        return known
    if 500 <= status_code <= 599 and status_code != 529:
        return LLMProviderErrorCategory.SERVER_ERROR
    return None


class LLMCallBudgetNotForwardableError(LLMCapabilityMismatchError):
    """An effective call timeout is set but the adapter cannot pass it to its SDK (fail closed, Q-RUN-02).

    Raised before the provider is touched. Subclasses ``LLMCapabilityMismatchError`` (no ``LLMFeature``
    member is added, so the declared-capability snapshot is unchanged).

    Not retriable: the refusal is a deterministic capability result, not a transient provider failure. Retry
    and failover (``llm_adapters._shared.retry``) classify an error without a status code by the
    transient words of its class name, so this name deliberately carries none of them (``Timeout``,
    ``Connection``, ``Overloaded``, ``Rate``): retry does not repeat the call and the default failover
    policy stops instead of moving to the next candidate.
    """

    def __init__(self, *, provider: str, model: str) -> None:
        super().__init__(
            (),
            provider=provider,
            model=model,
            detail=(
                f"capability mismatch: provider={provider!r} model={model!r} cannot forward the call timeout "
                "to its SDK; the call is refused instead of running without the budget"
            ),
        )


__all__ = [
    "LLM_PROVIDER_ERROR_CONTRACT_VERSION",
    "LLMCallBudgetNotForwardableError",
    "LLMProviderAuthenticationError",
    "LLMProviderConnectionError",
    "LLMProviderError",
    "LLMProviderErrorCategory",
    "LLMProviderErrorSignal",
    "LLMProviderInvalidRequestError",
    "LLMProviderModelNotFoundError",
    "LLMProviderOverloadedError",
    "LLMProviderPermissionError",
    "LLMProviderQuotaExhaustedError",
    "LLMProviderRateLimitError",
    "LLMProviderServerError",
    "LLMProviderTimeoutError",
    "LLMProviderUnknownError",
    "category_from_class_name",
    "category_from_status",
    "provider_error_class_for",
]
