# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from concurrent.futures import CancelledError
import random
from dataclasses import dataclass

from intergrax.fastapi_core.execution.governance.contracts import RetryDecision, RetryPolicy, RetryBudget, FailureClassifier
from intergrax.fastapi_core.execution.governance.models import FailureInfo, FailureKind


class DefaultFailureClassifier(FailureClassifier):
    """
    Maps concrete exception types to FailureInfo.
    No string comparisons.
    """

    def classify(self, exc: Exception) -> FailureInfo:
        if isinstance(exc, TimeoutError):
            return FailureInfo(
                error_type=exc.__class__.__name__,
                error_message=str(exc),
                kind=FailureKind.TIMEOUT,
            )

        if isinstance(exc, CancelledError):
            return FailureInfo(
                error_type=exc.__class__.__name__,
                error_message=str(exc),
                kind=FailureKind.CANCELED,
            )

        # Explicit default: permanent
        return FailureInfo(
            error_type=exc.__class__.__name__,
            error_message=str(exc),
            kind=FailureKind.PERMANENT,
        )



@dataclass(frozen=True)
class ExponentialBackoffRetryPolicy(RetryPolicy):
    budget: RetryBudget
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 8.0
    backoff_factor: float = 2.0
    jitter_ratio: float = 0.2

    def decide(self, attempt: int, failure: FailureInfo) -> RetryDecision:
        if attempt > self.budget.retries_max:
            return RetryDecision(
                should_retry=False,
                delay_seconds=0.0,
                reason="retry_budget_exhausted",
            )

        if not failure.is_retryable:
            return RetryDecision(
                should_retry=False,
                delay_seconds=0.0,
                reason=f"non_retryable:{failure.kind.value}",
            )

        if failure.retry_after_seconds is not None:
            return RetryDecision(
                should_retry=True,
                delay_seconds=failure.retry_after_seconds,
                reason="retry_after",
            )

        delay = self.base_delay_seconds * (self.backoff_factor ** (attempt - 1))
        delay = min(delay, self.max_delay_seconds)

        if self.jitter_ratio > 0:
            jitter = delay * self.jitter_ratio
            delay = random.uniform(delay - jitter, delay + jitter)

        return RetryDecision(
            should_retry=True,
            delay_seconds=delay,
            reason="exponential_backoff",
        )

