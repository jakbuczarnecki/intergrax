# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from intergrax.fastapi_core.execution.governance.models import FailureInfo


@dataclass(frozen=True)
class RetryDecision:
    should_retry: bool
    delay_seconds: float
    reason: str


class FailureClassifier(Protocol):
    """
    Maps an exception into typed FailureInfo.
    """

    def classify(self, exc: Exception) -> FailureInfo: ...


class RetryPolicy(Protocol):
    """
    Decides whether to retry given failure info and attempt number.

    attempt:
      1 for first failure (i.e. first retry decision point),
      2 for second failure, etc.
    """

    def decide(self, attempt: int, failure: FailureInfo) -> RetryDecision: ...


@dataclass(frozen=True)
class RetryBudget:
    """
    Hard cap on number of retries (not total attempts).
    retries_max=0 => never retry
    retries_max=2 => at most 2 retries after first failure
    """
    retries_max: int
