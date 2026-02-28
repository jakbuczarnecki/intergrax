# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RetryPolicy:
    """
    Strongly typed retry policy configuration for Tier-0 execution plane.

    This model is Celery-agnostic and does not implement retry logic.
    It only defines retry semantics that will later be consumed by
    the Celery dispatcher adapter.

    Designed to be injected during worker composition phase.
    """

    max_retries: int
    initial_backoff_seconds: float
    backoff_multiplier: float
    max_backoff_seconds: Optional[float]
    jitter: bool

    retry_on_lock_conflict: bool
    retry_on_handler_exception: bool

    def validate(self) -> None:
        """
        Validates policy configuration.

        Raises:
            ValueError: if configuration is invalid.
        """

        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0.")

        if self.initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds must be >= 0.")

        if self.backoff_multiplier < 1.0:
            raise ValueError("backoff_multiplier must be >= 1.0.")

        if self.max_backoff_seconds is not None:
            if self.max_backoff_seconds <= 0:
                raise ValueError("max_backoff_seconds must be > 0 when provided.")
            
    
    def max_retry_window_seconds(self) -> float:
        """
        Deterministically computes the maximum possible retry window
        (sum of all retry countdowns), excluding jitter.

        This is used to validate lease safety against retry duration.
        """

        total: float = 0.0

        for retry_index in range(self.max_retries):
            backoff = self.initial_backoff_seconds * (
                self.backoff_multiplier ** retry_index
            )

            countdown = min(backoff, self.max_backoff_seconds)

            total += countdown

        return total