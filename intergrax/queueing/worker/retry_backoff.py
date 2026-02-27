# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import random

from intergrax.queueing.worker.retry_policy import RetryPolicy


def calculate_retry_countdown(
    *,
    policy: RetryPolicy,
    current_retries: int,
) -> float:
    """
    Calculates retry countdown using exponential backoff with optional jitter.

    This function is deterministic when policy.jitter == False.

    Does not depend on Celery.
    """

    base_backoff = policy.initial_backoff_seconds * (
        policy.backoff_multiplier ** current_retries
    )

    if (
        policy.max_backoff_seconds is not None
        and base_backoff > policy.max_backoff_seconds
    ):
        base_backoff = policy.max_backoff_seconds

    if not policy.jitter:
        return base_backoff

    jitter_range = base_backoff * 0.2
    return base_backoff + random.uniform(0.0, jitter_range)