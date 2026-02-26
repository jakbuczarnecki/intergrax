# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.queueing.worker.retry_policy import RetryPolicy
from intergrax.queueing.worker.retry_backoff import calculate_retry_countdown

pytestmark = pytest.mark.unit

@pytest.mark.unit
def test_backoff_without_jitter_is_deterministic() -> None:
    policy = RetryPolicy(
        max_retries=5,
        initial_backoff_seconds=2.0,
        backoff_multiplier=2.0,
        max_backoff_seconds=None,
        jitter=False,
        retry_on_lock_conflict=True,
        retry_on_handler_exception=False,
    )

    # retries = 0 → initial
    assert calculate_retry_countdown(policy=policy, current_retries=0) == 2.0

    # retries = 1 → 2 * 2 = 4
    assert calculate_retry_countdown(policy=policy, current_retries=1) == 4.0

    # retries = 2 → 2 * 2^2 = 8
    assert calculate_retry_countdown(policy=policy, current_retries=2) == 8.0


@pytest.mark.unit
def test_backoff_respects_max_backoff() -> None:
    policy = RetryPolicy(
        max_retries=5,
        initial_backoff_seconds=2.0,
        backoff_multiplier=3.0,
        max_backoff_seconds=10.0,
        jitter=False,
        retry_on_lock_conflict=True,
        retry_on_handler_exception=False,
    )

    # retries = 2 → 2 * 3^2 = 18 → capped to 10
    assert calculate_retry_countdown(policy=policy, current_retries=2) == 10.0


@pytest.mark.unit
def test_backoff_with_jitter_within_expected_range() -> None:
    policy = RetryPolicy(
        max_retries=5,
        initial_backoff_seconds=10.0,
        backoff_multiplier=1.0,
        max_backoff_seconds=None,
        jitter=True,
        retry_on_lock_conflict=True,
        retry_on_handler_exception=False,
    )

    base = 10.0

    # Run multiple times to sample jitter distribution
    for _ in range(20):
        countdown = calculate_retry_countdown(
            policy=policy,
            current_retries=0,
        )

        assert countdown >= base
        assert countdown <= base * 1.2