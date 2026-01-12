# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Unit tests for TokenBucket rate limiter.

These tests define the behavioral contract of the rate limiter:
- token consumption semantics,
- refill behavior over time,
- capacity invariants,
- and handling of invalid inputs.

The goal is to ensure deterministic, safe, and predictable throttling behavior
across the entire websearch subsystem.

Notes on determinism:
- We patch time.monotonic() with a controlled fake clock to avoid flaky tests.
- No network calls, no real sleeping, no dependence on wall-clock time.
"""

from __future__ import annotations

import time

import pytest

from intergrax.websearch.utils.rate_limit import TokenBucket


pytestmark = pytest.mark.unit


class _FakeMonotonic:
    """
    A deterministic monotonic clock used to control time flow in tests.

    TokenBucket uses time.monotonic() to measure elapsed time for refills.
    This fake clock enables precise, reproducible tests for refill behavior
    without relying on the actual system clock.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        """
        Return the current fake time (seconds).
        This is injected as time.monotonic().
        """
        return self._t

    def advance(self, seconds: float) -> None:
        """
        Advance fake time forward by the given number of seconds.

        Negative time travel is forbidden to keep the tests aligned with
        monotonic clock assumptions.
        """
        if seconds < 0:
            raise ValueError("Cannot advance time backwards.")
        self._t += float(seconds)


def test_token_bucket_rejects_invalid_parameters() -> None:
    """
    The rate limiter must fail fast on invalid configuration.

    Contract:
    - rate_per_sec must be strictly positive,
    - capacity must be strictly positive.

    Failing fast prevents silent misconfiguration that could disable throttling
    or cause unpredictable behavior in production.
    """
    with pytest.raises(ValueError):
        TokenBucket(rate_per_sec=0.0, capacity=1)

    with pytest.raises(ValueError):
        TokenBucket(rate_per_sec=-1.0, capacity=1)

    with pytest.raises(ValueError):
        TokenBucket(rate_per_sec=1.0, capacity=0)

    with pytest.raises(ValueError):
        TokenBucket(rate_per_sec=1.0, capacity=-5)


def test_try_acquire_consumes_tokens_until_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    try_acquire() should consume tokens immediately until the bucket is empty.

    Contract:
    - When enough tokens are available, try_acquire(n) returns True and deducts n.
    - When insufficient tokens remain, try_acquire(n) returns False and does not
      allow the request.

    This is the core safety behavior that enforces rate limits deterministically.
    """
    fake = _FakeMonotonic(start=100.0)
    monkeypatch.setattr(time, "monotonic", fake.now)

    bucket = TokenBucket(rate_per_sec=1.0, capacity=3)

    assert bucket.try_acquire(1) is True
    assert bucket.try_acquire(1) is True
    assert bucket.try_acquire(1) is True
    assert bucket.try_acquire(1) is False  # Bucket is empty.


def test_try_acquire_refills_over_time_and_never_exceeds_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Tokens should refill proportionally to elapsed time, but never exceed capacity.

    Contract:
    - Refill amount is based on elapsed time * rate_per_sec.
    - Even if the system is idle for a long time, the bucket must cap at capacity.

    This protects against burst amplification after inactivity and guarantees
    predictable maximum burst size.
    """
    fake = _FakeMonotonic(start=100.0)
    monkeypatch.setattr(time, "monotonic", fake.now)

    # rate=2 tokens/sec, capacity=5
    bucket = TokenBucket(rate_per_sec=2.0, capacity=5)

    # Drain fully.
    assert bucket.try_acquire(5) is True
    assert bucket.try_acquire(1) is False

    # After 0.5s -> refill 1 token (2 * 0.5 = 1)
    fake.advance(0.5)
    assert bucket.try_acquire(1) is True
    assert bucket.try_acquire(1) is False

    # After 10s -> would refill 20 tokens, but must cap at capacity (5)
    fake.advance(10.0)
    assert bucket.try_acquire(5) is True
    assert bucket.try_acquire(1) is False


def test_try_acquire_tokens_leq_zero_are_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Non-positive token requests should be treated as no-ops.

    Contract:
    - try_acquire(0) returns True and does not change bucket state.
    - try_acquire(negative) returns True and does not change bucket state.

    Rationale:
    - This prevents surprising failures if higher-level code passes a computed
      token amount that can be zero in edge cases.
    - It makes the API more robust and easier to use safely.

    If your intended contract is different (e.g., raising ValueError), change
    this test accordingly to enforce that contract explicitly.
    """
    fake = _FakeMonotonic(start=100.0)
    monkeypatch.setattr(time, "monotonic", fake.now)

    bucket = TokenBucket(rate_per_sec=1.0, capacity=1)

    assert bucket.try_acquire(0) is True
    assert bucket.try_acquire(-10) is True

    # Still has full capacity (1).
    assert bucket.try_acquire(1) is True
    assert bucket.try_acquire(1) is False
