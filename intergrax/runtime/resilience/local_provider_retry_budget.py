# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Process-local provider retry budget (W2-C)."""

from __future__ import annotations

import threading
from collections.abc import Mapping

from intergrax.contracts.retry_budget import (
    RetryBudgetExhaustedError,
    RetryBudgetIdentity,
    RetryBudgetLogicalCall,
    RetryBudgetPolicy,
    RetryBudgetPort,
)


class _AggregateAttemptLedger:
    __slots__ = ("_active", "_capacity", "_lock")

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._active = 0
        self._lock = threading.Lock()

    def try_reserve(self) -> bool:
        with self._lock:
            if self._active >= self._capacity:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._active <= 0:
                raise RuntimeError("aggregate retry budget release without reservation")
            self._active -= 1


class _LocalRetryBudgetLogicalCall:
    __slots__ = (
        "_aggregate",
        "_aggregate_reserved",
        "_attempts_used",
        "_max_attempts",
    )

    def __init__(
        self,
        max_attempts: int,
        aggregate: _AggregateAttemptLedger | None,
    ) -> None:
        self._max_attempts = max_attempts
        self._attempts_used = 0
        self._aggregate = aggregate
        self._aggregate_reserved = False

    def begin_physical_attempt(self) -> None:
        if self._attempts_used >= self._max_attempts:
            raise RetryBudgetExhaustedError("per-call retry budget exhausted")
        if self._aggregate is not None:
            if not self._aggregate.try_reserve():
                raise RetryBudgetExhaustedError("aggregate retry budget exhausted")
            self._aggregate_reserved = True
        self._attempts_used += 1

    def complete_physical_attempt(self) -> None:
        if self._aggregate is not None and self._aggregate_reserved:
            self._aggregate.release()
            self._aggregate_reserved = False


class LocalProviderRetryBudget(RetryBudgetPort):
    """Bounded process-local retry budgets keyed by ``RetryBudgetIdentity``."""

    __slots__ = ("_aggregate_by_identity", "_lock")

    def __init__(
        self,
        aggregate_limits: Mapping[RetryBudgetIdentity, int] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._aggregate_by_identity: dict[RetryBudgetIdentity, _AggregateAttemptLedger] = {}
        if aggregate_limits is not None:
            for identity, cap in aggregate_limits.items():
                if not isinstance(identity, RetryBudgetIdentity):
                    raise TypeError("aggregate_limits keys must be RetryBudgetIdentity")
                if not isinstance(cap, int) or cap < 1:
                    raise ValueError("aggregate limit must be int >= 1")
                self._aggregate_by_identity[identity] = _AggregateAttemptLedger(cap)

    def open_logical_call(
        self,
        identity: RetryBudgetIdentity,
        policy: RetryBudgetPolicy,
    ) -> RetryBudgetLogicalCall:
        if not isinstance(identity, RetryBudgetIdentity):
            raise TypeError("identity must be RetryBudgetIdentity")
        if not isinstance(policy, RetryBudgetPolicy):
            raise TypeError("policy must be RetryBudgetPolicy")
        aggregate: _AggregateAttemptLedger | None = None
        if policy.max_aggregate_physical_attempts is not None:
            with self._lock:
                aggregate = self._aggregate_by_identity.get(identity)
                if aggregate is None:
                    aggregate = _AggregateAttemptLedger(
                        policy.max_aggregate_physical_attempts
                    )
                    self._aggregate_by_identity[identity] = aggregate
        elif identity in self._aggregate_by_identity:
            aggregate = self._aggregate_by_identity[identity]
        return _LocalRetryBudgetLogicalCall(
            policy.max_attempts_per_logical_call,
            aggregate,
        )
