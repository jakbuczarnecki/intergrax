# © Artur Czarnecki. All rights reserved.

"""HARNESS-02-R2 — canonical wall-time authority vs legacy BudgetEnforcer (Q15)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from intergrax.contracts.execution_deadline.admission import ExecutionProtectedWorkAdmissionResult
from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection
from intergrax.runtime.execution.deadline_scope import (
    bind_active_execution_deadline_scope,
    reset_active_execution_deadline_scope,
)
from intergrax.runtime.execution.protected_work_admission import (
    CanonicalHardProtectedWorkAdmission,
    StaticCancellationView,
    narrow_protected_work_admission_for_child,
)
from intergrax.runtime.nexus.budget.budget_enforcer import BudgetExceededError
from intergrax.runtime.nexus.budget.budget_models import BudgetEnforcementMode, BudgetPolicy, RunBudget
from intergrax.runtime.nexus.budget.budget_ticks import enforce_wall_time_budget
from testing_support.builder import governed_runtime_state_scope

pytestmark = pytest.mark.unit


class _FakeMonotonicClock:
    def __init__(self, value: float) -> None:
        self._value = value

    def monotonic(self) -> float:
        return self._value


def _bind_projection(
    *,
    monotonic_value: float,
    global_deadline_monotonic: float | None,
    deadline_at_utc: datetime | None = None,
) -> tuple:
    monotonic = _FakeMonotonicClock(monotonic_value)
    remaining = (
        0.0
        if global_deadline_monotonic is None
        else max(0.0, global_deadline_monotonic - monotonic_value)
    )
    projection = ExecutionDeadlineProjection(
        deadline_at_utc=deadline_at_utc,
        remaining_seconds=remaining,
        is_expired=remaining <= 0.0,
        global_deadline_monotonic=global_deadline_monotonic,
    )
    admission = CanonicalHardProtectedWorkAdmission(
        projection=projection,
        cancellation_view=StaticCancellationView(cancelled=False),
        monotonic_clock=monotonic,
    )
    return bind_active_execution_deadline_scope(
        projection=projection,
        admission=admission,
        monotonic_clock=monotonic,
    )


def test_q15_1_canonical_future_legacy_started_at_would_expire() -> None:
    """Legacy elapsed math would deny; canonical monotonic still allows."""
    tokens = _bind_projection(
        monotonic_value=50.0,
        global_deadline_monotonic=110.0,
        deadline_at_utc=datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc),
    )
    try:
        with governed_runtime_state_scope("q15-1", bind_budget=True) as st:
            st.context.config.run_budget = RunBudget(max_wall_time_seconds=1.0)
            st.context.config.budget_policy = BudgetPolicy(
                enforcement_mode=BudgetEnforcementMode.ABORT,
            )
            st.started_at_utc = (
                datetime.now(timezone.utc) - timedelta(days=365)
            ).isoformat()
            enforce_wall_time_budget(st)
    finally:
        reset_active_execution_deadline_scope(*tokens)


def test_q15_2_canonical_expired_legacy_started_at_fresh() -> None:
    tokens = _bind_projection(
        monotonic_value=200.0,
        global_deadline_monotonic=100.0,
        deadline_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    try:
        with governed_runtime_state_scope("q15-2", bind_budget=True) as st:
            st.context.config.run_budget = RunBudget(max_wall_time_seconds=3600.0)
            st.context.config.budget_policy = BudgetPolicy(
                enforcement_mode=BudgetEnforcementMode.ABORT,
            )
            st.started_at_utc = datetime.now(timezone.utc).isoformat()
            with pytest.raises(BudgetExceededError, match="max_wall_time_seconds"):
                enforce_wall_time_budget(st)
    finally:
        reset_active_execution_deadline_scope(*tokens)


def test_q15_3_redelivery_fresh_started_at_same_expired_deadline() -> None:
    durable_deadline = datetime(2020, 1, 1, tzinfo=timezone.utc)
    tokens = _bind_projection(
        monotonic_value=500.0,
        global_deadline_monotonic=400.0,
        deadline_at_utc=durable_deadline,
    )
    try:
        with governed_runtime_state_scope("q15-3-a", bind_budget=True) as st:
            st.context.config.run_budget = RunBudget(max_wall_time_seconds=9999.0)
            st.context.config.budget_policy = BudgetPolicy(
                enforcement_mode=BudgetEnforcementMode.ABORT,
            )
            st.started_at_utc = datetime.now(timezone.utc).isoformat()
            with pytest.raises(BudgetExceededError):
                enforce_wall_time_budget(st)
    finally:
        reset_active_execution_deadline_scope(*tokens)


def test_q15_4_child_narrowed_expired_root_started_at_ok() -> None:
    parent_tokens = _bind_projection(
        monotonic_value=10.0,
        global_deadline_monotonic=100.0,
        deadline_at_utc=datetime(2026, 9, 19, 13, 0, tzinfo=timezone.utc),
    )
    try:
        parent_projection = ExecutionDeadlineProjection(
            deadline_at_utc=datetime(2026, 9, 19, 13, 0, tzinfo=timezone.utc),
            remaining_seconds=90.0,
            is_expired=False,
            global_deadline_monotonic=100.0,
        )
        parent_monotonic = _FakeMonotonicClock(10.0)
        parent_admission = CanonicalHardProtectedWorkAdmission(
            projection=parent_projection,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=parent_monotonic,
        )
        child_projection = ExecutionDeadlineProjection(
            deadline_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
            remaining_seconds=0.0,
            is_expired=True,
            global_deadline_monotonic=10.0,
        )
        child_admission = narrow_protected_work_admission_for_child(
            child_projection,
            parent_admission,
            monotonic_clock=_FakeMonotonicClock(10.0),
        )
        child_tokens = bind_active_execution_deadline_scope(
            projection=child_projection,
            admission=child_admission,
            monotonic_clock=_FakeMonotonicClock(10.0),
        )
        try:
            with governed_runtime_state_scope("q15-4", bind_budget=True) as st:
                st.context.config.run_budget = RunBudget(max_wall_time_seconds=3600.0)
                st.context.config.budget_policy = BudgetPolicy(
                    enforcement_mode=BudgetEnforcementMode.ABORT,
                )
                st.started_at_utc = datetime.now(timezone.utc).isoformat()
                with pytest.raises(BudgetExceededError):
                    enforce_wall_time_budget(st)
                assert (
                    child_admission.assert_can_start_protected_work()
                    is ExecutionProtectedWorkAdmissionResult.EXPIRED
                )
        finally:
            reset_active_execution_deadline_scope(*child_tokens)
    finally:
        reset_active_execution_deadline_scope(*parent_tokens)


def test_q15_5_unbounded_canonical_legacy_old_started_at_no_expiry() -> None:
    tokens = _bind_projection(
        monotonic_value=1.0,
        global_deadline_monotonic=None,
        deadline_at_utc=None,
    )
    try:
        with governed_runtime_state_scope("q15-5", bind_budget=True) as st:
            st.context.config.run_budget = RunBudget(max_wall_time_seconds=1.0)
            st.context.config.budget_policy = BudgetPolicy(
                enforcement_mode=BudgetEnforcementMode.ABORT,
            )
            st.started_at_utc = (
                datetime.now(timezone.utc) - timedelta(days=30)
            ).isoformat()
            enforce_wall_time_budget(st)
    finally:
        reset_active_execution_deadline_scope(*tokens)
