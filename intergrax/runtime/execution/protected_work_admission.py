# © Artur Czarnecki. All rights reserved.

"""Canonical hard pre-effect protected work admission (HARNESS-02)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_deadline.admission import (
    ExecutionCancellationView,
    ExecutionProtectedWorkAdmissionPort,
    ExecutionProtectedWorkAdmissionResult,
)
from intergrax.contracts.execution_deadline.clock import MonotonicClockPort
from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection
from intergrax.runtime.execution.live_deadline_evaluator import execution_is_expired_now


@dataclass(frozen=True, slots=True)
class CanonicalHardProtectedWorkAdmission:
    """Non-replaceable deadline/cancellation guard."""

    projection: ExecutionDeadlineProjection
    cancellation_view: ExecutionCancellationView
    monotonic_clock: MonotonicClockPort

    def assert_can_start_protected_work(self) -> ExecutionProtectedWorkAdmissionResult:
        if self.cancellation_view.is_cancelled():
            return ExecutionProtectedWorkAdmissionResult.CANCELLED
        if execution_is_expired_now(self.projection, self.monotonic_clock):
            return ExecutionProtectedWorkAdmissionResult.EXPIRED
        return ExecutionProtectedWorkAdmissionResult.AVAILABLE


@dataclass(frozen=True, slots=True)
class ComposedProtectedWorkAdmission:
    """Canonical hard guard first; optional contributors may only further deny."""

    canonical: ExecutionProtectedWorkAdmissionPort
    contributors: tuple[ExecutionProtectedWorkAdmissionPort, ...] = ()

    def assert_can_start_protected_work(self) -> ExecutionProtectedWorkAdmissionResult:
        decision = self.canonical.assert_can_start_protected_work()
        if decision is not ExecutionProtectedWorkAdmissionResult.AVAILABLE:
            return decision
        for contributor in self.contributors:
            contributor_decision = contributor.assert_can_start_protected_work()
            if contributor_decision is not ExecutionProtectedWorkAdmissionResult.AVAILABLE:
                return contributor_decision
        return ExecutionProtectedWorkAdmissionResult.AVAILABLE


@dataclass(frozen=True, slots=True)
class StaticCancellationView:
    cancelled: bool
    reason: str | None = None

    def is_cancelled(self) -> bool:
        return self.cancelled

    def cancellation_reason(self) -> str | None:
        return self.reason


class ExecutionProtectedWorkAdmissionDeniedError(RuntimeError):
    """Raised when canonical admission blocks protected work."""

    def __init__(self, result: ExecutionProtectedWorkAdmissionResult) -> None:
        self.result = result
        super().__init__(result.value)


@dataclass(frozen=True, slots=True)
class _ParentAdmissionCancellationView:
    """Read-only cancellation probe via parent admission (custom contributors)."""

    parent_admission: ExecutionProtectedWorkAdmissionPort

    def is_cancelled(self) -> bool:
        return (
            self.parent_admission.assert_can_start_protected_work()
            is ExecutionProtectedWorkAdmissionResult.CANCELLED
        )

    def cancellation_reason(self) -> str | None:
        return None


def _cancellation_view_for_child_narrowing(
    parent_admission: ExecutionProtectedWorkAdmissionPort | None,
) -> ExecutionCancellationView:
    if parent_admission is None:
        return StaticCancellationView(cancelled=False)
    if isinstance(parent_admission, ComposedProtectedWorkAdmission):
        if isinstance(parent_admission.canonical, CanonicalHardProtectedWorkAdmission):
            return parent_admission.canonical.cancellation_view
    if isinstance(parent_admission, CanonicalHardProtectedWorkAdmission):
        return parent_admission.cancellation_view
    return _ParentAdmissionCancellationView(parent_admission)


def narrow_protected_work_admission_for_child(
    child_projection: ExecutionDeadlineProjection,
    parent_admission: ExecutionProtectedWorkAdmissionPort | None,
    *,
    monotonic_clock: MonotonicClockPort,
) -> ExecutionProtectedWorkAdmissionPort:
    """Rebind canonical admission to a narrowed child projection (parent scope unchanged)."""
    cancellation_view = _cancellation_view_for_child_narrowing(parent_admission)
    child_canonical = CanonicalHardProtectedWorkAdmission(
        projection=child_projection,
        cancellation_view=cancellation_view,
        monotonic_clock=monotonic_clock,
    )
    if parent_admission is None:
        return child_canonical
    if isinstance(parent_admission, CanonicalHardProtectedWorkAdmission):
        return child_canonical
    if isinstance(parent_admission, ComposedProtectedWorkAdmission):
        return ComposedProtectedWorkAdmission(
            canonical=child_canonical,
            contributors=parent_admission.contributors,
        )
    return ComposedProtectedWorkAdmission(
        canonical=child_canonical,
        contributors=(parent_admission,),
    )


@dataclass(frozen=True, slots=True)
class TaskMetadataCancellationView:
    """Adapter from cooperative task metadata to ExecutionCancellationView."""

    metadata_source: object

    def _live_metadata(self) -> dict[str, object]:
        source = self.metadata_source
        if callable(source):
            raw = source()
        else:
            raw = source
        if raw is None:
            return {}
        return dict(raw)

    def is_cancelled(self) -> bool:
        from intergrax.runtime.cancellation.coordinator import CancellationCoordinator

        return CancellationCoordinator.is_requested(self._live_metadata())

    def cancellation_reason(self) -> str | None:
        from intergrax.runtime.cancellation.coordinator import CANCELLATION_REASON_KEY

        reason = self._live_metadata().get(CANCELLATION_REASON_KEY)
        return reason if isinstance(reason, str) and reason else None
