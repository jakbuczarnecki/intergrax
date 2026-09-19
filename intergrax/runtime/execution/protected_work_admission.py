# © Artur Czarnecki. All rights reserved.

"""Canonical hard pre-effect protected work admission (HARNESS-02)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_deadline.admission import (
    ExecutionCancellationView,
    ExecutionProtectedWorkAdmissionPort,
    ExecutionProtectedWorkAdmissionResult,
)
from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection


@dataclass(frozen=True, slots=True)
class CanonicalHardProtectedWorkAdmission:
    """Non-replaceable deadline/cancellation guard."""

    projection: ExecutionDeadlineProjection
    cancellation_view: ExecutionCancellationView

    def assert_can_start_protected_work(self) -> ExecutionProtectedWorkAdmissionResult:
        if self.cancellation_view.is_cancelled():
            return ExecutionProtectedWorkAdmissionResult.CANCELLED
        if self.projection.is_expired:
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


def narrow_protected_work_admission_for_child(
    child_projection: ExecutionDeadlineProjection,
    parent_admission: ExecutionProtectedWorkAdmissionPort | None,
) -> ExecutionProtectedWorkAdmissionPort:
    """Rebind canonical admission to a narrowed child projection (parent scope unchanged)."""
    if parent_admission is None:
        return CanonicalHardProtectedWorkAdmission(
            projection=child_projection,
            cancellation_view=StaticCancellationView(cancelled=False),
        )
    if isinstance(parent_admission, ComposedProtectedWorkAdmission):
        return ComposedProtectedWorkAdmission(
            canonical=CanonicalHardProtectedWorkAdmission(
                projection=child_projection,
                cancellation_view=parent_admission.canonical.cancellation_view,
            ),
            contributors=parent_admission.contributors,
        )
    if isinstance(parent_admission, CanonicalHardProtectedWorkAdmission):
        return CanonicalHardProtectedWorkAdmission(
            projection=child_projection,
            cancellation_view=parent_admission.cancellation_view,
        )
    return ComposedProtectedWorkAdmission(
        canonical=CanonicalHardProtectedWorkAdmission(
            projection=child_projection,
            cancellation_view=StaticCancellationView(cancelled=False),
        ),
        contributors=(parent_admission,),
    )


@dataclass(frozen=True, slots=True)
class TaskMetadataCancellationView:
    """Adapter from cooperative task metadata to ExecutionCancellationView."""

    metadata: dict[str, object]

    def is_cancelled(self) -> bool:
        from intergrax.runtime.cancellation.coordinator import CancellationCoordinator

        return CancellationCoordinator.is_requested(self.metadata)

    def cancellation_reason(self) -> str | None:
        from intergrax.runtime.cancellation.coordinator import CANCELLATION_REASON_KEY

        raw = self.metadata.get(CANCELLATION_REASON_KEY)
        return raw if isinstance(raw, str) and raw else None
