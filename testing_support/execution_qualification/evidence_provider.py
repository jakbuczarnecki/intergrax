# © Artur Czarnecki. All rights reserved.

"""Receipt-based final gate evidence (vendor-neutral, plan-run scoped)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from testing_support.execution_qualification.contracts import (
    QualificationSuiteStatus,
)
from testing_support.execution_qualification.graph_contracts import (
    QualificationGateResult,
    QualificationPlanRunResult,
    QualificationReceiptConflictError,
)


class QualificationEvidenceProvider(Protocol):
    """Resolve terminal statuses for suite and gate IDs from one qualification run."""

    @property
    def run_id(self) -> str: ...

    def suite_status(self, suite_id: str) -> QualificationSuiteStatus: ...

    def gate_status(self, gate_id: str) -> QualificationSuiteStatus: ...


@dataclass(frozen=True, slots=True)
class PlanRunQualificationEvidenceProvider:
    """Adapter from ``QualificationPlanRunResult`` + run identity."""

    result: QualificationPlanRunResult
    run_id: str
    _suite_by_id: dict[str, QualificationSuiteStatus]
    _gate_by_id: dict[str, QualificationSuiteStatus]

    @classmethod
    def from_plan_run(
        cls,
        result: QualificationPlanRunResult,
        *,
        run_id: str,
    ) -> PlanRunQualificationEvidenceProvider:
        suite_by_id = {
            receipt.suite_id: receipt.status for receipt in result.suite_receipts
        }
        gate_by_id = {
            receipt.gate_id: receipt.status for receipt in result.gate_receipts
        }
        return cls(
            result=result,
            run_id=run_id,
            _suite_by_id=suite_by_id,
            _gate_by_id=gate_by_id,
        )

    def suite_status(self, suite_id: str) -> QualificationSuiteStatus:
        if suite_id not in self._suite_by_id:
            raise QualificationReceiptConflictError(
                f"missing suite receipt for {suite_id!r}",
            )
        return self._suite_by_id[suite_id]

    def gate_status(self, gate_id: str) -> QualificationSuiteStatus:
        if gate_id not in self._gate_by_id:
            raise QualificationReceiptConflictError(
                f"missing gate receipt for {gate_id!r}",
            )
        return self._gate_by_id[gate_id]


def resolve_dependency_status(
    provider: QualificationEvidenceProvider,
    dep_id: str,
) -> QualificationSuiteStatus:
    for lookup in (provider.suite_status, provider.gate_status):
        try:
            return lookup(dep_id)
        except QualificationReceiptConflictError:
            continue
    raise QualificationReceiptConflictError(
        f"missing receipt for dependency {dep_id!r}",
    )


def evaluate_final_gate_from_evidence(
    *,
    provider: QualificationEvidenceProvider,
    dependency_ids: tuple[str, ...],
) -> QualificationGateResult:
    """Fail closed when any dependency is missing or not PASS."""
    failure_dependencies: list[str] = []
    for dep_id in dependency_ids:
        dep_status = resolve_dependency_status(provider, dep_id)
        if dep_status is not QualificationSuiteStatus.PASS:
            failure_dependencies.append(dep_id)

    gate_status = (
        QualificationSuiteStatus.PASS
        if not failure_dependencies
        else QualificationSuiteStatus.FAIL
    )
    return QualificationGateResult(
        gate_id="final-gate.evidence",
        status=gate_status,
        consumed_node_ids=dependency_ids,
        mandatory=True,
        failure_dependencies=tuple(failure_dependencies),
    )


def assert_same_run_evidence(
    provider: QualificationEvidenceProvider,
    *,
    expected_run_id: str,
) -> None:
    if provider.run_id != expected_run_id:
        raise QualificationReceiptConflictError(
            f"evidence run_id mismatch: {provider.run_id!r} vs {expected_run_id!r}",
        )
