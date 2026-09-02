# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed fidelity gate results for functional qualification (DIAG-FUNCTIONAL-Q5)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class QualificationGateStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class QualificationGateResult:
    gate_id: str
    status: QualificationGateStatus
    summary: str | None = None


def core_gate_pass(gate_id: str, *, passed: bool, summary: str | None = None) -> QualificationGateResult:
    return QualificationGateResult(
        gate_id=gate_id,
        status=QualificationGateStatus.PASS if passed else QualificationGateStatus.FAIL,
        summary=summary,
    )


def count_gate_failures(gates: tuple[QualificationGateResult, ...]) -> int:
    return sum(1 for gate in gates if gate.status is QualificationGateStatus.FAIL)


__all__ = [
    "QualificationGateResult",
    "QualificationGateStatus",
    "core_gate_pass",
    "count_gate_failures",
]
