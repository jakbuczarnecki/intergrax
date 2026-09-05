# © Artur Czarnecki. All rights reserved.

"""Immutable DS-E2E qualification result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DecisionE2EProofId(StrEnum):
    """Canonical DS-E2E proof identifiers."""

    DS_E2E_01 = "DS-E2E-01"
    DS_E2E_02 = "DS-E2E-02"
    DS_E2E_03 = "DS-E2E-03"
    DS_E2E_04 = "DS-E2E-04"
    DS_E2E_05 = "DS-E2E-05"
    DS_E2E_06 = "DS-E2E-06"
    DS_E2E_07 = "DS-E2E-07"
    DS_E2E_08 = "DS-E2E-08"
    DS_E2E_09 = "DS-E2E-09"
    DS_E2E_10 = "DS-E2E-10"
    DS_E2E_11 = "DS-E2E-11"
    DS_E2E_12 = "DS-E2E-12"
    DS_E2E_13 = "DS-E2E-13"


EXPECTED_DECISION_E2E_PROOFS = frozenset(DecisionE2EProofId)


@dataclass(frozen=True, slots=True)
class QualificationCompleteness:
    """Exact DS-E2E proof-set completeness assessment."""

    expected: frozenset[DecisionE2EProofId]
    actual: frozenset[DecisionE2EProofId]

    @property
    def missing(self) -> frozenset[DecisionE2EProofId]:
        return self.expected - self.actual

    @property
    def unexpected(self) -> frozenset[DecisionE2EProofId]:
        return self.actual - self.expected

    @property
    def complete(self) -> bool:
        return (
            not self.missing
            and not self.unexpected
            and len(self.actual) == len(self.expected)
        )


class QualificationDisposition(StrEnum):
    """Machine-verifiable qualification outcome."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class QualificationEvidenceRef:
    """Safe, redaction-friendly evidence pointer."""

    kind: str
    ref: str
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class DecisionE2EQualificationResult:
    """One DS-E2E proof qualification row."""

    proof_id: DecisionE2EProofId
    disposition: QualificationDisposition
    evidence: tuple[QualificationEvidenceRef, ...]
    reason: str | None = None

    def to_report_row(self) -> dict[str, object]:
        return {
            "proof_id": self.proof_id.value,
            "disposition": self.disposition.value,
            "reason": self.reason,
            "evidence": [
                {"kind": item.kind, "ref": item.ref, "detail": item.detail}
                for item in self.evidence
            ],
        }
