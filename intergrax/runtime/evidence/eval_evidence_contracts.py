# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Eval regression evidence contracts (EVID-EVAL-01)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

EVAL_EVIDENCE_SCHEMA_VERSION = "1.0.0"
EVAL_EVIDENCE_KIND = "eval_regression_evidence"

EVAL_EVIDENCE_OUTPUT_DIR = "build/evidence/eval"
EVAL_EVIDENCE_REPORT_JSON = "report.json"
EVAL_EVIDENCE_REPORT_MARKDOWN = "report.md"


class EvalEvidenceStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    UNAVAILABLE = "UNAVAILABLE"


class EvalEvidenceBasis(str, Enum):
    EXISTING_EVAL_CHECK = "EXISTING_EVAL_CHECK"
    DETERMINISTIC_LOCAL = "DETERMINISTIC_LOCAL"
    NO_NETWORK = "NO_NETWORK"
    NO_PROVIDER_CALLS = "NO_PROVIDER_CALLS"
    NO_REAL_LLM = "NO_REAL_LLM"
    UNKNOWN = "UNKNOWN"


class EvalEvidenceArtifactKind(str, Enum):
    EVAL_REPORT_JSON = "EVAL_REPORT_JSON"
    EVAL_REPORT_MARKDOWN = "EVAL_REPORT_MARKDOWN"
    SOURCE_CHECK = "SOURCE_CHECK"
    OTHER = "OTHER"


class EvalEvidenceCheckKind(str, Enum):
    SCENARIO_LIBRARY = "SCENARIO_LIBRARY"
    REGRESSION_SURFACE = "REGRESSION_SURFACE"
    OTHER = "OTHER"


class EvalEvidenceArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: EvalEvidenceArtifactKind
    path: str
    description: str = ""


class EvalEvidenceCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    check_kind: EvalEvidenceCheckKind
    status: EvalEvidenceStatus
    title: str
    message: str = ""
    basis: list[EvalEvidenceBasis]
    artifact_refs: list[EvalEvidenceArtifactRef] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class EvalEvidenceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = EVAL_EVIDENCE_SCHEMA_VERSION
    report_id: str
    title: str = "Eval regression evidence"
    summary: str = ""
    status: EvalEvidenceStatus
    results: list[EvalEvidenceCheckResult]
    artifact_refs: list[EvalEvidenceArtifactRef] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


_REQUIRED_EVIDENCE_BASIS: frozenset[EvalEvidenceBasis] = frozenset(
    {
        EvalEvidenceBasis.EXISTING_EVAL_CHECK,
        EvalEvidenceBasis.DETERMINISTIC_LOCAL,
        EvalEvidenceBasis.NO_NETWORK,
        EvalEvidenceBasis.NO_PROVIDER_CALLS,
        EvalEvidenceBasis.NO_REAL_LLM,
    }
)

_DEFAULT_EVIDENCE_BASIS: list[EvalEvidenceBasis] = [
    EvalEvidenceBasis.EXISTING_EVAL_CHECK,
    EvalEvidenceBasis.DETERMINISTIC_LOCAL,
    EvalEvidenceBasis.NO_NETWORK,
    EvalEvidenceBasis.NO_PROVIDER_CALLS,
    EvalEvidenceBasis.NO_REAL_LLM,
]


def generate_eval_evidence_report_id(*, root_label: str = "local") -> str:
    """Return a deterministic eval evidence report identifier."""
    return f"eval-evidence:{root_label}"


def create_eval_evidence_check_result(
    *,
    check_id: str,
    check_kind: EvalEvidenceCheckKind,
    status: EvalEvidenceStatus,
    title: str,
    message: str = "",
    basis: list[EvalEvidenceBasis] | None = None,
    artifact_refs: list[EvalEvidenceArtifactRef] | None = None,
    metadata: dict[str, str] | None = None,
) -> EvalEvidenceCheckResult:
    """Build an eval evidence check result with default basis when omitted."""
    return EvalEvidenceCheckResult(
        check_id=check_id,
        check_kind=check_kind,
        status=status,
        title=title,
        message=message,
        basis=list(basis or _DEFAULT_EVIDENCE_BASIS),
        artifact_refs=list(artifact_refs or []),
        metadata=dict(metadata or {}),
    )


def derive_eval_evidence_report_status(
    results: list[EvalEvidenceCheckResult],
) -> EvalEvidenceStatus:
    """Derive overall report status from eval check results."""
    if not results:
        return EvalEvidenceStatus.UNAVAILABLE

    statuses = {result.status for result in results}

    if EvalEvidenceStatus.FAILED in statuses:
        return EvalEvidenceStatus.FAILED

    if all(result.status is EvalEvidenceStatus.PASSED for result in results):
        return EvalEvidenceStatus.PASSED

    if EvalEvidenceStatus.PASSED in statuses and (
        EvalEvidenceStatus.SKIPPED in statuses
        or EvalEvidenceStatus.UNAVAILABLE in statuses
    ):
        return EvalEvidenceStatus.SKIPPED

    return EvalEvidenceStatus.UNAVAILABLE


def validate_eval_evidence_report(report: EvalEvidenceReport) -> None:
    """Raise ``ValueError`` when ``report`` violates eval evidence contracts."""
    violations: list[str] = []

    if not report.report_id.strip():
        violations.append("report_id must not be empty")

    if not report.title.strip():
        violations.append("title must not be empty")

    if not report.results:
        violations.append("results must not be empty")

    check_id_counts: dict[str, int] = {}
    for result in report.results:
        if not result.check_id.strip():
            violations.append("check_id must not be empty")
        else:
            check_id_counts[result.check_id] = check_id_counts.get(result.check_id, 0) + 1

        if not result.title.strip():
            violations.append(
                f"result title must not be empty ({result.check_id or '<empty>'})"
            )

        if not result.basis:
            violations.append(
                f"result basis must not be empty ({result.check_id or '<empty>'})"
            )
        else:
            basis_set = frozenset(result.basis)
            for required in _REQUIRED_EVIDENCE_BASIS:
                if required not in basis_set:
                    violations.append(
                        f"result must include {required.value} "
                        f"({result.check_id or '<empty>'})"
                    )

        for artifact_ref in result.artifact_refs:
            if not artifact_ref.path.strip():
                violations.append(
                    f"result artifact path must not be empty "
                    f"({result.check_id or '<empty>'})"
                )

    for check_id, count in check_id_counts.items():
        if count > 1:
            violations.append(f"duplicate check_id: {check_id}")

    for artifact_ref in report.artifact_refs:
        if not artifact_ref.path.strip():
            violations.append("report artifact path must not be empty")

    expected_status = derive_eval_evidence_report_status(report.results)
    if report.status is not expected_status:
        violations.append(
            f"report status must be {expected_status.value} "
            f"(got {report.status.value})"
        )

    if violations:
        raise ValueError("; ".join(violations))
