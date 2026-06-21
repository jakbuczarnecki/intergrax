# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Cost evidence contracts (EVID-COST-01)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

COST_EVIDENCE_SCHEMA_VERSION = "1.0.0"
COST_EVIDENCE_KIND = "cost_evidence"

COST_EVIDENCE_OUTPUT_DIR = "build/evidence/cost"
COST_EVIDENCE_REPORT_JSON = "report.json"
COST_EVIDENCE_REPORT_MARKDOWN = "report.md"


class CostEvidenceStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    UNAVAILABLE = "UNAVAILABLE"


class CostEvidenceBasis(str, Enum):
    LOCAL_TRACE_BUDGET = "LOCAL_TRACE_BUDGET"
    LOCAL_COST_SIGNAL = "LOCAL_COST_SIGNAL"
    DETERMINISTIC_LOCAL = "DETERMINISTIC_LOCAL"
    REPORT_DERIVED = "REPORT_DERIVED"
    NO_PROVIDER_PRICING = "NO_PROVIDER_PRICING"
    NO_REAL_LLM_METERING = "NO_REAL_LLM_METERING"
    NO_NETWORK = "NO_NETWORK"
    UNKNOWN = "UNKNOWN"


class CostEvidenceArtifactKind(str, Enum):
    COST_REPORT_JSON = "COST_REPORT_JSON"
    COST_REPORT_MARKDOWN = "COST_REPORT_MARKDOWN"
    TRACE_TIMELINE = "TRACE_TIMELINE"
    CORE_CERTIFICATION_REPORT = "CORE_CERTIFICATION_REPORT"
    SOURCE_CHECK = "SOURCE_CHECK"
    OTHER = "OTHER"


class CostEvidenceCheckKind(str, Enum):
    TRACE_BUDGET_FACETS = "TRACE_BUDGET_FACETS"
    CORE_BUDGET_SIGNALS = "CORE_BUDGET_SIGNALS"
    COST_SUMMARY = "COST_SUMMARY"
    OTHER = "OTHER"


class CostEvidenceArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: CostEvidenceArtifactKind
    path: str
    description: str = ""


class CostEvidenceCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    check_kind: CostEvidenceCheckKind
    status: CostEvidenceStatus
    title: str
    message: str = ""
    basis: list[CostEvidenceBasis]
    artifact_refs: list[CostEvidenceArtifactRef] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class CostEvidenceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = COST_EVIDENCE_SCHEMA_VERSION
    report_id: str
    title: str = "Cost evidence"
    summary: str = ""
    status: CostEvidenceStatus
    results: list[CostEvidenceCheckResult]
    artifact_refs: list[CostEvidenceArtifactRef] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


_REQUIRED_SAFETY_BASIS: frozenset[CostEvidenceBasis] = frozenset(
    {
        CostEvidenceBasis.DETERMINISTIC_LOCAL,
        CostEvidenceBasis.NO_PROVIDER_PRICING,
        CostEvidenceBasis.NO_REAL_LLM_METERING,
        CostEvidenceBasis.NO_NETWORK,
    }
)

_LOCAL_SOURCE_BASIS: frozenset[CostEvidenceBasis] = frozenset(
    {
        CostEvidenceBasis.LOCAL_TRACE_BUDGET,
        CostEvidenceBasis.LOCAL_COST_SIGNAL,
        CostEvidenceBasis.REPORT_DERIVED,
    }
)

_DEFAULT_EVIDENCE_BASIS: list[CostEvidenceBasis] = [
    CostEvidenceBasis.LOCAL_TRACE_BUDGET,
    CostEvidenceBasis.DETERMINISTIC_LOCAL,
    CostEvidenceBasis.NO_PROVIDER_PRICING,
    CostEvidenceBasis.NO_REAL_LLM_METERING,
    CostEvidenceBasis.NO_NETWORK,
]


def generate_cost_evidence_report_id(*, root_label: str = "local") -> str:
    """Return a deterministic cost evidence report identifier."""
    return f"cost-evidence:{root_label}"


def create_cost_evidence_check_result(
    *,
    check_id: str,
    check_kind: CostEvidenceCheckKind,
    status: CostEvidenceStatus,
    title: str,
    message: str = "",
    basis: list[CostEvidenceBasis] | None = None,
    artifact_refs: list[CostEvidenceArtifactRef] | None = None,
    metadata: dict[str, str] | None = None,
) -> CostEvidenceCheckResult:
    """Build a cost evidence check result with default basis when omitted."""
    return CostEvidenceCheckResult(
        check_id=check_id,
        check_kind=check_kind,
        status=status,
        title=title,
        message=message,
        basis=list(basis or _DEFAULT_EVIDENCE_BASIS),
        artifact_refs=list(artifact_refs or []),
        metadata=dict(metadata or {}),
    )


def derive_cost_evidence_report_status(
    results: list[CostEvidenceCheckResult],
) -> CostEvidenceStatus:
    """Derive overall report status from cost check results."""
    if not results:
        return CostEvidenceStatus.UNAVAILABLE

    statuses = {result.status for result in results}

    if CostEvidenceStatus.FAILED in statuses:
        return CostEvidenceStatus.FAILED

    if all(result.status is CostEvidenceStatus.PASSED for result in results):
        return CostEvidenceStatus.PASSED

    if CostEvidenceStatus.PASSED in statuses and (
        CostEvidenceStatus.SKIPPED in statuses
        or CostEvidenceStatus.UNAVAILABLE in statuses
    ):
        return CostEvidenceStatus.SKIPPED

    return CostEvidenceStatus.UNAVAILABLE


def validate_cost_evidence_report(report: CostEvidenceReport) -> None:
    """Raise ``ValueError`` when ``report`` violates cost evidence contracts."""
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
            for required in _REQUIRED_SAFETY_BASIS:
                if required not in basis_set:
                    violations.append(
                        f"result must include {required.value} "
                        f"({result.check_id or '<empty>'})"
                    )
            if not basis_set.intersection(_LOCAL_SOURCE_BASIS):
                violations.append(
                    f"result must include at least one local source basis "
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

    expected_status = derive_cost_evidence_report_status(report.results)
    if report.status is not expected_status:
        violations.append(
            f"report status must be {expected_status.value} "
            f"(got {report.status.value})"
        )

    if violations:
        raise ValueError("; ".join(violations))
