# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Eval regression evidence contracts (EVID-EVAL-01)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.evaluation_contracts import (
    EvalComparison,
    EvalDigestMismatchError,
    EvalRun,
    EvalTenantBoundaryError,
    EvaluatorOutcomeStatus,
)

EVAL_EVIDENCE_SCHEMA_VERSION = "1.0.0"
EVAL_EVIDENCE_KIND = "eval_regression_evidence"

EVAL_EVIDENCE_OUTPUT_DIR = "build/evidence/eval"
EVAL_EVIDENCE_REPORT_JSON = "report.json"
EVAL_EVIDENCE_REPORT_MARKDOWN = "report.md"


class EvalEvidenceStatus(str, Enum):  # noqa: UP042 - str mixin kept: str() output is part of the public contract
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    UNAVAILABLE = "UNAVAILABLE"


class EvalEvidenceBasis(str, Enum):  # noqa: UP042 - str mixin kept: str() output is part of the public contract
    EXISTING_EVAL_CHECK = "EXISTING_EVAL_CHECK"
    DETERMINISTIC_LOCAL = "DETERMINISTIC_LOCAL"
    NO_NETWORK = "NO_NETWORK"
    NO_PROVIDER_CALLS = "NO_PROVIDER_CALLS"
    NO_REAL_LLM = "NO_REAL_LLM"
    UNKNOWN = "UNKNOWN"


class EvalEvidenceArtifactKind(str, Enum):  # noqa: UP042 - str mixin kept: str() output is part of the public contract
    EVAL_REPORT_JSON = "EVAL_REPORT_JSON"
    EVAL_REPORT_MARKDOWN = "EVAL_REPORT_MARKDOWN"
    SOURCE_CHECK = "SOURCE_CHECK"
    OTHER = "OTHER"


class EvalEvidenceCheckKind(str, Enum):  # noqa: UP042 - str mixin kept: str() output is part of the public contract
    SCENARIO_LIBRARY = "SCENARIO_LIBRARY"
    REGRESSION_SURFACE = "REGRESSION_SURFACE"
    EVAL_RUN = "EVAL_RUN"
    EVAL_COMPARISON = "EVAL_COMPARISON"
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

# Typed run and comparison evidence may stem from a model-assisted judge or a live
# executor, so NO_REAL_LLM / NO_PROVIDER_CALLS cannot be claimed for them.
_RUN_EVIDENCE_REQUIRED_BASIS: frozenset[EvalEvidenceBasis] = frozenset({EvalEvidenceBasis.EXISTING_EVAL_CHECK})

_TYPED_EVIDENCE_KINDS: frozenset[EvalEvidenceCheckKind] = frozenset(
    {EvalEvidenceCheckKind.EVAL_RUN, EvalEvidenceCheckKind.EVAL_COMPARISON}
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
        EvalEvidenceStatus.SKIPPED in statuses or EvalEvidenceStatus.UNAVAILABLE in statuses
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
            violations.append(f"result title must not be empty ({result.check_id or '<empty>'})")

        if not result.basis:
            violations.append(f"result basis must not be empty ({result.check_id or '<empty>'})")
        else:
            basis_set = frozenset(result.basis)
            required_basis = (
                _RUN_EVIDENCE_REQUIRED_BASIS if result.check_kind in _TYPED_EVIDENCE_KINDS else _REQUIRED_EVIDENCE_BASIS
            )
            for required in required_basis:
                if required not in basis_set:
                    violations.append(f"result must include {required.value} ({result.check_id or '<empty>'})")

        for artifact_ref in result.artifact_refs:
            if not artifact_ref.path.strip():
                violations.append(f"result artifact path must not be empty ({result.check_id or '<empty>'})")

    for check_id, count in check_id_counts.items():
        if count > 1:
            violations.append(f"duplicate check_id: {check_id}")

    for artifact_ref in report.artifact_refs:
        if not artifact_ref.path.strip():
            violations.append("report artifact path must not be empty")

    expected_status = derive_eval_evidence_report_status(report.results)
    if report.status is not expected_status:
        violations.append(f"report status must be {expected_status.value} (got {report.status.value})")

    if violations:
        raise ValueError("; ".join(violations))


# --- Typed run and comparison evidence (PDC-D5-R013, R015) ---------------------------

EVAL_RUN_EVIDENCE_PREFIX = "eval_run:"
EVAL_COMPARISON_EVIDENCE_PREFIX = "eval_comparison:"
_PAYLOAD_KEY = "payload_json"
_DIGEST_KEY = "result_digest"
_TENANT_KEY = "tenant_id"


def _typed_basis() -> list[EvalEvidenceBasis]:
    return [EvalEvidenceBasis.EXISTING_EVAL_CHECK]


def _run_status(run: EvalRun) -> EvalEvidenceStatus:
    """Evidence status from rule outcomes only; judge outcomes never decide it."""
    rule = run.rule_outcomes
    if any(item.status is EvaluatorOutcomeStatus.FAILED for item in rule):
        return EvalEvidenceStatus.FAILED
    if rule and all(item.status is EvaluatorOutcomeStatus.PASSED for item in rule):
        return EvalEvidenceStatus.PASSED
    return EvalEvidenceStatus.UNAVAILABLE


def build_eval_run_evidence_result(run: EvalRun) -> EvalEvidenceCheckResult:
    """Record one typed run as canonical eval evidence (advisory, never an authorization)."""
    metrics = run.metrics()
    return create_eval_evidence_check_result(
        check_id=f"{EVAL_RUN_EVIDENCE_PREFIX}{run.run_id}",
        check_kind=EvalEvidenceCheckKind.EVAL_RUN,
        status=_run_status(run),
        title=f"Eval run {run.run_id}",
        message="Advisory evaluation evidence; promotion is governance-owned.",
        basis=_typed_basis(),
        metadata={
            _TENANT_KEY: run.tenant_id,
            _DIGEST_KEY: run.result_digest,
            "inputs_digest": run.inputs_digest,
            "dataset_digest": run.dataset_digest,
            "code_sha": run.code_sha,
            "seed": str(run.seed),
            "scenario_count": str(metrics.scenario_count),
            "advisory": "true",
            _PAYLOAD_KEY: run.model_dump_json(),
        },
    )


def build_eval_comparison_evidence_result(
    comparison: EvalComparison,
) -> EvalEvidenceCheckResult:
    """Record one comparison as canonical eval evidence (advisory)."""
    status = EvalEvidenceStatus.PASSED if comparison.clean else EvalEvidenceStatus.FAILED
    digest = comparison.comparison_digest
    return create_eval_evidence_check_result(
        check_id=f"{EVAL_COMPARISON_EVIDENCE_PREFIX}{digest[7:31]}",
        check_kind=EvalEvidenceCheckKind.EVAL_COMPARISON,
        status=status,
        title=(f"Eval comparison {comparison.baseline_run_id} -> {comparison.candidate_run_id}"),
        message="Advisory comparison; an evaluation PASS never authorizes promotion.",
        basis=_typed_basis(),
        metadata={
            _TENANT_KEY: comparison.tenant_id,
            _DIGEST_KEY: digest,
            "dataset_digest": comparison.dataset_digest,
            "regressed": str(comparison.regressed).lower(),
            "threshold_breached": str(comparison.threshold_breached).lower(),
            "advisory": "true",
            _PAYLOAD_KEY: comparison.model_dump_json(),
        },
    )


def build_eval_run_evidence_report(
    run: EvalRun,
    comparison: EvalComparison | None = None,
) -> EvalEvidenceReport:
    """Build a canonical ``EvalEvidenceReport`` for a run and an optional comparison."""
    results = [build_eval_run_evidence_result(run)]
    if comparison is not None:
        if comparison.tenant_id != run.tenant_id:
            raise EvalTenantBoundaryError("comparison and run belong to different tenants")
        results.append(build_eval_comparison_evidence_result(comparison))
    report = EvalEvidenceReport(
        report_id=f"eval-evidence:{run.run_id}",
        title="Eval run evidence",
        summary="Typed evaluation evidence (advisory).",
        status=derive_eval_evidence_report_status(results),
        results=results,
    )
    validate_eval_evidence_report(report)
    return report


def _typed_results(
    report: EvalEvidenceReport,
    kind: EvalEvidenceCheckKind,
    *,
    tenant_id: str,
) -> list[EvalEvidenceCheckResult]:
    selected = [item for item in report.results if item.check_kind is kind]
    for item in selected:
        if item.metadata.get(_TENANT_KEY) != tenant_id:
            raise EvalTenantBoundaryError("eval evidence belongs to a different tenant")
    return selected


def read_eval_run_evidence(report: EvalEvidenceReport, *, tenant_id: str) -> list[EvalRun]:
    """Read runs back from evidence; cross-tenant reads and tampered payloads are rejected."""
    runs: list[EvalRun] = []
    for item in _typed_results(report, EvalEvidenceCheckKind.EVAL_RUN, tenant_id=tenant_id):
        run = EvalRun.model_validate_json(item.metadata[_PAYLOAD_KEY])
        if run.tenant_id != tenant_id:
            raise EvalTenantBoundaryError("eval run payload belongs to a different tenant")
        if run.result_digest != item.metadata.get(_DIGEST_KEY):
            raise EvalDigestMismatchError("eval run evidence digest mismatch")
        runs.append(run)
    return runs


def read_eval_comparison_evidence(report: EvalEvidenceReport, *, tenant_id: str) -> list[EvalComparison]:
    """Read comparisons back from evidence with the same tenant and digest checks."""
    comparisons: list[EvalComparison] = []
    for item in _typed_results(report, EvalEvidenceCheckKind.EVAL_COMPARISON, tenant_id=tenant_id):
        comparison = EvalComparison.model_validate_json(item.metadata[_PAYLOAD_KEY])
        if comparison.tenant_id != tenant_id:
            raise EvalTenantBoundaryError("eval comparison payload belongs to another tenant")
        if comparison.comparison_digest != item.metadata.get(_DIGEST_KEY):
            raise EvalDigestMismatchError("eval comparison evidence digest mismatch")
        comparisons.append(comparison)
    return comparisons
