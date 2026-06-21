# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Read-only cost evidence runner (EVID-COST-02)."""

from __future__ import annotations

from typing import Final

from intergrax.runtime.evidence.cost_evidence_contracts import (
    CostEvidenceArtifactKind,
    CostEvidenceArtifactRef,
    CostEvidenceCheckKind,
    CostEvidenceCheckResult,
    CostEvidenceReport,
    CostEvidenceStatus,
    create_cost_evidence_check_result,
    derive_cost_evidence_report_status,
    generate_cost_evidence_report_id,
    validate_cost_evidence_report,
)

COST_EVIDENCE_RUNNER_KIND = "cost_evidence_runner"

COST_EVIDENCE_OPERATOR_NOTE = (
    "Cost evidence packages existing local budget/cost/trace information. "
    "It does not implement provider pricing, billing, cloud cost estimation, "
    "real LLM usage metering, network execution, or a new budget policy framework."
)

_TRACE_BUDGET_FACETS_SOURCE_PATH = "intergrax/runtime/evidence/trace_timeline_facets.py"
_TRACE_BUDGET_FACETS_CHECK_ID = "trace_budget_facets"

_DEFAULT_REPORT_TITLE = "Cost evidence"

_STATUS_SUMMARIES: Final[dict[CostEvidenceStatus, str]] = {
    CostEvidenceStatus.PASSED: "All cost evidence checks passed.",
    CostEvidenceStatus.FAILED: "One or more cost evidence checks failed.",
    CostEvidenceStatus.SKIPPED: "Cost evidence checks partially completed.",
    CostEvidenceStatus.UNAVAILABLE: "Cost evidence checks are unavailable.",
}


def generate_cost_evidence_run_id(*, root_label: str = "local") -> str:
    """Return a deterministic cost evidence run identifier."""
    return f"cost-evidence-run:{root_label}"


def build_cost_evidence_report(
    *,
    results: list[CostEvidenceCheckResult],
    root_label: str = "local",
    summary: str = "",
) -> CostEvidenceReport:
    """Build and validate a cost evidence report from check results."""
    status = derive_cost_evidence_report_status(results)
    resolved_summary = summary or _STATUS_SUMMARIES[status]
    report = CostEvidenceReport(
        report_id=generate_cost_evidence_report_id(root_label=root_label),
        title=_DEFAULT_REPORT_TITLE,
        summary=resolved_summary,
        status=status,
        results=results,
    )
    validate_cost_evidence_report(report)
    return report


def _trace_budget_facets_metadata(*, root_label: str) -> dict[str, str]:
    return {
        "runner": COST_EVIDENCE_RUNNER_KIND,
        "run_id": generate_cost_evidence_run_id(root_label=root_label),
        "source": "trace_timeline_facets.TraceBudgetFacet",
        "cost_source": "local_trace_budget",
        "provider_pricing": "disabled",
        "billing": "disabled",
        "cloud_cost_estimation": "disabled",
        "real_llm_metering": "disabled",
        "network": "disabled",
    }


def _trace_budget_facets_artifact_ref() -> CostEvidenceArtifactRef:
    return CostEvidenceArtifactRef(
        kind=CostEvidenceArtifactKind.SOURCE_CHECK,
        path=_TRACE_BUDGET_FACETS_SOURCE_PATH,
        description="Trace budget facets source surface",
    )


def run_trace_budget_facets_cost_check(
    *,
    root_label: str = "local",
) -> CostEvidenceCheckResult:
    """Package trace budget facet availability into one local cost check result."""
    metadata = _trace_budget_facets_metadata(root_label=root_label)
    try:
        from intergrax.runtime.evidence.trace_timeline_facets import (  # noqa: PLC0415
            TraceBudgetFacet,
            TraceBudgetStatus,
        )

        if TraceBudgetFacet is None or TraceBudgetStatus is None:  # pragma: no cover
            raise RuntimeError("Trace budget facets are not available")

        return create_cost_evidence_check_result(
            check_id=_TRACE_BUDGET_FACETS_CHECK_ID,
            check_kind=CostEvidenceCheckKind.TRACE_BUDGET_FACETS,
            status=CostEvidenceStatus.PASSED,
            title="Trace budget facets available",
            message=(
                "Trace budget facets are available for local cost evidence packaging."
            ),
            artifact_refs=[_trace_budget_facets_artifact_ref()],
            metadata=metadata,
        )
    except Exception as exc:  # noqa: BLE001 — convert unexpected check failure
        return create_cost_evidence_check_result(
            check_id=_TRACE_BUDGET_FACETS_CHECK_ID,
            check_kind=CostEvidenceCheckKind.TRACE_BUDGET_FACETS,
            status=CostEvidenceStatus.FAILED,
            title="Trace budget facets unavailable",
            message=f"Cost evidence check failed: {exc}",
            metadata={
                **metadata,
                "error_type": type(exc).__name__,
            },
        )


def run_cost_evidence_checks(
    *,
    root_label: str = "local",
) -> CostEvidenceReport:
    """Run local cost evidence checks and return an in-memory report."""
    try:
        results = [
            run_trace_budget_facets_cost_check(root_label=root_label),
        ]
    except Exception as exc:  # noqa: BLE001 — convert unexpected check failure
        results = [
            create_cost_evidence_check_result(
                check_id=_TRACE_BUDGET_FACETS_CHECK_ID,
                check_kind=CostEvidenceCheckKind.TRACE_BUDGET_FACETS,
                status=CostEvidenceStatus.FAILED,
                title="Trace budget facets unavailable",
                message=f"Cost evidence check failed: {exc}",
                metadata={"error_type": type(exc).__name__},
            )
        ]

    return build_cost_evidence_report(results=results, root_label=root_label)
