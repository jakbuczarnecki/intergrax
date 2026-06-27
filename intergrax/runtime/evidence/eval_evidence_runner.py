# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Read-only eval regression evidence runner (EVID-EVAL-02)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from intergrax.runtime.evidence.eval_evidence_contracts import (
    EvalEvidenceArtifactKind,
    EvalEvidenceArtifactRef,
    EvalEvidenceCheckKind,
    EvalEvidenceCheckResult,
    EvalEvidenceReport,
    EvalEvidenceStatus,
    create_eval_evidence_check_result,
    derive_eval_evidence_report_status,
    generate_eval_evidence_report_id,
    validate_eval_evidence_report,
)

EVAL_EVIDENCE_RUNNER_KIND: Final = "eval_evidence_runner"

EVAL_EVIDENCE_OPERATOR_NOTE: Final = (
    "Eval regression evidence packages existing local eval/check mechanisms. "
    "It does not run real LLM evaluation, does not compare providers, does not use network, "
    "and does not create a new eval framework."
)

DEFAULT_EVAL_SCENARIO_LIBRARY_CHECK_PATH = Path("scripts/maintenance/check_eval_scenario_library.py")

_DEFAULT_REPORT_TITLE = "Eval regression evidence"

_STATUS_SUMMARIES: Final[dict[EvalEvidenceStatus, str]] = {
    EvalEvidenceStatus.PASSED: "All eval evidence checks passed.",
    EvalEvidenceStatus.FAILED: "One or more eval evidence checks failed.",
    EvalEvidenceStatus.SKIPPED: "Eval evidence checks partially completed.",
    EvalEvidenceStatus.UNAVAILABLE: "Eval evidence checks are unavailable.",
}

_SCENARIO_LIBRARY_CHECK_ID = "scenario_library"

_LOCAL_CONTROLLED_METADATA: Final[dict[str, str]] = {
    "runner": EVAL_EVIDENCE_RUNNER_KIND,
    "source_check": DEFAULT_EVAL_SCENARIO_LIBRARY_CHECK_PATH.as_posix(),
    "network": "disabled",
    "provider_calls": "disabled",
    "llm": "none",
    "real_llm_evaluation": "disabled",
}


def generate_eval_evidence_run_id(*, root_label: str = "local") -> str:
    """Return a deterministic eval evidence run identifier."""
    return f"eval-evidence-run:{root_label}"


def build_eval_evidence_report(
    *,
    results: list[EvalEvidenceCheckResult],
    root_label: str = "local",
    summary: str = "",
) -> EvalEvidenceReport:
    """Build and validate an eval evidence report from check results."""
    status = derive_eval_evidence_report_status(results)
    resolved_summary = summary or _STATUS_SUMMARIES[status]
    report = EvalEvidenceReport(
        report_id=generate_eval_evidence_report_id(root_label=root_label),
        title=_DEFAULT_REPORT_TITLE,
        summary=resolved_summary,
        status=status,
        results=results,
    )
    validate_eval_evidence_report(report)
    return report


def _scenario_library_metadata(*, root_label: str) -> dict[str, str]:
    return {
        **_LOCAL_CONTROLLED_METADATA,
        "run_id": generate_eval_evidence_run_id(root_label=root_label),
    }


def _failed_check_result(*, exc: Exception) -> EvalEvidenceCheckResult:
    return create_eval_evidence_check_result(
        check_id=_SCENARIO_LIBRARY_CHECK_ID,
        check_kind=EvalEvidenceCheckKind.SCENARIO_LIBRARY,
        status=EvalEvidenceStatus.FAILED,
        title="Eval scenario library check failed",
        message=f"Eval evidence check failed: {exc}",
        metadata={"error_type": type(exc).__name__},
    )


def run_eval_scenario_library_evidence_check(
    *,
    root: Path = Path.cwd(),
    root_label: str = "local",
) -> EvalEvidenceCheckResult:
    """Package eval scenario library source availability into one check result."""
    script_path = root / DEFAULT_EVAL_SCENARIO_LIBRARY_CHECK_PATH
    metadata = _scenario_library_metadata(root_label=root_label)

    if script_path.is_file():
        return create_eval_evidence_check_result(
            check_id=_SCENARIO_LIBRARY_CHECK_ID,
            check_kind=EvalEvidenceCheckKind.SCENARIO_LIBRARY,
            status=EvalEvidenceStatus.PASSED,
            title="Eval scenario library check available",
            message="Existing eval scenario library check is available for evidence packaging.",
            artifact_refs=[
                EvalEvidenceArtifactRef(
                    kind=EvalEvidenceArtifactKind.SOURCE_CHECK,
                    path=DEFAULT_EVAL_SCENARIO_LIBRARY_CHECK_PATH.as_posix(),
                )
            ],
            metadata=metadata,
        )

    return create_eval_evidence_check_result(
        check_id=_SCENARIO_LIBRARY_CHECK_ID,
        check_kind=EvalEvidenceCheckKind.SCENARIO_LIBRARY,
        status=EvalEvidenceStatus.UNAVAILABLE,
        title="Eval scenario library check unavailable",
        message="Eval scenario library source check is missing.",
        metadata=metadata,
    )


def run_eval_evidence_checks(
    *,
    root: Path = Path.cwd(),
    root_label: str = "local",
) -> EvalEvidenceReport:
    """Run local eval evidence checks and return an in-memory report."""
    try:
        results = [
            run_eval_scenario_library_evidence_check(root=root, root_label=root_label),
        ]
    except Exception as exc:  # noqa: BLE001 — convert unexpected check failure
        results = [_failed_check_result(exc=exc)]

    return build_eval_evidence_report(results=results, root_label=root_label)
