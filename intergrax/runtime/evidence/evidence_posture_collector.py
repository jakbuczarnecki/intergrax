# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Evidence posture read-only collector (HEP Band 2ae · EVID-POSTURE-02)."""

from __future__ import annotations

from pathlib import Path

from intergrax.runtime.evidence.certification_report import CoreCertificationReport
from intergrax.runtime.evidence.eval_evidence_contracts import (
    EvalEvidenceReport,
    EvalEvidenceStatus,
)
from intergrax.runtime.evidence.live_core_probe_contracts import (
    LiveCoreProbeReport,
    LiveCoreProbeStatus,
)
from intergrax.runtime.evidence.evidence_posture_contracts import (
    EvidenceBasis,
    EvidencePostureArtifactKind,
    EvidencePostureArtifactRef,
    EvidencePostureLevel,
    EvidencePostureSummary,
    EvidenceSignal,
    EvidenceSignalKind,
    EvidenceSignalStatus,
    create_evidence_signal,
    derive_posture_level,
    generate_evidence_posture_id,
    validate_evidence_posture_summary,
)
from intergrax.runtime.evidence.trace_timeline_contracts import TraceTimeline

DEFAULT_CORE_REPORT_PATH = Path("build/evidence/core_certification/report.json")
DEFAULT_TRACE_TIMELINE_PATH = Path("build/evidence/trace/timeline.json")
DEFAULT_LIVE_CORE_PROBE_REPORT_PATH = Path(
    "build/evidence/live_core_probes/live_core_report.json"
)
DEFAULT_EVAL_EVIDENCE_REPORT_PATH = Path("build/evidence/eval/report.json")

_POSTURE_TITLE = "Intergrax evidence posture"

_LEVEL_SUMMARY_TEXT: dict[EvidencePostureLevel, str] = {
    EvidencePostureLevel.ONBOARDING_READY: (
        "Core certification and report-derived trace timeline are available."
    ),
    EvidencePostureLevel.MISSING_EVIDENCE: "Required evidence artifacts are missing.",
    EvidencePostureLevel.FAILED: "One or more required evidence signals failed.",
    EvidencePostureLevel.PARTIAL: (
        "Some evidence is available, but the posture is incomplete."
    ),
    EvidencePostureLevel.UNKNOWN: "Evidence posture is unknown.",
}


def resolve_core_report_path(
    *,
    root: Path,
    core_report_path: Path | None = None,
) -> Path:
    """Return absolute core certification report path under ``root``."""
    if core_report_path is not None:
        return core_report_path.resolve()
    return (root / DEFAULT_CORE_REPORT_PATH).resolve()


def resolve_trace_timeline_path(
    *,
    root: Path,
    trace_timeline_path: Path | None = None,
) -> Path:
    """Return absolute trace timeline path under ``root``."""
    if trace_timeline_path is not None:
        return trace_timeline_path.resolve()
    return (root / DEFAULT_TRACE_TIMELINE_PATH).resolve()


def load_core_report_if_available(path: Path) -> CoreCertificationReport | None:
    """Load core certification report from ``path``, or ``None`` when missing."""
    if not path.is_file():
        return None
    try:
        return CoreCertificationReport.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"failed to parse core certification report at {path}: {exc}"
        ) from exc


def resolve_live_core_probe_report_path(
    *,
    root: Path,
    live_core_probe_report_path: Path | None = None,
) -> Path:
    """Return absolute live core probe report path under ``root``."""
    if live_core_probe_report_path is not None:
        return live_core_probe_report_path.resolve()
    return (root / DEFAULT_LIVE_CORE_PROBE_REPORT_PATH).resolve()


def load_live_core_probe_report_if_available(path: Path) -> LiveCoreProbeReport | None:
    """Load live core probe report from ``path``, or ``None`` when missing."""
    if not path.is_file():
        return None
    try:
        return LiveCoreProbeReport.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"failed to parse live core probe report at {path}"
        ) from exc


def resolve_eval_evidence_report_path(
    *,
    root: Path,
    eval_evidence_report_path: Path | None = None,
) -> Path:
    """Return absolute eval evidence report path under ``root``."""
    if eval_evidence_report_path is not None:
        return eval_evidence_report_path.resolve()
    return (root / DEFAULT_EVAL_EVIDENCE_REPORT_PATH).resolve()


def load_eval_evidence_report_if_available(path: Path) -> EvalEvidenceReport | None:
    """Load eval evidence report from ``path``, or ``None`` when missing."""
    if not path.is_file():
        return None
    try:
        return EvalEvidenceReport.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to parse eval evidence report at {path}") from exc


def load_trace_timeline_if_available(path: Path) -> TraceTimeline | None:
    """Load trace timeline from ``path``, or ``None`` when missing."""
    if not path.is_file():
        return None
    try:
        return TraceTimeline.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"failed to parse trace timeline at {path}: {exc}"
        ) from exc


def build_core_certification_signal(
    *,
    report: CoreCertificationReport | None,
    report_path: Path,
) -> EvidenceSignal:
    """Build CORE_CERTIFICATION signal from an optional report artifact."""
    if report is None:
        return create_evidence_signal(
            kind=EvidenceSignalKind.CORE_CERTIFICATION,
            status=EvidenceSignalStatus.MISSING,
            title="Core certification",
            message="Missing core certification report",
            basis=EvidenceBasis.DETERMINISTIC_MOCK,
            metadata={"path": str(report_path)},
        )

    artifact_refs = [
        EvidencePostureArtifactRef(
            kind=EvidencePostureArtifactKind.CORE_REPORT_JSON,
            path=str(report_path),
        ),
        EvidencePostureArtifactRef(
            kind=EvidencePostureArtifactKind.CORE_REPORT_MARKDOWN,
            path=str(Path(report.output_dir) / "report.md"),
        ),
    ]
    metadata = {
        "certification_level": report.certification_level.value,
        "certification_run_id": report.certification_run_id,
        "scenarios_total": str(report.scenarios_total),
        "scenarios_passed": str(report.scenarios_passed),
        "scenarios_failed": str(report.scenarios_failed),
    }

    if report.passed:
        return create_evidence_signal(
            kind=EvidenceSignalKind.CORE_CERTIFICATION,
            status=EvidenceSignalStatus.PASSED,
            title="Core certification",
            message=(
                f"{report.scenarios_passed}/{report.scenarios_total} scenarios passed"
            ),
            basis=EvidenceBasis.DETERMINISTIC_MOCK,
            artifact_refs=artifact_refs,
            metadata=metadata,
        )

    return create_evidence_signal(
        kind=EvidenceSignalKind.CORE_CERTIFICATION,
        status=EvidenceSignalStatus.FAILED,
        title="Core certification",
        message=(
            f"{report.scenarios_failed} failed; "
            f"{report.scenarios_passed}/{report.scenarios_total} passed"
        ),
        basis=EvidenceBasis.DETERMINISTIC_MOCK,
        artifact_refs=artifact_refs,
        metadata=metadata,
    )


def build_trace_timeline_signal(
    *,
    timeline: TraceTimeline | None,
    timeline_path: Path,
) -> EvidenceSignal:
    """Build TRACE_TIMELINE signal from an optional timeline artifact."""
    if timeline is None:
        return create_evidence_signal(
            kind=EvidenceSignalKind.TRACE_TIMELINE,
            status=EvidenceSignalStatus.MISSING,
            title="Trace timeline",
            message="Missing trace timeline artifact",
            basis=EvidenceBasis.REPORT_DERIVED,
            metadata={"path": str(timeline_path)},
        )

    return create_evidence_signal(
        kind=EvidenceSignalKind.TRACE_TIMELINE,
        status=EvidenceSignalStatus.AVAILABLE,
        title="Trace timeline",
        message=f"{len(timeline.events)} timeline events available",
        basis=EvidenceBasis.REPORT_DERIVED,
        artifact_refs=[
            EvidencePostureArtifactRef(
                kind=EvidencePostureArtifactKind.TRACE_TIMELINE_JSON,
                path=str(timeline_path),
            ),
            EvidencePostureArtifactRef(
                kind=EvidencePostureArtifactKind.TRACE_TIMELINE_MARKDOWN,
                path=str(timeline_path.parent / "timeline.md"),
            ),
        ],
        metadata={
            "timeline_id": timeline.timeline_id,
            "timeline_kind": timeline.kind.value,
            "events_count": str(len(timeline.events)),
            "source_report_path": timeline.source_report_path or "",
        },
    )


def _count_probe_statuses(
    report: LiveCoreProbeReport,
) -> dict[str, int]:
    counts = {
        "probes_total": len(report.results),
        "probes_passed": 0,
        "probes_failed": 0,
        "probes_skipped": 0,
        "probes_unavailable": 0,
    }
    for result in report.results:
        if result.status is LiveCoreProbeStatus.PASSED:
            counts["probes_passed"] += 1
        elif result.status is LiveCoreProbeStatus.FAILED:
            counts["probes_failed"] += 1
        elif result.status is LiveCoreProbeStatus.SKIPPED:
            counts["probes_skipped"] += 1
        elif result.status is LiveCoreProbeStatus.UNAVAILABLE:
            counts["probes_unavailable"] += 1
    return counts


def build_live_tier0_probe_signal(
    *,
    report: LiveCoreProbeReport | None,
    report_path: Path,
) -> EvidenceSignal:
    """Build LIVE_TIER0_PROBES signal from an optional live core probe report."""
    if report is None:
        return create_evidence_signal(
            kind=EvidenceSignalKind.LIVE_TIER0_PROBES,
            status=EvidenceSignalStatus.DEFERRED,
            title="Live Tier-0 probes",
            message="Deferred follow-up: EVID-CORE-FU-01",
            basis=EvidenceBasis.UNKNOWN,
            metadata={"path": str(report_path)},
        )

    counts = _count_probe_statuses(report)
    metadata = {
        "report_id": report.report_id,
        "report_status": report.status.value,
        "probes_total": str(counts["probes_total"]),
        "probes_passed": str(counts["probes_passed"]),
        "probes_failed": str(counts["probes_failed"]),
        "probes_skipped": str(counts["probes_skipped"]),
        "probes_unavailable": str(counts["probes_unavailable"]),
        "scope": "selected_tier0_probes",
        "llm": "mock",
        "network": "disabled",
        "provider_calls": "disabled",
    }
    artifact_refs = [
        EvidencePostureArtifactRef(
            kind=EvidencePostureArtifactKind.OTHER,
            path=str(report_path),
        ),
    ]

    if report.status is LiveCoreProbeStatus.PASSED:
        return create_evidence_signal(
            kind=EvidenceSignalKind.LIVE_TIER0_PROBES,
            status=EvidenceSignalStatus.PASSED,
            title="Live Tier-0 probes",
            message=(
                f"{counts['probes_passed']}/{counts['probes_total']} "
                "selected live Tier-0 probes passed"
            ),
            basis=EvidenceBasis.LIVE_RUNTIME,
            artifact_refs=artifact_refs,
            metadata=metadata,
        )

    if report.status is LiveCoreProbeStatus.FAILED:
        return create_evidence_signal(
            kind=EvidenceSignalKind.LIVE_TIER0_PROBES,
            status=EvidenceSignalStatus.FAILED,
            title="Live Tier-0 probes",
            message=(
                f"{counts['probes_failed']}/{counts['probes_total']} "
                "selected live Tier-0 probes failed"
            ),
            basis=EvidenceBasis.LIVE_RUNTIME,
            artifact_refs=artifact_refs,
            metadata=metadata,
        )

    return create_evidence_signal(
        kind=EvidenceSignalKind.LIVE_TIER0_PROBES,
        status=EvidenceSignalStatus.UNKNOWN,
        title="Live Tier-0 probes",
        message=report.summary,
        basis=EvidenceBasis.LIVE_RUNTIME,
        artifact_refs=artifact_refs,
        metadata=metadata,
    )


def _count_eval_check_statuses(
    report: EvalEvidenceReport,
) -> dict[str, int]:
    counts = {
        "checks_total": len(report.results),
        "checks_passed": 0,
        "checks_failed": 0,
        "checks_skipped": 0,
        "checks_unavailable": 0,
    }
    for result in report.results:
        if result.status is EvalEvidenceStatus.PASSED:
            counts["checks_passed"] += 1
        elif result.status is EvalEvidenceStatus.FAILED:
            counts["checks_failed"] += 1
        elif result.status is EvalEvidenceStatus.SKIPPED:
            counts["checks_skipped"] += 1
        elif result.status is EvalEvidenceStatus.UNAVAILABLE:
            counts["checks_unavailable"] += 1
    return counts


def build_eval_regression_signal(
    *,
    report: EvalEvidenceReport | None,
    report_path: Path,
) -> EvidenceSignal | None:
    """Build EVAL_REGRESSION signal from an optional eval evidence report."""
    if report is None:
        return None

    counts = _count_eval_check_statuses(report)
    metadata = {
        "report_id": report.report_id,
        "report_status": report.status.value,
        "checks_total": str(counts["checks_total"]),
        "checks_passed": str(counts["checks_passed"]),
        "checks_failed": str(counts["checks_failed"]),
        "checks_skipped": str(counts["checks_skipped"]),
        "checks_unavailable": str(counts["checks_unavailable"]),
        "scope": "eval_regression_evidence",
        "llm": "none",
        "network": "disabled",
        "provider_calls": "disabled",
        "real_llm_evaluation": "disabled",
    }
    artifact_refs = [
        EvidencePostureArtifactRef(
            kind=EvidencePostureArtifactKind.OTHER,
            path=str(report_path),
        ),
        EvidencePostureArtifactRef(
            kind=EvidencePostureArtifactKind.OTHER,
            path=str(report_path.parent / "report.md"),
        ),
    ]

    if report.status is EvalEvidenceStatus.PASSED:
        return create_evidence_signal(
            kind=EvidenceSignalKind.EVAL_REGRESSION,
            status=EvidenceSignalStatus.PASSED,
            title="Eval regression evidence",
            message=(
                f"{counts['checks_passed']}/{counts['checks_total']} "
                "eval evidence checks passed"
            ),
            basis=EvidenceBasis.REPORT_DERIVED,
            artifact_refs=artifact_refs,
            metadata=metadata,
        )

    if report.status is EvalEvidenceStatus.FAILED:
        return create_evidence_signal(
            kind=EvidenceSignalKind.EVAL_REGRESSION,
            status=EvidenceSignalStatus.FAILED,
            title="Eval regression evidence",
            message=(
                f"{counts['checks_failed']}/{counts['checks_total']} "
                "eval evidence checks failed"
            ),
            basis=EvidenceBasis.REPORT_DERIVED,
            artifact_refs=artifact_refs,
            metadata=metadata,
        )

    return create_evidence_signal(
        kind=EvidenceSignalKind.EVAL_REGRESSION,
        status=EvidenceSignalStatus.UNKNOWN,
        title="Eval regression evidence",
        message=report.summary,
        basis=EvidenceBasis.REPORT_DERIVED,
        artifact_refs=artifact_refs,
        metadata=metadata,
    )


def build_static_posture_signals(
    *,
    include_unknown_operational_signals: bool = True,
    include_deferred_live_tier0_probes: bool = True,
) -> list[EvidenceSignal]:
    """Return static posture signals that are not collected from artifacts."""
    signals: list[EvidenceSignal] = []
    if include_deferred_live_tier0_probes:
        signals.append(
            create_evidence_signal(
                kind=EvidenceSignalKind.LIVE_TIER0_PROBES,
                status=EvidenceSignalStatus.DEFERRED,
                title="Live Tier-0 probes",
                message="Deferred follow-up: EVID-CORE-FU-01",
                basis=EvidenceBasis.UNKNOWN,
            )
        )
    signals.append(
        create_evidence_signal(
            kind=EvidenceSignalKind.W_ADAPT_L4,
            status=EvidenceSignalStatus.SEPARATE,
            title="W-ADAPT L4",
            message=(
                "Separate adaptive utility/rollback semantics, not CORE posture"
            ),
            basis=EvidenceBasis.SEPARATE,
        )
    )

    if include_unknown_operational_signals:
        signals.extend(
            [
                create_evidence_signal(
                    kind=EvidenceSignalKind.REPO_HEALTH,
                    status=EvidenceSignalStatus.UNKNOWN,
                    title="Repo health",
                    message="Not executed by posture collector",
                    basis=EvidenceBasis.UNKNOWN,
                ),
                create_evidence_signal(
                    kind=EvidenceSignalKind.PYTEST_GATE,
                    status=EvidenceSignalStatus.UNKNOWN,
                    title="Pytest gate",
                    message="Not executed by posture collector",
                    basis=EvidenceBasis.UNKNOWN,
                ),
            ]
        )

    return signals


def _summary_artifact_refs(
    *,
    report: CoreCertificationReport | None,
    report_path: Path,
    timeline: TraceTimeline | None,
    timeline_path: Path,
) -> list[EvidencePostureArtifactRef]:
    refs: list[EvidencePostureArtifactRef] = []
    if report is not None:
        refs.append(
            EvidencePostureArtifactRef(
                kind=EvidencePostureArtifactKind.CORE_REPORT_JSON,
                path=str(report_path),
            )
        )
    if timeline is not None:
        refs.append(
            EvidencePostureArtifactRef(
                kind=EvidencePostureArtifactKind.TRACE_TIMELINE_JSON,
                path=str(timeline_path),
            )
        )
    return refs


def collect_evidence_posture(
    *,
    root: Path = Path.cwd(),
    core_report_path: Path | None = None,
    trace_timeline_path: Path | None = None,
    live_core_probe_report_path: Path | None = None,
    eval_evidence_report_path: Path | None = None,
    include_unknown_operational_signals: bool = True,
    root_label: str = "local",
) -> EvidencePostureSummary:
    """Collect evidence posture summary from existing artifacts (read-only)."""
    resolved_core_path = resolve_core_report_path(
        root=root,
        core_report_path=core_report_path,
    )
    resolved_timeline_path = resolve_trace_timeline_path(
        root=root,
        trace_timeline_path=trace_timeline_path,
    )
    resolved_live_core_path = resolve_live_core_probe_report_path(
        root=root,
        live_core_probe_report_path=live_core_probe_report_path,
    )
    resolved_eval_path = resolve_eval_evidence_report_path(
        root=root,
        eval_evidence_report_path=eval_evidence_report_path,
    )

    report = load_core_report_if_available(resolved_core_path)
    timeline = load_trace_timeline_if_available(resolved_timeline_path)
    live_core_report = load_live_core_probe_report_if_available(resolved_live_core_path)
    eval_report = load_eval_evidence_report_if_available(resolved_eval_path)

    signals = [
        build_core_certification_signal(report=report, report_path=resolved_core_path),
        build_trace_timeline_signal(timeline=timeline, timeline_path=resolved_timeline_path),
        build_live_tier0_probe_signal(
            report=live_core_report,
            report_path=resolved_live_core_path,
        ),
        *build_static_posture_signals(
            include_unknown_operational_signals=include_unknown_operational_signals,
            include_deferred_live_tier0_probes=False,
        ),
    ]
    eval_signal = build_eval_regression_signal(
        report=eval_report,
        report_path=resolved_eval_path,
    )
    if eval_signal is not None:
        signals.append(eval_signal)

    level = derive_posture_level(signals)
    summary = EvidencePostureSummary(
        posture_id=generate_evidence_posture_id(root_label=root_label),
        level=level,
        title=_POSTURE_TITLE,
        summary=_LEVEL_SUMMARY_TEXT[level],
        signals=signals,
        artifact_refs=_summary_artifact_refs(
            report=report,
            report_path=resolved_core_path,
            timeline=timeline,
            timeline_path=resolved_timeline_path,
        ),
    )
    validate_evidence_posture_summary(summary)
    return summary
