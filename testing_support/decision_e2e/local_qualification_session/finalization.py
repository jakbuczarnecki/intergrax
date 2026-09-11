# © Artur Czarnecki. All rights reserved.

"""Transactional qualification artifact finalization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from testing_support.decision_e2e.local_qualification_session.atomic_io import (
    atomic_write_json,
    atomic_write_text,
)
from testing_support.decision_e2e.local_qualification_session.classification_adapter import (
    ClassificationParseError,
    classification_from_persisted_dict,
    failure_view_from_classification,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    FinalizationPhase,
    QualificationSessionState,
    QualificationSpec,
    SessionIntegrityReport,
    TraceReadbackStatus,
)
from testing_support.decision_e2e.local_qualification_session.trace_readback import (
    read_typed_alignment_events,
)


class FinalizationError(RuntimeError):
    """Finalization failed; raw artifacts must remain intact."""


class ArtifactValidationError(FinalizationError):
    """Required artifact missing or inconsistent."""


@dataclass(frozen=True, slots=True)
class FinalizationContext:
    session_dir: Path
    spec: QualificationSpec
    session_id: str
    task_id: str


@dataclass(frozen=True, slots=True)
class FinalizationOutcome:
    state: QualificationSessionState
    phase: FinalizationPhase | None
    integrity: SessionIntegrityReport | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_runs_payload(session_dir: Path) -> dict[str, object]:
    runs_path = session_dir / "runs.json"
    if not runs_path.is_file():
        raise ArtifactValidationError("runs.json missing")
    payload = json.loads(runs_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactValidationError("runs.json must be an object")
    return payload


def _validate_run_count(payload: dict[str, object], spec: QualificationSpec) -> None:
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ArtifactValidationError("runs.json missing runs array")
    if len(runs) != spec.run_count:
        raise ArtifactValidationError("run count mismatch in runs.json")


def _build_analysis(payload: dict[str, object], spec: QualificationSpec) -> dict[str, object]:
    runs = payload.get("runs")
    alignment_statuses: list[str] = []
    platform_failures = 0
    if isinstance(runs, list):
        for item in runs:
            if not isinstance(item, dict):
                continue
            trace_events = item.get("trace_events")
            events: tuple[dict[str, object], ...] = ()
            if isinstance(trace_events, list):
                events = tuple(
                    dict(event) for event in trace_events if isinstance(event, dict)
                )
            alignment_statuses.append(read_typed_alignment_events(events).status.value)
            run_result = item.get("run_result")
            if isinstance(run_result, dict):
                classification = run_result.get("classification")
                if isinstance(classification, dict):
                    try:
                        typed = classification_from_persisted_dict(classification)
                        view = failure_view_from_classification(typed)
                        if view.is_platform_failure:
                            platform_failures += 1
                    except ClassificationParseError:
                        pass
    return {
        "qualification_session_schema_version": "qualification_analysis.v1",
        "planned_runs": spec.run_count,
        "typed_alignment_readback_statuses": alignment_statuses,
        "platform_failure_count": platform_failures,
        "max_evaluator_attempt_index": spec.max_evaluator_attempt_index,
    }


def _render_report(
    *,
    session_id: str,
    task_id: str,
    spec: QualificationSpec,
    integrity: SessionIntegrityReport,
) -> str:
    lines = [
        "# Local Qualification Session Report",
        "",
        f"- session_id: `{session_id}`",
        f"- task_id: `{task_id}`",
        f"- scenario: `{spec.experiment_identity.scenario_id}`",
        "",
        "## Session integrity",
        f"- runtime_identity_status: `{integrity.runtime_identity_status.value}`",
        f"- model_identity_status: `{integrity.model_identity_status.value}`",
        f"- source_identity_status: `{integrity.source_identity_status.value}`",
        f"- config_identity_status: `{integrity.config_identity_status.value}`",
        f"- trace_readback_status: `{integrity.trace_readback_status.value}`",
        f"- attempt_evidence_status: `{integrity.attempt_evidence_status.value}`",
        f"- artifact_completeness_status: `{integrity.artifact_completeness_status.value}`",
        f"- finalization_status: `{integrity.finalization_status.value}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def _write_manifest(session_dir: Path, required: tuple[str, ...]) -> None:
    lines: list[str] = []
    for name in required:
        if name == "artifact-manifest.txt":
            continue
        path = session_dir / name
        if not path.is_file():
            raise ArtifactValidationError(f"manifest artifact missing: {name}")
        lines.append(f"{name} sha256:{_sha256_file(path)}")
    body = "\n".join(lines) + ("\n" if lines else "")
    manifest_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    body += f"artifact-manifest.txt sha256:{manifest_hash}\n"
    atomic_write_text(session_dir / "artifact-manifest.txt", body)


def finalize_session_artifacts(
    context: FinalizationContext,
    *,
    integrity: SessionIntegrityReport,
    report_renderer: Callable[[], str] | None = None,
    inject_derived_failure: Callable[[], None] | None = None,
) -> FinalizationOutcome:
    session_dir = context.session_dir
    spec = context.spec
    runs_payload = _load_runs_payload(session_dir)
    _validate_run_count(runs_payload, spec)

    try:
        analysis = _build_analysis(runs_payload, spec)
        atomic_write_json(session_dir / "analysis.json", analysis)

        if inject_derived_failure is not None:
            inject_derived_failure()

        summary_path = session_dir / "summary.json"
        if not summary_path.is_file():
            summary_payload = {
                "session_id": context.session_id,
                "task_id": context.task_id,
                "planned_runs": spec.run_count,
                "session_integrity": {
                    "runtime_identity_status": integrity.runtime_identity_status.value,
                    "model_identity_status": integrity.model_identity_status.value,
                    "source_identity_status": integrity.source_identity_status.value,
                    "config_identity_status": integrity.config_identity_status.value,
                    "trace_readback_status": integrity.trace_readback_status.value,
                    "attempt_evidence_status": integrity.attempt_evidence_status.value,
                    "artifact_completeness_status": integrity.artifact_completeness_status.value,
                    "finalization_status": QualificationSessionState.FINALIZING.value,
                },
            }
            atomic_write_json(summary_path, summary_payload)

        report_body = (
            report_renderer()
            if report_renderer is not None
            else _render_report(
                session_id=context.session_id,
                task_id=context.task_id,
                spec=spec,
                integrity=integrity,
            )
        )
        atomic_write_text(session_dir / "report.md", report_body)
        atomic_write_text(session_dir / "final-report.md", report_body)
        _write_manifest(session_dir, spec.required_artifacts)
    except (OSError, FinalizationError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        if not isinstance(exc, FinalizationError):
            raise FinalizationError(str(exc)) from exc
        raise

    finalized_integrity = SessionIntegrityReport(
        runtime_identity_status=integrity.runtime_identity_status,
        model_identity_status=integrity.model_identity_status,
        source_identity_status=integrity.source_identity_status,
        config_identity_status=integrity.config_identity_status,
        trace_readback_status=integrity.trace_readback_status,
        attempt_evidence_status=integrity.attempt_evidence_status,
        artifact_completeness_status=integrity.artifact_completeness_status,
        finalization_status=QualificationSessionState.FINALIZED,
    )
    return FinalizationOutcome(
        state=QualificationSessionState.FINALIZED,
        phase=FinalizationPhase.SESSION_FINALIZED,
        integrity=finalized_integrity,
    )


def finalize_with_failure_capture(
    context: FinalizationContext,
    *,
    integrity: SessionIntegrityReport,
    inject_derived_failure: Callable[[], None] | None = None,
) -> FinalizationOutcome:
    try:
        return finalize_session_artifacts(
            context,
            integrity=integrity,
            inject_derived_failure=inject_derived_failure,
        )
    except FinalizationError:
        failed_integrity = SessionIntegrityReport(
            runtime_identity_status=integrity.runtime_identity_status,
            model_identity_status=integrity.model_identity_status,
            source_identity_status=integrity.source_identity_status,
            config_identity_status=integrity.config_identity_status,
            trace_readback_status=integrity.trace_readback_status,
            attempt_evidence_status=integrity.attempt_evidence_status,
            artifact_completeness_status=integrity.artifact_completeness_status,
            finalization_status=QualificationSessionState.FAILED_FINALIZATION,
        )
        return FinalizationOutcome(
            state=QualificationSessionState.FAILED_FINALIZATION,
            phase=FinalizationPhase.DERIVED_COMPLETE,
            integrity=failed_integrity,
        )


def validate_required_artifacts(session_dir: Path, required: tuple[str, ...]) -> None:
    for name in required:
        if not (session_dir / name).is_file():
            raise ArtifactValidationError(f"required artifact missing: {name}")


def aggregate_trace_readback_status(statuses: tuple[TraceReadbackStatus, ...]) -> TraceReadbackStatus:
    if not statuses:
        return TraceReadbackStatus.NOT_AVAILABLE
    if any(status is TraceReadbackStatus.FAILED for status in statuses):
        return TraceReadbackStatus.FAILED
    if any(status is TraceReadbackStatus.PARTIAL for status in statuses):
        return TraceReadbackStatus.PARTIAL
    return TraceReadbackStatus.PASS
