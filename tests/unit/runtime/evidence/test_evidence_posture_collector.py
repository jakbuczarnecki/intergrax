# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.evidence.evidence_posture_collector import (
    DEFAULT_CORE_REPORT_PATH,
    DEFAULT_TRACE_TIMELINE_PATH,
    build_core_certification_signal,
    build_static_posture_signals,
    build_trace_timeline_signal,
    collect_evidence_posture,
    load_core_report_if_available,
    load_trace_timeline_if_available,
    resolve_core_report_path,
    resolve_trace_timeline_path,
)
from intergrax.runtime.evidence.evidence_posture_contracts import (
    EvidenceBasis,
    EvidencePostureArtifactKind,
    EvidencePostureLevel,
    EvidenceSignalKind,
    EvidenceSignalStatus,
)
from intergrax.runtime.evidence.scenario_runner import run_core_certification
from intergrax.runtime.evidence.trace_timeline_adapter import (
    build_timeline_from_certification_report,
)

pytestmark = pytest.mark.unit

def test_resolve_core_report_path_uses_default_under_root(tmp_path: Path) -> None:
    resolved = resolve_core_report_path(root=tmp_path)
    assert resolved == (tmp_path / DEFAULT_CORE_REPORT_PATH).resolve()


def test_resolve_trace_timeline_path_uses_default_under_root(tmp_path: Path) -> None:
    resolved = resolve_trace_timeline_path(root=tmp_path)
    assert resolved == (tmp_path / DEFAULT_TRACE_TIMELINE_PATH).resolve()


def test_load_core_report_if_available_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_core_report_if_available(tmp_path / "report.json") is None


def test_load_trace_timeline_if_available_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_trace_timeline_if_available(tmp_path / "timeline.json") is None


def test_build_core_certification_signal_missing_report() -> None:
    signal = build_core_certification_signal(
        report=None,
        report_path=Path("build/evidence/core_certification/report.json"),
    )
    assert signal.kind is EvidenceSignalKind.CORE_CERTIFICATION
    assert signal.status is EvidenceSignalStatus.MISSING
    assert signal.basis is EvidenceBasis.DETERMINISTIC_MOCK


def test_build_trace_timeline_signal_missing_timeline() -> None:
    signal = build_trace_timeline_signal(
        timeline=None,
        timeline_path=Path("build/evidence/trace/timeline.json"),
    )
    assert signal.kind is EvidenceSignalKind.TRACE_TIMELINE
    assert signal.status is EvidenceSignalStatus.MISSING
    assert signal.basis is EvidenceBasis.REPORT_DERIVED


def test_build_core_certification_signal_passed_from_report(tmp_path: Path) -> None:
    report = run_core_certification("L2", output_dir=tmp_path)
    report_path = tmp_path / "report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    signal = build_core_certification_signal(report=report, report_path=report_path)

    assert signal.status is EvidenceSignalStatus.PASSED
    assert signal.basis is EvidenceBasis.DETERMINISTIC_MOCK
    assert signal.metadata["certification_level"] == report.certification_level.value


def test_build_trace_timeline_signal_available_from_timeline(tmp_path: Path) -> None:
    report = run_core_certification("L2", output_dir=tmp_path)
    timeline = build_timeline_from_certification_report(
        report,
        source_report_path=str(tmp_path / "report.json"),
    )
    timeline_path = tmp_path / "timeline.json"
    timeline_path.write_text(timeline.model_dump_json(indent=2), encoding="utf-8")

    signal = build_trace_timeline_signal(timeline=timeline, timeline_path=timeline_path)

    assert signal.status is EvidenceSignalStatus.AVAILABLE
    assert signal.basis is EvidenceBasis.REPORT_DERIVED
    assert signal.metadata["events_count"] == str(len(timeline.events))


def test_build_static_posture_signals_includes_unknown_operational() -> None:
    signals = build_static_posture_signals(include_unknown_operational_signals=True)
    by_kind = {signal.kind: signal for signal in signals}

    assert by_kind[EvidenceSignalKind.REPO_HEALTH].status is EvidenceSignalStatus.UNKNOWN
    assert by_kind[EvidenceSignalKind.PYTEST_GATE].status is EvidenceSignalStatus.UNKNOWN
    assert by_kind[EvidenceSignalKind.LIVE_TIER0_PROBES].status is EvidenceSignalStatus.DEFERRED
    assert by_kind[EvidenceSignalKind.W_ADAPT_L4].status is EvidenceSignalStatus.SEPARATE


def test_build_static_posture_signals_excludes_unknown_operational_when_disabled() -> None:
    signals = build_static_posture_signals(include_unknown_operational_signals=False)
    kinds = {signal.kind for signal in signals}

    assert EvidenceSignalKind.REPO_HEALTH not in kinds
    assert EvidenceSignalKind.PYTEST_GATE not in kinds
    assert EvidenceSignalKind.LIVE_TIER0_PROBES in kinds
    assert EvidenceSignalKind.W_ADAPT_L4 in kinds


def test_collect_evidence_posture_missing_artifacts_returns_missing_evidence(
    tmp_path: Path,
) -> None:
    summary = collect_evidence_posture(root=tmp_path)
    assert summary.level is EvidencePostureLevel.MISSING_EVIDENCE


def test_collect_evidence_posture_core_only_returns_missing_evidence(tmp_path: Path) -> None:
    report = run_core_certification("L2", output_dir=tmp_path / "core")
    report_path = tmp_path / "core" / "report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    summary = collect_evidence_posture(
        root=tmp_path,
        core_report_path=report_path,
    )

    assert summary.level is EvidencePostureLevel.MISSING_EVIDENCE


def test_collect_evidence_posture_core_and_timeline_returns_onboarding_ready(
    tmp_path: Path,
) -> None:
    core_dir = tmp_path / "core"
    report = run_core_certification("L2", output_dir=core_dir)
    report_path = core_dir / "report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    timeline = build_timeline_from_certification_report(
        report,
        source_report_path=str(report_path),
    )
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    timeline_path = trace_dir / "timeline.json"
    timeline_path.write_text(timeline.model_dump_json(indent=2), encoding="utf-8")

    summary = collect_evidence_posture(
        root=tmp_path,
        core_report_path=report_path,
        trace_timeline_path=timeline_path,
    )

    assert summary.level is EvidencePostureLevel.ONBOARDING_READY


def test_collect_evidence_posture_summary_artifact_refs_only_for_existing(
    tmp_path: Path,
) -> None:
    core_dir = tmp_path / "core"
    report = run_core_certification("L2", output_dir=core_dir)
    report_path = core_dir / "report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    core_only = collect_evidence_posture(
        root=tmp_path,
        core_report_path=report_path,
    )
    assert len(core_only.artifact_refs) == 1
    assert core_only.artifact_refs[0].kind is EvidencePostureArtifactKind.CORE_REPORT_JSON

    timeline = build_timeline_from_certification_report(
        report,
        source_report_path=str(report_path),
    )
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    timeline_path = trace_dir / "timeline.json"
    timeline_path.write_text(timeline.model_dump_json(indent=2), encoding="utf-8")

    both = collect_evidence_posture(
        root=tmp_path,
        core_report_path=report_path,
        trace_timeline_path=timeline_path,
    )
    kinds = {ref.kind for ref in both.artifact_refs}
    assert kinds == {
        EvidencePostureArtifactKind.CORE_REPORT_JSON,
        EvidencePostureArtifactKind.TRACE_TIMELINE_JSON,
    }


def test_collect_evidence_posture_includes_unknown_operational_signals(tmp_path: Path) -> None:
    summary = collect_evidence_posture(
        root=tmp_path,
        include_unknown_operational_signals=True,
    )
    by_kind = {signal.kind: signal for signal in summary.signals}
    assert by_kind[EvidenceSignalKind.REPO_HEALTH].status is EvidenceSignalStatus.UNKNOWN
    assert by_kind[EvidenceSignalKind.PYTEST_GATE].status is EvidenceSignalStatus.UNKNOWN


def test_evidence_posture_collector_has_no_applications_or_agents_imports() -> None:
    import intergrax.runtime.evidence.evidence_posture_collector as collector_module

    forbidden = ("applications.", "agents.", "from applications", "from agents")
    path = Path(collector_module.__file__)
    source = path.read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in source, f"{path.name} contains forbidden import token: {token}"
