# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.evidence.evidence_posture_contracts import (
    EvidenceBasis,
    EvidencePostureArtifactKind,
    EvidencePostureArtifactRef,
    EvidencePostureLevel,
    EvidencePostureSummary,
    EvidenceSignalKind,
    EvidenceSignalStatus,
    create_evidence_signal,
    generate_evidence_posture_id,
)
from intergrax.runtime.evidence.evidence_posture_export import (
    POSTURE_OPERATOR_NOTE,
    format_evidence_posture_cli,
    format_evidence_posture_markdown,
    write_evidence_posture,
)
from intergrax.runtime.evidence.evidence_posture_collector import collect_evidence_posture
from intergrax.runtime.evidence.scenario_runner import run_core_certification
from intergrax.runtime.evidence.trace_timeline_adapter import (
    build_timeline_from_certification_report,
)

pytestmark = pytest.mark.unit


def _onboarding_ready_summary(tmp_path: Path) -> EvidencePostureSummary:
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

    return collect_evidence_posture(
        root=tmp_path,
        core_report_path=report_path,
        trace_timeline_path=timeline_path,
    )


def test_format_evidence_posture_cli_shows_expected_sections(tmp_path: Path) -> None:
    rendered = format_evidence_posture_cli(_onboarding_ready_summary(tmp_path))

    assert "Intergrax evidence posture" in rendered
    assert "Level:" in rendered
    assert "CORE_CERTIFICATION" in rendered
    assert "TRACE_TIMELINE" in rendered
    assert "LIVE_TIER0_PROBES" in rendered
    assert "W_ADAPT_L4" in rendered
    assert POSTURE_OPERATOR_NOTE in rendered


def test_format_evidence_posture_markdown_shows_tables(tmp_path: Path) -> None:
    rendered = format_evidence_posture_markdown(_onboarding_ready_summary(tmp_path))

    assert "# Intergrax Evidence Posture" in rendered
    assert "## Signals" in rendered
    assert "## Artifacts" in rendered
    assert "| Kind | Status | Basis | Title | Message |" in rendered


def test_format_evidence_posture_markdown_escapes_pipe_characters() -> None:
    summary = EvidencePostureSummary(
        posture_id=generate_evidence_posture_id(root_label="local"),
        level=EvidencePostureLevel.UNKNOWN,
        title="Intergrax evidence posture",
        summary="summary|with|pipes",
        signals=[
            create_evidence_signal(
                kind=EvidenceSignalKind.CORE_CERTIFICATION,
                status=EvidenceSignalStatus.MISSING,
                title="title|pipe",
                message="message|pipe",
                basis=EvidenceBasis.DETERMINISTIC_MOCK,
            ),
        ],
        artifact_refs=[
            EvidencePostureArtifactRef(
                kind=EvidencePostureArtifactKind.CORE_REPORT_JSON,
                path="build/evidence/core|report.json",
                description="desc|ription",
            ),
        ],
    )

    rendered = format_evidence_posture_markdown(summary)

    assert "title\\|pipe" in rendered
    assert "message\\|pipe" in rendered
    assert "summary\\|with\\|pipes" in rendered
    assert "build/evidence/core\\|report.json" in rendered
    assert "desc\\|ription" in rendered


def test_write_evidence_posture_writes_json_and_markdown(tmp_path: Path) -> None:
    summary = _onboarding_ready_summary(tmp_path)
    output_dir = tmp_path / "posture"

    json_path, md_path = write_evidence_posture(summary, output_dir)

    assert json_path == output_dir / "posture.json"
    assert md_path == output_dir / "posture.md"
    assert json_path.is_file()
    assert md_path.is_file()


def test_write_evidence_posture_json_round_trips(tmp_path: Path) -> None:
    summary = _onboarding_ready_summary(tmp_path)
    output_dir = tmp_path / "posture"

    json_path, _ = write_evidence_posture(summary, output_dir)
    restored = EvidencePostureSummary.model_validate_json(json_path.read_text(encoding="utf-8"))

    assert restored.level is summary.level
    assert restored.posture_id == summary.posture_id
    assert len(restored.signals) == len(summary.signals)


def test_format_evidence_posture_empty_artifact_refs(tmp_path: Path) -> None:
    summary = collect_evidence_posture(root=tmp_path)

    cli = format_evidence_posture_cli(summary)
    markdown = format_evidence_posture_markdown(summary)

    assert "Artifacts: none" in cli
    assert "_No posture artifacts referenced._" in markdown
