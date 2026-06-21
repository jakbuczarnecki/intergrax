# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from intergrax.cli.evidence import run_evidence_posture, run_evidence_posture_export
from intergrax.runtime.evidence.scenario_runner import run_core_certification
from intergrax.runtime.evidence.trace_timeline_adapter import (
    build_timeline_from_certification_report,
)

pytestmark = pytest.mark.unit


def _posture_args(
    tmp_path: Path,
    *,
    core_report: Path | None = None,
    trace_timeline: Path | None = None,
    no_operational_unknowns: bool = False,
    output_dir: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        root=tmp_path,
        core_report=core_report,
        trace_timeline=trace_timeline,
        root_label="local",
        no_operational_unknowns=no_operational_unknowns,
        output_dir=output_dir,
    )


def _write_core_and_timeline(tmp_path: Path) -> tuple[Path, Path]:
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
    return report_path, timeline_path


def test_run_evidence_posture_missing_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = run_evidence_posture(_posture_args(tmp_path))
    captured = capsys.readouterr()

    assert code == 0
    assert "MISSING_EVIDENCE" in captured.out
    assert "CORE_CERTIFICATION" in captured.out
    assert "TRACE_TIMELINE" in captured.out


def test_run_evidence_posture_with_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path, timeline_path = _write_core_and_timeline(tmp_path)
    code = run_evidence_posture(
        _posture_args(tmp_path, core_report=report_path, trace_timeline=timeline_path)
    )
    captured = capsys.readouterr()

    assert code == 0
    assert "ONBOARDING_READY" in captured.out


def test_run_evidence_posture_export_with_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path, timeline_path = _write_core_and_timeline(tmp_path)
    out_dir = tmp_path / "posture"
    code = run_evidence_posture_export(
        _posture_args(
            tmp_path,
            core_report=report_path,
            trace_timeline=timeline_path,
            output_dir=out_dir,
        )
    )
    captured = capsys.readouterr()

    assert code == 0
    assert (out_dir / "posture.json").is_file()
    assert (out_dir / "posture.md").is_file()
    assert "posture.json" in captured.out
    assert "posture.md" in captured.out
    assert "note:" in captured.out


def test_run_evidence_posture_export_missing_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "posture"
    code = run_evidence_posture_export(_posture_args(tmp_path, output_dir=out_dir))
    captured = capsys.readouterr()

    assert code == 0
    assert (out_dir / "posture.json").is_file()
    assert (out_dir / "posture.md").is_file()
    assert "MISSING_EVIDENCE" in (out_dir / "posture.json").read_text(encoding="utf-8")


def test_run_evidence_posture_corrupt_report_exits_with_hint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        run_evidence_posture(_posture_args(tmp_path, core_report=report_path))

    captured = capsys.readouterr()
    combined = captured.out + captured.err

    assert exc_info.value.code == 1
    assert "error:" in combined
    assert "uv run intergrax certify core --level L2" in combined


def test_no_operational_unknowns_excludes_repo_health_and_pytest_gate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = run_evidence_posture(_posture_args(tmp_path, no_operational_unknowns=True))
    captured = capsys.readouterr()

    assert code == 0
    assert "REPO_HEALTH" not in captured.out
    assert "PYTEST_GATE" not in captured.out


def test_evidence_cli_has_no_applications_or_agents_imports() -> None:
    import intergrax.cli.evidence as evidence_module

    forbidden = ("applications.", "agents.", "from applications", "from agents")
    path = Path(evidence_module.__file__)
    source = path.read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in source, f"{path.name} contains forbidden import token: {token}"
