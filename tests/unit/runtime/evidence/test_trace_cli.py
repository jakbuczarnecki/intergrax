# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from intergrax.cli.trace import run_trace_export, run_trace_show
from intergrax.runtime.evidence.scenario_runner import run_core_certification

pytestmark = pytest.mark.unit


def test_run_trace_show_renders_timeline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cert_dir = tmp_path / "cert"
    run_core_certification("L1", output_dir=cert_dir)
    args = argparse.Namespace(
        report=cert_dir / "report.json",
        root=tmp_path,
    )
    code = run_trace_show(args)
    captured = capsys.readouterr()
    assert code == 0
    assert "certification_started" in captured.out
    assert "Trace timeline:" in captured.out


def test_run_trace_export_writes_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cert_dir = tmp_path / "cert"
    run_core_certification("L2", output_dir=cert_dir)
    out_dir = tmp_path / "trace"
    args = argparse.Namespace(
        report=cert_dir / "report.json",
        root=tmp_path,
        output_dir=out_dir,
    )
    code = run_trace_export(args)
    captured = capsys.readouterr()
    assert code == 0
    assert (out_dir / "timeline.json").is_file()
    assert (out_dir / "timeline.md").is_file()
    assert "timeline.json" in captured.out


def test_run_trace_show_missing_report_raises(tmp_path: Path) -> None:
    args = argparse.Namespace(
        report=tmp_path / "missing.json",
        root=tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="certification report not found"):
        run_trace_show(args)
