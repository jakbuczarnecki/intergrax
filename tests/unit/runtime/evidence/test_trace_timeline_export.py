# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.runtime.evidence.scenario_runner import run_core_certification
from intergrax.runtime.evidence.trace_timeline_adapter import build_timeline_from_certification_report
from intergrax.runtime.evidence.trace_timeline_export import (
    format_trace_timeline_cli,
    format_trace_timeline_markdown,
    write_trace_timeline,
)

pytestmark = pytest.mark.unit


def test_write_trace_timeline_creates_json_and_markdown(tmp_path: Path) -> None:
    report = run_core_certification("L1", output_dir=tmp_path / "cert")
    timeline = build_timeline_from_certification_report(report)
    out_dir = tmp_path / "trace"
    json_path, md_path = write_trace_timeline(timeline, out_dir)
    assert json_path.is_file()
    assert md_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["timeline_id"] == report.certification_run_id
    assert "Events" in md_path.read_text(encoding="utf-8")


def test_format_trace_timeline_cli_includes_event_kinds(tmp_path: Path) -> None:
    report = run_core_certification("L1", output_dir=tmp_path)
    timeline = build_timeline_from_certification_report(report)
    rendered = format_trace_timeline_cli(timeline)
    assert "certification_started" in rendered
    assert "certification_completed" in rendered
    assert report.certification_run_id in rendered


def test_format_trace_timeline_markdown_includes_table_header(tmp_path: Path) -> None:
    report = run_core_certification("L1", output_dir=tmp_path)
    timeline = build_timeline_from_certification_report(report)
    rendered = format_trace_timeline_markdown(timeline)
    assert "# Intergrax Trace Evidence Timeline" in rendered
    assert "| Seq | Kind | Scenario | Severity | Title |" in rendered
