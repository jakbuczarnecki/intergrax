# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.runtime.evidence.live_core_probe_contracts import (
    LIVE_CORE_PROBE_CATALOG_ORDER,
    LiveCoreProbeReport,
)
from intergrax.runtime.evidence.live_core_probe_export import (
    LIVE_CORE_PROBE_REPORT_JSON,
    LIVE_CORE_PROBE_REPORT_MARKDOWN,
    format_live_core_probe_cli,
    format_live_core_probe_markdown,
    write_live_core_probe_report,
)
from intergrax.runtime.evidence.live_core_probe_runner import (
    LIVE_CORE_PROBE_OPERATOR_NOTE,
    run_live_core_probes,
)

pytestmark = pytest.mark.unit


def test_format_live_core_probe_cli_includes_required_fields() -> None:
    report = run_live_core_probes(root_label="local")
    rendered = format_live_core_probe_cli(report)

    assert report.title in rendered
    assert f"Status: {report.status.value}" in rendered
    assert f"Report ID: {report.report_id}" in rendered
    assert LIVE_CORE_PROBE_OPERATOR_NOTE in rendered
    for probe_id in LIVE_CORE_PROBE_CATALOG_ORDER:
        assert probe_id in rendered


def test_format_live_core_probe_markdown_includes_results_and_artifacts() -> None:
    report = run_live_core_probes(root_label="local")
    rendered = format_live_core_probe_markdown(report)

    assert "# Selected Live Tier-0 Probes" in rendered
    assert "## Results" in rendered
    assert "| Probe | Status | Evidence basis | Title | Message |" in rendered
    assert "## Artifacts" in rendered
    for probe_id in LIVE_CORE_PROBE_CATALOG_ORDER:
        assert probe_id in rendered


def test_format_live_core_probe_markdown_escapes_pipe_in_cells() -> None:
    report = run_live_core_probes(root_label="local")
    mutated = report.model_copy(
        update={
            "results": [
                result.model_copy(update={"message": "value|with|pipes"})
                for result in report.results
            ]
        }
    )
    rendered = format_live_core_probe_markdown(mutated)
    assert "value\\|with\\|pipes" in rendered


def test_write_live_core_probe_report_writes_json_and_markdown(tmp_path: Path) -> None:
    report = run_live_core_probes(root_label="local")
    out_dir = tmp_path / "live_core_probes"
    json_path, md_path = write_live_core_probe_report(report, out_dir)

    assert json_path == out_dir / LIVE_CORE_PROBE_REPORT_JSON
    assert md_path == out_dir / LIVE_CORE_PROBE_REPORT_MARKDOWN
    assert json_path.is_file()
    assert md_path.is_file()
    assert "# Selected Live Tier-0 Probes" in md_path.read_text(encoding="utf-8")


def test_write_live_core_probe_report_json_round_trips(tmp_path: Path) -> None:
    report = run_live_core_probes(root_label="local")
    json_path, _ = write_live_core_probe_report(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    round_tripped = LiveCoreProbeReport.model_validate(payload)
    assert round_tripped.report_id == report.report_id
    assert len(round_tripped.results) == len(report.results)
