# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.runtime.evidence.eval_evidence_contracts import (
    EVAL_EVIDENCE_REPORT_JSON,
    EVAL_EVIDENCE_REPORT_MARKDOWN,
    EvalEvidenceReport,
)
from intergrax.runtime.evidence.eval_evidence_export import (
    format_eval_evidence_cli,
    format_eval_evidence_markdown,
    write_eval_evidence_report,
)
from intergrax.runtime.evidence.eval_evidence_runner import (
    EVAL_EVIDENCE_OPERATOR_NOTE,
    run_eval_evidence_checks,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_format_eval_evidence_cli_includes_required_fields() -> None:
    report = run_eval_evidence_checks(root=_REPO_ROOT, root_label="local")
    rendered = format_eval_evidence_cli(report)

    assert report.title in rendered
    assert f"Status: {report.status.value}" in rendered
    assert f"Report ID: {report.report_id}" in rendered
    assert EVAL_EVIDENCE_OPERATOR_NOTE in rendered
    assert "scenario_library" in rendered


def test_format_eval_evidence_markdown_includes_results_and_artifacts() -> None:
    report = run_eval_evidence_checks(root=_REPO_ROOT, root_label="local")
    rendered = format_eval_evidence_markdown(report)

    assert "# Eval Regression Evidence" in rendered
    assert "## Results" in rendered
    assert "| Check ID | Kind | Status | Basis | Title | Message |" in rendered
    assert "## Artifacts" in rendered
    assert "scenario_library" in rendered


def test_format_eval_evidence_markdown_escapes_pipe_in_cells() -> None:
    report = run_eval_evidence_checks(root=_REPO_ROOT, root_label="local")
    mutated = report.model_copy(
        update={
            "results": [
                result.model_copy(update={"message": "value|with|pipes"})
                for result in report.results
            ]
        }
    )
    rendered = format_eval_evidence_markdown(mutated)
    assert "value\\|with\\|pipes" in rendered


def test_write_eval_evidence_report_writes_json_and_markdown(tmp_path: Path) -> None:
    report = run_eval_evidence_checks(root=_REPO_ROOT, root_label="local")
    out_dir = tmp_path / "eval"
    json_path, md_path = write_eval_evidence_report(report, out_dir)

    assert json_path == out_dir / EVAL_EVIDENCE_REPORT_JSON
    assert md_path == out_dir / EVAL_EVIDENCE_REPORT_MARKDOWN
    assert json_path.is_file()
    assert md_path.is_file()
    assert "# Eval Regression Evidence" in md_path.read_text(encoding="utf-8")


def test_write_eval_evidence_report_json_round_trips(tmp_path: Path) -> None:
    report = run_eval_evidence_checks(root=_REPO_ROOT, root_label="local")
    json_path, _ = write_eval_evidence_report(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    round_tripped = EvalEvidenceReport.model_validate(payload)
    assert round_tripped.report_id == report.report_id
    assert len(round_tripped.results) == len(report.results)
