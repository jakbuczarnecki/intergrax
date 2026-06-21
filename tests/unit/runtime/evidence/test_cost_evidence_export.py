# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.runtime.evidence.cost_evidence_contracts import (
    COST_EVIDENCE_REPORT_JSON,
    COST_EVIDENCE_REPORT_MARKDOWN,
    CostEvidenceReport,
)
from intergrax.runtime.evidence.cost_evidence_export import (
    format_cost_evidence_cli,
    format_cost_evidence_markdown,
    write_cost_evidence_report,
)
from intergrax.runtime.evidence.cost_evidence_runner import (
    COST_EVIDENCE_OPERATOR_NOTE,
    run_cost_evidence_checks,
)

pytestmark = pytest.mark.unit


def test_format_cost_evidence_cli_includes_required_fields() -> None:
    report = run_cost_evidence_checks(root_label="local")
    rendered = format_cost_evidence_cli(report)

    assert report.title in rendered
    assert f"Status: {report.status.value}" in rendered
    assert f"Report ID: {report.report_id}" in rendered
    assert COST_EVIDENCE_OPERATOR_NOTE in rendered
    assert "trace_budget_facets" in rendered


def test_format_cost_evidence_markdown_includes_results_and_artifacts() -> None:
    report = run_cost_evidence_checks(root_label="local")
    rendered = format_cost_evidence_markdown(report)

    assert "# Cost Evidence" in rendered
    assert "## Results" in rendered
    assert "| Check ID | Kind | Status | Basis | Title | Message |" in rendered
    assert "## Artifacts" in rendered
    assert "trace_budget_facets" in rendered


def test_format_cost_evidence_markdown_escapes_pipe_in_cells() -> None:
    report = run_cost_evidence_checks(root_label="local")
    mutated = report.model_copy(
        update={
            "results": [
                result.model_copy(update={"message": "value|with|pipes"})
                for result in report.results
            ]
        }
    )
    rendered = format_cost_evidence_markdown(mutated)
    assert "value\\|with\\|pipes" in rendered


def test_write_cost_evidence_report_writes_json_and_markdown(tmp_path: Path) -> None:
    report = run_cost_evidence_checks(root_label="local")
    out_dir = tmp_path / "cost"
    json_path, md_path = write_cost_evidence_report(report, out_dir)

    assert json_path == out_dir / COST_EVIDENCE_REPORT_JSON
    assert md_path == out_dir / COST_EVIDENCE_REPORT_MARKDOWN
    assert json_path.is_file()
    assert md_path.is_file()
    assert "# Cost Evidence" in md_path.read_text(encoding="utf-8")


def test_write_cost_evidence_report_json_round_trips(tmp_path: Path) -> None:
    report = run_cost_evidence_checks(root_label="local")
    json_path, _ = write_cost_evidence_report(report, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    round_tripped = CostEvidenceReport.model_validate(payload)
    assert round_tripped.report_id == report.report_id
    assert len(round_tripped.results) == len(report.results)


def test_format_cost_evidence_does_not_claim_billing_or_pricing_implemented() -> None:
    report = run_cost_evidence_checks(root_label="local")
    cli_rendered = format_cost_evidence_cli(report)
    md_rendered = format_cost_evidence_markdown(report)
    forbidden = ("stripe", "invoice", "price_per_token", "openai", "anthropic")
    for token in forbidden:
        assert token not in cli_rendered.lower()
        assert token not in md_rendered.lower()
