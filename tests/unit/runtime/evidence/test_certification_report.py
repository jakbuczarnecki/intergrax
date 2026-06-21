# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.runtime.evidence.certification_report import (
    build_core_certification_report,
    format_core_certification_markdown,
    write_core_certification_report,
)
from intergrax.runtime.evidence.core_certification_spec import CoreCertificationLevel
from intergrax.runtime.evidence.scenario_contracts import (
    CoreEvidenceRef,
    CoreScenarioResult,
    CoreScenarioStatus,
    EvidenceRefKind,
)

pytestmark = pytest.mark.unit


def test_write_core_certification_report_creates_json_and_markdown(tmp_path: Path) -> None:
    result = CoreScenarioResult(
        scenario_id="basic_run_completed",
        status=CoreScenarioStatus.PASSED,
        evidence_refs=[
            CoreEvidenceRef(kind=EvidenceRefKind.RUNTIME_EVENT, ref="mock:basic_run_completed")
        ],
    )
    report = build_core_certification_report(
        level=CoreCertificationLevel.L1,
        results=[result],
        certification_run_id="test-run",
        output_dir=tmp_path,
    )
    json_path, md_path = write_core_certification_report(report, tmp_path)

    assert json_path.is_file()
    assert md_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["certification_level"] == "CORE-L1"
    assert payload["passed"] is True
    assert "basic_run_completed" in md_path.read_text(encoding="utf-8")


def test_format_core_certification_markdown_lists_failures() -> None:
    report = build_core_certification_report(
        level=CoreCertificationLevel.L2,
        results=[
            CoreScenarioResult(
                scenario_id="tool_denied_by_policy",
                status=CoreScenarioStatus.FAILED,
                message="deny missing",
            )
        ],
        certification_run_id="run-fail",
        output_dir=Path("build/evidence/core_certification"),
    )
    md = format_core_certification_markdown(report)
    assert "tool_denied_by_policy" in md
    assert "failed" in md
