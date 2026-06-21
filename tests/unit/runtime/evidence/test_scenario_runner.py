# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.runtime.evidence.core_certification_spec import CoreCertificationLevel
from intergrax.runtime.evidence.scenario_contracts import CoreScenarioStatus
from intergrax.runtime.evidence.scenario_runner import (
    require_core_scenario_contract,
    run_core_certification,
    run_core_scenario,
)
from intergrax.runtime.evidence.scenario_runner import CoreScenarioRunContext

pytestmark = pytest.mark.unit


def test_run_core_certification_l1_passes_four_scenarios(tmp_path: Path) -> None:
    report = run_core_certification(CoreCertificationLevel.L1, output_dir=tmp_path)
    assert report.passed is True
    assert report.scenarios_total == 4
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()


def test_run_core_certification_l2_passes_eight_scenarios(tmp_path: Path) -> None:
    report = run_core_certification("l2", output_dir=tmp_path)
    assert report.passed is True
    assert report.scenarios_total == 8


def test_run_core_certification_l3_passes_twelve_scenarios(tmp_path: Path) -> None:
    report = run_core_certification("CORE-L3", output_dir=tmp_path)
    assert report.passed is True
    assert report.scenarios_total == 12
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert payload["scenarios_passed"] == 12


def test_certification_report_emitted_in_l1_results(tmp_path: Path) -> None:
    report = run_core_certification("L1", output_dir=tmp_path)
    ids = {item.scenario_id for item in report.scenario_results}
    assert "certification_report_emitted" in ids
    cert_result = next(
        item for item in report.scenario_results if item.scenario_id == "certification_report_emitted"
    )
    assert cert_result.status is CoreScenarioStatus.PASSED


def test_require_core_scenario_contract_raises_for_unknown() -> None:
    with pytest.raises(ValueError, match="missing core scenario contract"):
        require_core_scenario_contract("unknown_scenario")


def test_run_core_scenario_raises_when_runner_missing() -> None:
    contract = require_core_scenario_contract("basic_run_completed")
    context = CoreScenarioRunContext(
        certification_run_id="x",
        output_dir=Path("build/evidence/core_certification"),
        level=CoreCertificationLevel.L1,
    )
    broken = contract.model_copy(update={"scenario_id": "not_registered"})
    with pytest.raises(ValueError, match="no runner registered"):
        run_core_scenario(broken, context)
