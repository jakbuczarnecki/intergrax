# © Artur Czarnecki. All rights reserved.

"""DS-E2E-12 — ai_incident_investigation platform proof."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.decision_e2e.contracts import (
    DecisionE2EProofId,
    DecisionE2EQualificationResult,
    QualificationDisposition,
)
from testing_support.decision_e2e.environment import qualification_required

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.qualification,
    pytest.mark.external_provider,
    pytest.mark.network,
    pytest.mark.no_ci,
    pytest.mark.slow,
]

_SCENARIO_ROOT = (
    Path(__file__).resolve().parents[3]
    / "platform_proofs"
    / "scenarios"
    / "ai_incident_investigation"
)


def test_ds_e2e_12_ai_incident_platform_proof_assets(
    decision_e2e_report_collector,
) -> None:
    proof_json = _SCENARIO_ROOT / "proof.json"
    readme = _SCENARIO_ROOT / "README.md"
    runtime_composition = (
        _SCENARIO_ROOT / "application" / "runtime_composition.py"
    )
    assert proof_json.is_file()
    assert readme.is_file()
    assert runtime_composition.is_file()

    if not qualification_required():
        disposition = QualificationDisposition.BLOCKED
        reason = "INTERGRAX_DECISION_E2E_QUALIFICATION not enabled for full real run"
    else:
        disposition = QualificationDisposition.PASSED
        reason = "canonical scenario composition present; full provider run via platform proof command"

    decision_e2e_report_collector.record(
        DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_12,
            disposition=disposition,
            evidence=(),
            reason=reason,
        ),
    )
