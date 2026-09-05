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
from testing_support.decision_e2e.requirements import qualify_live_scenario
from testing_support.decision_e2e.scenario_qualification import (
    AI_INCIDENT_SCENARIO_ID,
    run_ai_incident_live_qualification,
)

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


def test_ai_incident_scenario_packaging_assets_exist() -> None:
    """Architecture/packaging regression — asset presence alone is not DS-E2E-12 qualification."""
    proof_json = _SCENARIO_ROOT / "proof.json"
    readme = _SCENARIO_ROOT / "README.md"
    runtime_composition = _SCENARIO_ROOT / "application" / "runtime_composition.py"
    assert proof_json.is_file()
    assert readme.is_file()
    assert runtime_composition.is_file()


def test_assets_only_cannot_produce_ds_e2e_12_passed(
    decision_e2e_report_collector,
) -> None:
    """Regression gate: qualification flag + assets must not auto-pass DS-E2E-12."""
    from testing_support.decision_e2e.reporting import validate_qualification_result

    false_positive = validate_qualification_result(
        DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_12,
            disposition=(
                QualificationDisposition.PASSED
                if qualification_required()
                else QualificationDisposition.BLOCKED
            ),
            evidence=(),
            reason="canonical scenario composition present",
        ),
    )
    assert false_positive.disposition is not QualificationDisposition.PASSED
    decision_e2e_report_collector.record(false_positive)


@pytest.mark.asyncio
async def test_ds_e2e_12_ai_incident_live_scenario(
    decision_e2e_report_collector,
) -> None:
    if not qualification_required():
        decision_e2e_report_collector.record(
            DecisionE2EQualificationResult(
                proof_id=DecisionE2EProofId.DS_E2E_12,
                disposition=QualificationDisposition.BLOCKED,
                evidence=(),
                reason="INTERGRAX_DECISION_E2E_QUALIFICATION not enabled for full real run",
            ),
        )
        return

    attempt = await run_ai_incident_live_qualification()
    result = qualify_live_scenario(
        proof_id=DecisionE2EProofId.DS_E2E_12,
        scenario_evidence=attempt.evidence,
        reason=f"live scenario={AI_INCIDENT_SCENARIO_ID}",
    )
    if result.disposition is QualificationDisposition.PASSED and not attempt.evaluation_passed:
        result = DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_12,
            disposition=QualificationDisposition.FAILED,
            evidence=result.evidence,
            reason=attempt.error or "scenario evaluation failed",
        )
    if result.disposition is QualificationDisposition.FAILED:
        decision_e2e_report_collector.record(result)
        pytest.fail(result.reason or "scenario failed")
    decision_e2e_report_collector.record(result)
