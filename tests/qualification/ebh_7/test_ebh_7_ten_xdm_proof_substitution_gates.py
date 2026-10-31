# © Artur Czarnecki. All rights reserved.

"""TEN/XDM proof substitution — unrelated passing proof must fail binding gates."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._ebh7_scenario_evidence_validation import validate_xdm_scenario_status
from tests.qualification.ebh_7._ebh7_scenario_normative_audit import ScenarioVerifiedStatus
from tests.qualification.ebh_7._semantic_disposition_registry import disposition_for
from tests.qualification.ebh_7._tenant_e2e_scenarios import EBH7_TENANT_E2E_SCENARIOS

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_ebh7_ten_wrong_disposition_binding_detected_as_stale() -> None:
    scenario = EBH7_TENANT_E2E_SCENARIOS[0]
    disp = disposition_for(scenario.scenario_id)
    assert disp is not None
    assert disp.actual_test.startswith(scenario.proof_module or "MISSING")


def test_ebh7_ten_substituted_disposition_registry_fails_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.qualification.ebh_7._semantic_disposition_registry import VerifiedSemanticDisposition, disposition_for
    from tests.qualification.ebh_7._semantic_proof_binding_audit import assert_ebh7_semantic_proof_binding_audit
    from tests.qualification.ebh_7._semantic_proof_types import ProofBindingResult

    real = disposition_for("EBH7-TEN-E2E-01")
    assert real is not None

    def fake_disposition(sid: str) -> VerifiedSemanticDisposition | None:
        if sid == "EBH7-TEN-E2E-01":
            return VerifiedSemanticDisposition(
                scenario_id=sid,
                required_invariant=real.required_invariant,
                production_boundaries=real.production_boundaries,
                attack=real.attack,
                actual_test="tests.fake.unrelated::test_passes_unrelated_invariant",
                assertions_inspected=real.assertions_inspected,
                semantic_match=ProofBindingResult.EXACT_PROOF,
                test_execution_evidence=real.test_execution_evidence,
            )
        return disposition_for(sid)

    monkeypatch.setattr(
        "tests.qualification.ebh_7._semantic_proof_binding_audit.disposition_for",
        fake_disposition,
    )
    with pytest.raises(AssertionError):
        assert_ebh7_semantic_proof_binding_audit()


def test_ebh7_xdm_all_scenarios_resolve_or_classified() -> None:
    from tests.qualification.ebh_7._cross_domain_scenarios import EBH7_CROSS_DOMAIN_SCENARIOS

    for s in EBH7_CROSS_DOMAIN_SCENARIOS:
        status = validate_xdm_scenario_status(s)
        assert status is not ScenarioVerifiedStatus.PRODUCTION_GAP, s.scenario_id
