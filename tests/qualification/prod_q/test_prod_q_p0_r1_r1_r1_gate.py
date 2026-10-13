# © Artur Czarnecki. All rights reserved.

"""PROD-Q-P0-R1-R1-R1 fail-closed readiness gate."""

from __future__ import annotations

import pytest

from tests.qualification.prod_q._p0_audit_gate import (
    PROD_Q_P0_R1_R1_R1_STAGE_ID,
    prod_q_p0_r1_r1_r1_gate_violations,
    prod_q_p0_r1_r1_r1_ready_for_audit,
)
from tests.qualification.prod_q._p0_findings_register import (
    historical_prod_q_findings,
    known_finding_ids,
    known_prod_q_findings,
)
from tests.qualification.prod_q._p0_integrity import (
    probe_false_auxiliary_na_without_symbol_evidence,
    probe_mutated_alternate_mechanism_detected,
    probe_readiness_method_not_auto_auxiliary,
    probe_unresolved_classification_role_explicit,
)
from tests.qualification.prod_q._p0_ownership import (
    probe_alternate_contract_conflict_detected,
    probe_legal_port_providers_not_duplicate,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_prod_q_p0_r1_r1_r1_stage_id() -> None:
    assert PROD_Q_P0_R1_R1_R1_STAGE_ID == "PROD-Q-P0-R1-R1-R1"


def test_prod_q_p0_r1_r1_r1_ready_for_audit_gate() -> None:
    assert prod_q_p0_r1_r1_r1_ready_for_audit() is True
    assert prod_q_p0_r1_r1_r1_gate_violations() == ()


def test_prod_q_p0_r1_r1_r1_known_findings_register_eight() -> None:
    assert len(historical_prod_q_findings()) == 8
    findings = known_prod_q_findings()
    assert len(findings) >= 8
    assert historical_prod_q_findings()[0].finding_id in known_finding_ids()


def test_prod_q_p0_r1_r1_r1_negative_mutation_probes() -> None:
    assert probe_readiness_method_not_auto_auxiliary() is None
    assert probe_unresolved_classification_role_explicit() is None
    assert probe_false_auxiliary_na_without_symbol_evidence() is None
    assert probe_alternate_contract_conflict_detected() is None
    assert probe_legal_port_providers_not_duplicate() is None
    assert probe_mutated_alternate_mechanism_detected() is None
