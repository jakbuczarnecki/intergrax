# © Artur Czarnecki. All rights reserved.

"""Q2-D2 dual path — Path B reconciliation; independent audit still required."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_6._findings_register import FindingStatus, all_hardcoding_findings
from tests.qualification.ebh_6._l3_aj_group_reconciliation import AjControlOutcome, aj_group_by_layer_class
from tests.qualification.ebh_6._q2d2_technical_qualification import (
    AUTHORITATIVE_ADR_PATH,
    PATH_A_NOT_ELIGIBLE,
    PATH_B_PRODUCTION_SCOPE,
    PATH_B_REMEDIATED_PENDING,
    POST_PATH_B_RECON_EXIT,
    assess_path_b_remediation,
    assess_q2d2_adr_scope,
    historical_path_a_not_eligible_when_nexus_erasure_present,
    post_path_b_reconciliation_status,
    replay_q2d2_fail_closed_behavioral_gates,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO = Path(__file__).resolve().parents[3]
_OD_REGISTER = _REPO / "docs/project/maintainers/qualification/EBH_6_OPERATOR_DECISION_REGISTER.md"
_CLOSURE_PLAN = _REPO / "docs/project/maintainers/qualification/EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md"


def test_ebh6_q2d2_semantic_core_f_pass_candidate_pending_independent_acceptance() -> None:
    group = aj_group_by_layer_class()[("semantic_core", "F")]
    assert group.outcome == AjControlOutcome.PASS_CANDIDATE, group.outcome
    debt = next(r for r in all_hardcoding_findings() if r.finding_id == "EBH6-DEBT-Q2-D2")
    assert debt.status == FindingStatus.CANDIDATE_RESOLVED
    assert "Q2-D2" in group.invariant_proves or "EBH6-DEBT-Q2-D2" in group.invariant_proves


def test_ebh6_q2d2_path_b_documented_no_prod_without_od01() -> None:
    plan = _CLOSURE_PLAN.read_text(encoding="utf-8")
    assert "T-REM-Q2D2" in plan
    assert "OD-EBH6-01" in plan
    assert PATH_B_PRODUCTION_SCOPE.startswith("OD-EBH6-01")


def test_ebh6_q2d2_operator_path_b_approved_not_open() -> None:
    text = _OD_REGISTER.read_text(encoding="utf-8")
    q2_row = [line for line in text.splitlines() if "OD-EBH6-Q2D2" in line and line.startswith("|")][0]
    assert "OPEN" not in q2_row or "APPROVED" in q2_row
    assert "PATH B" in q2_row
    assert "APPROVED" in q2_row


def test_ebh6_q2d2_adr_authority_and_scope_recorded() -> None:
    assessment = assess_q2d2_adr_scope()
    assert assessment.adr_exists
    assert assessment.adr_path == AUTHORITATIVE_ADR_PATH
    assert assessment.adr_covers_exposure_model
    assert assessment.resolver_returns_typed_exposure


def test_ebh6_q2d2_production_exposure_fail_closed_behavior_replayed() -> None:
    replay_q2d2_fail_closed_behavioral_gates()


def test_ebh6_q2d2_historical_path_a_not_eligible_before_nexus_erasure() -> None:
    assert historical_path_a_not_eligible_when_nexus_erasure_present() == PATH_A_NOT_ELIGIBLE


def test_ebh6_q2d2_path_b_remediation_proves_registered_violation_removed() -> None:
    rem = assess_path_b_remediation()
    assert rem.remediation_complete
    assert post_path_b_reconciliation_status() == PATH_B_REMEDIATED_PENDING
    assert POST_PATH_B_RECON_EXIT.startswith("Q2-D2 PATH B")
