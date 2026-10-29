# © Artur Czarnecki. All rights reserved.

"""Q2-D2 dual path — ADR scope, behavioral replay, Path B scope; OD stays OPEN."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_6._findings_register import all_hardcoding_findings
from tests.qualification.ebh_6._l3_aj_group_reconciliation import AjControlOutcome, aj_group_by_layer_class
from tests.qualification.ebh_6._q2d2_technical_qualification import (
    AUTHORITATIVE_ADR_PATH,
    PATH_A_ELIGIBLE,
    PATH_B_PRODUCTION_SCOPE,
    assess_q2d2_adr_scope,
    replay_q2d2_fail_closed_behavioral_gates,
    technical_q2d2_disposition,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO = Path(__file__).resolve().parents[3]
_OD_REGISTER = _REPO / "docs/project/maintainers/qualification/EBH_6_OPERATOR_DECISION_REGISTER.md"
_CLOSURE_PLAN = _REPO / "docs/project/maintainers/qualification/EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md"


def test_ebh6_q2d2_path_a_evidence_pack_present() -> None:
    group = aj_group_by_layer_class()[("semantic_core", "F")]
    assert group.outcome == AjControlOutcome.FINDING
    debt = next(r for r in all_hardcoding_findings() if r.finding_id == "EBH6-DEBT-Q2-D2")
    assert debt.status.value == "RATIFIED"
    assert "Q2-D2" in group.invariant_proves or "EBH6-DEBT-Q2-D2" in group.invariant_proves


def test_ebh6_q2d2_path_b_documented_no_prod_without_od01() -> None:
    plan = _CLOSURE_PLAN.read_text(encoding="utf-8")
    assert "T-REM-Q2D2" in plan
    assert "OD-EBH6-01" in plan
    assert PATH_B_PRODUCTION_SCOPE.startswith("OD-EBH6-01")


def test_ebh6_q2d2_operator_path_b_approved_implementation_wave_open() -> None:
    text = _OD_REGISTER.read_text(encoding="utf-8")
    assert "OD-EBH6-Q2D2" in text
    q2_row = [line for line in text.splitlines() if "OD-EBH6-Q2D2" in line and line.startswith("|")][0]
    assert "PATH B" in q2_row or "APPROVED" in q2_row
    assert "OD-EBH6-01" in text
    od01_row = [line for line in text.splitlines() if "OD-EBH6-01" in line and line.startswith("|")][0]
    assert "APPROVED" in od01_row or "WAVE OPEN" in od01_row


def test_ebh6_q2d2_adr_authority_and_scope_recorded() -> None:
    assessment = assess_q2d2_adr_scope()
    assert assessment.adr_exists
    assert assessment.adr_path == AUTHORITATIVE_ADR_PATH
    assert assessment.adr_covers_exposure_model
    assert assessment.resolver_returns_typed_exposure


def test_ebh6_q2d2_production_exposure_fail_closed_behavior_replayed() -> None:
    replay_q2d2_fail_closed_behavioral_gates()


def test_ebh6_q2d2_technical_disposition_after_path_b_nexus_typing_remediation() -> None:
    """Post OD-EBH6-01 Q2-D2 — NexusLoop projection matches resolver (FRZ-TYP-03)."""
    assessment = assess_q2d2_adr_scope()
    assert assessment.resolver_returns_typed_exposure
    assert not assessment.nexus_loop_uses_object_erasure
    disposition = technical_q2d2_disposition(assessment)
    assert disposition == PATH_A_ELIGIBLE
