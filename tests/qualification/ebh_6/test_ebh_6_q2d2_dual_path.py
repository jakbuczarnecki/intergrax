# © Artur Czarnecki. All rights reserved.

"""Q2-D2 dual path — evidence pack (A) vs conditional remediation doc (B); OD stays OPEN."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_6._findings_register import all_hardcoding_findings
from tests.qualification.ebh_6._l3_aj_group_reconciliation import AjControlOutcome, aj_group_by_layer_class

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO = Path(__file__).resolve().parents[3]
_OD_REGISTER = _REPO / "docs/project/maintainers/qualification/EBH_6_OPERATOR_DECISION_REGISTER.md"
_CLOSURE_PLAN = _REPO / "docs/project/maintainers/qualification/EBH_6_FINAL_BLOCKER_CLOSURE_PLAN.md"
_NEXUS = _REPO / "intergrax/runtime/nexus/nexus_loop.py"


def test_ebh6_q2d2_path_a_evidence_pack_present() -> None:
    group = aj_group_by_layer_class()[("semantic_core", "F")]
    assert group.outcome == AjControlOutcome.FINDING
    assert _NEXUS.is_file()
    debt = next(r for r in all_hardcoding_findings() if r.finding_id == "EBH6-DEBT-Q2-D2")
    assert debt.status.value == "RATIFIED"
    assert "Q2-D2" in group.invariant_proves or "EBH6-DEBT-Q2-D2" in group.invariant_proves
    assert any("_findings_register" in ref or "q2d2" in ref.lower() for ref in group.evidence_refs)


def test_ebh6_q2d2_path_b_documented_no_prod_without_od01() -> None:
    plan = _CLOSURE_PLAN.read_text(encoding="utf-8")
    assert "T-REM-Q2D2" in plan
    assert "OD-EBH6-01" in plan
    assert "nexus_loop.py" in plan


def test_ebh6_q2d2_operator_decision_stays_open_without_verified_entry() -> None:
    text = _OD_REGISTER.read_text(encoding="utf-8")
    assert "OD-EBH6-Q2D2" in text
    row = [line for line in text.splitlines() if "OD-EBH6-Q2D2" in line and line.startswith("|")][0]
    assert "OPEN" in row
    assert "APPROVED" not in row and "CLOSED" not in row
