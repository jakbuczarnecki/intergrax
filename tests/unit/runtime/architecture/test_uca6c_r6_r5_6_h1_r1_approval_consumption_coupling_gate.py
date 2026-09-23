# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.6-H1-R1 architecture gates — verified/consumption coupling."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_INVOKER = _REPO / "intergrax" / "runtime" / "nexus" / "tools" / "invoker.py"


@pytest.mark.gate
def test_invoker_enforces_verified_consumption_pair_before_governance() -> None:
    source = _INVOKER.read_text(encoding="utf-8")
    assert "_require_agent_governance_approval_consumption_pair" in source
    assert "agent_governance_approval_consumption_missing" in source
    assert "agent_governance_verified_approval_missing" in source
    pair_pos = source.index("_require_agent_governance_approval_consumption_pair(")
    authorize_pos = source.index("governance.authorize_tool(", pair_pos)
    mark_applied_pos = source.index(
        "mark_applied_after_governance_allow", authorize_pos
    )
    assert pair_pos < authorize_pos < mark_applied_pos
