# © Artur Czarnecki. All rights reserved.

"""GR-10-R11 — architecture gates for ORCHESTRATION HITL."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_GATE = _REPO / "intergrax/runtime/policy/mse_hitl_effect_gate.py"
_INVOKER = _REPO / "intergrax/runtime/nexus/tools/invoker.py"
_SLOT = _REPO / "intergrax/runtime/nexus/orchestration/governed_consequential_operation.py"
_BRIDGE = _REPO / "intergrax/runtime/nexus/tools/mse_governed_continuation_hitl_bridge.py"
_BOUNDARY = _REPO / "intergrax/runtime/policy/meaningful_side_effect_authorization.py"


def test_gate_human_approval_does_not_imply_governance_allow() -> None:
    source = _GATE.read_text(encoding="utf-8")
    assert "never become Governance ALLOW" in source or "Human judgment" in source
    assert "PolicyAction.ALLOW" in source
    assert "matches_current_requirement" in source
    # Grant consume only under REQUIRE_HUMAN, not as ALLOW mapper
    assert "REQUIRE_HUMAN" in source


def test_gate_continuation_authority_required_for_require_human() -> None:
    source = _GATE.read_text(encoding="utf-8")
    assert "consume_matching_grant" in source
    assert "REQUIRE_HITL" in source


def test_invoker_fresh_governance_after_human() -> None:
    source = _INVOKER.read_text(encoding="utf-8")
    assert "boundary.authorize(" in source
    assert "evaluate_mse_hitl_effect_gate" in source


def test_slot_continue_fresh_governance() -> None:
    source = _SLOT.read_text(encoding="utf-8")
    assert "evaluate_mse_hitl_effect_gate" in source
    assert "boundary.authorize(" in source


def test_no_approval_bool_as_authority_in_gate() -> None:
    source = _GATE.read_text(encoding="utf-8")
    assert "approved: bool" not in source
    assert "approved=True" not in source


def test_bridge_does_not_map_approval_to_allow() -> None:
    source = _BRIDGE.read_text(encoding="utf-8")
    assert "PolicyAction.ALLOW" not in source
    assert "apply_governed_continuation_pause" in source


def test_boundary_authorize_and_execute_remains_canonical_reauth() -> None:
    source = _BOUNDARY.read_text(encoding="utf-8")
    assert "def authorize_and_execute" in source
    assert "Fresh ``DENY`` is absolute" in source or "Fresh DENY" in source
    assert "matches_current_requirement" in source


def test_no_service_locator_in_hitl_gate() -> None:
    for path in (_GATE, _BRIDGE, _SLOT):
        source = path.read_text(encoding="utf-8")
        assert "get_instance" not in source
        assert "AppContext" not in source
        assert "service_locator" not in source
