# © Artur Czarnecki. All rights reserved.

"""GR-10-R11 / R11-R1 — architecture gates for ORCHESTRATION HITL."""

from __future__ import annotations

import ast
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
    assert "GovernedContinuationApprovalGrant" in source
    assert "REQUIRE_HUMAN" in source
    assert "RESUMED" in source


def test_gate_require_human_never_maps_to_proceed_ast() -> None:
    source = _GATE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_GATE))
    gate_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_mse_hitl_effect_gate":
            gate_fn = node
            break
    assert gate_fn is not None

    def _mentions_require_human(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr == "REQUIRE_HUMAN":
                return True
        return False

    def _returns_proceed(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr == "PROCEED":
                return True
        return False

    for stmt in gate_fn.body:
        if isinstance(stmt, ast.If) and _mentions_require_human(stmt.test):
            assert not _returns_proceed(stmt), (
                "REQUIRE_HUMAN branch must not return MseHitlEffectGateDisposition.PROCEED"
            )


def test_gate_matching_grant_alone_cannot_proceed_without_allow_and_continuation() -> None:
    source = _GATE.read_text(encoding="utf-8")
    assert "resolve_canonical_continuation_authority" in source
    assert "continuation_port" in source
    assert "consume_matching_grant" in source


def test_invoker_fresh_governance_after_human() -> None:
    source = _INVOKER.read_text(encoding="utf-8")
    assert "boundary.authorize(" in source
    assert "evaluate_mse_hitl_effect_gate" in source
    assert "resolve_continuation_port_for_mse_hitl_gate" in source


def test_slot_continue_fresh_governance() -> None:
    source = _SLOT.read_text(encoding="utf-8")
    assert "evaluate_mse_hitl_effect_gate" in source
    assert "boundary.authorize(" in source
    assert "continuation_port" in source


def test_no_approval_bool_as_authority_in_gate() -> None:
    source = _GATE.read_text(encoding="utf-8")
    assert "approved: bool" not in source
    assert "approved=True" not in source
    assert "continuation_authorized: bool" not in source


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


def test_gate_no_concrete_continuation_implementation_import() -> None:
    source = _GATE.read_text(encoding="utf-8")
    assert "ExecutionContinuationService" not in source
    assert "sqlite" not in source.lower()
    assert "InMemory" not in source
    tree = ast.parse(source, filename=str(_GATE))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = (
                node.module
                if isinstance(node, ast.ImportFrom)
                else ".".join(alias.name for alias in node.names)
            )
            if mod is None:
                continue
            assert "continuation.service" not in mod
            assert "continuation.persistence" not in mod
