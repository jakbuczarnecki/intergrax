# © Artur Czarnecki. All rights reserved.

"""GR-6-R1 architecture gates — requirement policy at canonical boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]


def test_boundary_enforces_decision_requirement_before_governance() -> None:
    source = (
        _REPO / "intergrax/runtime/policy/meaningful_side_effect_authorization.py"
    ).read_text(encoding="utf-8")
    assert "_enforce_decision_requirement" in source
    assert "decision provenance required but absent" in source
    assert "classify_decision_requirement" in source


def test_requirement_policy_contract_does_not_return_policy_action() -> None:
    source = (
        _REPO / "intergrax/contracts/decision_requirement_policy.py"
    ).read_text(encoding="utf-8")
    assert "from intergrax.contracts.runtime_policy import PolicyAction" not in source
    assert 'REQUIRED = "required"' in source


def test_boundary_depends_on_policy_protocol_not_default_type() -> None:
    path = _REPO / "intergrax/runtime/policy/meaningful_side_effect_authorization.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "isinstance":
                hits.append(ast.unparse(node))
    assert not any("ConfiguredDecisionRequirementPolicy" in hit for hit in hits)
    assert not any("PermissiveDecisionRequirementPolicy" in hit for hit in hits if "is not None" not in hit)
