# © Artur Czarnecki. All rights reserved.

"""GR-6 architecture gates — Decision System inside Execution Engine, Governance owns auth."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]


def _module_imports(module_path: Path, forbidden: tuple[str, ...]) -> list[str]:
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for forbidden_fragment in forbidden:
                if forbidden_fragment in node.module:
                    hits.append(f"{module_path}: from {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden_fragment in forbidden:
                    if forbidden_fragment in alias.name:
                        hits.append(f"{module_path}: import {alias.name}")
    return hits


def test_governance_contracts_do_not_import_execution_decision_engine() -> None:
    hits: list[str] = []
    for path in (
        _REPO / "intergrax/contracts/decision_authorization.py",
        _REPO / "intergrax/contracts/decision_governance_material.py",
    ):
        hits.extend(
            _module_imports(
                path,
                (
                    "runtime.execution.council_deliberation",
                    "runtime.decision_flow",
                    "runtime.nexus",
                ),
            ),
        )
    assert hits == []


def test_decision_governed_coordinator_lives_under_execution_engine() -> None:
    path = _REPO / "intergrax/runtime/execution/decision_governed_side_effect.py"
    assert path.is_file()
    hits = _module_imports(
        path,
        (
            "agents.",
            "applications.",
            "external_contractor",
        ),
    )
    assert hits == []


def test_decision_governed_coordinator_uses_canonical_authorization_boundary() -> None:
    source = (
        _REPO / "intergrax/runtime/execution/decision_governed_side_effect.py"
    ).read_text(encoding="utf-8")
    assert "authorize_and_execute" in source
    assert "decision_and_execute" not in source


def test_canonical_boundary_wires_decision_requirement_policy() -> None:
    source = (
        _REPO / "intergrax/runtime/policy/meaningful_side_effect_authorization.py"
    ).read_text(encoding="utf-8")
    assert "decision_requirement_policy" in source
    assert "_enforce_decision_requirement" in source


def test_meaningful_side_effect_carries_typed_decision_material_field() -> None:
    source = (_REPO / "intergrax/contracts/meaningful_side_effect.py").read_text(
        encoding="utf-8",
    )
    assert "decision_governance_material: DecisionGovernanceMaterialRef" in source
    assert 'metadata["decision"]' not in source
