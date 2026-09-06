# © Artur Czarnecki. All rights reserved.

"""Architecture gates for production STRICT governed-discovery composition."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSITION_MODULE = (
    REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "production_capability_discovery_composition.py"
)


def _function_defs(tree: ast.AST) -> list[ast.FunctionDef]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


def _annotation_names(node: ast.expr | None) -> set[str]:
    if node is None:
        return set()
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    if isinstance(node, ast.Subscript):
        return _annotation_names(node.value) | _annotation_names(node.slice)
    if isinstance(node, ast.Tuple):
        names: set[str] = set()
        for element in node.elts:
            names.update(_annotation_names(element))
        return names
    return set()


def test_production_capability_discovery_composition_orchestrates_governance() -> None:
    text = COMPOSITION_MODULE.read_text(encoding="utf-8")
    assert "discover_rank_and_govern_capabilities" in text
    assert "govern_capability_candidates" in text
    assert "rank_capability_candidates" in text
    assert "discover_capability_candidates" in text


def test_production_capability_discovery_composition_has_no_ranked_downstream_api() -> None:
    tree = ast.parse(COMPOSITION_MODULE.read_text(encoding="utf-8"))
    violations: list[str] = []
    for function in _function_defs(tree):
        arg_annotations = [
            _annotation_names(arg.annotation)
            for arg in function.args.args
        ]
        return_names = _annotation_names(function.returns)
        accepts_ranked = any("RankedCapabilityCandidate" in names for names in arg_annotations)
        returns_governed = "GovernedDiscoveryResult" in return_names or (
            "GovernedCapabilityCandidate" in return_names
        )
        if accepts_ranked and not returns_governed:
            violations.append(function.name)
        if function.name.startswith("consume_ranked"):
            violations.append(function.name)
    assert not violations, (
        "production composition exposes ranked downstream API: "
        + ", ".join(sorted(violations))
    )


def test_production_capability_discovery_composition_downstream_requires_governed_result() -> None:
    tree = ast.parse(COMPOSITION_MODULE.read_text(encoding="utf-8"))
    downstream = next(
        node
        for node in _function_defs(tree)
        if node.name == "consume_governed_discovery_for_downstream"
    )
    arg_names = _annotation_names(downstream.args.args[0].annotation)
    assert "GovernedDiscoveryResult" in arg_names
