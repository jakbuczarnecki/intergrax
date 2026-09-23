# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R6-R1-R3-R3 — canonical typed routing observer contracts."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from intergrax.runtime.nexus.engine import runtime_state as runtime_state_mod

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HOOKS_SOURCE = _REPO_ROOT / "intergrax/llm_adapters/routing/evaluating_hooks.py"
_ADAPTER_SOURCE = _REPO_ROOT / "intergrax/llm_adapters/routing/evaluating_adapter.py"
_CONTRACTS_SOURCE = _REPO_ROOT / "intergrax/llm_adapters/routing/contracts.py"
_EVALUATOR_SOURCE = _REPO_ROOT / "intergrax/llm_adapters/routing/evaluator.py"


def _alias_assignments(path: Path, name: str) -> list[ast.Assign]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[ast.Assign] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                found.append(node)
    return found


def test_ebh_2e_r6_r1_r3_r3_canonical_observer_aliases_single_owner() -> None:
    for alias in (
        "RoutingEvaluationObserver",
        "AllowlistViolationObserver",
        "InnerSwappedObserver",
    ):
        assert len(_alias_assignments(_HOOKS_SOURCE, alias)) == 1
        assert len(_alias_assignments(_ADAPTER_SOURCE, alias)) == 0


def test_ebh_2e_r6_r1_r3_r3_allowlist_observer_not_object() -> None:
    text = _HOOKS_SOURCE.read_text(encoding="utf-8")
    assert "AllowlistViolationObserver = Callable[[AllowlistViolationError, RoutingContext], None]" in text
    assert "Callable[[object," not in text


def test_ebh_2e_r6_r1_r3_r3_allowlist_error_single_owner() -> None:
    contract_defs = [
        n
        for n in ast.walk(ast.parse(_CONTRACTS_SOURCE.read_text(encoding="utf-8")))
        if isinstance(n, ast.ClassDef) and n.name == "AllowlistViolationError"
    ]
    evaluator_defs = [
        n
        for n in ast.walk(ast.parse(_EVALUATOR_SOURCE.read_text(encoding="utf-8")))
        if isinstance(n, ast.ClassDef) and n.name == "AllowlistViolationError"
    ]
    assert len(contract_defs) == 1
    assert len(evaluator_defs) == 0


def test_ebh_2e_r6_r1_r3_r3_runtime_state_typed_observer_callbacks() -> None:
    source = inspect.getsource(runtime_state_mod.RuntimeState.configure_llm_tracker)
    assert "def _on_evaluated(evaluation: RoutingEvaluation)" in source
    assert "exc: AllowlistViolationError" in source
    assert "context: RoutingContext" in source
    assert "inner: LLMAdapter" in source
    assert "evaluation: RoutingEvaluation" in source
    assert "evaluation: object" not in source
    assert "assert isinstance(evaluation, RoutingEvaluation)" not in source
    assert "assert isinstance(exc, AllowlistViolationError)" not in source
    assert "_cached_identity" not in source
