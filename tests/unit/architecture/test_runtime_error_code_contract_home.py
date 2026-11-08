# © Artur Czarnecki. All rights reserved.

"""Regression: RuntimeErrorCode contract home and Nexus re-export alias."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from intergrax.contracts import runtime_error_code as contract_module
from intergrax.runtime.nexus.errors import (  # noqa: I001 — gate imports alias under test
    error_codes as nexus_alias_module,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[3]
_RECOVERY = _REPO / "intergrax" / "runtime" / "tools" / "tool_operation_recovery.py"

_EXPECTED_MEMBERS: tuple[tuple[str, str], ...] = (
    ("INTERNAL_ERROR", "internal_error"),
    ("VALIDATION_ERROR", "validation_error"),
    ("TIMEOUT", "timeout"),
    ("LLM_ERROR", "llm_error"),
    ("TOOL_ERROR", "tool_error"),
    ("USER_ERROR", "user_error"),
    ("POLICY_ERROR", "policy_error"),
    ("DEPENDENCY_ERROR", "dependency_error"),
    ("QUALITY_ERROR", "quality_error"),
    ("PERMISSION_ERROR", "permission_error"),
    ("RUNTIME_ERROR", "runtime_error"),
    ("ABANDONED", "abandoned"),
    ("STALLED", "stalled"),
)


def _imported_intergrax_prefixes(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "intergrax" or node.module.startswith("intergrax."):
                top = node.module.split(".", 2)
                if len(top) >= 2:
                    found.add(f"intergrax.{top[1]}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "intergrax" or alias.name.startswith("intergrax."):
                    top = alias.name.split(".", 2)
                    if len(top) >= 2:
                        found.add(f"intergrax.{top[1]}")
    return found


def test_runtime_error_code_contract_and_nexus_alias_same_class() -> None:
    contract_cls = contract_module.RuntimeErrorCode
    alias_cls = nexus_alias_module.RuntimeErrorCode
    assert contract_cls is alias_cls
    assert issubclass(contract_cls, str)
    pairs = tuple((m.name, m.value) for m in contract_cls)
    assert pairs == _EXPECTED_MEMBERS


def test_runtime_error_code_contract_module_import_purity() -> None:
    path = Path(contract_module.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    prefixes = _imported_intergrax_prefixes(tree)
    assert prefixes == set()
    assert "intergrax.runtime" not in path.read_text(encoding="utf-8")


def test_runtime_error_code_contract_ast_detects_synthetic_offender() -> None:
    snippet = "from intergrax.runtime.nexus import x\n"
    tree = ast.parse(snippet)
    prefixes = _imported_intergrax_prefixes(tree)
    assert "intergrax.runtime" in prefixes


def test_tool_operation_recovery_has_no_nexus_import() -> None:
    text = _RECOVERY.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not (
                node.module == "intergrax.runtime.nexus" or node.module.startswith("intergrax.runtime.nexus.")
            ), node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not (
                    alias.name == "intergrax.runtime.nexus" or alias.name.startswith("intergrax.runtime.nexus.")
                )
