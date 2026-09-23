# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5-H2-R1 — canonical orchestration tool MSE operation identifier ownership."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution.suspended_operation.authority_scope_compat import (
    CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID as COMPAT_EXPORT,
)
from intergrax.contracts.orchestration_tool_invocation_mse_operation import (
    CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID,
)
from intergrax.runtime.nexus.tools.tool_invocation_inner_governance import (
    TOOL_INVOCATION_INNER_ACTION_PREFIX,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OWNER_MODULE = (
    _REPO_ROOT / "intergrax/contracts/orchestration_tool_invocation_mse_operation.py"
)
_LITERAL = "orchestration.tool_invocation_authorization"


def _module_level_literal_definitions(tree: ast.Module, literal: str) -> int:
    count = 0
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if node.value is not None and _expr_is_string_constant(node.value, literal):
                count += 1
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None and _expr_is_string_constant(node.value, literal):
                count += 1
    return count


def _expr_is_string_constant(node: ast.expr, literal: str) -> bool:
    if isinstance(node, ast.Constant) and node.value == literal:
        return True
    if isinstance(node, ast.JoinedStr):
        return False
    return False


def _production_literal_definition_count(literal: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    intergrax_root = _REPO_ROOT / "intergrax"
    for path in intergrax_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        if not isinstance(tree, ast.Module):
            continue
        n = _module_level_literal_definitions(tree, literal)
        if n:
            rel = str(path.relative_to(_REPO_ROOT)).replace("\\", "/")
            counts[rel] = n
    return counts


@pytest.mark.gate
def test_canonical_mse_operation_id_has_single_production_literal_owner() -> None:
    counts = _production_literal_definition_count(_LITERAL)
    assert counts == {
        "intergrax/contracts/orchestration_tool_invocation_mse_operation.py": 1,
    }


@pytest.mark.gate
def test_runtime_and_compat_reuse_contract_constant() -> None:
    assert (
        TOOL_INVOCATION_INNER_ACTION_PREFIX
        is CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID
    )
    assert COMPAT_EXPORT is CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID
    assert CANONICAL_ORCHESTRATION_TOOL_MSE_OPERATION_ID == _LITERAL


@pytest.mark.gate
def test_owner_module_is_contract_not_nexus() -> None:
    assert _OWNER_MODULE.is_file()
    source = _OWNER_MODULE.read_text(encoding="utf-8")
    assert "intergrax.runtime.nexus" not in source
