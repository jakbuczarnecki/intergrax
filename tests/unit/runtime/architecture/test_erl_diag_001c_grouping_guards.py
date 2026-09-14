# © Artur Czarnecki. All rights reserved.

"""ERL-DIAG-001C architecture guards."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONTRACTS_GROUPING = (
    _REPO_ROOT
    / "intergrax"
    / "contracts"
    / "enterprise_reliability"
    / "diagnostics"
    / "grouping.py"
)


def test_grouping_contracts_do_not_import_runtime() -> None:
    imports = _collect_imports(_CONTRACTS_GROUPING)
    for module in imports:
        assert not module.startswith("intergrax.runtime"), module


def test_grouping_contracts_have_no_payment_terms() -> None:
    source = _CONTRACTS_GROUPING.read_text(encoding="utf-8").lower()
    assert "payment" not in source


def test_runtime_adapter_imports_public_grouping_contract() -> None:
    adapter = (
        _REPO_ROOT
        / "intergrax"
        / "runtime"
        / "diagnostics"
        / "reliability"
        / "reliability_observation_grouping_adapter.py"
    )
    imports = _collect_imports(adapter)
    assert "intergrax.contracts.enterprise_reliability.diagnostics.grouping" in imports


def _collect_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports
