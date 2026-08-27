# © Artur Czarnecki. All rights reserved.

"""Architecture gate — investigation contracts must not depend on proof scenarios (DIAG-8B)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN_IMPORT_MODULES = frozenset(
    {
        "platform_proofs",
        "platform_proofs.scenarios",
        "platform_proofs.scenarios.ai_incident_investigation",
    }
)

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


_CONTRACT_MODULE = (
    _repo_root()
    / "intergrax"
    / "runtime"
    / "diagnostics"
    / "investigation_contracts.py"
)


def _collect_forbidden_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    rel = path.relative_to(_repo_root()).as_posix()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                if any(
                    module == forbidden or module.startswith(f"{forbidden}.")
                    for forbidden in _FORBIDDEN_IMPORT_MODULES
                ):
                    violations.append(f"{rel}:{node.lineno} imports {module}")
        if isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            if any(
                module == forbidden or module.startswith(f"{forbidden}.")
                for forbidden in _FORBIDDEN_IMPORT_MODULES
            ):
                violations.append(f"{rel}:{node.lineno} imports from {module}")
    return violations


def test_investigation_contracts_do_not_import_ai_incident_investigation_proof() -> None:
    violations = _collect_forbidden_imports(_CONTRACT_MODULE)
    assert violations == []
