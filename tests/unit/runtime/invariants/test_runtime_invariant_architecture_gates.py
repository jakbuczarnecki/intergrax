# © Artur Czarnecki. All rights reserved.

"""Static architecture gates for shared RI foundation."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FOUNDATION_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "invariants"
_CONTRACT_PATH = _REPO_ROOT / "intergrax" / "contracts" / "runtime_invariants.py"
_FORBIDDEN_DOMAIN_PREFIXES = (
    "intergrax.runtime.execution",
    "intergrax.runtime.governance",
    "intergrax.runtime.nexus",
    "intergrax.runtime.diagnostics",
)
_COMPOSITION_ALLOWLIST = frozenset({"foundation_composition.py"})


def _foundation_modules() -> list[Path]:
    paths = sorted(_FOUNDATION_ROOT.glob("*.py"))
    return [p for p in paths if p.name not in _COMPOSITION_ALLOWLIST]


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_foundation_does_not_import_domain_implementations() -> None:
    violations: list[str] = []
    for path in _foundation_modules():
        for module in _imports(path):
            for prefix in _FORBIDDEN_DOMAIN_PREFIXES:
                if module == prefix or module.startswith(prefix + "."):
                    violations.append(f"{path.name}: {module}")
    assert not violations, violations


def test_foundation_no_global_registry_tokens() -> None:
    combined = "\n".join(p.read_text(encoding="utf-8") for p in _foundation_modules())
    assert "register_global" not in combined
    assert "GLOBAL_INVARIANT" not in combined


def test_foundation_no_reflection_adaptation() -> None:
    combined = "\n".join(p.read_text(encoding="utf-8") for p in _foundation_modules())
    assert "getattr(" not in combined
    assert "hasattr(" not in combined
    assert "setattr(" not in combined


def test_public_contract_no_any_stable_abi() -> None:
    source = _CONTRACT_PATH.read_text(encoding="utf-8")
    assert "dict[str, Any]" not in source
    assert re.search(r"\bAny\b", source) is None
