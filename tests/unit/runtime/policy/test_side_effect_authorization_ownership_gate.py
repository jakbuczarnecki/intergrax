# © Artur Czarnecki. All rights reserved.

"""PLATFORM-SE-FAIL-CLOSED-1-R1 — runtime policy authorization ownership gate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_POLICY_ROOT = Path(__file__).resolve().parents[4] / "intergrax" / "runtime" / "policy"
_AUTHORIZATION_GATE = _POLICY_ROOT / "declarative_tool_authorization_gate.py"
_ERROR_MODULE = "intergrax/runtime/policy/side_effect_authorization_errors.py"


def _static_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if node.module:
                modules.append(node.module)
    return modules


@pytest.mark.gate
def test_declarative_tool_authorization_gate_has_no_nexus_dependency() -> None:
    nexus_imports = [
        module
        for module in _static_imports(_AUTHORIZATION_GATE)
        if module.startswith("intergrax.runtime.nexus")
    ]
    assert not nexus_imports, (
        "declarative_tool_authorization_gate.py must not import Nexus internals; "
        f"found: {nexus_imports}"
    )


def _class_definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    }


@pytest.mark.gate
def test_side_effect_authorization_error_has_single_authoritative_definition() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    scan_roots = (repo_root / "intergrax", repo_root / "tests")
    error_def_paths: set[str] = set()
    reason_def_paths: set[str] = set()
    for root in scan_roots:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                class_names = _class_definitions(path)
            except (SyntaxError, UnicodeDecodeError):
                continue
            rel = str(path.relative_to(repo_root)).replace("\\", "/")
            if "MeaningfulSideEffectAuthorizationRequiredError" in class_names:
                error_def_paths.add(rel)
            if "SideEffectAuthorizationFailureReason" in class_names:
                reason_def_paths.add(rel)
    assert error_def_paths == {_ERROR_MODULE}, (
        f"expected single error definition, found: {sorted(error_def_paths)}"
    )
    assert reason_def_paths == {_ERROR_MODULE}, (
        f"expected single reason enum definition, found: {sorted(reason_def_paths)}"
    )
