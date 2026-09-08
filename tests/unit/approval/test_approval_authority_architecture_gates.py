# © Artur Czarnecki. All rights reserved.

"""Static architecture gates for MP-4D Approval authority integration."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_APPROVAL_PACKAGE = _REPO_ROOT / "intergrax" / "approval"
_FORBIDDEN_IMPORT_PREFIXES = ("intergrax.runtime.nexus",)
_FORBIDDEN_IMPORT_SUBSTRINGS = (
    "GraphExecutor",
    "TaskState",
    "Repository",
    "Database",
    "ApprovalACL",
    "NewPrincipal",
    "ApprovalPrincipal",
    "ApprovalRole",
    "ApprovalPermission",
    "ApprovalPolicy",
)
_FORBIDDEN_SYMBOLS = frozenset(
    {
        "ApprovalPrincipal",
        "ApprovalRole",
        "ApprovalPermission",
        "ApprovalPolicy",
        "ApprovalACL",
        "NewPrincipal",
        "GraphExecutor",
        "TaskState",
    },
)
_REQUIRED_SYMBOLS = (
    "TRUSTED_OPERATION_APPROVAL_CREATE",
    "TRUSTED_OPERATION_APPROVAL_ACTION",
    "ApprovalService",
    "ApprovalAuthorizationDenied",
    "ApprovalAuthorityContextFactory",
    "require_approval_allow",
)


def _approval_modules() -> list[Path]:
    return sorted(path for path in _APPROVAL_PACKAGE.rglob("*.py") if path.is_file())


def _collect_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
    return imports


def _collect_forbidden_symbol_usage(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_SYMBOLS:
            violations.append(
                f"{path.relative_to(_REPO_ROOT)}:{node.lineno} references {node.id}"
            )
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_SYMBOLS:
            violations.append(
                f"{path.relative_to(_REPO_ROOT)}:{node.lineno} references {node.attr}",
            )
    return violations


def test_mp4d_approval_package_has_no_forbidden_imports() -> None:
    violations: list[str] = []
    for module_path in _approval_modules():
        for lineno, module in _collect_imports(module_path):
            if any(
                module == prefix or module.startswith(f"{prefix}.")
                for prefix in _FORBIDDEN_IMPORT_PREFIXES
            ):
                violations.append(f"{module_path.name}:{lineno} imports {module}")
            if any(token in module for token in _FORBIDDEN_IMPORT_SUBSTRINGS):
                violations.append(f"{module_path.name}:{lineno} imports {module}")
    assert not violations, "\n".join(violations)


def test_mp4d_approval_package_has_no_forbidden_symbols() -> None:
    violations: list[str] = []
    for module_path in _approval_modules():
        violations.extend(_collect_forbidden_symbol_usage(module_path))
    assert not violations, "\n".join(violations)


def test_mp4d_approval_package_exposes_authority_surface() -> None:
    service_source = (_APPROVAL_PACKAGE / "service.py").read_text(encoding="utf-8")
    enforcement_source = (_APPROVAL_PACKAGE / "_authority_enforcement.py").read_text(
        encoding="utf-8",
    )
    combined = f"{service_source}\n{enforcement_source}"
    for symbol in _REQUIRED_SYMBOLS:
        assert symbol in combined
