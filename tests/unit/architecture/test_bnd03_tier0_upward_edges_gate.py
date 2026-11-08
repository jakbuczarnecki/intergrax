# © Artur Czarnecki. All rights reserved.

"""BND-03 — Tier-0 must not import agents/applications or name Tier-2/3 modules in CLI code."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[3]
_TIER0 = _REPO / "intergrax"
_TIER0_CLI = _TIER0 / "cli"


def _tier23_roots(repo: Path) -> set[str]:
    roots = {"agents", "applications"}
    for tier in ("agents", "applications"):
        base = repo / tier
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir() and (child / "__init__.py").is_file():
                roots.add(child.name)
    return roots


def _static_upward_imports(root: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(_REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".", 1)[0]
                if top in {"agents", "applications"}:
                    violations.append(f"{rel} -> {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".", 1)[0]
                    if top in {"agents", "applications"}:
                        violations.append(f"{rel} -> {alias.name}")
    return violations


def _cli_tier23_module_string_violations(repo: Path) -> list[str]:
    tier_roots = _tier23_roots(repo)
    violations: list[str] = []
    if not _TIER0_CLI.is_dir():
        return violations
    for path in sorted(_TIER0_CLI.rglob("*.py")):
        rel = path.relative_to(_REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if "." in value:
                    head = value.split(".", 1)[0]
                    if head in tier_roots and not value.startswith("intergrax."):
                        violations.append(f"{rel}: literal {value!r}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"import_module", "__import__"}:
                    if node.args and isinstance(node.args[0], ast.Constant):
                        arg = node.args[0].value
                        if isinstance(arg, str):
                            head = arg.split(".", 1)[0]
                            if head in tier_roots:
                                violations.append(f"{rel}: import_module({arg!r})")
    return violations


def test_tier0_has_no_static_upward_imports() -> None:
    assert _static_upward_imports(_TIER0) == []


def test_tier0_cli_has_no_tier2_tier3_module_strings() -> None:
    assert _cli_tier23_module_string_violations(_REPO) == []


def test_negative_control_static_upward_import_detected(tmp_path: Path) -> None:
    snippet = "from applications.foo import bar\n"
    fake = tmp_path / "intergrax" / "fake.py"
    fake.parent.mkdir(parents=True)
    fake.write_text(snippet, encoding="utf-8")
    violations = _static_upward_imports(tmp_path / "intergrax")
    assert any("applications.foo" in item for item in violations)


def test_intergrax_applications_path_not_flagged(tmp_path: Path) -> None:
    snippet = 'x = "intergrax.applications._shared"\n'
    fake = tmp_path / "intergrax" / "cli" / "x.py"
    fake.parent.mkdir(parents=True)
    fake.write_text(snippet, encoding="utf-8")
    assert _cli_tier23_module_string_violations(tmp_path) == []
