# © Artur Czarnecki. All rights reserved.

"""AST-based Tier-1 / Tier-2 / Tier-3 dependency boundary gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _agent_dirs() -> set[str]:
    root = REPO / "agents"
    return {
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "__init__.py").is_file() and not p.name.startswith("_")
    }


def _app_dirs() -> set[str]:
    root = REPO / "applications"
    return {
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "pyproject.toml").is_file()
    }


def _iter_py_files(root: Path):
    skip = {".venv", "__pycache__", "build", ".git", "runtime-context"}
    for path in root.rglob("*.py"):
        if any(part in skip for part in path.parts):
            continue
        yield path


def _static_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(
            path.read_text(encoding="utf-8-sig"), filename=str(path)
        )
    except SyntaxError as exc:
        raise AssertionError(f"syntax error in {path}: {exc}") from exc
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


def _top(module: str) -> str:
    return module.split(".", 1)[0]


@pytest.mark.gate
def test_platform_does_not_statically_import_agents_or_applications() -> None:
    agents = _agent_dirs()
    apps = _app_dirs()
    violations: list[str] = []
    for path in _iter_py_files(REPO / "intergrax"):
        for module in _static_imports(path):
            top = _top(module)
            if top == "agents" or top in agents:
                violations.append(
                    f"{path.relative_to(REPO)} imports {module} (agent)"
                )
            if top == "applications" or top in apps:
                violations.append(
                    f"{path.relative_to(REPO)} imports {module} (application)"
                )
    assert not violations, "platform → agent/application edges:\n" + "\n".join(
        violations
    )


@pytest.mark.gate
def test_agents_do_not_statically_import_applications() -> None:
    apps = _app_dirs()
    violations: list[str] = []
    for path in _iter_py_files(REPO / "agents"):
        for module in _static_imports(path):
            top = _top(module)
            if top == "applications" or top in apps:
                violations.append(
                    f"{path.relative_to(REPO)} imports {module} (application)"
                )
    assert not violations, "agent → application edges:\n" + "\n".join(violations)


@pytest.mark.gate
def test_applications_do_not_import_other_applications() -> None:
    apps = _app_dirs()
    violations: list[str] = []
    for app in apps:
        for path in _iter_py_files(REPO / "applications" / app):
            for module in _static_imports(path):
                top = _top(module)
                if top in apps and top != app:
                    violations.append(
                        f"{path.relative_to(REPO)} imports {module} "
                        f"(application {app} → {top})"
                    )
                if top == "applications":
                    parts = module.split(".")
                    if len(parts) > 1 and parts[1] in apps and parts[1] != app:
                        violations.append(
                            f"{path.relative_to(REPO)} imports {module} "
                            f"(application {app} → {parts[1]})"
                        )
    assert not violations, "application → application edges:\n" + "\n".join(
        violations
    )
