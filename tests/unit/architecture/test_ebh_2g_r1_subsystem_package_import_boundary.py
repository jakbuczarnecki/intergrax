# © Artur Czarnecki. All rights reserved.

"""EBH-2G-R1 — subsystem package root leaf-import boundary gates."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_ROOT_PACKAGES: tuple[tuple[str, Path], ...] = (
    ("intergrax.integrations", _REPO_ROOT / "intergrax" / "integrations" / "__init__.py"),
    (
        "intergrax.integrations.contracts",
        _REPO_ROOT / "intergrax" / "integrations" / "contracts" / "__init__.py",
    ),
    ("intergrax.skills", _REPO_ROOT / "intergrax" / "skills" / "__init__.py"),
    ("intergrax.tools", _REPO_ROOT / "intergrax" / "tools" / "__init__.py"),
)

_LEAF_ISOLATION_CASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "import intergrax.integrations.contracts.base",
        (
            "intergrax.tools.registry.wiring",
            "intergrax.skills.resolver",
            "intergrax.integrations.registry.bootstrap",
        ),
    ),
    (
        "import intergrax.skills.core.contracts",
        (
            "intergrax.tools.registry.wiring",
            "intergrax.integrations.registry.bootstrap",
            "intergrax.skills.resolver",
        ),
    ),
    (
        "import intergrax.tools.core.contracts",
        (
            "intergrax.tools.registry.wiring",
            "intergrax.integrations.registry.bootstrap",
            "intergrax.skills.resolver",
        ),
    ),
)


def _import_lines(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    lines: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            lines.append(f"from {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                lines.append(f"import {alias.name}")
    return lines


def _run_import_probe(statement: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", statement],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(("label", "init_path"), _ROOT_PACKAGES)
def test_subsystem_root_init_is_side_effect_free(label: str, init_path: Path) -> None:
    source = init_path.read_text(encoding="utf-8")
    assert "__getattr__" not in source, f"{label} root must not define __getattr__ compatibility"
    assert "__all__" not in source, f"{label} root must not re-export via __all__"
    import_lines = _import_lines(init_path)
    assert import_lines == [], f"{label} root must not import submodules; found: {import_lines}"


@pytest.mark.parametrize("root_import", [f"import {name}" for name, _ in _ROOT_PACKAGES])
def test_subsystem_root_package_import_succeeds(root_import: str) -> None:
    completed = _run_import_probe(f"{root_import}; print('ok')")
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(("leaf_import", "forbidden_modules"), _LEAF_ISOLATION_CASES)
def test_leaf_contract_import_does_not_materialize_runtime_graph(
    leaf_import: str, forbidden_modules: tuple[str, ...]
) -> None:
    forbidden_literal = repr(forbidden_modules)
    statement = (
        f"{leaf_import}\n"
        "import sys\n"
        f"forbidden = {forbidden_literal}\n"
        "loaded = [m for m in forbidden if m in sys.modules]\n"
        "assert not loaded, f'unexpected modules loaded: {{loaded}}'\n"
    )
    completed = _run_import_probe(statement)
    assert completed.returncode == 0, completed.stdout + completed.stderr
