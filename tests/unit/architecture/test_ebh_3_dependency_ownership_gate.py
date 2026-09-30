# © Artur Czarnecki. All rights reserved.

"""EBH-3 — dependency direction and Integrations health composition ownership gates."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_FORBIDDEN_CROSS_LAYER_HEALTH_IMPORT = "intergrax.integrations._shared.health"

_CROSS_LAYER_ROOTS = (
    "intergrax/tools/",
    "intergrax/applications/",
)


def _python_files_under(prefix: str) -> list[Path]:
    root = _REPO_ROOT / prefix.replace("/", "\\") if sys.platform == "win32" else _REPO_ROOT / prefix
    if not root.exists():
        root = _REPO_ROOT / prefix
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _imports_shared_health(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == _FORBIDDEN_CROSS_LAYER_HEALTH_IMPORT:
            return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _FORBIDDEN_CROSS_LAYER_HEALTH_IMPORT:
                    return True
    return False


def test_ebh_3_cross_layer_must_not_import_shared_health_module() -> None:
    violations: list[str] = []
    for prefix in _CROSS_LAYER_ROOTS:
        for path in _python_files_under(prefix):
            if _imports_shared_health(path):
                rel = path.relative_to(_REPO_ROOT).as_posix()
                violations.append(rel)
    assert not violations, (
        "Use intergrax.integrations.registry.health_probes for cross-layer health composition: "
        + ", ".join(violations)
    )


def test_ebh_3_shared_health_module_imports_without_package_init_cycle() -> None:
    probe = textwrap.dedent(
        """
        import intergrax.integrations._shared.health as health

        if not callable(health.health_check_all):
            raise SystemExit(2)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_ebh_3_sanctioned_health_probes_surface_imports() -> None:
    from intergrax.integrations.registry import health_probes

    assert callable(health_probes.health_check_all)
    assert callable(health_probes.health_check_catalog_slugs)
