# © Artur Czarnecki. All rights reserved.

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[4]
_PRODUCTION_ROOTS = (_REPO / "intergrax",)
_FORBIDDEN_ROOTS = frozenset({"tests", "testing_support"})


def _is_production_source(path: Path) -> bool:
    rel = path.relative_to(_REPO).as_posix()
    if "/tests/" in f"/{rel}/" or rel.endswith("/tests/conftest.py"):
        return False
    if "/test_" in rel and "/tests/" not in rel:
        return False
    return True


def _violations(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except SyntaxError:
        return []
    rel = path.relative_to(_REPO).as_posix()
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _FORBIDDEN_ROOTS:
                    out.append(f"{rel}:{node.lineno}:{alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in _FORBIDDEN_ROOTS:
                out.append(f"{rel}:{node.lineno}:{node.module}")
    return out


def test_intergrax_production_tree_does_not_import_tests_or_testing_support() -> None:
    violations: list[str] = []
    root = _PRODUCTION_ROOTS[0]
    for path in sorted(root.rglob("*.py")):
        if not _is_production_source(path):
            continue
        rel = path.relative_to(_REPO).as_posix()
        if "/docker/" in rel:
            continue
        violations.extend(_violations(path))
    assert violations == [], "; ".join(violations[:20])
