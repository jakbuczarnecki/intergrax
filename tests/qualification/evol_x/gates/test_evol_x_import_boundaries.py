# © Artur Czarnecki. All rights reserved.

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[4]
_CHANGE_SAFETY = _REPO / "testing_support" / "change_safety"
_FORBIDDEN = ("tests", "tests.qualification")


def _violations_in_tree(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    rel = path.relative_to(_REPO).as_posix()
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _FORBIDDEN or alias.name.startswith("tests."):
                    out.append(f"{rel}:{node.lineno}:{alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in _FORBIDDEN or node.module.startswith("tests."):
                out.append(f"{rel}:{node.lineno}:{node.module}")
    return out


def test_change_safety_does_not_import_tests_package() -> None:
    violations: list[str] = []
    for path in sorted(_CHANGE_SAFETY.rglob("*.py")):
        violations.extend(_violations_in_tree(path))
    assert violations == [], "; ".join(violations)


def test_contracts_roundtrip_enums() -> None:
    from testing_support.change_safety.contracts.enums import ImpactStatus

    assert ImpactStatus.COMPLETE.value == "COMPLETE"
