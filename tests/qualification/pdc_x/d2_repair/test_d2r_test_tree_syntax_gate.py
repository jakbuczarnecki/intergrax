# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-B pinned test: F-D2-09 a syntax error cannot hide behind a skip marker in the D2-cited trees."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_TREES = (
    "tests/unit/rag",
    "tests/integration/rag",
    "tests/integration/context",
    "tests/unit/context",
    "tests/qualification/pdc_x/d2",
    "tests/qualification/pdc_x/d2_repair",
)


def _syntax_errors(root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:  # IndentationError and TabError are subclasses
            errors.append(f"{path.relative_to(_REPO).as_posix()}:{exc.lineno}: {exc.msg}")
    return errors


def test_f_d2_09_every_python_file_of_the_d2_cited_test_trees_parses() -> None:
    errors: list[str] = []
    for tree in _TREES:
        root = _REPO / tree
        assert root.is_dir(), f"missing D2-cited test tree: {tree}"
        errors.extend(_syntax_errors(root))
    assert errors == []


def test_f_d2_09_the_gate_detects_an_indentation_error(tmp_path: Path) -> None:
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (tmp_path / "test_broken.py").write_text("def test_broken():\n    x = 1\n      y = 2\n", encoding="utf-8")

    errors = _syntax_errors(tmp_path)

    assert len(errors) == 1
    assert "test_broken.py" not in "".join(e for e in errors if "test_ok.py" in e)
    assert errors[0].count(":") >= 2
