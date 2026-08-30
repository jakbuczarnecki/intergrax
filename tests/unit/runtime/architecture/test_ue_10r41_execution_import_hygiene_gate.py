# © Artur Czarnecki. All rights reserved.

"""UE-10R4.1 — runtime/execution module-level import hygiene gate."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXECUTION_DIR = _REPO_ROOT / "intergrax" / "runtime" / "execution"


@dataclass(frozen=True, slots=True)
class LocalImportViolation:
    relative_path: str
    line_number: int
    enclosing_function: str
    import_statement: str

    def format(self) -> str:
        return (
            f"{self.relative_path}:{self.line_number}\n"
            f"local import in {self.enclosing_function}():\n"
            f"{self.import_statement}"
        )


def _import_statement(node: ast.Import | ast.ImportFrom) -> str:
    if isinstance(node, ast.Import):
        return "import " + ", ".join(alias.name for alias in node.names)
    module = node.module or ""
    names = ", ".join(alias.name for alias in node.names)
    return f"from {module} import {names}"


def _enclosing_function_name(ancestors: tuple[ast.AST, ...]) -> str | None:
    for ancestor in reversed(ancestors):
        if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ancestor.name
        if isinstance(ancestor, ast.Lambda):
            return "<lambda>"
    return None


def _collect_local_import_violations(path: Path) -> list[LocalImportViolation]:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    relative_path = path.relative_to(_REPO_ROOT).as_posix()
    violations: list[LocalImportViolation] = []

    def visit(node: ast.AST, ancestors: tuple[ast.AST, ...]) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            enclosing_function = _enclosing_function_name(ancestors)
            if enclosing_function is not None:
                violations.append(
                    LocalImportViolation(
                        relative_path=relative_path,
                        line_number=node.lineno,
                        enclosing_function=enclosing_function,
                        import_statement=_import_statement(node),
                    )
                )
        child_ancestors = ancestors + (node,)
        for child in ast.iter_child_nodes(node):
            visit(child, child_ancestors)

    visit(tree, ())
    return violations


def _execution_package_local_import_violations() -> list[LocalImportViolation]:
    violations: list[LocalImportViolation] = []
    for path in sorted(_EXECUTION_DIR.rglob("*.py")):
        violations.extend(_collect_local_import_violations(path))
    return violations


def test_execution_package_has_no_local_imports() -> None:
    violations = _execution_package_local_import_violations()
    assert violations == [], (
        "runtime/execution must use module-level imports only: "
        + "; ".join(violation.format() for violation in violations)
    )


def test_local_import_scan_reports_import_node_kinds() -> None:
    violations = _execution_package_local_import_violations()
    import_count = sum(
        1 for violation in violations if violation.import_statement.startswith("import ")
    )
    import_from_count = sum(
        1 for violation in violations if violation.import_statement.startswith("from ")
    )
    assert import_count == 0
    assert import_from_count == 0
