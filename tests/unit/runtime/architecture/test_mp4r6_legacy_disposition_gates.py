# © Artur Czarnecki. All rights reserved.

"""MP-4R6 — legacy human decision disposition architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MIGRATION_MODULE = (
    _REPO_ROOT / "intergrax" / "runtime" / "migration" / "human_decision_legacy_disposition.py"
)
_RUNTIME_HUMAN_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "human"
_NEXUS_HUMAN_RESPONSE = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "orchestration" / "human_response.py"
)
_FORBIDDEN_MIGRATION_IMPORT = "intergrax.runtime.migration.human_decision_legacy_disposition"
_FORBIDDEN_ARCHIVE_IMPORT = "intergrax.contracts.human_decision_legacy_disposition"


def _collect_import_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_mp4r6_disposition_migration_does_not_map_user_id_to_approver() -> None:
    source = _MIGRATION_MODULE.read_text(encoding="utf-8-sig")
    assert 'row["user_id"]' not in source
    assert "legacy_unknown_approver" not in source
    assert "local_development_approver_evidence" not in source


def test_mp4r6_runtime_human_store_does_not_import_disposition_migration() -> None:
    violations: list[str] = []
    for path in _RUNTIME_HUMAN_ROOT.rglob("*.py"):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for module in _collect_import_modules(path):
            if module == _FORBIDDEN_MIGRATION_IMPORT or module.startswith(
                f"{_FORBIDDEN_MIGRATION_IMPORT}."
            ):
                violations.append(rel)
    assert not violations, "\n".join(violations)


def test_mp4r6_nexus_hitl_restore_does_not_import_legacy_archive_contract() -> None:
    modules = _collect_import_modules(_NEXUS_HUMAN_RESPONSE)
    for module in modules:
        assert module != _FORBIDDEN_ARCHIVE_IMPORT
        assert not module.startswith(f"{_FORBIDDEN_ARCHIVE_IMPORT}.")
