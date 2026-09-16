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
_CLI_MODULE = _REPO_ROOT / "scripts" / "maintenance" / "human_decision_legacy_disposition_cli.py"
_RUNTIME_HUMAN_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "human"
_NEXUS_HUMAN_RESPONSE = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "orchestration" / "human_response.py"
)
_FORBIDDEN_MIGRATION_IMPORT = "intergrax.runtime.migration.human_decision_legacy_disposition"
_FORBIDDEN_ARCHIVE_IMPORT = "intergrax.contracts.human_decision_legacy_disposition"
_REFLECTION_FORBIDDEN = (
    "getattr(",
    ".__dataclass_fields__",
    "vars(",
    ".__dict__",
    "setattr(",
    "hasattr(",
    "object.__setattr__",
)
_ALLOWED_MIGRATION_IMPORTERS = (
    _REPO_ROOT / "scripts" / "maintenance" / "human_decision_legacy_disposition_cli.py",
    _MIGRATION_MODULE,
)


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


def test_mp4r6_disposition_migration_archive_path_has_no_reflection() -> None:
    source = _MIGRATION_MODULE.read_text(encoding="utf-8-sig")
    for token in _REFLECTION_FORBIDDEN:
        assert token not in source, token


def test_mp4r6_cli_exposes_only_wired_disposition_strategies() -> None:
    source = _CLI_MODULE.read_text(encoding="utf-8-sig")
    assert "CLI_DISPOSITION_STRATEGIES" in source
    assert "provenance_recovery" not in source.split("CLI_DISPOSITION_STRATEGIES", 1)[1].split(
        "def _parse_args", 1
    )[0]
    assert "controlled_archive" not in source
    assert "no_data_present" not in source.split("choices=", 1)[1].split(")", 1)[0]


def test_mp4r6_disposition_migration_imported_only_by_admin_surfaces() -> None:
    violations: list[str] = []
    scan_roots = (
        _REPO_ROOT / "intergrax",
        _REPO_ROOT / "agents",
        _REPO_ROOT / "applications",
        _REPO_ROOT / "scripts",
    )
    allowed = {path.resolve() for path in _ALLOWED_MIGRATION_IMPORTERS}
    for root in scan_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            resolved = path.resolve()
            if resolved in allowed:
                continue
            if "tests" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8-sig")
            except OSError:
                continue
            if _FORBIDDEN_MIGRATION_IMPORT not in text:
                continue
            violations.append(path.relative_to(_REPO_ROOT).as_posix())
    assert not violations, "\n".join(sorted(set(violations)))
