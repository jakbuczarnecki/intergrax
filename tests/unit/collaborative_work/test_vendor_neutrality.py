# © Artur Czarnecki. All rights reserved.

"""Vendor-neutrality and replaceability proofs for Collaborative Work."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.collaborative_work.persistence import open_sqlite_collaborative_work_repositories
from intergrax.collaborative_work.repository import WorkspaceMembershipRepository
from intergrax.collaborative_work.in_memory_repository import InMemoryWorkspaceMembershipRepository

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOMAIN_FILES = (
    _REPO_ROOT / "intergrax" / "collaborative_work" / "repository.py",
    _REPO_ROOT / "intergrax" / "collaborative_work" / "enforcement_gate.py",
    _REPO_ROOT / "intergrax" / "collaborative_work" / "policy_composition.py",
    _REPO_ROOT / "intergrax" / "collaborative_work" / "authority.py",
    _REPO_ROOT / "intergrax" / "collaborative_work" / "policy_source.py",
)
_ADAPTER_FILES = (
    _REPO_ROOT / "intergrax" / "collaborative_work" / "postgresql_repository.py",
)
_FORBIDDEN_IMPORT_PREFIXES = (
    "boto3",
    "psycopg",
    "sqlalchemy",
    "supabase",
    "sentry_sdk",
    "datadog",
    "opentelemetry",
)


def _import_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_domain_core_imports_no_database_or_observability_vendors() -> None:
    imported: set[str] = set()
    for path in _DOMAIN_FILES:
        imported |= _import_names(path)
    assert not imported.intersection(_FORBIDDEN_IMPORT_PREFIXES)


def test_postgresql_adapter_does_not_import_psycopg_driver() -> None:
    imported = _import_names(_ADAPTER_FILES[0])
    assert "psycopg" not in imported


def test_durable_adapter_is_replaceable_behind_port(tmp_path: Path) -> None:
    in_memory: WorkspaceMembershipRepository = InMemoryWorkspaceMembershipRepository()
    durable = open_sqlite_collaborative_work_repositories(str(tmp_path / "vendor.sqlite"))
    try:
        assert isinstance(in_memory, WorkspaceMembershipRepository)
        assert isinstance(durable.membership, WorkspaceMembershipRepository)
        assert in_memory.capabilities.reference_only is True
        assert durable.membership.capabilities.durable is True
        assert durable.membership.capabilities.reference_only is False
    finally:
        durable.close()
