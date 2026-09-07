# © Artur Czarnecki. All rights reserved.

"""Composition-contract tests for Collaborative Work persistence bundles (COLLAB-WORK-2D)."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from typing import get_type_hints
from unittest.mock import patch

import pytest

from intergrax.collaborative_work.persistence import (
    CollaborativeWorkRepositories,
    CollaborativeWorkRepositoriesWithSharedWork,
    CollaborativeWorkSharedWorkRepositories,
    collaborative_work_core_repositories,
    open_postgresql_collaborative_work_repositories,
    open_sqlite_collaborative_work_repositories,
)
from intergrax.collaborative_work.persistence_provider import (
    resolve_collaborative_work_repositories,
)
from intergrax.collaborative_work.repository import AssignmentRepository, WorkItemRepository
from intergrax.integrations.providers.relational_store.postgresql.register import (
    register_postgresql_integration,
)
from intergrax.integrations.providers.relational_store.sqlite.register import (
    register_sqlite_integration,
)
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.catalog_manifests import POSTGRESQL, SQLITE
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PERSISTENCE_MODULE = _REPO_ROOT / "intergrax" / "collaborative_work" / "persistence.py"


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    yield
    clear_catalog()


def test_sqlite_factory_returns_full_shared_work_bundle(tmp_path: Path) -> None:
    bundle = open_sqlite_collaborative_work_repositories(str(tmp_path / "sqlite.sqlite"))
    try:
        assert isinstance(bundle, CollaborativeWorkRepositoriesWithSharedWork)
        assert isinstance(bundle.shared_work, CollaborativeWorkSharedWorkRepositories)
        assert isinstance(bundle.work_item, WorkItemRepository)
        assert isinstance(bundle.assignment, AssignmentRepository)
        assert isinstance(bundle.core, CollaborativeWorkRepositories)
        assert collaborative_work_core_repositories(bundle) is bundle.core
    finally:
        bundle.close()


def test_postgresql_factory_returns_full_shared_work_bundle() -> None:
    from intergrax.integrations.providers.relational_store.postgresql.config import (
        PostgreSQLIntegrationConfig,
    )

    with patch(
        "intergrax.collaborative_work.persistence.PostgreSQLCollaborativeWorkStore",
    ) as store_cls:
        store = store_cls.return_value
        bundle = open_postgresql_collaborative_work_repositories(
            config=PostgreSQLIntegrationConfig(dsn="postgresql://localhost/test"),
        )
    try:
        assert isinstance(bundle, CollaborativeWorkRepositoriesWithSharedWork)
        assert isinstance(bundle.shared_work, CollaborativeWorkSharedWorkRepositories)
        assert isinstance(bundle.work_item, WorkItemRepository)
        assert isinstance(bundle.assignment, AssignmentRepository)
        assert isinstance(bundle.core, CollaborativeWorkRepositories)
        assert collaborative_work_core_repositories(bundle) is bundle.core
    finally:
        bundle.close()
        store.close.assert_called_once()


def test_postgresql_open_return_annotation_guarantees_shared_work() -> None:
    hints = get_type_hints(open_postgresql_collaborative_work_repositories)
    assert hints["return"] is CollaborativeWorkRepositoriesWithSharedWork


def test_core_bundle_has_no_optional_shared_work_fields() -> None:
    fields = dataclasses.fields(CollaborativeWorkRepositories)
    field_names = {field.name for field in fields}
    assert "shared_work" not in field_names
    assert "work_item" not in field_names
    assert "assignment" not in field_names
    for field in fields:
        assert field.type is not None
        assert "None" not in str(field.type)


def test_shared_work_bundle_has_required_shared_work_field() -> None:
    fields = dataclasses.fields(CollaborativeWorkRepositoriesWithSharedWork)
    shared_work_field = next(field for field in fields if field.name == "shared_work")
    assert "None" not in str(shared_work_field.type)


def test_persistence_module_has_no_runtime_error_capability_discovery() -> None:
    tree = ast.parse(_PERSISTENCE_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            func = node.exc.func
            if isinstance(func, ast.Name) and func.id == "RuntimeError":
                pytest.fail(
                    f"persistence.py:{node.lineno} uses RuntimeError for capability discovery"
                )


def test_sqlite_profile_materializes_full_shared_work_bundle(tmp_path: Path) -> None:
    register_sqlite_integration()
    profile = IntegrationProfile(
        relational_store=SQLITE,
        options={SQLITE.slug: {"data_dir": str(tmp_path)}},
    )
    bundle = resolve_collaborative_work_repositories(profile)
    try:
        assert isinstance(bundle, CollaborativeWorkRepositoriesWithSharedWork)
        assert isinstance(bundle.work_item, WorkItemRepository)
        assert isinstance(bundle.assignment, AssignmentRepository)
    finally:
        bundle.close()


def test_postgresql_profile_materializes_full_shared_work_bundle() -> None:
    register_postgresql_integration()
    profile = IntegrationProfile(
        relational_store=POSTGRESQL,
        options={POSTGRESQL.slug: {"dsn": "postgresql://localhost/test"}},
    )
    with patch(
        "intergrax.collaborative_work.persistence.open_postgresql_collaborative_work_repositories",
    ) as open_fn:
        open_fn.return_value = CollaborativeWorkRepositoriesWithSharedWork(
            core=CollaborativeWorkRepositories(
                membership=object(),  # type: ignore[arg-type]
                delegation=object(),  # type: ignore[arg-type]
                principal_authority=object(),  # type: ignore[arg-type]
                policy=object(),  # type: ignore[arg-type]
                operation_profile=object(),  # type: ignore[arg-type]
                store=type("Store", (), {"close": lambda self: None})(),
            ),
            shared_work=CollaborativeWorkSharedWorkRepositories(
                work_item=object(),  # type: ignore[arg-type]
                assignment=object(),  # type: ignore[arg-type]
            ),
        )
        bundle = resolve_collaborative_work_repositories(profile)
    assert isinstance(bundle, CollaborativeWorkRepositoriesWithSharedWork)
    bundle.close()


def test_mp1_callers_use_core_from_sqlite_bundle(tmp_path: Path) -> None:
    bundle = open_sqlite_collaborative_work_repositories(str(tmp_path / "mp1.sqlite"))
    try:
        core = collaborative_work_core_repositories(bundle)
        assert isinstance(core, CollaborativeWorkRepositories)
        assert core.membership is bundle.membership
        assert core.delegation is bundle.delegation
    finally:
        bundle.close()


def test_sqlite_open_return_annotation_guarantees_shared_work() -> None:
    hints = get_type_hints(open_sqlite_collaborative_work_repositories)
    assert hints["return"] is CollaborativeWorkRepositoriesWithSharedWork
