# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for SQLite integration provider (Phase M.4)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.collaborative_work.materialization_factory import (
    CollaborativeWorkMaterializationBinder,
    CollaborativeWorkPersistenceFactory,
)
from intergrax.integrations._shared.conformance import assert_relational_store
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.relational_store.sqlite.adapter import (
    _SQLiteRelationalStore,
)
from intergrax.integrations.providers.relational_store.sqlite.bundle import (
    create_sqlite_relational_store,
)
from intergrax.integrations.providers.relational_store.sqlite.paths import (
    RELATIONAL_DB_NAME,
)
from intergrax.integrations.providers.relational_store.sqlite.integration import (
    SqliteRelationalStoreIntegration,
)
from intergrax.integrations.providers.relational_store.sqlite.register import (
    register_sqlite_integration,
)
from intergrax.integrations.registry.bootstrap import (
    register_default_integrations,
    reset_default_integrations_state,
)
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit

_BUNDLE_PATH = (
    Path(__file__).resolve().parents[5]
    / "intergrax"
    / "integrations"
    / "providers"
    / "relational_store"
    / "sqlite"
    / "bundle.py"
)


def _bundle_module_level_imports(module: str) -> list[int]:
    tree = ast.parse(
        _BUNDLE_PATH.read_text(encoding="utf-8"), filename=str(_BUNDLE_PATH)
    )
    lines: list[int] = []

    def _scan_body(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
                if node.test.id == "TYPE_CHECKING":
                    continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == module or alias.name.startswith(f"{module}."):
                        lines.append(node.lineno)
            elif isinstance(node, ast.ImportFrom):
                if node.module == module or (
                    node.module is not None and node.module.startswith(f"{module}.")
                ):
                    lines.append(node.lineno)
            elif isinstance(node, ast.ClassDef):
                _scan_body(node.body)

    _scan_body(tree.body)
    return lines


def _public_method_return_annotations() -> list[str]:
    tree = ast.parse(
        _BUNDLE_PATH.read_text(encoding="utf-8"), filename=str(_BUNDLE_PATH)
    )
    missing: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if (
            node.name.startswith("_")
            and node.name != "_sqlite_materialization_paths_from_options"
        ):
            continue
        if node.name not in {
            "materialize_collaborative_work_repositories",
            "bind_collaborative_work_materialization",
            "_sqlite_materialization_paths_from_options",
        }:
            continue
        if node.returns is None:
            missing.append(node.name)
        elif isinstance(node.returns, ast.Name) and node.returns.id in {
            "Any",
            "object",
        }:
            missing.append(f"{node.name}:returns={node.returns.id}")
    return missing


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


def test_sqlite_relational_store_execute_and_fetch(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    store = _SQLiteRelationalStore(db_path)
    assert_relational_store(store)

    store.connect()
    store.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    store.execute("INSERT INTO items (name) VALUES (?)", ("alpha",))
    rows = store.fetch_all("SELECT name FROM items")
    store.close()

    assert [row["name"] for row in rows] == ["alpha"]


def test_register_and_resolve_via_lab_profile(tmp_path: Path) -> None:
    register_sqlite_integration()
    profile = IntegrationProfile(relational_store="sqlite")

    store = resolve(
        IntegrationCategory.RELATIONAL_STORE,
        profile=profile,
        config={"data_dir": str(tmp_path)},
    )

    assert_relational_store(store)
    assert isinstance(store, SqliteRelationalStoreIntegration)
    assert store.db_path == tmp_path / RELATIONAL_DB_NAME


def test_register_default_integrations_includes_sqlite(tmp_path: Path) -> None:
    register_default_integrations()
    profile = IntegrationProfile.lab()

    store = resolve(
        IntegrationCategory.RELATIONAL_STORE,
        profile=profile,
        config={"data_dir": str(tmp_path)},
    )

    assert isinstance(store, SqliteRelationalStoreIntegration)


def test_create_sqlite_relational_store_catalog_factory(tmp_path: Path) -> None:
    store = create_sqlite_relational_store(data_dir=tmp_path)
    assert_relational_store(store)
    assert store.db_path == tmp_path / RELATIONAL_DB_NAME


def test_sqlite_factory_structurally_conforms_to_cw_materialization_binder() -> None:
    assert isinstance(
        create_sqlite_relational_store, CollaborativeWorkMaterializationBinder
    )


def test_sqlite_bound_materializer_conforms_to_cw_persistence_factory(
    tmp_path: Path,
) -> None:
    materializer = (
        create_sqlite_relational_store.bind_collaborative_work_materialization(
            {"data_dir": str(tmp_path)},
        )
    )
    assert isinstance(materializer, CollaborativeWorkPersistenceFactory)


def test_sqlite_bundle_has_no_module_level_cw_persistence_import() -> None:
    lines = _bundle_module_level_imports("intergrax.collaborative_work.persistence")
    assert not lines, f"module-level CW persistence imports at lines: {lines}"


def test_sqlite_bundle_collaborative_work_seam_return_annotations() -> None:
    missing = _public_method_return_annotations()
    assert not missing, f"missing or weak return annotations: {missing}"
