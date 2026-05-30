# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for MongoDB integration provider (Phase M.6 P2)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations._shared.conformance import assert_document_store
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.integrations.providers.mongodb.adapter import MongoDBDocumentStore
from intergrax.integrations.providers.mongodb.bundle import (
    MongoDBIntegrationBundle,
    create_mongodb_document_store,
    create_mongodb_integration,
)
from intergrax.integrations.providers.mongodb.config import (
    ENV_MONGODB_DATABASE,
    ENV_MONGODB_URI,
    MongoDBIntegrationConfig,
)
from intergrax.integrations.providers.mongodb.register import register_mongodb_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MONGODB_PKG = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "mongodb"
_THIS_TEST = Path(__file__).resolve()
_SCAN_ROOTS = ("intergrax", "applications", "agents", "tests")
_SKIP_DIR_NAMES = {".venv", "build", "__pycache__", "node_modules"}
_FORBIDDEN_OUTSIDE_PROVIDER = (
    "MongoCollectionClient(",
    "MongoDBDocumentStore(",
    "integrations.providers.mongodb.client",
    "integrations.providers.mongodb.opens",
    "import pymongo",
    "from pymongo",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


class _FakeCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def sort(self, _field: str, _direction: int) -> _FakeCursor:
        return self

    def limit(self, _count: int) -> _FakeCursor:
        return self

    def __iter__(self) -> Iterable[dict[str, Any]]:
        return iter(self._rows)


class _FakeCollection:
    def __init__(self) -> None:
        self.storage: dict[tuple[str, str], dict[str, Any]] = {}
        self.operations: list[tuple[str, Any]] = []

    def find_one(self, query: dict[str, Any]) -> Optional[dict[str, Any]]:
        self.operations.append(("find_one", query))
        if "partition_key" in query and "row_key" in query:
            return self.storage.get((query["partition_key"], query["row_key"]))
        return None

    def replace_one(self, query: dict[str, Any], doc: dict[str, Any], *, upsert: bool = False) -> None:
        self.operations.append(("replace_one", query, doc, upsert))
        key = (doc["partition_key"], doc["row_key"])
        self.storage[key] = dict(doc)

    def delete_one(self, query: dict[str, Any]) -> None:
        self.operations.append(("delete_one", query))
        if "partition_key" in query and "row_key" in query:
            self.storage.pop((query["partition_key"], query["row_key"]), None)

    def find(self, query: dict[str, Any]) -> _FakeCursor:
        self.operations.append(("find", query))
        rows: list[dict[str, Any]] = []
        prefix = None
        row_key_filter = query.get("row_key")
        if isinstance(row_key_filter, dict) and "$regex" in row_key_filter:
            prefix = str(row_key_filter["$regex"]).removeprefix("^")
        for (partition_key, row_key), doc in sorted(self.storage.items()):
            if partition_key != query.get("partition_key"):
                continue
            if prefix is not None and not row_key.startswith(prefix):
                continue
            rows.append(dict(doc))
        return _FakeCursor(rows)


class _FakeMongoClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _mongodb_config() -> MongoDBIntegrationConfig:
    return MongoDBIntegrationConfig(
        uri="mongodb://localhost:27017",
        database="intergrax",
        collection_name="intergrax_documents",
    )


def _collection_factory(collection: _FakeCollection | None = None):
    fake = collection or _FakeCollection()

    def _factory() -> _FakeCollection:
        return fake

    return _factory, fake


def _iter_python_files(*roots: str):
    for root_name in roots:
        root = _PROJECT_ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if any(part in _SKIP_DIR_NAMES for part in path.parts):
                continue
            yield path


def test_pymongo_only_imported_in_opens_module() -> None:
    violations: list[str] = []
    for path in _MONGODB_PKG.glob("*.py"):
        if path.name == "opens.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "pymongo" in text:
            violations.append(path.name)
    assert violations == []


def test_mongodb_not_constructed_outside_provider_package() -> None:
    violations: list[str] = []
    for path in _iter_python_files(*_SCAN_ROOTS):
        if path.resolve() == _THIS_TEST.resolve():
            continue
        if _MONGODB_PKG in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_OUTSIDE_PROVIDER:
            if pattern in text:
                violations.append(f"{path.relative_to(_PROJECT_ROOT).as_posix()}: {pattern}")
    assert violations == []


def test_mongodb_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_MONGODB_URI, "mongodb://db.example:27017")
    monkeypatch.setenv(ENV_MONGODB_DATABASE, "agents")
    config = MongoDBIntegrationConfig.from_env()
    assert config.uri == "mongodb://db.example:27017"
    assert config.qualified_collection() == ("agents", "intergrax_documents")


def test_mongodb_config_requires_uri() -> None:
    factory, _ = _collection_factory()
    with pytest.raises(IntegrationConfigurationError, match="uri is required"):
        create_mongodb_document_store(uri="", collection_factory=factory)


def test_put_and_get_round_trip() -> None:
    factory, collection = _collection_factory()
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)

    store.put(
        DocumentRecord(
            partition_key="tenant-1",
            row_key="evt-1",
            data={"status": "ok", "count": 2},
        )
    )
    doc = store.get("tenant-1", "evt-1")

    assert doc is not None
    assert doc.data["status"] == "ok"
    assert doc.data["count"] == 2
    assert any(op[0] == "replace_one" for op in collection.operations)
    assert_document_store(store)


def test_get_returns_none_for_missing_row() -> None:
    factory, _ = _collection_factory()
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)

    assert store.get("tenant-1", "missing") is None


def test_delete_removes_row() -> None:
    factory, collection = _collection_factory()
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)
    store.put(DocumentRecord(partition_key="t1", row_key="r1", data={"x": 1}))

    store.delete("t1", "r1")

    assert store.get("t1", "r1") is None
    assert any(op[0] == "delete_one" for op in collection.operations)


def test_query_lists_partition_rows() -> None:
    factory, _ = _collection_factory()
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)
    store.put(DocumentRecord(partition_key="t1", row_key="a", data={"n": 1}))
    store.put(DocumentRecord(partition_key="t1", row_key="b", data={"n": 2}))
    store.put(DocumentRecord(partition_key="t2", row_key="c", data={"n": 3}))

    result = store.query("t1", limit=10)

    assert result.total == 2
    assert {doc.row_key for doc in result.documents} == {"a", "b"}


def test_query_with_row_key_prefix() -> None:
    factory, collection = _collection_factory()
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)
    store.put(DocumentRecord(partition_key="t1", row_key="2026-05-01", data={}))
    store.put(DocumentRecord(partition_key="t1", row_key="2026-05-02", data={}))
    store.put(DocumentRecord(partition_key="t1", row_key="2026-06-01", data={}))

    result = store.query("t1", row_key_prefix="2026-05", limit=10)

    assert result.total == 2
    assert all(doc.row_key.startswith("2026-05") for doc in result.documents)
    find_ops = [op for op in collection.operations if op[0] == "find"]
    assert find_ops[-1][1]["row_key"] == {"$regex": "^2026-05"}


def test_put_with_ttl_sets_expires_at() -> None:
    factory, collection = _collection_factory()
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)

    store.put(
        DocumentRecord(
            partition_key="t1",
            row_key="ttl-row",
            data={"temp": True},
            ttl_seconds=3600,
        )
    )

    doc = collection.storage[("t1", "ttl-row")]
    assert doc["expires_at"] is not None
    assert doc["expires_at"] > datetime.now(tz=UTC)


def test_get_expired_document_returns_none_and_deletes() -> None:
    factory, collection = _collection_factory()
    collection.storage[("t1", "expired")] = {
        "partition_key": "t1",
        "row_key": "expired",
        "data": {"x": 1},
        "expires_at": datetime.now(tz=UTC) - timedelta(seconds=30),
    }
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)

    assert store.get("t1", "expired") is None
    assert ("t1", "expired") not in collection.storage


def test_close_closes_underlying_client() -> None:
    factory, _ = _collection_factory()
    fake_client = _FakeMongoClient()
    store = create_mongodb_document_store(
        **_mongodb_config().model_dump(),
        collection_factory=factory,
        client=fake_client,
    )

    store.close()

    assert fake_client.closed
    with pytest.raises(IntegrationConfigurationError, match="closed"):
        store.get("t1", "r1")


def test_create_mongodb_integration_bundle() -> None:
    factory, _ = _collection_factory()
    bundle = create_mongodb_integration(**_mongodb_config().model_dump(), collection_factory=factory)

    assert isinstance(bundle, MongoDBIntegrationBundle)
    assert isinstance(bundle.document_store, MongoDBDocumentStore)
    assert bundle.config.database == "intergrax"


def test_register_and_resolve_via_profile() -> None:
    register_mongodb_integration()
    profile = IntegrationProfile(document_store=IntegrationSlug.MONGODB)
    factory, _ = _collection_factory()

    store = resolve(
        IntegrationCategory.DOCUMENT_STORE,
        profile=profile,
        config={**_mongodb_config().model_dump(), "collection_factory": factory},
    )

    assert_document_store(store)
    assert isinstance(store, MongoDBDocumentStore)


def test_register_default_integrations_includes_mongodb() -> None:
    register_default_integrations()
    profile = IntegrationProfile(document_store=IntegrationSlug.MONGODB)
    factory, _ = _collection_factory()

    store = resolve(
        IntegrationCategory.DOCUMENT_STORE,
        profile=profile,
        config={**_mongodb_config().model_dump(), "collection_factory": factory},
    )

    assert isinstance(store, MongoDBDocumentStore)


def test_opens_creates_driver_client_when_not_injected() -> None:
    config = _mongodb_config()
    mock_collection = MagicMock()
    mock_client = MagicMock()

    with patch(
        "intergrax.integrations.providers.mongodb.opens._open_collection",
        return_value=(mock_collection, mock_client),
    ) as open_mock:
        from intergrax.integrations.providers.mongodb.opens import open_mongodb_collection_client

        client = open_mongodb_collection_client(config)

    open_mock.assert_called_once_with(config)
    assert client.config is config
