# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for MongoDB integration provider (Phase M.6 P2)."""

from __future__ import annotations

import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations._shared.conformance import (
    assert_conditional_document_store,
    assert_document_store,
)
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.integrations.providers.document_store.mongodb import bundle as mongodb_bundle
from intergrax.integrations.providers.document_store.mongodb import integration as mongodb_integration
from intergrax.integrations.providers.document_store.mongodb.client import MongoCollectionClient
from intergrax.integrations.providers.document_store.mongodb.integration import (
    MongoDBDocumentStoreIntegration,
)
from intergrax.integrations.providers.document_store.mongodb.opens import (
    DOCUMENT_KEY_INDEX_KEYS,
    DOCUMENT_KEY_INDEX_NAME,
)
from intergrax.integrations.providers.document_store.mongodb.bundle import (
    MongoDBIntegrationBundle,
    create_mongodb_document_store,
    create_mongodb_document_store_integration,
    create_mongodb_integration,
)
from intergrax.integrations.providers.document_store.mongodb.config import (
    ENV_MONGODB_DATABASE,
    ENV_MONGODB_URI,
    MongoDBIntegrationConfig,
)
from intergrax.integrations.providers.document_store.mongodb.register import register_mongodb_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MONGODB_PKG = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "mongodb"
_THIS_TEST = Path(__file__).resolve()
_SCAN_ROOTS = ("intergrax", "applications", "agents", "tests")
_SKIP_DIR_NAMES = {".venv", "build", "__pycache__", "node_modules"}
_FORBIDDEN_OUTSIDE_PROVIDER = (
    "MongoCollectionClient(",
    "MongoDBDocumentStoreIntegration(",
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


class _FakeWriteResult:
    def __init__(
        self,
        *,
        matched_count: int = 0,
        modified_count: int = 0,
        upserted_id: object | None = None,
        deleted_count: int = 0,
    ) -> None:
        self.matched_count = matched_count
        self.modified_count = modified_count
        self.upserted_id = upserted_id
        self.deleted_count = deleted_count


class _FakeCollection:
    def __init__(self) -> None:
        self.storage: dict[tuple[str, str], dict[str, Any]] = {}
        self.operations: list[tuple[str, Any]] = []
        self.indexes: list[tuple[Any, bool, str | None]] = []
        self.fail_create_index = False
        self.create_index_error: BaseException | None = None
        self.update_one_error: BaseException | None = None

    def create_index(
        self,
        keys: Any,
        *,
        unique: bool = False,
        name: str | None = None,
    ) -> str:
        self.operations.append(("create_index", keys, unique, name))
        if self.create_index_error is not None:
            raise self.create_index_error
        if self.fail_create_index:
            raise RuntimeError(
                "duplicate key tenant=secret-tenant "
                "mongodb://user:password@example.test/private"
            )
        self.indexes.append((keys, unique, name))
        return name or "index"

    def find_one(self, query: dict[str, Any]) -> Optional[dict[str, Any]]:
        self.operations.append(("find_one", query))
        if "partition_key" in query and "row_key" in query:
            return self.storage.get((query["partition_key"], query["row_key"]))
        return None

    def _matches(self, query: dict[str, Any], doc: dict[str, Any]) -> bool:
        if query.get("partition_key") != doc.get("partition_key"):
            return False
        if query.get("row_key") != doc.get("row_key"):
            return False
        if "data" in query and query["data"] != doc.get("data"):
            return False
        return True

    def replace_one(
        self,
        query: dict[str, Any],
        doc: dict[str, Any],
        *,
        upsert: bool = False,
    ) -> _FakeWriteResult:
        self.operations.append(("replace_one", query, doc, upsert))
        key = (query.get("partition_key"), query.get("row_key"))
        if not isinstance(key[0], str) or not isinstance(key[1], str):
            key = (doc["partition_key"], doc["row_key"])
        existing = self.storage.get(key)  # type: ignore[arg-type]
        if existing is None:
            if upsert:
                self.storage[key] = dict(doc)  # type: ignore[index]
                return _FakeWriteResult(matched_count=0, modified_count=1, upserted_id="upserted")
            return _FakeWriteResult(matched_count=0, modified_count=0)
        if not self._matches(query, existing):
            return _FakeWriteResult(matched_count=0, modified_count=0)
        modified = 0 if existing == doc else 1
        self.storage[key] = dict(doc)  # type: ignore[index]
        return _FakeWriteResult(matched_count=1, modified_count=modified)

    def update_one(
        self,
        query: dict[str, Any],
        update: dict[str, Any],
        *,
        upsert: bool = False,
    ) -> _FakeWriteResult:
        self.operations.append(("update_one", query, update, upsert))
        if self.update_one_error is not None:
            raise self.update_one_error
        key = (query["partition_key"], query["row_key"])
        existing = self.storage.get(key)
        if existing is not None:
            return _FakeWriteResult(matched_count=1, modified_count=0, upserted_id=None)
        if upsert and "$setOnInsert" in update:
            self.storage[key] = dict(update["$setOnInsert"])
            return _FakeWriteResult(matched_count=0, modified_count=0, upserted_id="new")
        return _FakeWriteResult(matched_count=0, modified_count=0, upserted_id=None)

    def delete_one(self, query: dict[str, Any]) -> _FakeWriteResult:
        self.operations.append(("delete_one", query))
        if "partition_key" not in query or "row_key" not in query:
            return _FakeWriteResult(deleted_count=0)
        key = (query["partition_key"], query["row_key"])
        existing = self.storage.get(key)
        if existing is None:
            return _FakeWriteResult(deleted_count=0)
        if not self._matches(query, existing):
            return _FakeWriteResult(deleted_count=0)
        del self.storage[key]
        return _FakeWriteResult(deleted_count=1)

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
    assert isinstance(bundle.document_store, MongoDBDocumentStoreIntegration)
    assert bundle.config.database == "intergrax"


def test_register_and_resolve_via_profile() -> None:
    register_mongodb_integration()
    profile = IntegrationProfile(document_store="mongodb")
    factory, _ = _collection_factory()

    store = resolve(
        IntegrationCategory.DOCUMENT_STORE,
        profile=profile,
        config={**_mongodb_config().model_dump(), "collection_factory": factory},
    )

    assert_document_store(store)
    integration = create_mongodb_document_store_integration(
        client=store,
        enabled=True,
    )
    assert isinstance(integration, MongoDBDocumentStoreIntegration)
    assert integration.as_document_store() is store


def test_register_default_integrations_includes_mongodb() -> None:
    register_default_integrations()
    profile = IntegrationProfile(document_store="mongodb")
    factory, _ = _collection_factory()

    store = resolve(
        IntegrationCategory.DOCUMENT_STORE,
        profile=profile,
        config={**_mongodb_config().model_dump(), "collection_factory": factory},
    )

    assert_document_store(store)
    integration = create_mongodb_document_store_integration(client=store, enabled=True)
    assert isinstance(integration, MongoDBDocumentStoreIntegration)
    assert integration.as_document_store() is store


def test_opens_creates_driver_client_when_not_injected() -> None:
    config = _mongodb_config()
    mock_collection = MagicMock()
    mock_client = MagicMock()

    with patch(
        "intergrax.integrations.providers.document_store.mongodb.opens._open_collection",
        return_value=(mock_collection, mock_client),
    ) as open_mock:
        from intergrax.integrations.providers.document_store.mongodb.opens import open_mongodb_collection_client

        client = open_mongodb_collection_client(config)

    open_mock.assert_called_once_with(config)
    assert client.config is config
    mock_collection.create_index.assert_called_once_with(
        list(DOCUMENT_KEY_INDEX_KEYS),
        unique=True,
        name=DOCUMENT_KEY_INDEX_NAME,
    )


def test_open_creates_unique_document_key_index() -> None:
    factory, collection = _collection_factory()
    create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)

    assert collection.indexes == [
        (list(DOCUMENT_KEY_INDEX_KEYS), True, DOCUMENT_KEY_INDEX_NAME)
    ]
    create_ops = [op for op in collection.operations if op[0] == "create_index"]
    assert create_ops == [
        ("create_index", list(DOCUMENT_KEY_INDEX_KEYS), True, DOCUMENT_KEY_INDEX_NAME)
    ]


def test_open_with_injected_collection_creates_unique_index() -> None:
    collection = _FakeCollection()
    create_mongodb_document_store(**_mongodb_config().model_dump(), collection=collection)

    assert collection.indexes[0][2] == DOCUMENT_KEY_INDEX_NAME
    assert collection.indexes[0][1] is True


def test_index_creation_failure_maps_to_safe_configuration_error() -> None:
    collection = _FakeCollection()
    collection.fail_create_index = True
    factory, _ = _collection_factory(collection)

    with pytest.raises(IntegrationConfigurationError, match=DOCUMENT_KEY_INDEX_NAME) as exc_info:
        create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)

    error = exc_info.value
    assert error.__cause__ is None
    assert error.__suppress_context__ is True
    message = str(error)
    assert DOCUMENT_KEY_INDEX_NAME in message
    rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    assert "secret-tenant" not in rendered
    assert "user:password" not in rendered
    assert "example.test" not in rendered
    assert "/private" not in rendered
    assert "duplicate key" not in rendered
    assert DOCUMENT_KEY_INDEX_NAME in rendered


def test_bundle_preserves_compatibility_aliases() -> None:
    assert (
        mongodb_bundle.MongoDBDocumentStoreClient
        is mongodb_integration.MongoDBDocumentStoreClient
    )
    assert (
        mongodb_bundle.MongodbDocumentStoreIntegration
        is mongodb_integration.MongodbDocumentStoreIntegration
    )
    assert (
        mongodb_bundle.MongodbDocumentStoreIntegrationConfig
        is mongodb_integration.MongodbDocumentStoreIntegrationConfig
    )
    assert (
        mongodb_bundle.MongodbDocumentStoreClient
        is mongodb_integration.MongodbDocumentStoreClient
    )


def test_put_if_absent_duplicate_key_race_returns_false_without_overwrite() -> None:
    collection = _FakeCollection()
    existing = {
        "partition_key": "t1",
        "row_key": "r1",
        "data": {"v": 1},
        "etag": None,
        "expires_at": None,
        "created_at": None,
        "updated_at": None,
    }
    collection.storage[("t1", "r1")] = dict(existing)
    duplicate_key_error = RuntimeError("sentinel-duplicate-key")
    collection.update_one_error = duplicate_key_error
    client = MongoCollectionClient(
        _mongodb_config(),
        collection=collection,
        is_duplicate_key_error=lambda exc: exc is duplicate_key_error,
    )
    document = DocumentRecord(partition_key="t1", row_key="r1", data={"v": 99})

    assert client.put_if_absent(document) is False
    assert collection.storage[("t1", "r1")]["data"] == {"v": 1}


def test_put_if_absent_unknown_update_error_is_reraised() -> None:
    collection = _FakeCollection()
    unknown_error = RuntimeError("unrelated-mongo-failure")
    collection.update_one_error = unknown_error
    client = MongoCollectionClient(
        _mongodb_config(),
        collection=collection,
        is_duplicate_key_error=lambda exc: False,
    )
    document = DocumentRecord(partition_key="t1", row_key="r1", data={"v": 1})

    with pytest.raises(RuntimeError, match="unrelated-mongo-failure") as exc_info:
        client.put_if_absent(document)

    assert exc_info.value is unknown_error
    assert ("t1", "r1") not in collection.storage


def test_put_if_absent_creates_document() -> None:
    factory, collection = _collection_factory()
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)
    document = DocumentRecord(partition_key="t1", row_key="r1", data={"v": 1})

    assert store.put_if_absent(document) is True
    assert store.get("t1", "r1") is not None
    assert any(op[0] == "update_one" for op in collection.operations)


def test_second_put_if_absent_does_not_overwrite() -> None:
    factory, _ = _collection_factory()
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)
    first = DocumentRecord(partition_key="t1", row_key="r1", data={"v": 1})
    second = DocumentRecord(partition_key="t1", row_key="r1", data={"v": 2})

    assert store.put_if_absent(first) is True
    assert store.put_if_absent(second) is False
    doc = store.get("t1", "r1")
    assert doc is not None
    assert doc.data == {"v": 1}


def test_replace_if_match_updates_current_document() -> None:
    factory, _ = _collection_factory()
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)
    expected = DocumentRecord(partition_key="t1", row_key="r1", data={"v": 1})
    replacement = DocumentRecord(partition_key="t1", row_key="r1", data={"v": 2})
    store.put(expected)

    assert store.replace_if_match(expected=expected, replacement=replacement) is True
    doc = store.get("t1", "r1")
    assert doc is not None
    assert doc.data == {"v": 2}


def test_stale_replace_if_match_returns_false_and_preserves() -> None:
    factory, _ = _collection_factory()
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)
    current = DocumentRecord(partition_key="t1", row_key="r1", data={"v": 1})
    stale = DocumentRecord(partition_key="t1", row_key="r1", data={"v": 0})
    replacement = DocumentRecord(partition_key="t1", row_key="r1", data={"v": 2})
    store.put(current)

    assert store.replace_if_match(expected=stale, replacement=replacement) is False
    doc = store.get("t1", "r1")
    assert doc is not None
    assert doc.data == {"v": 1}


def test_delete_if_match_removes_matching_document() -> None:
    factory, _ = _collection_factory()
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)
    expected = DocumentRecord(partition_key="t1", row_key="r1", data={"v": 1})
    store.put(expected)

    assert store.delete_if_match(expected=expected) is True
    assert store.get("t1", "r1") is None


def test_stale_delete_if_match_does_not_remove() -> None:
    factory, _ = _collection_factory()
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)
    current = DocumentRecord(partition_key="t1", row_key="r1", data={"v": 1})
    stale = DocumentRecord(partition_key="t1", row_key="r1", data={"v": 0})
    store.put(current)

    assert store.delete_if_match(expected=stale) is False
    assert store.get("t1", "r1") is not None


def test_closed_store_rejects_conditional_operations() -> None:
    factory, _ = _collection_factory()
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)
    document = DocumentRecord(partition_key="t1", row_key="r1", data={"v": 1})
    store.close()

    with pytest.raises(IntegrationConfigurationError, match="closed"):
        store.put_if_absent(document)
    with pytest.raises(IntegrationConfigurationError, match="closed"):
        store.replace_if_match(expected=document, replacement=document)
    with pytest.raises(IntegrationConfigurationError, match="closed"):
        store.delete_if_match(expected=document)


def test_mongodb_store_implements_conditional_document_store() -> None:
    factory, _ = _collection_factory()
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)
    assert_document_store(store)
    assert_conditional_document_store(store)
