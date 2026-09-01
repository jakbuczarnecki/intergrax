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
from intergrax.integrations.contracts.document_store import (
    DocumentDataEquality,
    DocumentDataSort,
    DocumentRecord,
    validate_document_query_limit,
)
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
        self._limit: int | None = None
        self._sort_spec: list[tuple[str, int]] = []

    def sort(
        self,
        key: str | list[tuple[str, int]],
        direction: int | None = None,
    ) -> _FakeCursor:
        if isinstance(key, list):
            self._sort_spec = key
        else:
            self._sort_spec = [(key, direction or 1)]
        return self

    def limit(self, count: int) -> _FakeCursor:
        self._limit = count
        return self

    def _sorted_rows(self) -> list[dict[str, Any]]:
        if not self._sort_spec:
            return list(self._rows)
        reverse = self._sort_spec[0][1] == -1

        def _value(doc: dict[str, Any], field: str) -> Any:
            if field == "row_key":
                return doc.get("row_key")
            if field.startswith("data."):
                return _nested_value(doc.get("data", {}), field.split(".", 1)[1])
            return doc.get(field)

        return sorted(
            self._rows,
            key=lambda doc: tuple(_value(doc, field) for field, _ in self._sort_spec),
            reverse=reverse,
        )

    def __iter__(self) -> Iterable[dict[str, Any]]:
        rows = self._sorted_rows()
        if self._limit is None:
            return iter(rows)
        return iter(rows[:self._limit])


def _nested_value(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for segment in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current


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

    def _matches_find_query(self, query: dict[str, Any], doc: dict[str, Any]) -> bool:
        if query.get("partition_key") != doc.get("partition_key"):
            return False
        row_key_filter = query.get("row_key")
        row_key = str(doc.get("row_key", ""))
        if isinstance(row_key_filter, dict):
            prefix = row_key_filter.get("$regex")
            if isinstance(prefix, str) and not row_key.startswith(prefix.removeprefix("^")):
                return False
            upper = row_key_filter.get("$lte")
            if upper is not None and row_key > str(upper):
                return False
            lower = row_key_filter.get("$gt")
            if lower is not None and row_key <= str(lower):
                return False
            less_than = row_key_filter.get("$lt")
            if less_than is not None and row_key >= str(less_than):
                return False
        for key, expected in query.items():
            if key in {"partition_key", "row_key", "$or"}:
                continue
            if key.startswith("data."):
                if _nested_value(doc.get("data", {}), key.split(".", 1)[1]) != expected:
                    return False
        or_clauses = query.get("$or")
        if isinstance(or_clauses, list):
            if not any(self._matches_find_clause(clause, doc) for clause in or_clauses):
                return False
        return True

    def _matches_find_clause(self, clause: dict[str, Any], doc: dict[str, Any]) -> bool:
        for key, expected in clause.items():
            if key == "row_key":
                row_key = str(doc.get("row_key", ""))
                if isinstance(expected, dict) and "$lt" in expected:
                    if row_key >= str(expected["$lt"]):
                        return False
                elif row_key != expected:
                    return False
            elif key.startswith("data."):
                actual = _nested_value(doc.get("data", {}), key.split(".", 1)[1])
                if isinstance(expected, dict) and "$lt" in expected:
                    if actual >= expected["$lt"]:
                        return False
                elif actual != expected:
                    return False
            elif doc.get(key) != expected:
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
        for (partition_key, row_key), doc in sorted(self.storage.items()):
            if not self._matches_find_query(query, doc):
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


_MAX_QUERY_LIMIT = validate_document_query_limit(5000)


def _seed_partition_rows(
    collection: _FakeCollection,
    partition_key: str,
    count: int,
    *,
    prefix: str = "row-",
) -> list[str]:
    row_keys: list[str] = []
    for index in range(count):
        row_key = f"{prefix}{index:05d}"
        row_keys.append(row_key)
        collection.storage[(partition_key, row_key)] = {
            "partition_key": partition_key,
            "row_key": row_key,
            "data": {},
            "expires_at": None,
        }
    return row_keys


def test_query_limit_one_with_multiple_rows_returns_next_cursor() -> None:
    factory, collection = _collection_factory()
    _seed_partition_rows(collection, "p1", 2)
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)

    page = store.query("p1", limit=1)

    assert len(page.documents) == 1
    assert page.next_cursor is not None


def test_query_exact_limit_rows_returns_no_next_cursor() -> None:
    factory, collection = _collection_factory()
    _seed_partition_rows(collection, "p1", 3)
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)

    page = store.query("p1", limit=3)

    assert len(page.documents) == 3
    assert page.next_cursor is None


def test_query_limit_plus_one_rows_returns_next_cursor() -> None:
    factory, collection = _collection_factory()
    _seed_partition_rows(collection, "p1", 4)
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)

    page = store.query("p1", limit=3)

    assert len(page.documents) == 3
    assert page.next_cursor is not None


def test_query_max_limit_exact_rows_returns_no_next_cursor() -> None:
    factory, collection = _collection_factory()
    _seed_partition_rows(collection, "p1", _MAX_QUERY_LIMIT)
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)

    page = store.query("p1", limit=_MAX_QUERY_LIMIT)

    assert len(page.documents) == _MAX_QUERY_LIMIT
    assert page.next_cursor is None


def test_query_max_limit_plus_one_rows_paginates_remaining_row() -> None:
    factory, collection = _collection_factory()
    expected_keys = _seed_partition_rows(collection, "p1", _MAX_QUERY_LIMIT + 1)
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)

    first_page = store.query("p1", limit=_MAX_QUERY_LIMIT)
    assert len(first_page.documents) == _MAX_QUERY_LIMIT
    assert first_page.next_cursor is not None
    assert [doc.row_key for doc in first_page.documents] == expected_keys[:_MAX_QUERY_LIMIT]

    second_page = store.query("p1", limit=_MAX_QUERY_LIMIT, cursor=first_page.next_cursor)
    assert len(second_page.documents) == 1
    assert second_page.documents[0].row_key == expected_keys[_MAX_QUERY_LIMIT]
    assert second_page.next_cursor is None


def test_query_prefix_cursor_continuation_returns_expected_slice() -> None:
    factory, collection = _collection_factory()
    _seed_partition_rows(collection, "p1", 5, prefix="scope-")
    _seed_partition_rows(collection, "p1", 2, prefix="other-")
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)

    first_page = store.query("p1", row_key_prefix="scope-", limit=2)
    assert [doc.row_key for doc in first_page.documents] == ["scope-00000", "scope-00001"]
    assert first_page.next_cursor is not None

    second_page = store.query(
        "p1",
        row_key_prefix="scope-",
        limit=2,
        cursor=first_page.next_cursor,
    )
    assert [doc.row_key for doc in second_page.documents] == ["scope-00002", "scope-00003"]
    assert second_page.next_cursor is not None

    third_page = store.query(
        "p1",
        row_key_prefix="scope-",
        limit=2,
        cursor=second_page.next_cursor,
    )
    assert [doc.row_key for doc in third_page.documents] == ["scope-00004"]
    assert third_page.next_cursor is None


def test_query_cursor_partition_mismatch_fails_closed() -> None:
    factory, collection = _collection_factory()
    _seed_partition_rows(collection, "p1", 2)
    _seed_partition_rows(collection, "p2", 2)
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)

    cursor = store.query("p1", limit=1).next_cursor
    assert cursor is not None

    with pytest.raises(ValueError, match="document_store_cursor_query_mismatch"):
        store.query("p2", limit=1, cursor=cursor)


def test_query_with_data_equalities_and_sort_cursor() -> None:
    factory, collection = _collection_factory()
    store = create_mongodb_document_store(**_mongodb_config().model_dump(), collection_factory=factory)
    store.put(
        DocumentRecord(
            partition_key="qual",
            row_key="proof/provider_qualification/run-a",
            data={
                "recorded_at": "2026-08-17T12:02:00Z",
                "run_id": "run-a",
                "domain_evidence": {"subject": {"provider_id": "postgresql"}},
            },
        ),
    )
    store.put(
        DocumentRecord(
            partition_key="qual",
            row_key="proof/provider_qualification/run-b",
            data={
                "recorded_at": "2026-08-17T12:01:00Z",
                "run_id": "run-b",
                "domain_evidence": {"subject": {"provider_id": "postgresql"}},
            },
        ),
    )
    store.put(
        DocumentRecord(
            partition_key="qual",
            row_key="proof/provider_qualification/run-c",
            data={
                "recorded_at": "2026-08-17T12:03:00Z",
                "run_id": "run-c",
                "domain_evidence": {"subject": {"provider_id": "sqlite"}},
            },
        ),
    )

    sort = (
        DocumentDataSort(path="recorded_at", direction="desc"),
        DocumentDataSort(path="run_id", direction="desc"),
        DocumentDataSort(path="$row_key", direction="desc"),
    )
    equalities = (
        DocumentDataEquality(
            path="domain_evidence.subject.provider_id",
            value="postgresql",
        ),
    )
    first_page = store.query(
        "qual",
        row_key_prefix="proof/provider_qualification/",
        limit=1,
        data_equalities=equalities,
        sort=sort,
    )
    assert [doc.data["run_id"] for doc in first_page.documents] == ["run-a"]
    assert first_page.next_cursor is not None

    find_ops = [op for op in collection.operations if op[0] == "find"]
    assert find_ops[-1][1]["data.domain_evidence.subject.provider_id"] == "postgresql"

    second_page = store.query(
        "qual",
        row_key_prefix="proof/provider_qualification/",
        limit=1,
        data_equalities=equalities,
        sort=sort,
        cursor=first_page.next_cursor,
    )
    assert [doc.data["run_id"] for doc in second_page.documents] == ["run-b"]
    assert second_page.next_cursor is None
