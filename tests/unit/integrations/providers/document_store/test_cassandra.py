# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Cassandra integration provider (Phase M.6 P2)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations._shared.conformance import assert_document_store
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.integrations.providers.document_store.cassandra.adapter import CassandraDocumentStore
from intergrax.integrations.providers.document_store.cassandra.bundle import (
    CassandraIntegrationBundle,
    create_cassandra_document_store,
    create_cassandra_integration,
)
from intergrax.integrations.providers.document_store.cassandra.config import (
    ENV_CASSANDRA_CONTACT_POINTS,
    ENV_CASSANDRA_KEYSPACE,
    CassandraIntegrationConfig,
)
from intergrax.integrations.providers.document_store.cassandra.register import register_cassandra_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CASSANDRA_PKG = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "cassandra"
_THIS_TEST = Path(__file__).resolve()
_SCAN_ROOTS = ("intergrax", "applications", "agents", "tests")
_SKIP_DIR_NAMES = {".venv", "build", "__pycache__", "node_modules"}
_FORBIDDEN_OUTSIDE_PROVIDER = (
    "CassandraCqlClient(",
    "CassandraDocumentStore(",
    "integrations.providers.cassandra.client",
    "integrations.providers.cassandra.opens",
    "import cassandra",
    "from cassandra",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


@dataclass
class _FakePrepared:
    query: str


@dataclass
class _FakeRow:
    payload: str = ""
    row_key: str = ""


class _FakeResult:
    def __init__(self, rows: Iterable[_FakeRow] | _FakeRow | None) -> None:
        if rows is None:
            self._rows: list[_FakeRow] = []
        elif isinstance(rows, _FakeRow):
            self._rows = [rows]
        else:
            self._rows = list(rows)

    def one(self) -> Optional[_FakeRow]:
        return self._rows[0] if self._rows else None

    def __iter__(self):
        return iter(self._rows)


class _FakeSession:
    def __init__(self) -> None:
        self.storage: dict[tuple[str, str], str] = {}
        self.prepared: list[str] = []
        self.executions: list[tuple[str, tuple[Any, ...]]] = []
        self.shutdown_called = False
        self.cluster = MagicMock()

    def prepare(self, query: str) -> _FakePrepared:
        self.prepared.append(query)
        return _FakePrepared(query=query)

    def execute(self, statement: _FakePrepared, params: tuple[Any, ...] = ()) -> _FakeResult:
        self.executions.append((statement.query, params))
        query = statement.query
        if query.startswith("SELECT payload FROM"):
            partition_key, row_key = params
            payload = self.storage.get((str(partition_key), str(row_key)))
            if payload is None:
                return _FakeResult(None)
            return _FakeResult(_FakeRow(payload=payload))
        if query.startswith("INSERT INTO"):
            if "USING TTL" in query:
                partition_key, row_key, payload, _ttl = params
            else:
                partition_key, row_key, payload = params
            self.storage[(str(partition_key), str(row_key))] = str(payload)
            return _FakeResult([])
        if query.startswith("DELETE FROM"):
            partition_key, row_key = params
            self.storage.pop((str(partition_key), str(row_key)), None)
            return _FakeResult([])
        if "row_key >=" in query:
            partition_key, prefix, end_key, limit = params
            rows = []
            for (pk, rk), payload in sorted(self.storage.items()):
                if pk != str(partition_key):
                    continue
                if rk < str(prefix) or rk >= str(end_key):
                    continue
                rows.append(_FakeRow(row_key=rk, payload=payload))
                if len(rows) >= int(limit):
                    break
            return _FakeResult(rows)
        if query.startswith("SELECT row_key, payload FROM"):
            partition_key, limit = params
            rows = []
            for (pk, rk), payload in sorted(self.storage.items()):
                if pk != str(partition_key):
                    continue
                rows.append(_FakeRow(row_key=rk, payload=payload))
                if len(rows) >= int(limit):
                    break
            return _FakeResult(rows)
        raise AssertionError(f"Unexpected CQL in fake session: {query}")

    def shutdown(self) -> None:
        self.shutdown_called = True


def _cassandra_config() -> CassandraIntegrationConfig:
    return CassandraIntegrationConfig(
        contact_points="127.0.0.1",
        keyspace="intergrax",
        table="intergrax_documents",
        user="cassandra",
        password="secret",
    )


def _session_factory(session: _FakeSession | None = None):
    fake = session or _FakeSession()

    def _factory() -> _FakeSession:
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


def test_cassandra_driver_only_imported_in_opens_module() -> None:
    violations: list[str] = []
    for path in _CASSANDRA_PKG.glob("*.py"):
        if path.name == "opens.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "cassandra" in text:
            violations.append(path.name)
    assert violations == []


def test_cassandra_not_constructed_outside_provider_package() -> None:
    violations: list[str] = []
    for path in _iter_python_files(*_SCAN_ROOTS):
        if path.resolve() == _THIS_TEST.resolve():
            continue
        if _CASSANDRA_PKG in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_OUTSIDE_PROVIDER:
            if pattern in text:
                violations.append(f"{path.relative_to(_PROJECT_ROOT).as_posix()}: {pattern}")
    assert violations == []


def test_cassandra_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_CASSANDRA_CONTACT_POINTS, "10.0.0.1,10.0.0.2")
    monkeypatch.setenv(ENV_CASSANDRA_KEYSPACE, "events")
    config = CassandraIntegrationConfig.from_env()
    assert config.contact_points_list() == ["10.0.0.1", "10.0.0.2"]
    assert config.qualified_table() == "events.intergrax_documents"


def test_cassandra_config_requires_contact_points_and_keyspace() -> None:
    factory, _ = _session_factory()
    with pytest.raises(IntegrationConfigurationError, match="contact_points"):
        create_cassandra_document_store(keyspace="ks", contact_points="", session_factory=factory)
    with pytest.raises(IntegrationConfigurationError, match="keyspace"):
        create_cassandra_document_store(
            contact_points="127.0.0.1",
            keyspace="",
            session_factory=factory,
        )


def test_put_and_get_round_trip() -> None:
    factory, session = _session_factory()
    store = create_cassandra_document_store(**_cassandra_config().model_dump(), session_factory=factory)

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
    assert any("INSERT INTO" in query for query, _ in session.executions)
    assert_document_store(store)


def test_get_returns_none_for_missing_row() -> None:
    factory, _ = _session_factory()
    store = create_cassandra_document_store(**_cassandra_config().model_dump(), session_factory=factory)

    assert store.get("tenant-1", "missing") is None


def test_delete_removes_row() -> None:
    factory, session = _session_factory()
    store = create_cassandra_document_store(**_cassandra_config().model_dump(), session_factory=factory)
    store.put(DocumentRecord(partition_key="t1", row_key="r1", data={"x": 1}))

    store.delete("t1", "r1")

    assert store.get("t1", "r1") is None
    assert any(query.startswith("DELETE FROM") for query, _ in session.executions)


def test_query_lists_partition_rows() -> None:
    factory, _ = _session_factory()
    store = create_cassandra_document_store(**_cassandra_config().model_dump(), session_factory=factory)
    store.put(DocumentRecord(partition_key="t1", row_key="a", data={"n": 1}))
    store.put(DocumentRecord(partition_key="t1", row_key="b", data={"n": 2}))
    store.put(DocumentRecord(partition_key="t2", row_key="c", data={"n": 3}))

    result = store.query("t1", limit=10)

    assert result.total == 2
    assert {doc.row_key for doc in result.documents} == {"a", "b"}


def test_query_with_row_key_prefix() -> None:
    factory, _ = _session_factory()
    store = create_cassandra_document_store(**_cassandra_config().model_dump(), session_factory=factory)
    store.put(DocumentRecord(partition_key="t1", row_key="2026-05-01", data={}))
    store.put(DocumentRecord(partition_key="t1", row_key="2026-05-02", data={}))
    store.put(DocumentRecord(partition_key="t1", row_key="2026-06-01", data={}))

    result = store.query("t1", row_key_prefix="2026-05", limit=10)

    assert result.total == 2
    assert all(doc.row_key.startswith("2026-05") for doc in result.documents)


def test_put_with_ttl_uses_ttl_statement() -> None:
    factory, session = _session_factory()
    store = create_cassandra_document_store(**_cassandra_config().model_dump(), session_factory=factory)

    store.put(
        DocumentRecord(
            partition_key="t1",
            row_key="ttl-row",
            data={"temp": True},
            ttl_seconds=3600,
        )
    )

    assert any("USING TTL" in query for query in session.prepared)


def test_close_shuts_down_session() -> None:
    factory, session = _session_factory()
    store = create_cassandra_document_store(**_cassandra_config().model_dump(), session_factory=factory)

    store.close()

    assert session.shutdown_called
    session.cluster.shutdown.assert_called_once()
    with pytest.raises(IntegrationConfigurationError, match="closed"):
        store.get("t1", "r1")


def test_create_cassandra_integration_bundle() -> None:
    factory, _ = _session_factory()
    bundle = create_cassandra_integration(**_cassandra_config().model_dump(), session_factory=factory)

    assert isinstance(bundle, CassandraIntegrationBundle)
    assert isinstance(bundle.document_store, CassandraDocumentStore)
    assert bundle.cql_client.config.keyspace == "intergrax"


def test_register_and_resolve_via_profile() -> None:
    register_cassandra_integration()
    profile = IntegrationProfile(document_store="cassandra")
    factory, _ = _session_factory()

    store = resolve(
        IntegrationCategory.DOCUMENT_STORE,
        profile=profile,
        config={**_cassandra_config().model_dump(), "session_factory": factory},
    )

    assert_document_store(store)
    assert isinstance(store, CassandraDocumentStore)


def test_register_default_integrations_includes_cassandra() -> None:
    register_default_integrations()
    profile = IntegrationProfile(document_store="cassandra")
    factory, _ = _session_factory()

    store = resolve(
        IntegrationCategory.DOCUMENT_STORE,
        profile=profile,
        config={**_cassandra_config().model_dump(), "session_factory": factory},
    )

    assert isinstance(store, CassandraDocumentStore)


def test_opens_creates_driver_session_when_not_injected() -> None:
    config = _cassandra_config()
    mock_session = MagicMock()

    with patch(
        "intergrax.integrations.providers.document_store.cassandra.opens._open_session",
        return_value=mock_session,
    ) as open_mock:
        from intergrax.integrations.providers.document_store.cassandra.opens import open_cassandra_cql_client

        client = open_cassandra_cql_client(config)

    open_mock.assert_called_once_with(config, session_factory=None)
    assert client.config is config


def test_payload_json_encoding() -> None:
    factory, session = _session_factory()
    store = create_cassandra_document_store(**_cassandra_config().model_dump(), session_factory=factory)
    store.put(DocumentRecord(partition_key="t1", row_key="r1", data={"nested": {"a": 1}}))

    _, params = next(item for item in session.executions if item[0].startswith("INSERT INTO"))
    payload = params[2]
    assert json.loads(payload) == {"nested": {"a": 1}}
