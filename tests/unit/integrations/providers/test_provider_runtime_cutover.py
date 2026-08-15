# © Artur Czarnecki. All rights reserved.

"""Runtime cutover guards — INTEGRATIONS-2E single provider entrypoint."""

from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import re
from pathlib import Path
from typing import Any, Callable, Sequence
from unittest.mock import MagicMock

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.providers.layout import SLUG_CATEGORY
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFERRED_LLM_GUARDRAIL_SLUGS: frozenset[str] = frozenset(
    {
        "llm_guard",
        "guardrails_ai",
        "nemo_guardrails",
        "openguardrails",
        "presidio",
        "llama_guard",
        "lakera",
        "azure_content_safety",
        "bedrock_guardrails",
    }
)

# Providers fully cut over to single Integration entrypoint (behavior tests must pass).
CUTOVER_SLUGS: frozenset[str] = frozenset(
    slug
    for slug, category in SLUG_CATEGORY.items()
    if category != "llm_guardrail" and slug not in DEFERRED_LLM_GUARDRAIL_SLUGS
)

_CLASS_NAME_OVERRIDES: dict[str, str] = {
    "newrelic": "NewRelic",
    "opentelemetry_collector": "OpenTelemetryCollector",
    "aws": "Aws",
    "gcp": "Gcp",
    "azure_sql": "AzureSql",
    "cloud_sql": "CloudSql",
    "mssql": "Mssql",
    "pgvector": "Pgvector",
    "yt_dlp": "YtDlp",
    "e2b": "E2b",
    "n8n": "N8n",
    "okta": "Okta",
    "auth0": "Auth0",
}


def _slug_to_pascal(slug: str) -> str:
    if slug in _CLASS_NAME_OVERRIDES:
        return _CLASS_NAME_OVERRIDES[slug]
    return "".join(part.capitalize() for part in slug.split("_"))


def _category_to_pascal(category: str) -> str:
    return "".join(part.capitalize() for part in category.split("_"))


def _class_prefix(slug: str, category: str) -> str:
    return f"{_slug_to_pascal(slug)}{_category_to_pascal(category)}"


def _provider_pkg(slug: str, category: str) -> str:
    return f"intergrax.integrations.providers.{category}.{slug}"


def _integration_class_name(slug: str, category: str) -> str:
    if category == "observability_backend":
        return f"{_slug_to_pascal(slug)}ObservabilityIntegration"
    return f"{_class_prefix(slug, category)}Integration"


VECTOR_STORE_CUTOVER_SLUGS = frozenset(
    slug for slug in CUTOVER_SLUGS if SLUG_CATEGORY[slug] == "vector_store"
)
OBSERVABILITY_CUTOVER_SLUGS = frozenset(
    slug for slug in CUTOVER_SLUGS if SLUG_CATEGORY[slug] == "observability_backend"
)
CLIENT_CONTRACT_CUTOVER_SLUGS = CUTOVER_SLUGS - OBSERVABILITY_CUTOVER_SLUGS

# Non-vector category representatives — legacy factory must return Integration at runtime.
NON_VECTOR_LEGACY_SMOKE_SLUGS: frozenset[str] = frozenset(
    {
        "pagerduty",
        "ms365_graph",
        "slack",
        "filesystem",
        "github",
        "redis",
        "postgresql",
        "tavily",
        "s3",
        "mongodb",
    }
)


def _provider_dir(slug: str, category: str) -> Path:
    return _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / category / slug


def _bundle_module(slug: str, category: str) -> Any:
    return importlib.import_module(f"{_provider_pkg(slug, category)}.bundle")


def _integration_module(slug: str, category: str) -> Any:
    return importlib.import_module(f"{_provider_pkg(slug, category)}.integration")


def _init_module(slug: str, category: str) -> Any:
    return importlib.import_module(_provider_pkg(slug, category))


def _legacy_factory_name(slug: str, category: str) -> str:
    register_mod = importlib.import_module(f"{_provider_pkg(slug, category)}.register")
    register_source = inspect.getsource(register_mod)
    contract_name = f"create_{slug}_{category}_integration"
    for token in re.findall(r"(create_\w+)", register_source):
        if token != contract_name:
            return token
    for token in re.findall(r"(_create_\w+)", register_source):
        if "integration" not in token:
            return token.replace("_create_", "create_", 1)
    bundle = _bundle_module(slug, category)
    for name in getattr(bundle, "__all__", ()):
        if name.startswith("create_") and name != contract_name:
            return name
    msg = f"{slug}: legacy factory not found"
    raise AssertionError(msg)


def _contract_factory(slug: str, category: str) -> Any:
    bundle = _bundle_module(slug, category)
    if category == "observability_backend":
        return getattr(bundle, f"create_{slug}_observability_integration")
    return getattr(bundle, f"create_{slug}_{category}_integration")


_PRIMARY_VECTOR_STORE_CLASS_NAMES = {
    "chroma": "ChromaVectorStore",
    "milvus": "MilvusVectorStore",
    "pgvector": "PgVectorRagStore",
    "pinecone": "PineconeVectorStore",
    "qdrant": "QdrantVectorStore",
    "weaviate": "WeaviateVectorStore",
}
PRIMARY_VECTOR_STORE_CUTOVER_SLUGS = frozenset(_PRIMARY_VECTOR_STORE_CLASS_NAMES)


def _native_document(
    *,
    document_id: str = "doc-1",
    scope: VectorStoreScope,
) -> KnowledgeDocument:
    document_scope: dict[str, str] = {
        "tenant_id": scope.tenant_id,
        "namespace": scope.namespace,
    }
    if scope.workspace_id is not None:
        document_scope["workspace_id"] = scope.workspace_id
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": document_scope,
            "content": "hello",
            "metadata": {"source": "runtime-cutover-matrix"},
            "provenance": {
                "source_kind": "test",
                "source_id": document_id,
            },
        }
    )


def _native_record(
    *,
    vector_id: str = "doc-1",
    scope: VectorStoreScope,
) -> VectorStoreRecord:
    return VectorStoreRecord(
        document=_native_document(document_id=vector_id, scope=scope),
        embedding=[0.1, 0.2],
        vector_id=vector_id,
    )


class _FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.records: list[VectorStoreRecord] = []
        self.add_scopes: list[VectorStoreScope] = []
        self.query_scopes: list[VectorStoreScope] = []
        self.deleted: list[str] = []
        self.delete_scopes: list[VectorStoreScope] = []
        self.count_scopes: list[VectorStoreScope] = []

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str]:
        self.records.extend(records)
        self.add_scopes.append(scope)
        return [record.vector_id for record in records]

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> Sequence[VectorStoreHit]:
        del query_embedding, top_k, metadata_filter
        self.query_scopes.append(scope)
        return [
            VectorStoreHit(
                vector_id="doc-1",
                document=_native_document(scope=scope),
                similarity_score=0.9,
                rank=0,
                embedding=[0.1, 0.2] if include_embeddings else None,
            )
        ]

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
        self.deleted.extend(ids)
        self.delete_scopes.append(scope)
        deleted_ids = set(ids)
        self.records = [
            record
            for record in self.records
            if record.vector_id not in deleted_ids
            or not scope.matches_document(record.document)
        ]

    def count(self, *, scope: VectorStoreScope) -> int:
        self.count_scopes.append(scope)
        return sum(scope.matches_document(record.document) for record in self.records)


class _FakeClient:
    async def ping(self) -> None:
        return None


class _FakeSearchClient:
    def search(self, query: str, limit: int) -> dict[str, Any]:
        return {"results": [{"title": query, "url": "https://example", "content": "hit"}]}


class _FakeIssueClient:
    def get_issue(self, issue_key: str) -> dict[str, Any]:
        return {"key": issue_key, "summary": "Task", "status": "open"}

    def add_comment(self, issue_key: str, body: str) -> dict[str, Any]:
        return {"id": "c1", "body": body, "issue": issue_key}

    def search_issues(self, jql: str, *, limit: int) -> list[dict[str, Any]]:
        del limit
        return [{"key": "1", "summary": jql, "status": "open"}]

    def health(self) -> bool:
        return True


class _FakePagerDutyEventsClient:
    def trigger_incident(self, *args: object, **kwargs: object) -> str:
        del args, kwargs
        return "d1"

    def send_notification(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def acknowledge_incident(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def health(self) -> bool:
        return True


class _FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **kwargs: object) -> None:
        del Bucket, kwargs
        self.objects[Key] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        del Bucket
        from io import BytesIO

        body = self.objects.get(Key)
        if body is None:
            raise _FakeS3NotFound(Key)
        return {"Body": BytesIO(body), "ContentType": "application/octet-stream", "Metadata": {}}

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        del Bucket
        self.objects.pop(Key, None)

    def generate_presigned_url(self, *args: object, **kwargs: object) -> str:
        del args, kwargs
        return "https://example/presigned"


class _FakeS3NotFound(Exception):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.response = {"Error": {"Code": "NoSuchKey"}}


class _FakeMongoCollection:
    def __init__(self) -> None:
        self.docs: dict[tuple[str, str], dict[str, Any]] = {}

    def find_one(self, query: dict[str, str]) -> dict[str, Any] | None:
        return self.docs.get((query["partition_key"], query["row_key"]))

    def replace_one(self, query: dict[str, str], payload: dict[str, Any], *, upsert: bool) -> None:
        del upsert
        self.docs[(query["partition_key"], query["row_key"])] = payload

    def delete_one(self, query: dict[str, str]) -> None:
        self.docs.pop((query["partition_key"], query["row_key"]), None)

    def find(self, query: dict[str, Any]) -> "_FakeMongoCursor":
        partition_key = query["partition_key"]
        rows = [
            doc
            for (pk, _rk), doc in self.docs.items()
            if pk == partition_key
        ]
        return _FakeMongoCursor(rows)


class _FakeMongoCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def sort(self, _field: str, _direction: int) -> "_FakeMongoCursor":
        return self

    def limit(self, _count: int) -> "_FakeMongoCursor":
        return self

    def __iter__(self) -> Any:
        return iter(self._rows)


class _FakePostgresqlConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.committed = 0
        self.closed = False

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> object:
        self.executed.append((sql, params))

        class _Cursor:
            def fetchall(self) -> list[dict[str, str]]:
                return [{"name": "alpha"}]

        return _Cursor()

    def commit(self) -> None:
        self.committed += 1

    def close(self) -> None:
        self.closed = True


def _postgresql_connection_factory() -> Callable[[], _FakePostgresqlConnection]:
    conn = _FakePostgresqlConnection()

    def _factory() -> _FakePostgresqlConnection:
        return conn

    return _factory


def _redis_smoke_client() -> MagicMock:
    storage: dict[str, bytes] = {}
    client = MagicMock()
    client.get.side_effect = lambda key: storage.get(key)
    client.set.side_effect = lambda key, value, ex=None: storage.__setitem__(key, value)
    client.delete.side_effect = lambda *keys: sum(1 for key in keys if storage.pop(key, None) is not None)
    client.register_script.return_value = MagicMock(return_value=1)
    client.pipeline.return_value = MagicMock()
    client.hget.return_value = None
    client.hgetall.return_value = {}
    client.eval.return_value = 1
    return client


def _legacy_smoke_default_kwargs(slug: str, tmp_path: Path) -> dict[str, Any]:
    if slug == "pagerduty":
        return {"client": _FakePagerDutyEventsClient(), "routing_key": "test-key"}
    if slug == "ms365_graph":
        from intergrax.integrations.contracts.collaboration_suite import UserRecord
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.client import GraphRestClient

        fake_client = MagicMock(spec=GraphRestClient)
        fake_client.get_user.return_value = UserRecord(id="u1", display_name="Smoke")
        return {"client": fake_client}
    if slug == "slack":
        return {
            "integration_category": IntegrationCategory.NOTIFICATION_CHANNEL,
            "webhook_url": "",
        }
    if slug == "filesystem":
        return {"root_dir": str(tmp_path)}
    if slug == "github":
        return {"client": _FakeIssueClient()}
    if slug == "redis":
        return {"client": _redis_smoke_client(), "key_prefix": "smoke"}
    if slug == "postgresql":
        return {
            "connection_factory": _postgresql_connection_factory(),
            "dsn": "postgresql://localhost/test",
        }
    if slug == "tavily":
        return {"client": _FakeSearchClient(), "api_key": "test-key"}
    if slug == "s3":
        return {"bucket": "smoke-bucket", "s3_client": _FakeS3Client()}
    if slug == "mongodb":
        return {
            "uri": "mongodb://localhost:27017",
            "database": "intergrax",
            "collection_name": "intergrax_documents",
            "collection_factory": _FakeMongoCollection,
        }
    msg = f"{slug}: no default legacy smoke factory kwargs configured"
    raise AssertionError(msg)


def _legacy_smoke_custom_kwargs(slug: str, tmp_path: Path) -> dict[str, Any]:
    kwargs = _legacy_smoke_default_kwargs(slug, tmp_path)
    if slug == "pagerduty":
        from intergrax.integrations.providers.notification_channel.pagerduty.adapter import (
            _PagerDutyNotificationChannel,
        )

        client = kwargs.pop("client")
        kwargs["notification_channel"] = _PagerDutyNotificationChannel(client)
        return kwargs
    if slug == "ms365_graph":
        from intergrax.integrations.providers.collaboration_suite.ms365_graph.adapter import (
            _Ms365GraphCollaborationSuite,
        )

        client = kwargs.pop("client")
        kwargs["collaboration_suite"] = _Ms365GraphCollaborationSuite(client)
        return kwargs
    if slug == "slack":
        kwargs["notification_adapter"] = MagicMock()
        return kwargs
    if slug == "filesystem":
        kwargs["object_storage"] = MagicMock()
        return kwargs
    if slug == "github":
        kwargs["issue_tracker"] = MagicMock()
        return kwargs
    if slug == "postgresql":
        kwargs["relational_store"] = MagicMock()
        return kwargs
    if slug == "tavily":
        kwargs["search_provider"] = MagicMock()
        return kwargs
    if slug == "s3":
        kwargs["object_storage"] = MagicMock()
        return kwargs
    if slug == "mongodb":
        kwargs["document_store"] = MagicMock()
        return kwargs
    if slug == "redis":
        return kwargs
    msg = f"{slug}: no custom legacy smoke factory kwargs configured"
    raise AssertionError(msg)


def _assert_non_vector_runtime_operations(slug: str, integration: Any, tmp_path: Path) -> None:
    if slug == "pagerduty":
        assert integration.trigger_incident(summary="smoke") == "d1"
        assert integration.health().healthy is True
        return
    if slug == "ms365_graph":
        user = integration.get_user("u1")
        assert user.id == "u1"
        assert user.display_name == "Smoke"
        return
    if slug == "slack":
        from intergrax.runtime.notifications.models import NotificationMessage

        asyncio.run(
            integration.notify(
                NotificationMessage(
                    task_id="smoke",
                    subject="subject",
                    body="body",
                    channel="slack",
                    tenant_id="tenant-1",
                ),
            ),
        )
        return
    if slug == "filesystem":
        integration.put("smoke.txt", b"payload")
        stored = integration.get("smoke.txt")
        assert stored is not None
        assert stored.body == b"payload"
        return
    if slug == "github":
        issue = integration.get_issue("IGX-1")
        assert issue.key == "IGX-1"
        return
    if slug == "redis":
        integration.set("tenant", "key", b"value")
        assert integration.get("tenant", "key") == b"value"
        return
    if slug == "postgresql":
        integration.execute("SELECT 1")
        rows = integration.fetch_all("SELECT name FROM tenants")
        assert rows == [{"name": "alpha"}]
        return
    if slug == "tavily":
        hits = integration.search("intergrax", limit=1)
        assert hits[0].title == "intergrax"
        return
    if slug == "s3":
        integration.put("smoke.txt", b"payload")
        stored = integration.get("smoke.txt")
        assert stored is not None
        assert stored.body == b"payload"
        return
    if slug == "mongodb":
        from intergrax.integrations.contracts.document_store import DocumentRecord

        integration.put(DocumentRecord(partition_key="p1", row_key="r1", data={"k": "v"}))
        stored = integration.get("p1", "r1")
        assert stored is not None
        assert stored.data == {"k": "v"}
        return
    msg = f"{slug}: no runtime operation smoke configured"
    raise AssertionError(msg)


def test_non_vector_legacy_smoke_slugs_are_cutover_representatives() -> None:
    assert NON_VECTOR_LEGACY_SMOKE_SLUGS <= CUTOVER_SLUGS
    assert SLUG_CATEGORY["redis"] == "key_value_cache"
    assert SLUG_CATEGORY["postgresql"] == "relational_store"
    assert SLUG_CATEGORY["ms365_graph"] == "collaboration_suite"
    assert SLUG_CATEGORY["s3"] == "object_storage"
    assert SLUG_CATEGORY["mongodb"] == "document_store"
    assert SLUG_CATEGORY["tavily"] == "search_provider"


def test_cutover_registry_documents_deferred_llm_guardrail() -> None:
    for slug in DEFERRED_LLM_GUARDRAIL_SLUGS:
        assert slug not in CUTOVER_SLUGS
        assert SLUG_CATEGORY[slug] == "llm_guardrail"


@pytest.mark.parametrize("slug", sorted(CUTOVER_SLUGS))
def test_bundle_does_not_shadow_integration_class_name(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    bundle_path = _provider_dir(slug, category) / "bundle.py"
    tree = ast.parse(bundle_path.read_text(encoding="utf-8"))
    integration_name = _integration_class_name(slug, category)
    imported_from_adapter = False
    imported_from_integration = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        for alias in node.names:
            if alias.name != integration_name:
                continue
            if module.endswith(".adapter"):
                imported_from_adapter = True
            if module.endswith(".integration"):
                imported_from_integration = True
    assert not imported_from_adapter, f"{slug}: bundle imports {integration_name} from adapter.py"
    assert imported_from_integration, f"{slug}: bundle must import {integration_name} from integration.py"


@pytest.mark.parametrize("slug", sorted(CUTOVER_SLUGS))
def test_no_public_legacy_adapter_export(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    integration_name = _integration_class_name(slug, category)
    pkg = _init_module(slug, category)
    integration_cls = getattr(pkg, integration_name)
    assert integration_cls.__module__.endswith(".integration")
    adapter_path = _provider_dir(slug, category) / "adapter.py"
    if adapter_path.is_file():
        adapter_source = adapter_path.read_text(encoding="utf-8")
        tree = ast.parse(adapter_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                msg = f"{slug}: adapter.py must not expose public class {node.name!r}"
                raise AssertionError(msg)


@pytest.mark.parametrize("slug", sorted(CUTOVER_SLUGS))
def test_legacy_factory_delegates_to_integration_model(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    integration_cls = getattr(_integration_module(slug, category), _integration_class_name(slug, category))
    legacy_factory = getattr(_bundle_module(slug, category), _legacy_factory_name(slug, category))
    bundle_source = inspect.getsource(_bundle_module(slug, category))
    opens_path = _provider_dir(slug, category) / "opens.py"
    opens_source = opens_path.read_text(encoding="utf-8") if opens_path.is_file() else ""
    combined = bundle_source + opens_source
    assert hasattr(integration_cls, "from_runtime") or hasattr(integration_cls, "from_store") or hasattr(
        integration_cls, "from_client"
    )
    assert (
        f"{integration_cls.__name__}.from_runtime" in combined
        or f"{integration_cls.__name__}.from_store" in combined
        or f"{integration_cls.__name__}.from_client" in combined
        or f"return {integration_cls.__name__}." in combined
    )
    if category == "vector_store":
        fake = _FakeVectorStore()

        def _factory() -> _FakeVectorStore:
            return fake

        store = legacy_factory(store_factory=_factory, collection_name="c1", tenant_id="t1")
        assert isinstance(store, integration_cls)


@pytest.mark.parametrize("slug", sorted(NON_VECTOR_LEGACY_SMOKE_SLUGS))
def test_legacy_factory_non_vector_default_returns_integration(slug: str, tmp_path: Path) -> None:
    category = SLUG_CATEGORY[slug]
    integration_cls = getattr(_integration_module(slug, category), _integration_class_name(slug, category))
    legacy_factory = getattr(_bundle_module(slug, category), _legacy_factory_name(slug, category))
    result = legacy_factory(**_legacy_smoke_default_kwargs(slug, tmp_path))
    assert isinstance(result, integration_cls)


@pytest.mark.parametrize("slug", sorted(NON_VECTOR_LEGACY_SMOKE_SLUGS))
def test_legacy_factory_non_vector_custom_returns_integration(slug: str, tmp_path: Path) -> None:
    category = SLUG_CATEGORY[slug]
    integration_cls = getattr(_integration_module(slug, category), _integration_class_name(slug, category))
    legacy_factory = getattr(_bundle_module(slug, category), _legacy_factory_name(slug, category))
    result = legacy_factory(**_legacy_smoke_custom_kwargs(slug, tmp_path))
    assert isinstance(result, integration_cls)


@pytest.mark.parametrize("slug", sorted(NON_VECTOR_LEGACY_SMOKE_SLUGS))
def test_legacy_factory_non_vector_runtime_operations(slug: str, tmp_path: Path) -> None:
    category = SLUG_CATEGORY[slug]
    legacy_factory = getattr(_bundle_module(slug, category), _legacy_factory_name(slug, category))
    integration = legacy_factory(**_legacy_smoke_default_kwargs(slug, tmp_path))
    _assert_non_vector_runtime_operations(slug, integration, tmp_path)


@pytest.mark.parametrize("slug", sorted(VECTOR_STORE_CUTOVER_SLUGS))
def test_legacy_factory_vector_store_operations(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    legacy_factory = getattr(_bundle_module(slug, category), _legacy_factory_name(slug, category))
    fake = _FakeVectorStore()
    scope = VectorStoreScope(tenant_id="t1", namespace="rag")
    records = [_native_record(scope=scope)]

    def _factory() -> _FakeVectorStore:
        return fake

    store = legacy_factory(store_factory=_factory, collection_name="c1", tenant_id="t1")
    assert isinstance(store, VectorStore)
    assert store.add_records(records, scope=scope) == ["doc-1"]
    assert fake.records == records
    assert fake.add_scopes == [scope]
    hits = store.query([0.1, 0.2], scope=scope, top_k=1, include_embeddings=True)
    assert hits[0].document.scope == records[0].document.scope
    assert fake.query_scopes == [scope]
    assert store.count(scope=scope) == 1
    assert fake.count_scopes == [scope]
    store.delete(["doc-1"], scope=scope)
    assert fake.deleted == ["doc-1"]
    assert fake.delete_scopes == [scope]
    assert store.count(scope=scope) == 0


@pytest.mark.parametrize("slug", sorted(PRIMARY_VECTOR_STORE_CUTOVER_SLUGS))
def test_primary_vector_store_native_method_signatures(slug: str) -> None:
    module = importlib.import_module(
        f"intergrax.integrations.providers.vector_store.{slug}.rag_store"
    )
    provider_cls = getattr(module, _PRIMARY_VECTOR_STORE_CLASS_NAMES[slug])

    add_records = inspect.signature(provider_cls.add_records)
    assert "records" in add_records.parameters
    assert "scope" in add_records.parameters
    assert add_records.parameters["scope"].kind is inspect.Parameter.KEYWORD_ONLY
    assert not {"documents", "embeddings", "ids"} & add_records.parameters.keys()
    assert "add_documents" not in provider_cls.__dict__

    query = inspect.signature(provider_cls.query)
    assert "scope" in query.parameters
    assert query.parameters["scope"].kind is inspect.Parameter.KEYWORD_ONLY
    assert {"top_k", "metadata_filter", "include_embeddings"} <= query.parameters.keys()

    delete = inspect.signature(provider_cls.delete)
    assert {"ids", "scope"} <= delete.parameters.keys()
    assert delete.parameters["scope"].kind is inspect.Parameter.KEYWORD_ONLY

    count = inspect.signature(provider_cls.count)
    assert "scope" in count.parameters
    assert count.parameters["scope"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("slug", sorted(CLIENT_CONTRACT_CUTOVER_SLUGS))
def test_contract_factory_disabled_without_client(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    integration = _contract_factory(slug, category)(enabled=False, client=None)
    assert integration.config.enabled is False
    assert integration.client is None


@pytest.mark.parametrize("slug", sorted(OBSERVABILITY_CUTOVER_SLUGS))
def test_contract_factory_disabled_without_transport(slug: str) -> None:
    integration = _contract_factory(slug, "observability_backend")(enabled=False, transport=None)
    assert integration.config.enabled is False
    assert integration.transport is None


@pytest.mark.parametrize("slug", sorted(CLIENT_CONTRACT_CUTOVER_SLUGS))
def test_contract_factory_enabled_without_client_raises(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    with pytest.raises(IntegrationConfigurationError, match="client"):
        _contract_factory(slug, category)(enabled=True, client=None)


@pytest.mark.parametrize("slug", sorted(OBSERVABILITY_CUTOVER_SLUGS))
def test_contract_factory_enabled_without_transport_raises(slug: str) -> None:
    with pytest.raises(IntegrationConfigurationError, match="transport"):
        _contract_factory(slug, "observability_backend")(enabled=True, transport=None)


@pytest.mark.parametrize("slug", sorted(CLIENT_CONTRACT_CUTOVER_SLUGS))
def test_contract_factory_enabled_with_fake_client(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    client = _FakeClient()
    integration = _contract_factory(slug, category)(enabled=True, client=client)
    assert integration.client is client
    assert integration.config.enabled is True


class _FakeTransport:
    async def send_observability_payload(self, payload: object) -> None:
        return None


@pytest.mark.parametrize("slug", sorted(OBSERVABILITY_CUTOVER_SLUGS))
def test_contract_factory_enabled_with_fake_transport(slug: str) -> None:
    transport = _FakeTransport()
    integration = _contract_factory(slug, "observability_backend")(enabled=True, transport=transport)
    assert integration.transport is transport
    assert integration.config.enabled is True


@pytest.mark.parametrize("slug", sorted(CUTOVER_SLUGS))
def test_register_remains_compatible(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    register_mod = importlib.import_module(f"{_provider_pkg(slug, category)}.register")
    register_fn = getattr(register_mod, f"register_{slug}_integration")
    assert callable(register_fn)
    assert "register_from_manifest" in inspect.getsource(register_mod)


@pytest.mark.parametrize("slug", sorted(CUTOVER_SLUGS))
def test_integration_module_has_no_vendor_sdk_imports(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    path = _provider_dir(slug, category) / "integration.py"
    allowed_roots = frozenset({"__future__", "typing", "pydantic", "intergrax"})
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("from "):
            root = stripped.split()[1].split(".")[0]
            assert root in allowed_roots, f"unexpected import root {root!r} in {slug}"
        elif stripped.startswith("import ") and not stripped.startswith("import ("):
            root = stripped.split()[1].split(".")[0]
            assert root in allowed_roots, f"unexpected import root {root!r} in {slug}"


def test_pinecone_legacy_factory_delegates_to_integration_class() -> None:
    from intergrax.integrations.providers.vector_store.pinecone.bundle import (
        create_pinecone_vector_store,
        create_pinecone_vector_store_integration,
    )
    from intergrax.integrations.providers.vector_store.pinecone.integration import (
        PineconeVectorStoreIntegration,
    )

    fake = _FakeVectorStore()
    store = create_pinecone_vector_store(
        api_key="pc-test",
        index_name="idx",
        tenant_id="t1",
        store_factory=lambda: fake,
    )
    contract = create_pinecone_vector_store_integration(enabled=False)
    assert isinstance(store, PineconeVectorStoreIntegration)
    assert isinstance(contract, PineconeVectorStoreIntegration)
    assert store.rag_store is fake


def test_qdrant_legacy_factory_delegates_to_integration_class() -> None:
    from intergrax.integrations.providers.vector_store.qdrant.bundle import (
        create_qdrant_vector_store,
        create_qdrant_vector_store_integration,
    )
    from intergrax.integrations.providers.vector_store.qdrant.integration import (
        QdrantVectorStoreIntegration,
    )

    fake = _FakeVectorStore()
    store = create_qdrant_vector_store(
        collection_name="c1",
        tenant_id="t1",
        store_factory=lambda: fake,
    )
    contract = create_qdrant_vector_store_integration(enabled=False)
    assert isinstance(store, QdrantVectorStoreIntegration)
    assert isinstance(contract, QdrantVectorStoreIntegration)
    assert store.rag_store is fake
