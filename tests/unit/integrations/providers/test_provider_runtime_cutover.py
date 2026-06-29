# © Artur Czarnecki. All rights reserved.

"""Runtime cutover guards — INTEGRATIONS-2E single provider entrypoint."""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import MetadataFilter, VectorStore, VectorStoreHit
from intergrax.integrations.providers.layout import SLUG_CATEGORY

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
        "slack",
        "filesystem",
        "github",
        "redis",
        "postgresql",
        "tavily",
        "exa",
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


class _FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.deleted: list[str] = []

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        self.documents.extend(documents)

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        return [
            VectorStoreHit(
                id="doc-1",
                content="hello",
                metadata={},
                similarity_score=0.9,
                rank=0,
            )
        ]

    def delete(self, ids: Sequence[str]) -> None:
        self.deleted.extend(ids)

    def count(self) -> int:
        return len(self.documents)


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
    def trigger_incident(self, *args: object, **kwargs: object) -> dict[str, str]:
        del args, kwargs
        return {"status": "success", "message": "Event processed", "dedup_key": "d1"}


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


def _legacy_smoke_factory_kwargs(slug: str, tmp_path: Path) -> dict[str, Any]:
    if slug == "pagerduty":
        return {"client": _FakePagerDutyEventsClient()}
    if slug == "slack":
        return {
            "integration_category": IntegrationCategory.NOTIFICATION_CHANNEL,
            "notification_adapter": MagicMock(),
        }
    if slug == "filesystem":
        return {"root_dir": str(tmp_path)}
    if slug == "github":
        return {"client": _FakeIssueClient()}
    if slug == "redis":
        return {"client": MagicMock(), "key_prefix": "smoke"}
    if slug == "postgresql":
        return {
            "connection_factory": _postgresql_connection_factory(),
            "dsn": "postgresql://localhost/test",
        }
    if slug in {"tavily", "exa"}:
        return {"client": _FakeSearchClient()}
    msg = f"{slug}: no legacy smoke factory kwargs configured"
    raise AssertionError(msg)


def test_non_vector_legacy_smoke_slugs_are_cutover_representatives() -> None:
    assert NON_VECTOR_LEGACY_SMOKE_SLUGS <= CUTOVER_SLUGS
    assert SLUG_CATEGORY["redis"] == "key_value_cache"
    assert SLUG_CATEGORY["postgresql"] == "relational_store"


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
        integration_cls, "from_backend"
    )
    assert (
        f"{integration_cls.__name__}.from_runtime" in combined
        or f"{integration_cls.__name__}.from_store" in combined
        or f"{integration_cls.__name__}.from_backend" in combined
        or f"return {integration_cls.__name__}." in combined
    )
    if category == "vector_store":
        fake = _FakeVectorStore()

        def _factory() -> _FakeVectorStore:
            return fake

        store = legacy_factory(store_factory=_factory, collection_name="c1", tenant_id="t1")
        assert isinstance(store, integration_cls)


@pytest.mark.parametrize("slug", sorted(NON_VECTOR_LEGACY_SMOKE_SLUGS))
def test_legacy_factory_non_vector_runtime_smoke(slug: str, tmp_path: Path) -> None:
    category = SLUG_CATEGORY[slug]
    integration_cls = getattr(_integration_module(slug, category), _integration_class_name(slug, category))
    legacy_factory = getattr(_bundle_module(slug, category), _legacy_factory_name(slug, category))
    result = legacy_factory(**_legacy_smoke_factory_kwargs(slug, tmp_path))
    assert isinstance(result, integration_cls)


@pytest.mark.parametrize("slug", sorted(VECTOR_STORE_CUTOVER_SLUGS))
def test_legacy_factory_vector_store_operations(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    legacy_factory = getattr(_bundle_module(slug, category), _legacy_factory_name(slug, category))
    fake = _FakeVectorStore()

    def _factory() -> _FakeVectorStore:
        return fake

    store = legacy_factory(store_factory=_factory, collection_name="c1", tenant_id="t1")
    assert isinstance(store, VectorStore)
    store.add_documents([Document(page_content="x")], [[0.1, 0.2]])
    store.delete(["doc-1"])
    assert fake.documents
    assert fake.deleted == ["doc-1"]
    assert store.count() == 1
    hits = store.query([0.1, 0.2], top_k=1)
    assert hits[0].content == "hello"


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
