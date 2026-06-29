# © Artur Czarnecki. All rights reserved.

"""Runtime cutover guards — INTEGRATIONS-2E single provider entrypoint."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any, Optional, Sequence

import pytest
from langchain_core.documents import Document

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import MetadataFilter, VectorStore, VectorStoreHit
from intergrax.integrations.providers.layout import SLUG_CATEGORY

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Providers fully cut over to single Integration entrypoint (behavior tests must pass).
CUTOVER_SLUGS: frozenset[str] = frozenset({"pinecone", "qdrant"})

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
    return f"{_class_prefix(slug, category)}Integration"


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
    for line in register_source.splitlines():
        if "register_from_manifest" in line or "factory=" in line:
            for token in line.replace("(", " ").replace(",", " ").split():
                if token.startswith("create_") and token != contract_name:
                    return token
    bundle = _bundle_module(slug, category)
    for name in getattr(bundle, "__all__", ()):
        if name.startswith("create_") and name != contract_name:
            return name
    msg = f"{slug}: legacy factory not found"
    raise AssertionError(msg)


def _contract_factory(slug: str, category: str) -> Any:
    return getattr(_bundle_module(slug, category), f"create_{slug}_{category}_integration")


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
def test_legacy_factory_returns_integration_entrypoint(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    integration_cls = getattr(_integration_module(slug, category), _integration_class_name(slug, category))
    legacy_factory = getattr(_bundle_module(slug, category), _legacy_factory_name(slug, category))
    fake = _FakeVectorStore()

    def _factory() -> _FakeVectorStore:
        return fake

    store = legacy_factory(store_factory=_factory, collection_name="c1", tenant_id="t1")
    assert isinstance(store, integration_cls)


@pytest.mark.parametrize("slug", sorted(CUTOVER_SLUGS))
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


@pytest.mark.parametrize("slug", sorted(CUTOVER_SLUGS))
def test_contract_factory_disabled_without_client(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    integration = _contract_factory(slug, category)(enabled=False, client=None)
    assert integration.config.enabled is False
    assert integration.client is None


@pytest.mark.parametrize("slug", sorted(CUTOVER_SLUGS))
def test_contract_factory_enabled_without_client_raises(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    with pytest.raises(IntegrationConfigurationError, match="client"):
        _contract_factory(slug, category)(enabled=True, client=None)


@pytest.mark.parametrize("slug", sorted(CUTOVER_SLUGS))
def test_contract_factory_enabled_with_fake_client(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    client = _FakeClient()
    integration = _contract_factory(slug, category)(enabled=True, client=client)
    assert integration.client is client
    assert integration.config.enabled is True


@pytest.mark.parametrize("slug", sorted(CUTOVER_SLUGS))
def test_register_remains_compatible(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    register_mod = importlib.import_module(f"{_provider_pkg(slug, category)}.register")
    register_fn = getattr(register_mod, f"register_{slug}_integration")
    assert callable(register_fn)
    legacy_name = _legacy_factory_name(slug, category)
    assert legacy_name in inspect.getsource(register_mod)


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
