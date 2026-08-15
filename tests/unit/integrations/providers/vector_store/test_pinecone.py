# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Pinecone integration provider (Phase M.6 P2)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.integrations._shared.conformance import assert_vector_store
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import MetadataFilter, VectorStore, VectorStoreHit
from intergrax.integrations.providers.vector_store.pinecone.integration import PineconeVectorStoreIntegration
from intergrax.integrations.providers.vector_store.pinecone.bundle import (
    PineconeIntegrationBundle,
    create_pinecone_integration,
    create_pinecone_vector_store,
)
from intergrax.integrations.providers.vector_store.pinecone.config import (
    ENV_PINECONE_API_KEY,
    ENV_PINECONE_INDEX,
    ENV_PINECONE_TENANT_ID,
    PineconeIntegrationConfig,
)
from intergrax.integrations.providers.vector_store.pinecone.register import register_pinecone_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreRecord, VectorStoreScope

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PINECONE_PKG = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "pinecone"
_THIS_TEST = Path(__file__).resolve()
_SCAN_ROOTS = ("intergrax", "applications", "agents", "tests")
_SKIP_DIR_NAMES = {".venv", "build", "__pycache__", "node_modules"}
_FORBIDDEN_OUTSIDE_PROVIDER = (
    "PineconeVectorStoreIntegration(",
    "integrations.providers.pinecone.opens",
    "from pinecone import",
    "import pinecone",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> Iterator[None]:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


class _FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.records: list[VectorStoreRecord] = []
        self.deleted: list[str] = []
        self.last_query: tuple[Sequence[float], int] | None = None

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str]:
        self.records.extend(records)
        return [record.vector_id for record in records]

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        self.last_query = (list(query_embedding), top_k)
        document = _document("hello", tenant_id=scope.tenant_id)
        return [
            VectorStoreHit(
                vector_id="doc-1",
                document=document,
                similarity_score=0.95,
                rank=0,
            )
        ]

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
        self.deleted.extend(ids)

    def count(self, *, scope: VectorStoreScope) -> int:
        return len(self.records)


def _document(content: str, *, tenant_id: str = "tenant-a") -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": "doc-1", "root_document_id": "doc-1"},
            "scope": {"tenant_id": tenant_id},
            "content": content,
            "metadata": {"source": "test"},
            "provenance": {"source_kind": "test", "source_id": "doc-1"},
        }
    )


def _record(content: str = "hello") -> VectorStoreRecord:
    return VectorStoreRecord(document=_document(content), embedding=[0.1, 0.2, 0.3], vector_id="doc-1")


def _pinecone_config() -> PineconeIntegrationConfig:
    return PineconeIntegrationConfig(
        api_key="pc-test",
        index_name="intergrax-rag",
        tenant_id="tenant-a",
    )


def _store_factory(store: _FakeVectorStore | None = None):
    fake = store or _FakeVectorStore()

    def _factory() -> _FakeVectorStore:
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


def test_pinecone_sdk_only_imported_in_opens_module() -> None:
    violations: list[str] = []
    for path in _PINECONE_PKG.glob("*.py"):
        if path.name == "opens.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "pinecone" in text.lower():
            violations.append(path.name)
    assert violations == []


def test_pinecone_not_constructed_outside_provider_package() -> None:
    violations: list[str] = []
    for path in _iter_python_files(*_SCAN_ROOTS):
        if path.resolve() == _THIS_TEST.resolve():
            continue
        if _PINECONE_PKG in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_OUTSIDE_PROVIDER:
            if pattern in text:
                violations.append(f"{path.relative_to(_PROJECT_ROOT).as_posix()}: {pattern}")
    assert violations == []


def test_pinecone_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_PINECONE_API_KEY, "pc-test")
    monkeypatch.setenv(ENV_PINECONE_INDEX, "prod-index")
    monkeypatch.setenv(ENV_PINECONE_TENANT_ID, "tenant-a")
    config = PineconeIntegrationConfig.from_env()
    assert config.api_key == "pc-test"
    assert config.resolved_index_name() == "prod-index"
    assert config.tenant_id == "tenant-a"


def test_pinecone_config_requires_api_key() -> None:
    with patch("intergrax.integrations.providers.vector_store.pinecone.opens._import_pinecone"):
        with pytest.raises(IntegrationConfigurationError, match="api_key"):
            create_pinecone_vector_store(api_key="")


def test_add_documents_and_query_delegate_to_rag_store() -> None:
    factory, inner = _store_factory()
    store = create_pinecone_vector_store(**_pinecone_config().model_dump(), store_factory=factory)
    scope = VectorStoreScope(tenant_id="tenant-a")

    store.add_records([_record()], scope=scope)
    hits = store.query([0.1, 0.2, 0.3], scope=scope, top_k=3)

    assert len(inner.records) == 1
    assert hits[0].content == "hello"
    assert_vector_store(store)


def test_delete_and_count_delegate() -> None:
    factory, inner = _store_factory()
    store = create_pinecone_vector_store(**_pinecone_config().model_dump(), store_factory=factory)
    scope = VectorStoreScope(tenant_id="tenant-a")
    store.add_records([_record("x")], scope=scope)

    store.delete(["doc-1"], scope=scope)

    assert inner.deleted == ["doc-1"]
    assert store.count(scope=scope) == 1


def test_opens_imports_pinecone_and_builds_rag_store() -> None:
    config = _pinecone_config()
    mock_rag_store = MagicMock()

    with patch("intergrax.integrations.providers.vector_store.pinecone.opens._import_pinecone") as import_mock:
        with patch(
            "intergrax.integrations.providers.vector_store.pinecone.rag_store.PineconeVectorStore",
            return_value=mock_rag_store,
        ) as rag_cls:
            from intergrax.integrations.providers.vector_store.pinecone.opens import open_pinecone_vector_store

            result = open_pinecone_vector_store(config)

    import_mock.assert_called_once()
    rag_cls.assert_called_once()
    assert isinstance(result, PineconeVectorStoreIntegration)
    assert result.rag_store is mock_rag_store


def test_missing_pinecone_package_raises_configuration_error() -> None:
    config = _pinecone_config()
    with patch(
        "intergrax.integrations.providers.vector_store.pinecone.opens._import_pinecone",
        side_effect=IntegrationConfigurationError("missing pinecone"),
    ):
        from intergrax.integrations.providers.vector_store.pinecone.opens import open_pinecone_vector_store

        with pytest.raises(IntegrationConfigurationError, match="missing pinecone"):
            open_pinecone_vector_store(config)


def test_create_pinecone_integration_bundle() -> None:
    factory, _ = _store_factory()
    bundle = create_pinecone_integration(**_pinecone_config().model_dump(), store_factory=factory)

    assert isinstance(bundle, PineconeIntegrationBundle)
    assert isinstance(bundle.vector_store, PineconeVectorStoreIntegration)
    assert bundle.config.tenant_id == "tenant-a"


def test_register_and_resolve_via_profile() -> None:
    register_pinecone_integration()
    profile = IntegrationProfile(vector_store="pinecone")
    factory, _ = _store_factory()

    store = resolve(
        IntegrationCategory.VECTOR_STORE,
        profile=profile,
        config={**_pinecone_config().model_dump(), "store_factory": factory},
    )

    assert_vector_store(store)
    assert isinstance(store, PineconeVectorStoreIntegration)


def test_register_default_integrations_includes_pinecone() -> None:
    register_default_integrations()
    profile = IntegrationProfile(vector_store="pinecone")
    factory, _ = _store_factory()

    store = resolve(
        IntegrationCategory.VECTOR_STORE,
        profile=profile,
        config={**_pinecone_config().model_dump(), "store_factory": factory},
    )

    assert isinstance(store, PineconeVectorStoreIntegration)
