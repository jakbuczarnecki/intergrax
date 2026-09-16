# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

from intergrax.applications._shared.memory_vector_wiring import build_session_turn_index_store
from intergrax.applications._shared.session_turn_index_rag_adapters import (
    SessionTurnIndexEmbeddingAdapter,
    SessionTurnIndexVectorstoreAdapter,
    adapt_rag_managers_to_session_turn_index_ports,
    build_session_turn_index_creation_context,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.memory.contracts.session_turn_index import (
    SessionTurnIndexEmbeddingPort,
    SessionTurnIndexMetadataFilter,
    SessionTurnIndexStoreCreationContext,
    SessionTurnIndexVectorQueryHit,
    SessionTurnIndexVectorScope,
    SessionTurnIndexVectorUpsertRecord,
    SessionTurnIndexVectorstorePort,
)
from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore
from intergrax.rag.bootstrap.rag_stack_bootstrap import RagStack
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True)
class _MemoryScope:
    tenant_id: str
    namespace: str | None
    workspace_id: str | None


@dataclass(frozen=True, slots=True)
class _MemoryFilter:
    conditions: Mapping[str, str | int | float]


@dataclass(frozen=True, slots=True)
class _MemoryUpsertRecord:
    vector_id: str
    document_content: str
    document_metadata: Mapping[str, str | int | float]
    embedding: Sequence[float]


class _FakeEmbeddingManager(BaseEmbeddingManager):
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        rows = [[float(index), float(len(text))] for index, text in enumerate(texts)]
        return np.asarray(rows, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_documents(self, documents):
        raise NotImplementedError


class _CapturingVectorstoreManager(BaseVectorstoreManager):
    def __init__(self) -> None:
        self.added_records: list[VectorStoreRecord] = []
        self.add_scopes: list[VectorStoreScope] = []
        self.query_scopes: list[VectorStoreScope] = []
        self.query_filters: list[MetadataFilter | None] = []
        self.deleted_ids: list[str] = []
        self.delete_scopes: list[VectorStoreScope] = []
        self.query_embedding: Sequence[float] | None = None

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope | None = None,
    ) -> None:
        assert scope is not None
        self.added_records.extend(records)
        self.add_scopes.append(scope)

    def query(
        self,
        query_embedding,
        *,
        scope: VectorStoreScope | None = None,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> Sequence[VectorStoreHit]:
        assert scope is not None
        self.query_scopes.append(scope)
        self.query_filters.append(metadata_filter)
        self.query_embedding = list(query_embedding)
        return []

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope | None = None) -> None:
        assert scope is not None
        self.deleted_ids.extend(ids)
        self.delete_scopes.append(scope)

    def list_source_record_ids(self, *, source_id: str, scope=None, root_document_id=None):
        return []

    def count(self, *, scope=None) -> int:
        return 0


def test_embedding_adapter_normalizes_ndarray_to_float_sequences() -> None:
    manager = _FakeEmbeddingManager()
    adapter = SessionTurnIndexEmbeddingAdapter(_delegate=manager)
    assert isinstance(adapter, SessionTurnIndexEmbeddingPort)
    out = adapter.embed_texts(["a", "b"])
    assert out == ((0.0, 1.0), (1.0, 1.0))


def test_vectorstore_adapter_add_translates_record_and_scope() -> None:
    manager = _CapturingVectorstoreManager()
    adapter = SessionTurnIndexVectorstoreAdapter(_delegate=manager)
    scope = _MemoryScope(tenant_id="tenant-a", namespace="ns-a", workspace_id="ws-a")
    record = _MemoryUpsertRecord(
        vector_id="vec-1",
        document_content="hello",
        document_metadata={"session_id": "sess-1", "deleted": 0},
        embedding=(0.1, 0.2),
    )
    adapter.add_records([record], scope=scope)
    assert len(manager.added_records) == 1
    rag_record = manager.added_records[0]
    assert rag_record.vector_id == "vec-1"
    assert rag_record.document.content == "hello"
    assert manager.add_scopes == [
        VectorStoreScope(tenant_id="tenant-a", namespace="ns-a", workspace_id="ws-a")
    ]


def test_vectorstore_adapter_query_translates_filter_and_scope() -> None:
    manager = _CapturingVectorstoreManager()
    adapter = SessionTurnIndexVectorstoreAdapter(_delegate=manager)
    scope = _MemoryScope(tenant_id="tenant-a", namespace="ns-a", workspace_id="ws-a")
    metadata_filter = _MemoryFilter(conditions={"deleted": 0, "session_id": "sess-1"})
    adapter.query((0.5, 0.625), scope=scope, top_k=3, metadata_filter=metadata_filter)
    assert manager.query_scopes == [
        VectorStoreScope(tenant_id="tenant-a", namespace="ns-a", workspace_id="ws-a")
    ]
    assert manager.query_filters == [MetadataFilter(conditions={"deleted": 0, "session_id": "sess-1"})]
    assert [float(x) for x in manager.query_embedding] == [0.5, 0.625]


def test_vectorstore_adapter_delete_preserves_ids_and_scope() -> None:
    manager = _CapturingVectorstoreManager()
    adapter = SessionTurnIndexVectorstoreAdapter(_delegate=manager)
    scope = _MemoryScope(tenant_id="tenant-a", namespace="ns-a", workspace_id="ws-a")
    adapter.delete(["id-1", "id-2"], scope=scope)
    assert manager.deleted_ids == ["id-1", "id-2"]
    assert manager.delete_scopes == [
        VectorStoreScope(tenant_id="tenant-a", namespace="ns-a", workspace_id="ws-a")
    ]


def test_vectorstore_adapter_query_hit_translation() -> None:
    from intergrax.knowledge.contracts import KnowledgeDocument

    manager = _CapturingVectorstoreManager()

    def _query(*args, **kwargs):
        doc = KnowledgeDocument.model_validate(
            {
                "schema_version": 1,
                "identity": {"document_id": "doc-1", "root_document_id": "doc-1"},
                "scope": {"tenant_id": "tenant-a"},
                "content": "turn text",
                "metadata": {"session_id": "sess-1", "role": "user", "deleted": 0},
                "provenance": {
                    "source_kind": "conversation_turn",
                    "source_id": "doc-1",
                    "source_parent_id": "sess-1",
                },
            }
        )
        return [
            VectorStoreHit(
                vector_id="doc-1",
                document=doc,
                similarity_score=0.88,
                rank=0,
            )
        ]

    manager.query = _query  # type: ignore[method-assign]
    adapter = SessionTurnIndexVectorstoreAdapter(_delegate=manager)
    hits = adapter.query(
        (0.1,),
        scope=_MemoryScope(tenant_id="tenant-a", namespace=None, workspace_id=None),
        top_k=1,
    )
    assert len(hits) == 1
    hit = hits[0]
    assert isinstance(hit, SessionTurnIndexVectorQueryHit)
    assert hit.document_id == "doc-1"
    assert hit.document_content == "turn text"
    assert hit.similarity_score == 0.88
    assert hit.document_metadata["session_id"] == "sess-1"


def test_provider_exception_propagates_from_vectorstore_adapter() -> None:
    manager = MagicMock(spec=BaseVectorstoreManager)
    manager.delete.side_effect = RuntimeError("backend down")
    adapter = SessionTurnIndexVectorstoreAdapter(_delegate=manager)
    with pytest.raises(RuntimeError, match="backend down"):
        adapter.delete(
            ["x"],
            scope=_MemoryScope(tenant_id="t", namespace=None, workspace_id=None),
        )


def test_memory_session_turn_index_contract_has_no_rag_imports() -> None:
    import ast
    from pathlib import Path

    path = Path("intergrax/memory/contracts/session_turn_index.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "rag" not in alias.name
                assert "runtime" not in alias.name.lower()
        if isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            assert "intergrax.rag" not in module
            assert "intergrax.runtime" not in module


class _PortCapturingPlugin:
    captured: SessionTurnIndexStoreCreationContext | None = None

    @classmethod
    def plugin_id(cls) -> str:
        return "test.port_capture"

    @classmethod
    def create_session_turn_index(
        cls,
        context: SessionTurnIndexStoreCreationContext,
    ) -> VectorSessionTurnIndexStore:
        cls.captured = context
        if context.embedding_manager is None or context.vectorstore_manager is None:
            raise ValueError("ports required")
        vectors = context.vectorstore_manager.query(
            (0.0,),
            scope=_MemoryScope(
                tenant_id=context.tenant_id,
                namespace=context.vector_index_namespace,
                workspace_id=context.workspace_id,
            ),
            top_k=1,
        )
        assert len(vectors) == 0
        return VectorSessionTurnIndexStore(
            embedding_port=context.embedding_manager,
            vectorstore_port=context.vectorstore_manager,
            tenant_id=context.tenant_id,
            vector_index_namespace=context.vector_index_namespace,
            workspace_id=context.workspace_id,
        )


def test_wiring_passes_adapted_ports_not_raw_rag_managers() -> None:
    embedding = _FakeEmbeddingManager()
    vectorstore = _CapturingVectorstoreManager()
    stack = RagStack(
        profile=RagProfile(),
        vectorstore_manager=vectorstore,
        embedding_manager=embedding,
        retriever_manager=MagicMock(),
        reranker_manager=MagicMock(),
        retrieval_service=MagicMock(),
    )
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.ent13b.r2.wiring")
    env.memory_profile = MemoryProfile(enable_session_vector_index=True)
    _PortCapturingPlugin.captured = None
    store = build_session_turn_index_store(
        env,
        tenant_id="tenant-wire",
        rag_stack=stack,
        session_turn_index_plugins=[_PortCapturingPlugin],
    )
    assert store is not None
    context = _PortCapturingPlugin.captured
    assert context is not None
    assert isinstance(context.embedding_manager, SessionTurnIndexEmbeddingAdapter)
    assert isinstance(context.vectorstore_manager, SessionTurnIndexVectorstoreAdapter)
    assert not isinstance(context.embedding_manager, BaseEmbeddingManager)
    assert not isinstance(context.vectorstore_manager, BaseVectorstoreManager)


def test_default_vector_store_path_uses_same_adapters_as_context_builder() -> None:
    embedding = _FakeEmbeddingManager()
    vectorstore = _CapturingVectorstoreManager()
    context = build_session_turn_index_creation_context(
        tenant_id="tenant-default",
        embedding_manager=embedding,
        vectorstore_manager=vectorstore,
    )
    store = VectorSessionTurnIndexStore(
        embedding_port=context.embedding_manager,
        vectorstore_port=context.vectorstore_manager,
        tenant_id=context.tenant_id,
    )
    assert isinstance(context.embedding_manager, SessionTurnIndexEmbeddingAdapter)
    assert isinstance(context.vectorstore_manager, SessionTurnIndexVectorstoreAdapter)
    assert store is not None


def test_adapt_rag_managers_returns_runtime_checkable_ports() -> None:
    embedding_port, vectorstore_port = adapt_rag_managers_to_session_turn_index_ports(
        embedding_manager=_FakeEmbeddingManager(),
        vectorstore_manager=_CapturingVectorstoreManager(),
    )
    assert isinstance(embedding_port, SessionTurnIndexEmbeddingPort)
    assert isinstance(vectorstore_port, SessionTurnIndexVectorstorePort)
