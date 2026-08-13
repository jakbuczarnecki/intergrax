from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.distributed.source_operation import (
    DocumentStoreSourceOperationCoordinator,
    InProcessSourceOperationCoordinator,
    RagSourceOperationKey,
    SOURCE_PUBLICATION_GENERATION_METADATA_KEY,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreHit, VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = pytest.mark.unit

SCOPE = VectorStoreScope(
    tenant_id="tenant-a",
    namespace="namespace-a",
    workspace_id="workspace-a",
)
SOURCE_ID = "/data/user_docs/fresh.txt"


def _document(
    *,
    document_id: str = "fresh-source",
    generation: str | None = None,
) -> KnowledgeDocument:
    metadata: dict[str, object] = {}
    if generation is not None:
        metadata[SOURCE_PUBLICATION_GENERATION_METADATA_KEY] = generation
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": {
                "tenant_id": SCOPE.tenant_id,
                "namespace": SCOPE.namespace,
                "workspace_id": SCOPE.workspace_id,
            },
            "content": "alpha unique marker persisted payload",
            "metadata": metadata,
            "provenance": {
                "source_kind": "test",
                "source_id": SOURCE_ID,
            },
        }
    )


class _DenseOnlyProvider:
    def __init__(self, hits: list[VectorStoreHit]) -> None:
        self._hits = hits

    def query(
        self,
        query_embedding,
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter=None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        return list(self._hits[:top_k])

    def count(self, *, scope: VectorStoreScope) -> int:
        return len(self._hits)


def _manager_with_hits(
    *,
    generation: str | None,
    coordinator: object | None = None,
) -> VectorstoreManager:
    provider = _DenseOnlyProvider(
        [
            VectorStoreHit(
                vector_id="fresh-source",
                document=_document(generation=generation),
                similarity_score=0.7,
                rank=0,
            )
        ]
    )
    manager = VectorstoreManager(provider, scope=SCOPE)
    if coordinator is not None:
        manager.set_source_operation_coordinator(coordinator)
    return manager


def _source_key(*, document_id: str = "fresh-source") -> RagSourceOperationKey:
    return RagSourceOperationKey(
        tenant_id=SCOPE.tenant_id,
        namespace=SCOPE.namespace,
        workspace_id=SCOPE.workspace_id,
        source_id=SOURCE_ID,
        publication_scope_id=document_id,
    )


def test_legacy_generationless_record_stays_visible_with_coordinator() -> None:
    coordinator = InProcessSourceOperationCoordinator()
    manager = _manager_with_hits(generation=None, coordinator=coordinator)

    hits = manager.query([1.0, 0.0], scope=SCOPE, top_k=5)

    assert len(hits) == 1


def test_active_generation_match_is_visible() -> None:
    document_store = InMemoryDocumentStore()
    coordinator = DocumentStoreSourceOperationCoordinator(
        document_store,
        owner_id="worker-a",
    )
    key = _source_key()
    lease = coordinator.acquire(key=key)
    assert lease is not None
    assert coordinator.promote_publication(lease=lease)
    active_generation = coordinator.active_publication_generation(key=key)
    assert active_generation is not None

    manager = _manager_with_hits(
        generation=active_generation,
        coordinator=coordinator,
    )
    hits = manager.query([1.0, 0.0], scope=SCOPE, top_k=5)

    assert len(hits) == 1


def test_stale_generation_is_hidden() -> None:
    document_store = InMemoryDocumentStore()
    coordinator = DocumentStoreSourceOperationCoordinator(
        document_store,
        owner_id="worker-a",
    )
    key = _source_key()
    lease = coordinator.acquire(key=key)
    assert lease is not None
    assert coordinator.promote_publication(lease=lease)

    manager = _manager_with_hits(
        generation="999:stale-token",
        coordinator=coordinator,
    )
    hits = manager.query([1.0, 0.0], scope=SCOPE, top_k=5)

    assert hits == []


def test_unknown_authority_hides_generation_managed_record() -> None:
    coordinator = InProcessSourceOperationCoordinator()
    manager = _manager_with_hits(
        generation="1:cold-start-token",
        coordinator=coordinator,
    )

    hits = manager.query([1.0, 0.0], scope=SCOPE, top_k=5)

    assert hits == []


def test_document_store_coordinator_recovers_active_generation() -> None:
    document_store = InMemoryDocumentStore()
    writer = DocumentStoreSourceOperationCoordinator(
        document_store,
        owner_id="worker-a",
    )
    key = _source_key()
    lease = writer.acquire(key=key)
    assert lease is not None
    assert writer.promote_publication(lease=lease)
    active_generation = writer.active_publication_generation(key=key)
    assert active_generation is not None

    reader = DocumentStoreSourceOperationCoordinator(
        document_store,
        owner_id="worker-b",
    )
    assert reader.active_publication_generation(key=key) == active_generation


def test_reconstructed_document_store_coordinator_preserves_visibility() -> None:
    document_store = InMemoryDocumentStore()
    writer = DocumentStoreSourceOperationCoordinator(
        document_store,
        owner_id="worker-a",
    )
    key = _source_key()
    lease = writer.acquire(key=key)
    assert lease is not None
    assert writer.promote_publication(lease=lease)
    active_generation = writer.active_publication_generation(key=key)
    assert active_generation is not None

    before_restart = _manager_with_hits(
        generation=active_generation,
        coordinator=writer,
    )
    assert len(before_restart.query([1.0, 0.0], scope=SCOPE, top_k=5)) == 1

    after_restart = DocumentStoreSourceOperationCoordinator(
        document_store,
        owner_id="worker-restarted",
    )
    restarted_manager = _manager_with_hits(
        generation=active_generation,
        coordinator=after_restart,
    )
    assert len(restarted_manager.query([1.0, 0.0], scope=SCOPE, top_k=5)) == 1


def test_inprocess_empty_state_does_not_disable_fencing() -> None:
    coordinator = InProcessSourceOperationCoordinator()
    manager = _manager_with_hits(
        generation="1:token",
        coordinator=coordinator,
    )

    hits = manager.query([1.0, 0.0], scope=SCOPE, top_k=5)

    assert hits == []


def test_tenant_workspace_source_isolation() -> None:
    document_store = InMemoryDocumentStore()
    coordinator = DocumentStoreSourceOperationCoordinator(
        document_store,
        owner_id="worker-a",
    )
    source_a_key = RagSourceOperationKey(
        tenant_id=SCOPE.tenant_id,
        namespace=SCOPE.namespace,
        workspace_id=SCOPE.workspace_id,
        source_id=SOURCE_ID,
    )
    source_b_key = RagSourceOperationKey(
        tenant_id=SCOPE.tenant_id,
        namespace=SCOPE.namespace,
        workspace_id="workspace-b",
        source_id=SOURCE_ID,
    )
    lease_a = coordinator.acquire(key=source_a_key)
    assert lease_a is not None
    assert coordinator.promote_publication(lease=lease_a)
    active_a = coordinator.active_publication_generation(key=source_a_key)
    assert active_a is not None

    workspace_b_scope = VectorStoreScope(
        tenant_id=SCOPE.tenant_id,
        namespace=SCOPE.namespace,
        workspace_id="workspace-b",
    )
    other_workspace_doc = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": "other-workspace",
                "root_document_id": "other-workspace",
            },
            "scope": {
                "tenant_id": workspace_b_scope.tenant_id,
                "namespace": workspace_b_scope.namespace,
                "workspace_id": workspace_b_scope.workspace_id,
            },
            "content": "other workspace payload",
            "metadata": {SOURCE_PUBLICATION_GENERATION_METADATA_KEY: active_a},
            "provenance": {
                "source_kind": "test",
                "source_id": SOURCE_ID,
            },
        }
    )
    provider = _DenseOnlyProvider(
        [
            VectorStoreHit(
                vector_id="other-workspace",
                document=other_workspace_doc,
                similarity_score=0.7,
                rank=0,
            )
        ]
    )
    manager = VectorstoreManager(provider, scope=workspace_b_scope)
    manager.set_source_operation_coordinator(coordinator)

    hits = manager.query([1.0, 0.0], scope=workspace_b_scope, top_k=5)

    assert hits == []
    assert coordinator.active_publication_generation(key=source_b_key) is None


_CRITICAL_MODULES = (
    "intergrax/rag/vectorstore/vectorstore_manager.py",
    "intergrax/rag/vectorstore/contracts/hybrid_search.py",
    "intergrax/rag/retrievers/providers/hybrid_retriever.py",
    "intergrax/rag/vectorstore/publication_visibility.py",
    "intergrax/tools/providers/rag/source_operation_wiring.py",
    "intergrax/rag/ingest/ingest_pipeline.py",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _forbidden_patterns(source: str) -> list[str]:
    violations: list[str] = []
    if "getattr(" in source:
        violations.append("getattr(")
    if "setattr(" in source:
        violations.append("setattr(")
    if "hasattr(" in source:
        violations.append("hasattr(")
    if "delattr(" in source:
        violations.append("delattr(")
    if "._active_publications" in source:
        violations.append("_active_publications")
    if '"_inner"' in source or "'_inner'" in source:
        violations.append("_inner")
    if 'extras.get("source_operation_coordinator"' in source:
        violations.append('extras coordinator lookup')
    if 'extras["source_operation_coordinator"' in source:
        violations.append('extras coordinator lookup')
    if "self._store.query_hybrid(" in source:
        violations.append("untyped query_hybrid on VectorStore")
    return violations


_CRITICAL_FUNCTION_NAMES = frozenset(
    {
        "supports_native_hybrid_search",
        "query_hybrid",
        "_filter_visible_publication_hits",
        "retrieve",
        "vector_record_visible",
        "provider_supports_native_hybrid_search",
        "resolve_native_hybrid_search_provider",
        "shared_source_operation_coordinator",
        "bind_source_operation_coordinator",
        "list_source_record_ids",
        "_list_current_source_ids",
    }
)


def _function_source(node: ast.AST, source: str) -> str:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    return "\n".join(source.splitlines()[start:end])


def _collect_scoped_sources(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    scoped: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in _CRITICAL_FUNCTION_NAMES:
                scoped.append(_function_source(node, source))
    if path.name in {"hybrid_search.py", "publication_visibility.py", "source_operation_wiring.py"}:
        scoped.append(source)
    return scoped


def test_critical_rag_modules_forbid_reflection_wiring() -> None:
    root = _repo_root()
    for relative in _CRITICAL_MODULES:
        path = root / relative
        scoped_sources = _collect_scoped_sources(path)
        assert scoped_sources, f"{relative} has no scoped critical functions"
        for scoped in scoped_sources:
            violations = _forbidden_patterns(scoped)
            assert not violations, f"{relative} contains forbidden patterns: {violations}"


def test_critical_rag_modules_parse_cleanly() -> None:
    root = _repo_root()
    for relative in _CRITICAL_MODULES:
        path = root / relative
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_vector_store_declares_list_source_record_ids() -> None:
    import inspect

    from intergrax.rag.vectorstore.contracts.vector_store import VectorStore

    signature = inspect.signature(VectorStore.list_source_record_ids)
    assert tuple(signature.parameters) == (
        "self",
        "source_id",
        "scope",
        "root_document_id",
    )
