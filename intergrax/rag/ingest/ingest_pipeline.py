# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Configurable ingest pipeline — loader, chunking strategy, optional contextual enrich."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from intergrax.distributed.source_operation import (
    InProcessSourceOperationCoordinator,
    RagSourceOperationKey,
    SourceOperationCoordinator,
    SourceOperationLease,
)
from intergrax.knowledge.contracts import KnowledgeDocument, KnowledgeDocumentScope
from intergrax.knowledge.contracts.validation import knowledge_metadata_to_plain
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.contextual.chunk_enricher import ContextualChunkEnricher
from intergrax.rag.document_loaders.compat.legacy_runtime_document import (
    copy_parser_runtime_state,
)
from intergrax.rag.document_loaders.contracts.base_document_loader import (
    BaseDocumentsLoader,
)
from intergrax.rag.document_loaders.pipeline.parser_pipeline import TRACE_METADATA_KEY
from intergrax.rag.document_splitters.contracts.base_documents_splitter import (
    BaseDocumentsSplitter,
)
from intergrax.rag.embedding.contracts.base_embedding_manager import (
    BaseEmbeddingManager,
)
from intergrax.rag.governance.embedding_version_policy import (
    evaluate_ingest_embedding_version,
)
from intergrax.rag.graph.contracts.graph_store import GraphStore
from intergrax.rag.graph.indexer.graph_indexer_factory import resolve_graph_indexer
from intergrax.rag.graph.lifecycle.graph_lifecycle_sync import (
    sync_graph_delete_documents,
)
from intergrax.rag.indexing.indexing_manager import IndexingManager
from intergrax.rag.indexing.strategies.dual_index_strategy import DualIndexStrategy
from intergrax.rag.ingest.ingest_policy import (
    semantic_chunking_allowed,
    sync_ingest_allowed,
)
from intergrax.rag.ingest.native_document_metadata import (
    add_native_metadata,
    filter_native_metadata,
)
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.tracking.rag_spans import rag_span
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import (
    BaseVectorstoreManager,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreRecord,
    VectorStoreScope,
)


@dataclass(frozen=True)
class IngestRequest:
    source_path: str
    base_metadata: dict[str, Any] = field(default_factory=dict)
    chunking_strategy_id: str | None = None
    workspace_id: str | None = None


@dataclass
class IngestResult:
    used: bool
    reason: str
    file_size_bytes: int = 0
    async_job_recommended: bool = False
    num_chunks: int = 0
    vector_ids: list[str] = field(default_factory=list)
    parser_id: str | None = None
    parser_trace: dict[str, Any] = field(default_factory=dict)
    version_warnings: list[str] = field(default_factory=list)
    reindex_recommended: bool = False


class _RecordingVectorstore:
    """Capture provider-returned IDs while preserving the vectorstore ABI."""

    def __init__(self, delegate: BaseVectorstoreManager) -> None:
        self._delegate = delegate
        self.persisted_ids: set[str] = set()

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope | None = None,
    ) -> Sequence[str] | None:
        stored_ids = self._delegate.add_records(records, scope=scope)
        ids = (
            list(stored_ids)
            if stored_ids is not None
            else [record.vector_id for record in records]
        )
        self.persisted_ids.update(ids)
        return stored_ids


class IngestPipeline:
    _SOURCE_LOOKUP_UNSUPPORTED = "vectorstore_source_record_lookup_not_supported"
    _SOURCE_REINGEST_UNSUPPORTED = "source_reingest_not_supported"

    def __init__(
        self,
        *,
        loader: BaseDocumentsLoader,
        splitter: BaseDocumentsSplitter,
        embedding_manager: BaseEmbeddingManager,
        vectorstore: BaseVectorstoreManager,
        toc_vectorstore: BaseVectorstoreManager | None = None,
        profile: RagProfile | None = None,
        contextual_enricher: ContextualChunkEnricher | None = None,
        graph_store: GraphStore | None = None,
        llm_for_graph: LLMAdapter | None = None,
        metadata_callback: Callable[..., dict[str, Any]] | None = None,
        source_coordinator: SourceOperationCoordinator | None = None,
    ) -> None:
        self._loader = loader
        self._splitter = splitter
        self._embedding_manager = embedding_manager
        self._vectorstore = vectorstore
        self._toc_vectorstore = toc_vectorstore
        self._profile = profile or RagProfile()
        self._contextual = contextual_enricher
        self._graph_store = graph_store
        self._llm_for_graph = llm_for_graph
        self._metadata_callback = metadata_callback
        self._source_coordinator = (
            source_coordinator or InProcessSourceOperationCoordinator()
        )

    def run(self, request: IngestRequest) -> IngestResult:
        path = Path(request.source_path)
        with rag_span(
            "rag.ingest",
            attributes={
                "rag.ingest.source": str(path),
                "rag.ingest.chunking_strategy": request.chunking_strategy_id
                or self._profile.chunking_strategy_id,
            },
        ):
            if not path.exists():
                return IngestResult(used=False, reason="source_not_found")

            allowed, size_reason, file_size = sync_ingest_allowed(
                path=path, profile=self._profile
            )
            if not allowed:
                return IngestResult(
                    used=False,
                    reason=size_reason,
                    file_size_bytes=file_size,
                    async_job_recommended=True,
                )

            base_metadata = dict(request.base_metadata)
            tenant_id = base_metadata.get("tenant_id")
            if not isinstance(tenant_id, str) or not tenant_id.strip():
                return IngestResult(used=False, reason="missing_tenant_id")

            namespace = base_metadata.get("namespace")
            loader_namespace = (
                namespace if isinstance(namespace, str) and namespace.strip() else None
            )

            version_policy = evaluate_ingest_embedding_version(
                profile=self._profile,
                base_metadata=base_metadata,
                source_path=str(path),
                indexed_version_hint=base_metadata.get(
                    "indexed_embedding_model_version"
                ),
            )
            if self._profile.embedding_model_version:
                base_metadata["embedding_model_version"] = (
                    self._profile.embedding_model_version
                )

            def _cb(doc: Any, source: str) -> dict[str, Any]:
                if self._metadata_callback is not None:
                    return filter_native_metadata(self._metadata_callback(doc, source))
                return filter_native_metadata(base_metadata)

            with rag_span(
                "rag.ingest.load", attributes={"rag.ingest.source": str(path)}
            ):
                native_docs = self._loader.load_document(
                    str(path),
                    tenant_id=tenant_id,
                    namespace=loader_namespace,
                    use_default_metadata=True,
                    call_custom_metadata=_cb,
                )
            if request.workspace_id is not None:
                scoped_docs: list[KnowledgeDocument] = []
                for document in native_docs:
                    payload = document.model_dump(mode="python")
                    payload["scope"] = KnowledgeDocumentScope(
                        tenant_id=document.scope.tenant_id,
                        namespace=document.scope.namespace,
                        workspace_id=request.workspace_id,
                    ).model_dump(mode="python")
                    scoped_docs.append(
                        copy_parser_runtime_state(
                            document,
                            KnowledgeDocument.model_validate(payload),
                        )
                    )
                native_docs = scoped_docs
            if not native_docs:
                return IngestResult(used=False, reason="no_documents_loaded")

            source_id, source_scope, ownership_error = self._source_ownership(
                native_docs
            )
            if ownership_error is not None:
                return IngestResult(used=False, reason=ownership_error)
            assert source_id is not None
            assert source_scope is not None

            return self._run_source_replacement(
                source_id=source_id,
                source_scope=source_scope,
                native_docs=native_docs,
                base_metadata=base_metadata,
                file_size=file_size,
                version_policy=version_policy,
                strategy_id=request.chunking_strategy_id
                or self._profile.chunking_strategy_id,
            )

    def _run_source_replacement(
        self,
        *,
        source_id: str,
        source_scope: VectorStoreScope,
        native_docs: list[KnowledgeDocument],
        base_metadata: dict[str, Any],
        file_size: int,
        version_policy: Any,
        strategy_id: str,
    ) -> IngestResult:
        key = RagSourceOperationKey(
            tenant_id=source_scope.tenant_id,
            namespace=source_scope.namespace,
            workspace_id=source_scope.workspace_id,
            source_id=source_id,
        )
        lease = self._source_coordinator.acquire(key=key)
        if lease is None:
            return IngestResult(used=False, reason="source_ingest_conflict")
        try:
            try:
                old_main_ids = self._list_current_source_ids(
                    source_id=source_id,
                    scope=source_scope,
                )
                old_toc_ids = (
                    self._list_current_source_ids(
                        source_id=source_id,
                        scope=source_scope,
                        vectorstore=self._toc_vectorstore,
                    )
                    if self._uses_dual_index()
                    else set()
                )
            except RuntimeError as exc:
                if str(exc) == self._SOURCE_REINGEST_UNSUPPORTED:
                    return IngestResult(
                        used=False,
                        reason=self._SOURCE_REINGEST_UNSUPPORTED,
                    )
                raise

            sem_allowed, sem_reason, _ = semantic_chunking_allowed(
                docs=native_docs,
                strategy_id=strategy_id,
                profile=self._profile,
            )
            if not sem_allowed:
                return IngestResult(
                    used=False,
                    reason=sem_reason,
                    file_size_bytes=file_size,
                    async_job_recommended=True,
                )

            parser_trace: dict[str, Any] = {}
            parser_id: str | None = None
            first_meta = dict(native_docs[0].metadata)
            if TRACE_METADATA_KEY in first_meta:
                parser_trace = dict(first_meta.get(TRACE_METADATA_KEY) or {})
                parser_id = parser_trace.get("parser_id") or first_meta.get(
                    "integration_parser_id"
                )

            with rag_span(
                "rag.ingest.chunk",
                attributes={"rag.ingest.chunking_strategy": strategy_id},
            ):
                native_chunks = self._splitter.split_documents(
                    native_docs,
                    strategy_id=strategy_id,
                )
            if not native_chunks:
                return IngestResult(
                    used=False,
                    reason="no_chunks_generated",
                    parser_id=parser_id,
                    parser_trace=parser_trace,
                )

            if self._profile.contextual_enrich == "on" and self._contextual is not None:
                native_chunks = self._contextual.enrich(
                    native_docs,
                    native_chunks,
                )
            native_chunks = [
                add_native_metadata(chunk, base_metadata) for chunk in native_chunks
            ]
            if any(
                VectorStoreScope.from_document(chunk) != source_scope
                or str(chunk.provenance.source_id) != source_id
                for chunk in native_chunks
            ):
                return IngestResult(
                    used=False,
                    reason="source_scope_or_ownership_mismatch",
                )
            self._ensure_source_lease(lease)
            texts = [chunk.content for chunk in native_chunks]

            with rag_span(
                "rag.ingest.index",
                attributes={
                    "rag.ingest.num_chunks": len(native_chunks),
                    "rag.ingest.dual_index": self._uses_dual_index(),
                },
            ):
                if self._uses_dual_index():
                    assert self._toc_vectorstore is not None
                    toc_recording_store = _RecordingVectorstore(self._toc_vectorstore)
                    vector_ids = list(
                        IndexingManager(
                            embed_manager=self._embedding_manager,
                            vectorstore=self._vectorstore,
                            strategy=DualIndexStrategy(
                                toc_vectorstore=toc_recording_store  # type: ignore[arg-type]
                            ),
                        ).index_documents(native_chunks)
                    )
                else:
                    embeddings = self._embedding_manager.embed_texts(texts)
                    records = [
                        VectorStoreRecord(
                            document=chunk,
                            embedding=embeddings[index],
                            vector_id=chunk.identity.document_id,
                        )
                        for index, chunk in enumerate(native_chunks)
                    ]
                    stored_ids = self._vectorstore.add_records(
                        records,
                        scope=source_scope,
                    )
                    if stored_ids is None:
                        vector_ids = [record.vector_id for record in records]
                    else:
                        vector_ids = list(stored_ids)
                        if len(vector_ids) != len(records):
                            raise ValueError(
                                "vectorstore returned an unexpected number of vector IDs"
                            )

            self._ensure_source_lease(lease)
            if self._uses_dual_index():
                published_main_ids = self._list_current_source_ids(
                    source_id=source_id,
                    scope=source_scope,
                )
                published_toc_ids = self._list_current_source_ids(
                    source_id=source_id,
                    scope=source_scope,
                    vectorstore=self._toc_vectorstore,
                )
                new_main_ids = set(vector_ids)
                new_toc_ids = toc_recording_store.persisted_ids
                if not new_main_ids.issubset(published_main_ids):
                    raise RuntimeError("source_reingest_main_ownership_mismatch")
                if not new_toc_ids.issubset(published_toc_ids):
                    raise RuntimeError("source_reingest_toc_ownership_mismatch")
            else:
                new_main_ids = set(vector_ids)
                new_toc_ids = set()

            self._ensure_source_lease(lease)
            if self._graph_store is not None and self._profile.graph_rag_enabled:
                with rag_span("rag.ingest.graph_index"):
                    try:
                        indexer = resolve_graph_indexer(
                            self._graph_store,
                            self._profile,
                            llm=self._llm_for_graph,
                        )
                        indexer.index_documents(native_chunks, chunk_ids=vector_ids)
                    except Exception as exc:
                        raise RuntimeError(
                            "source_reingest_graph_publish_failed"
                        ) from exc

                self._ensure_source_lease(lease)
                stale_graph_ids = old_main_ids - new_main_ids
                if stale_graph_ids:
                    try:
                        sync_graph_delete_documents(
                            self._graph_store,
                            sorted(stale_graph_ids),
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "source_reingest_graph_stale_unlink_failed"
                        ) from exc

            self._ensure_source_lease(lease)
            stale_main_ids = old_main_ids - new_main_ids
            if stale_main_ids:
                try:
                    self._vectorstore.delete(
                        sorted(stale_main_ids),
                        scope=source_scope,
                    )
                except Exception as exc:
                    raise RuntimeError("source_reingest_stale_delete_failed") from exc

            self._ensure_source_lease(lease)
            stale_toc_ids = old_toc_ids - new_toc_ids
            if stale_toc_ids:
                assert self._toc_vectorstore is not None
                try:
                    self._toc_vectorstore.delete(
                        sorted(stale_toc_ids),
                        scope=source_scope,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "source_reingest_toc_stale_delete_failed"
                    ) from exc

            self._ensure_source_lease(lease)
            return IngestResult(
                used=True,
                reason="ok",
                num_chunks=len(native_chunks),
                vector_ids=vector_ids,
                parser_id=parser_id,
                parser_trace=knowledge_metadata_to_plain(parser_trace),
                version_warnings=list(version_policy.warnings),
                reindex_recommended=version_policy.reindex_enqueued,
            )
        finally:
            self._source_coordinator.release(lease=lease)

    def _ensure_source_lease(self, lease: SourceOperationLease) -> None:
        if not self._source_coordinator.is_owned(lease=lease):
            raise RuntimeError("source_ingest_conflict")

    def _source_ownership(
        self,
        documents: list[KnowledgeDocument],
    ) -> tuple[str | None, VectorStoreScope | None, str | None]:
        source_ids = {str(document.provenance.source_id) for document in documents}
        if len(source_ids) != 1:
            return None, None, "source_ownership_ambiguous"

        scopes = {VectorStoreScope.from_document(document) for document in documents}
        if len(scopes) != 1:
            return None, None, "source_scope_or_ownership_mismatch"

        return next(iter(source_ids)), next(iter(scopes)), None

    def _list_current_source_ids(
        self,
        *,
        source_id: str,
        scope: VectorStoreScope,
        vectorstore: BaseVectorstoreManager | None = None,
    ) -> set[str]:
        target_vectorstore = vectorstore or self._vectorstore
        lookup = getattr(target_vectorstore, "list_source_record_ids", None)
        if not callable(lookup):
            lookup_error = RuntimeError(self._SOURCE_LOOKUP_UNSUPPORTED)
        else:
            try:
                return set(lookup(source_id=source_id, scope=scope))
            except RuntimeError as exc:
                if str(exc) != self._SOURCE_LOOKUP_UNSUPPORTED:
                    raise
                lookup_error = exc

        if target_vectorstore.count(scope=scope) == 0:
            return set()
        raise RuntimeError(self._SOURCE_REINGEST_UNSUPPORTED) from lookup_error

    def _uses_dual_index(self) -> bool:
        return (
            self._profile.uses_hierarchical_index()
            and self._toc_vectorstore is not None
        )
