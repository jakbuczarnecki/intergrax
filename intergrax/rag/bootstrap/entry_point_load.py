# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""RAG public entry-point discovery with shared platform load evidence (PLUG-04)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.core.plugin_env import discover_plugins_enabled
from intergrax.core.plugins.admission import (
    DomainPluginLoadReport,
    PluginAdmissionReasonCode,
    PluginAdmissionRejection,
)
from intergrax.core.plugins.discovery import (
    EP_RAG_CHUNKERS,
    EP_RAG_RERANKERS,
    EP_RAG_RETRIEVERS,
    EntryPointSpec,
    register_plugins_with_report,
)
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.document_splitters.contracts.base_chunking_strategy import (
    BaseChunkingStrategy,
)
from intergrax.rag.document_splitters.registry.strategy_registry import (
    ChunkingStrategyRegistry,
)
from intergrax.rag.embedding.bootstrap.default_embedding_engine import (
    create_default_embedding_manager,
)
from intergrax.rag.embedding.contracts.base_embedding_manager import (
    BaseEmbeddingManager,
)
from intergrax.rag.graph.contracts.graph_store import GraphStore
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.rerankers.contracts.base_reranker import (
    BaseReranker,
    BaseRerankerPlugin,
)
from intergrax.rag.rerankers.registry.reranker_registry import RerankerRegistry
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    BaseRetrieverPlugin,
)
from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry
from intergrax.rag.vectorstore.bootstrap.vectorstore_bootstrap import (
    create_default_vectorstore_manager,
)
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import (
    BaseVectorstoreManager,
)


@dataclass(frozen=True, slots=True)
class RagPluginLoadEvidence:
    """Immutable snapshot of RAG public EP bootstrap evidence (metadata only)."""

    chunker_report: DomainPluginLoadReport
    retriever_report: DomainPluginLoadReport
    reranker_report: DomainPluginLoadReport


def register_rag_chunker_entry_points(
    registry: ChunkingStrategyRegistry,
    *,
    discover_entry_points: bool,
) -> DomainPluginLoadReport:
    def _register_entry_point(
        plugin_type: type,
        spec: EntryPointSpec,
    ) -> tuple[bool, PluginAdmissionRejection | None]:
        if not issubclass(plugin_type, BaseChunkingStrategy):
            message = (
                f"RAG chunker entry point {spec.name!r} must subclass "
                f"BaseChunkingStrategy: {plugin_type!r}"
            )
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                reason=message,
                fail_closed=True,
            )
        registry.register(plugin_type())
        return True, None

    return register_plugins_with_report(
        EP_RAG_CHUNKERS,
        _register_entry_point,
        discover_entry_points=discover_entry_points,
    )


def register_rag_retriever_entry_points(
    registry: RetrieverRegistry,
    *,
    vector_store: BaseVectorstoreManager,
    embedding_manager: BaseEmbeddingManager,
    toc_vector_store: BaseVectorstoreManager | None,
    graph_store: GraphStore | None,
    profile: RagProfile | None,
    llm_for_query_expansion: LLMAdapter | None,
    discover_entry_points: bool,
) -> DomainPluginLoadReport:
    def _register_entry_point(
        plugin_type: type,
        spec: EntryPointSpec,
    ) -> tuple[bool, PluginAdmissionRejection | None]:
        if issubclass(plugin_type, BaseRetrieverPlugin):
            retriever = plugin_type.create(
                vector_store=vector_store,
                embedding_manager=embedding_manager,
                toc_vector_store=toc_vector_store,
                graph_store=graph_store,
                profile=profile,
                llm_for_query_expansion=llm_for_query_expansion,
            )
        elif issubclass(plugin_type, BaseRetriever):
            retriever = plugin_type()
        else:
            message = (
                "RAG retriever entry point must subclass BaseRetriever or "
                f"BaseRetrieverPlugin: {plugin_type!r}"
            )
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                reason=message,
                fail_closed=True,
            )
        if not isinstance(retriever, BaseRetriever):
            message = (
                f"RAG retriever plugin factory must return BaseRetriever: "
                f"{plugin_type!r}"
            )
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                reason=message,
                fail_closed=True,
            )
        registry.register(retriever)
        return True, None

    return register_plugins_with_report(
        EP_RAG_RETRIEVERS,
        _register_entry_point,
        discover_entry_points=discover_entry_points,
    )


def register_rag_reranker_entry_points(
    registry: RerankerRegistry,
    *,
    embedding_manager: BaseEmbeddingManager,
    discover_entry_points: bool,
) -> DomainPluginLoadReport:
    def _register_entry_point(
        plugin_type: type,
        spec: EntryPointSpec,
    ) -> tuple[bool, PluginAdmissionRejection | None]:
        if issubclass(plugin_type, BaseRerankerPlugin):
            reranker = plugin_type.create(embedding_manager=embedding_manager)
        elif issubclass(plugin_type, BaseReranker):
            reranker = plugin_type()
        else:
            message = (
                "RAG reranker entry point must subclass BaseReranker or "
                f"BaseRerankerPlugin: {plugin_type!r}"
            )
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                reason=message,
                fail_closed=True,
            )
        if not isinstance(reranker, BaseReranker):
            message = (
                f"RAG reranker plugin factory must return BaseReranker: "
                f"{plugin_type!r}"
            )
            return False, PluginAdmissionRejection(
                spec=spec,
                reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                reason=message,
                fail_closed=True,
            )
        registry.register(reranker)
        return True, None

    return register_plugins_with_report(
        EP_RAG_RERANKERS,
        _register_entry_point,
        discover_entry_points=discover_entry_points,
    )


def empty_rag_plugin_load_evidence() -> RagPluginLoadEvidence:
    """Deterministic empty snapshot when discovery is disabled."""
    return RagPluginLoadEvidence(
        chunker_report=DomainPluginLoadReport.empty(EP_RAG_CHUNKERS),
        retriever_report=DomainPluginLoadReport.empty(EP_RAG_RETRIEVERS),
        reranker_report=DomainPluginLoadReport.empty(EP_RAG_RERANKERS),
    )


def bootstrap_rag_chunker_plugin_load_report(
    *,
    discover_entry_points: bool,
) -> DomainPluginLoadReport:
    """Run chunker EP admission once against a fresh strategy registry."""
    return register_rag_chunker_entry_points(
        ChunkingStrategyRegistry(),
        discover_entry_points=discover_entry_points,
    )


def compose_rag_plugin_load_evidence(
    *,
    chunker_report: DomainPluginLoadReport,
    retriever_report: DomainPluginLoadReport,
    reranker_report: DomainPluginLoadReport,
) -> RagPluginLoadEvidence:
    return RagPluginLoadEvidence(
        chunker_report=chunker_report,
        retriever_report=retriever_report,
        reranker_report=reranker_report,
    )


def bootstrap_rag_plugin_load_evidence_for_host_context(
    *,
    discover_entry_points: bool,
    embedding_manager: BaseEmbeddingManager,
    vector_store: BaseVectorstoreManager,
    toc_vector_store: BaseVectorstoreManager | None = None,
    graph_store: GraphStore | None = None,
    profile: RagProfile | None = None,
    llm_for_query_expansion: LLMAdapter | None = None,
) -> RagPluginLoadEvidence:
    """Discover RAG public EP groups with host-composed runtime dependencies."""
    if not discover_entry_points:
        return empty_rag_plugin_load_evidence()

    chunker_report = bootstrap_rag_chunker_plugin_load_report(
        discover_entry_points=True,
    )
    retriever_report = register_rag_retriever_entry_points(
        RetrieverRegistry(),
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        toc_vector_store=toc_vector_store,
        graph_store=graph_store,
        profile=profile,
        llm_for_query_expansion=llm_for_query_expansion,
        discover_entry_points=True,
    )
    reranker_report = register_rag_reranker_entry_points(
        RerankerRegistry(),
        embedding_manager=embedding_manager,
        discover_entry_points=True,
    )
    return compose_rag_plugin_load_evidence(
        chunker_report=chunker_report,
        retriever_report=retriever_report,
        reranker_report=reranker_report,
    )


def collect_rag_plugin_load_evidence(
    *,
    discover_entry_points: bool | None = None,
) -> RagPluginLoadEvidence:
    """Diagnostic-only duplicate bootstrap; not for production application evidence."""
    if discover_entry_points is None:
        discover_entry_points = discover_plugins_enabled()
    if not discover_entry_points:
        return empty_rag_plugin_load_evidence()

    embedding_manager = create_default_embedding_manager()
    vector_store = create_default_vectorstore_manager(
        tenant_id="__platform_plugin_evidence__",
    )
    return bootstrap_rag_plugin_load_evidence_for_host_context(
        discover_entry_points=True,
        embedding_manager=embedding_manager,
        vector_store=vector_store,
    )
