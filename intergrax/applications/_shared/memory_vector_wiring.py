# © Artur Czarnecki. All rights reserved.

"""Vector memory wiring helpers (Phase MEM-VEC-1.1–1.4)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.applications._shared.memory_provider_admission import (
    validate_session_turn_index_store_admission,
)
from intergrax.memory.memory_vector_errors import MemoryVectorBackendUnavailableError
from intergrax.applications._shared.session_turn_index_rag_adapters import (
    build_session_turn_index_creation_context,
)
from intergrax.memory.contracts.provider_identity import (
    BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
    MemoryProviderIdentity,
    builtin_session_turn_index_store_identity,
    plugin_session_turn_index_store_identity,
)
from intergrax.memory.contracts.session_turn_index import (
    SessionTurnIndexStore,
    SessionTurnIndexStorePlugin,
)
from intergrax.memory.resolver.classifier import (
    ClassifiedMemoryStorePlugin,
    MemoryStorePluginKind,
    classify_memory_store_plugin_record,
)
from intergrax.memory.resolver.discovery import (
    MemoryStorePluginCatalog,
    index_classified_memory_store_plugins,
)
from intergrax.memory.resolver.errors import MemoryStorePluginResolutionError
from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_ltm_vector_projection import UserProfileLtmVectorProjection
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.integrations.contracts.integration_profile import IntegrationProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager

if TYPE_CHECKING:
    from intergrax.memory.contracts.entity_temporal_memory import (
        EntityMemoryIndexer,
        EntityTemporalMemoryCapability,
    )
    from intergrax.memory.contracts.memory_lifecycle import UserProfileMemoryProjection
    from intergrax.memory.contracts.provider_admission_evidence import (
        MemoryProviderAdmissionEvidenceContext,
    )
    from intergrax.memory.contracts.provider_qualification_evidence import (
        MemoryProviderQualificationEvidenceRegistry,
    )
    from intergrax.rag.bootstrap.rag_stack_bootstrap import RagStack


def memory_vector_flags_require_backend(env: ApplicationEnvironmentProfile) -> bool:
    profile = env.memory_profile
    return bool(profile.enable_long_term_memory or profile.enable_session_vector_index)


def vector_store_backing_provider_id(profile: IntegrationProfile | None) -> str | None:
    """Platform-owned vector backend slug from integration profile binding."""
    if profile is None:
        return None
    binding = profile.vector_store
    if binding is None:
        return None
    return binding.resolved_slug()


def resolve_builtin_session_turn_index_provider_identity(
    integration_profile: IntegrationProfile | None,
) -> MemoryProviderIdentity:
    backing = vector_store_backing_provider_id(integration_profile)
    return builtin_session_turn_index_store_identity(
        BUILTIN_VECTOR_SESSION_TURN_INDEX_ID,
        backing_provider_id=backing,
    )


def resolve_plugin_session_turn_index_provider_identity(
    plugin: ClassifiedMemoryStorePlugin,
) -> MemoryProviderIdentity:
    if plugin.kind is not MemoryStorePluginKind.SESSION_TURN_INDEX:
        raise MemoryStorePluginResolutionError(
            f"Session turn index plugin expected, got {plugin.kind!r} for {plugin.plugin_id!r}"
        )
    return plugin_session_turn_index_store_identity(plugin.plugin_id)


def resolve_session_turn_index_provider_identity(
    integration_profile: IntegrationProfile | None,
) -> MemoryProviderIdentity:
    """Resolve builtin vector ``SessionTurnIndexStore`` provider identity only."""
    return resolve_builtin_session_turn_index_provider_identity(integration_profile)


def _classify_session_turn_index_plugin_type(plugin_type: type) -> ClassifiedMemoryStorePlugin:
    record = classify_memory_store_plugin_record(plugin_type)
    if record is None or record.kind is not MemoryStorePluginKind.SESSION_TURN_INDEX:
        raise MemoryStorePluginResolutionError(
            f"Invalid session turn index plugin type {plugin_type!r}: "
            "must implement SessionTurnIndexStorePlugin with a non-empty plugin_id"
        )
    return record


def _select_session_turn_index_plugin(
    plugin_types: Sequence[type],
) -> ClassifiedMemoryStorePlugin | None:
    if not plugin_types:
        return None
    records = [_classify_session_turn_index_plugin_type(plugin_type) for plugin_type in plugin_types]
    index_classified_memory_store_plugins(records)
    if len(records) > 1:
        plugin_ids = ", ".join(sorted(record.plugin_id for record in records))
        raise MemoryStorePluginResolutionError(
            f"Ambiguous session turn index plugin selection: {plugin_ids}"
        )
    return records[0]


def _session_turn_index_plugins_from_catalog(
    catalog: MemoryStorePluginCatalog,
) -> tuple[ClassifiedMemoryStorePlugin, ...]:
    records = [
        item
        for item in catalog.index.values()
        if item.kind is MemoryStorePluginKind.SESSION_TURN_INDEX
    ]
    return tuple(sorted(records, key=lambda record: record.plugin_id))


def _discover_classified_session_turn_index_plugins(
    *,
    discover_entry_points: bool,
    memory_store_plugin_catalog: MemoryStorePluginCatalog | None = None,
) -> tuple[ClassifiedMemoryStorePlugin, ...]:
    if memory_store_plugin_catalog is not None:
        return _session_turn_index_plugins_from_catalog(memory_store_plugin_catalog)
    if not discover_entry_points:
        return ()

    from intergrax.memory.resolver.discovery import discover_classified_memory_store_plugins

    discovery = discover_classified_memory_store_plugins(
        discover_entry_points=discover_entry_points,
    )
    records = [
        item
        for item in discovery.plugins
        if item.kind is MemoryStorePluginKind.SESSION_TURN_INDEX
    ]
    if records:
        index_classified_memory_store_plugins(records)
    return tuple(records)


def _require_runtime_tenant(tenant_id: str | None) -> str:
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise MemoryVectorBackendUnavailableError(reason="tenant_required")
    return tenant_id.strip()


def _product_session_turn_index_enabled(env: ApplicationEnvironmentProfile) -> bool:
    return (
        env.application_profile is ApplicationProfile.PRODUCT
        and env.memory_profile.enable_session_vector_index
    )


def resolve_rag_stack_for_memory_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    tenant_id: str | None = None,
    integration_profile: IntegrationProfile | None = None,
    llm_adapter: LLMAdapter | None = None,
) -> RagStack | None:
    """Resolve RAG stack for memory vector indexes — independent of ``enable_rag``."""
    from intergrax.applications._shared.rag_runtime_bridge import (
        resolve_rag_stack_for_environment,
    )
    from intergrax.rag.bootstrap.rag_stack_bootstrap import create_default_rag_stack

    if not memory_vector_flags_require_backend(env):
        if tenant_id is None and env.context_profile.enable_rag:
            return None
        return resolve_rag_stack_for_environment(
            env,
            tenant_id=tenant_id,
            integration_profile=integration_profile,
            llm_adapter=llm_adapter,
        )
    profile = integration_profile or env.integration_profile
    resolved_tenant_id = _require_runtime_tenant(tenant_id)
    return create_default_rag_stack(
        integration_profile=profile,
        tenant_id=resolved_tenant_id,
        llm_for_contextual=llm_adapter,
    )


def assert_memory_vector_backend_available(
    env: ApplicationEnvironmentProfile,
    rag_stack: RagStack | None,
) -> None:
    """Fail closed when vector memory flags are true but RAG stack lacks vector backends."""
    if not memory_vector_flags_require_backend(env):
        return
    if rag_stack is None:
        raise MemoryVectorBackendUnavailableError(reason="vector_backend_unavailable")
    if rag_stack.vectorstore_manager is None or rag_stack.embedding_manager is None:
        raise MemoryVectorBackendUnavailableError(reason="vector_backend_unavailable")


def build_user_profile_manager(
    store: UserProfileStore,
    env: ApplicationEnvironmentProfile,
    *,
    tenant_id: str | None = None,
    rag_stack: RagStack | None = None,
    entity_memory_indexer: EntityMemoryIndexer | None = None,
    entity_temporal_memory_capability: EntityTemporalMemoryCapability | None = None,
) -> UserProfileManager | None:
    """Construct ``UserProfileManager`` with optional LTM vector dependencies."""
    profile = env.memory_profile
    if not (profile.enable_user_memory or profile.enable_long_term_memory):
        return None

    resolved_tenant_id = _require_runtime_tenant(tenant_id)
    embedding_manager: BaseEmbeddingManager | None = None
    vectorstore_manager: BaseVectorstoreManager | None = None
    retrieval_service: RetrievalService | None = None
    rag_profile: RagProfile | None = None
    if profile.enable_long_term_memory and rag_stack is not None:
        embedding_manager = rag_stack.embedding_manager
        vectorstore_manager = rag_stack.vectorstore_manager
        retrieval_service = rag_stack.retrieval_service
        rag_profile = rag_stack.profile

    projections: list[UserProfileMemoryProjection] = []
    if (
        profile.enable_entity_graph_memory
        and entity_memory_indexer is not None
        and entity_temporal_memory_capability is not None
    ):
        from intergrax.applications._shared.entity_user_profile_memory_projection import (
            EntityIndexerUserProfileMemoryProjection,
        )

        projections.append(
            EntityIndexerUserProfileMemoryProjection(
                indexer=entity_memory_indexer,
                entity_capability=entity_temporal_memory_capability,
            )
        )
    if profile.enable_long_term_memory and rag_stack is not None:
        projections.append(
            UserProfileLtmVectorProjection(
                embedding_manager=rag_stack.embedding_manager,
                vectorstore_manager=rag_stack.vectorstore_manager,
                tenant_id=resolved_tenant_id,
                vector_index_namespace=profile.vector_index_namespace,
                workspace_id=None,
            )
        )
    memory_projections = tuple(projections) if projections else None

    return UserProfileManager(
        store,
        tenant_id=resolved_tenant_id,
        vector_index_namespace=profile.vector_index_namespace,
        embedding_manager=embedding_manager,
        vectorstore_manager=vectorstore_manager,
        retrieval_service=retrieval_service,
        rag_profile=rag_profile,
        memory_projections=memory_projections,
    )


def build_session_turn_index_store(
    env: ApplicationEnvironmentProfile,
    *,
    tenant_id: str | None = None,
    rag_stack: RagStack | None = None,
    integration_profile: IntegrationProfile | None = None,
    session_turn_index_plugins: Sequence[type[SessionTurnIndexStorePlugin]] = (),
    discover_entry_points: bool = True,
    memory_store_plugin_catalog: MemoryStorePluginCatalog | None = None,
    qualification_evidence_registry: MemoryProviderQualificationEvidenceRegistry | None = None,
    admission_evidence: MemoryProviderAdmissionEvidenceContext | None = None,
) -> SessionTurnIndexStore | None:
    """Construct episodic index when ``enable_session_vector_index`` is true."""
    profile = env.memory_profile
    if not profile.enable_session_vector_index:
        return None
    resolved_tenant_id = _require_runtime_tenant(tenant_id)
    resolved_integration = integration_profile

    if _product_session_turn_index_enabled(env):
        assert_memory_vector_backend_available(env, rag_stack)

    if rag_stack is None:
        if _product_session_turn_index_enabled(env):
            raise MemoryVectorBackendUnavailableError(reason="vector_backend_unavailable")
        return None

    if rag_stack.embedding_manager is None or rag_stack.vectorstore_manager is None:
        if _product_session_turn_index_enabled(env):
            raise MemoryVectorBackendUnavailableError(reason="vector_backend_unavailable")
        return None

    creation_context = build_session_turn_index_creation_context(
        tenant_id=resolved_tenant_id,
        embedding_manager=rag_stack.embedding_manager,
        vectorstore_manager=rag_stack.vectorstore_manager,
        index_roles=tuple(profile.session_index_roles),
        vector_index_namespace=profile.vector_index_namespace,
    )

    if session_turn_index_plugins:
        selected_plugin = _select_session_turn_index_plugin(session_turn_index_plugins)
    else:
        discovered = _discover_classified_session_turn_index_plugins(
            discover_entry_points=discover_entry_points,
            memory_store_plugin_catalog=memory_store_plugin_catalog,
        )
        selected_plugin = _select_session_turn_index_plugin(
            tuple(record.plugin_type for record in discovered),
        )

    if selected_plugin is not None:
        resolved_identity = resolve_plugin_session_turn_index_provider_identity(
            selected_plugin,
        )
        validate_session_turn_index_store_admission(
            env,
            provider_identity=resolved_identity,
            qualification_evidence_registry=qualification_evidence_registry,
            admission_evidence=admission_evidence,
        )
        plugin_type = cast(type[SessionTurnIndexStorePlugin], selected_plugin.plugin_type)
        return plugin_type.create_session_turn_index(creation_context)

    resolved_identity = resolve_builtin_session_turn_index_provider_identity(
        resolved_integration,
    )
    validate_session_turn_index_store_admission(
        env,
        provider_identity=resolved_identity,
        qualification_evidence_registry=qualification_evidence_registry,
        admission_evidence=admission_evidence,
    )

    embedding_port = creation_context.embedding_manager
    vectorstore_port = creation_context.vectorstore_manager
    if embedding_port is None or vectorstore_port is None:
        raise MemoryVectorBackendUnavailableError(reason="vector_backend_unavailable")

    return VectorSessionTurnIndexStore(
        embedding_port=embedding_port,
        vectorstore_port=vectorstore_port,
        index_roles=profile.session_index_roles,
        tenant_id=resolved_tenant_id,
        vector_index_namespace=profile.vector_index_namespace,
        workspace_id=creation_context.workspace_id,
    )
