# © Artur Czarnecki. All rights reserved.

"""Vector memory wiring helpers (Phase MEM-VEC-1.1–1.4)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.memory.memory_vector_errors import MemoryVectorBackendUnavailableError
from intergrax.applications._shared.session_turn_index_rag_adapters import (
    build_session_turn_index_creation_context,
)
from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStore
from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_ltm_vector_projection import UserProfileLtmVectorProjection
from intergrax.memory.user_profile_store import UserProfileStore

if TYPE_CHECKING:
    from intergrax.memory.contracts.entity_temporal_memory import (
        EntityMemoryIndexer,
        EntityTemporalMemoryCapability,
    )
    from intergrax.memory.contracts.memory_lifecycle import UserProfileMemoryProjection
    from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStore
    from intergrax.rag.bootstrap.rag_stack_bootstrap import RagStack


def memory_vector_flags_require_backend(env: ApplicationEnvironmentProfile) -> bool:
    profile = env.memory_profile
    return bool(profile.enable_long_term_memory or profile.enable_session_vector_index)


def _require_runtime_tenant(tenant_id: str | None) -> str:
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise MemoryVectorBackendUnavailableError(reason="tenant_required")
    return tenant_id.strip()


def resolve_rag_stack_for_memory_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    tenant_id: str | None = None,
    integration_profile: object | None = None,
    llm_adapter: object | None = None,
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
            integration_profile=integration_profile,  # type: ignore[arg-type]
            llm_adapter=llm_adapter,  # type: ignore[arg-type]
        )
    profile = integration_profile or env.integration_profile
    resolved_tenant_id = _require_runtime_tenant(tenant_id)
    return create_default_rag_stack(
        integration_profile=profile,  # type: ignore[arg-type]
        tenant_id=resolved_tenant_id,
        llm_for_contextual=llm_adapter,  # type: ignore[arg-type]
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
    kwargs: dict[str, object] = {
        "tenant_id": resolved_tenant_id,
        "vector_index_namespace": profile.vector_index_namespace,
    }
    if profile.enable_long_term_memory and rag_stack is not None:
        kwargs["embedding_manager"] = rag_stack.embedding_manager
        kwargs["vectorstore_manager"] = rag_stack.vectorstore_manager
        kwargs["retrieval_service"] = rag_stack.retrieval_service
        kwargs["rag_profile"] = rag_stack.profile

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
    if projections:
        kwargs["memory_projections"] = tuple(projections)

    return UserProfileManager(store, **kwargs)


def build_session_turn_index_store(
    env: ApplicationEnvironmentProfile,
    *,
    tenant_id: str | None = None,
    rag_stack: RagStack | None = None,
    session_turn_index_plugins: Sequence[type] = (),
) -> SessionTurnIndexStore | None:
    """Construct episodic index when ``enable_session_vector_index`` is true."""
    from intergrax.core.memory_bootstrap import discover_session_turn_index_plugin_types

    profile = env.memory_profile
    if not profile.enable_session_vector_index:
        return None
    resolved_tenant_id = _require_runtime_tenant(tenant_id)
    if rag_stack is None:
        return None

    if rag_stack.embedding_manager is None or rag_stack.vectorstore_manager is None:
        return None

    creation_context = build_session_turn_index_creation_context(
        tenant_id=resolved_tenant_id,
        embedding_manager=rag_stack.embedding_manager,
        vectorstore_manager=rag_stack.vectorstore_manager,
        index_roles=tuple(profile.session_index_roles),
        vector_index_namespace=profile.vector_index_namespace,
    )

    plugin_types = list(session_turn_index_plugins) or discover_session_turn_index_plugin_types()
    for plugin_type in plugin_types:
        return plugin_type.create_session_turn_index(creation_context)

    return VectorSessionTurnIndexStore(
        embedding_port=creation_context.embedding_manager,
        vectorstore_port=creation_context.vectorstore_manager,
        index_roles=profile.session_index_roles,
        tenant_id=resolved_tenant_id,
        vector_index_namespace=profile.vector_index_namespace,
        workspace_id=creation_context.workspace_id,
    )
