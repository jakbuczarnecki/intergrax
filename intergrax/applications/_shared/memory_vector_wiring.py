# © Artur Czarnecki. All rights reserved.

"""Vector memory wiring helpers (Phase MEM-VEC-1.1–1.4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.memory.memory_vector_errors import MemoryVectorBackendUnavailableError
from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_store import UserProfileStore

if TYPE_CHECKING:
    from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStore
    from intergrax.rag.bootstrap.rag_stack_bootstrap import RagStack


if TYPE_CHECKING:
    from intergrax.rag.bootstrap.rag_stack_bootstrap import RagStack


def memory_vector_flags_require_backend(env: ApplicationEnvironmentProfile) -> bool:
    profile = env.memory_profile
    return bool(profile.enable_long_term_memory or profile.enable_session_vector_index)


def resolve_rag_stack_for_memory_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    integration_profile: object | None = None,
    llm_adapter: object | None = None,
) -> RagStack | None:
    """Resolve RAG stack for memory vector indexes — independent of ``enable_rag``."""
    from intergrax.applications._shared.rag_runtime_bridge import resolve_rag_stack_for_environment
    from intergrax.rag.bootstrap.rag_stack_bootstrap import create_default_rag_stack

    if not memory_vector_flags_require_backend(env):
        return resolve_rag_stack_for_environment(
            env,
            integration_profile=integration_profile,  # type: ignore[arg-type]
            llm_adapter=llm_adapter,  # type: ignore[arg-type]
        )
    profile = integration_profile or env.integration_profile
    return create_default_rag_stack(
        integration_profile=profile,  # type: ignore[arg-type]
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
    rag_stack: RagStack | None = None,
) -> UserProfileManager | None:
    """Construct ``UserProfileManager`` with optional LTM vector dependencies."""
    profile = env.memory_profile
    if not (profile.enable_user_memory or profile.enable_long_term_memory):
        return None

    kwargs: dict[str, object] = {}
    if profile.enable_long_term_memory and rag_stack is not None:
        kwargs["embedding_manager"] = rag_stack.embedding_manager
        kwargs["vectorstore_manager"] = rag_stack.vectorstore_manager
        kwargs["rag_profile"] = rag_stack.profile

    return UserProfileManager(store, **kwargs)


def build_session_turn_index_store(
    env: ApplicationEnvironmentProfile,
    *,
    rag_stack: RagStack | None = None,
) -> SessionTurnIndexStore | None:
    """Construct episodic index when ``enable_session_vector_index`` is true."""
    profile = env.memory_profile
    if not profile.enable_session_vector_index:
        return None
    if rag_stack is None:
        return None
    return VectorSessionTurnIndexStore(
        embedding_manager=rag_stack.embedding_manager,
        vectorstore_manager=rag_stack.vectorstore_manager,
        index_roles=profile.session_index_roles,
    )
