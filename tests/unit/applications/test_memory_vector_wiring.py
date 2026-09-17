# © Artur Czarnecki. All rights reserved.

"""Unit tests for MEM-VEC vector memory wiring."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.memory_vector_wiring import (
    assert_memory_vector_backend_available,
    build_user_profile_manager,
    memory_vector_flags_require_backend,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.memory.memory_vector_errors import MemoryVectorBackendUnavailableError
from intergrax.rag.bootstrap.rag_stack_bootstrap import RagStack
from intergrax.rag.profiles.rag_profile import RagProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_memory_vector_flags_require_backend_when_ltm_or_episodic_enabled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.vec.flags")
    env.memory_profile = MemoryProfile(enable_long_term_memory=True)
    assert memory_vector_flags_require_backend(env) is True

    env.memory_profile = MemoryProfile(enable_session_vector_index=True)
    assert memory_vector_flags_require_backend(env) is True

    env.memory_profile = MemoryProfile()
    assert memory_vector_flags_require_backend(env) is False


def test_assert_memory_vector_backend_unavailable_raises() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.vec.fail")
    env.memory_profile = MemoryProfile(enable_long_term_memory=True)
    with pytest.raises(MemoryVectorBackendUnavailableError):
        assert_memory_vector_backend_available(env, None)


def test_assert_memory_vector_backend_available_with_stack() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.vec.ok")
    env.memory_profile = MemoryProfile(enable_long_term_memory=True)
    stack = RagStack(
        profile=RagProfile(),
        vectorstore_manager=MagicMock(),
        embedding_manager=MagicMock(),
        retriever_manager=MagicMock(),
        reranker_manager=MagicMock(),
        retrieval_service=MagicMock(),
    )
    assert_memory_vector_backend_available(env, stack)


def test_build_user_profile_manager_requires_and_preserves_runtime_tenant() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.profile.tenant")
    env.memory_profile = MemoryProfile(enable_user_memory=True)

    with pytest.raises(MemoryVectorBackendUnavailableError) as exc_info:
        build_user_profile_manager(MagicMock(), env)

    assert exc_info.value.reason == "tenant_required"

    manager = build_user_profile_manager(
        MagicMock(),
        env,
        tenant_id="tenant-explicit",
    )

    assert manager is not None
    assert manager._tenant_id == "tenant-explicit"


def test_build_user_profile_manager_injects_ltm_vector_projection_from_composition() -> None:
    from intergrax.memory.user_profile_ltm_vector_projection import UserProfileLtmVectorProjection

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="mem.profile.ltm.proj")
    env.memory_profile = MemoryProfile(
        enable_user_memory=True,
        enable_long_term_memory=True,
    )
    stack = RagStack(
        profile=RagProfile(),
        vectorstore_manager=MagicMock(),
        embedding_manager=MagicMock(),
        retriever_manager=MagicMock(),
        reranker_manager=MagicMock(),
        retrieval_service=MagicMock(),
    )
    manager = build_user_profile_manager(
        MagicMock(),
        env,
        tenant_id="tenant-ltm",
        rag_stack=stack,
    )
    assert manager is not None
    projections = manager._memory_lifecycle._projections
    assert len(projections) == 1
    assert isinstance(projections[0], UserProfileLtmVectorProjection)


@pytest.mark.asyncio
async def test_user_profile_manager_uses_injected_custom_projection_only() -> None:
    from intergrax.memory.contracts.memory_lifecycle import (
        MemoryProjectionReconciliationDisposition,
        MemoryProjectionReconciliationResult,
        UserProfileMemoryReconciliationContext,
    )
    from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
    from intergrax.memory.user_profile_manager import UserProfileManager

    class _CustomProjection:
        projection_id = "custom.test"

        async def reconcile(
            self,
            context: UserProfileMemoryReconciliationContext,
        ) -> MemoryProjectionReconciliationResult:
            return MemoryProjectionReconciliationResult(
                projection_id=self.projection_id,
                disposition=MemoryProjectionReconciliationDisposition.CONSISTENT,
            )

    custom = _CustomProjection()
    mgr = UserProfileManager(
        InMemoryUserProfileStore(),
        tenant_id="tenant-custom",
        memory_projections=(custom,),
    )
    assert mgr._memory_lifecycle._projections == (custom,)
