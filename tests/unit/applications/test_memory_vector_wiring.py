# © Artur Czarnecki. All rights reserved.

"""Unit tests for MEM-VEC vector memory wiring."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.memory_vector_wiring import (
    assert_memory_vector_backend_available,
    memory_vector_flags_require_backend,
)
from intergrax.memory.memory_vector_errors import MemoryVectorBackendUnavailableError
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.rag.bootstrap.rag_stack_bootstrap import create_default_rag_stack

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
    stack = create_default_rag_stack()
    assert_memory_vector_backend_available(env, stack)
