# © Artur Czarnecki. All rights reserved.

"""Test-only ContextEngine attachment for iterative bounded tool loop tests."""

from __future__ import annotations

from intergrax.applications._shared.context_wiring import apply_context_engine_to_runtime_config
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.context.protocols import ContextEngine
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


def attach_test_context_engine_for_iterative_tool_loop(state: RuntimeState) -> ContextEngine:
    """Wire ContextEngine via composition resolver when tests use minimal RuntimeState."""
    config = state.context.config
    if config.llm_adapter is None:
        from testing_support.builder import FakeLLMAdapter

        config.llm_adapter = FakeLLMAdapter()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="test.iterative_tool_loop")
    apply_context_engine_to_runtime_config(config, env)
    if config.context_engine is None:
        raise RuntimeError("test_context_engine_attachment_failed")
    return config.context_engine
