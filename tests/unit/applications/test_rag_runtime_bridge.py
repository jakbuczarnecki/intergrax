# © Artur Czarnecki. All rights reserved.

"""RAG-1: RAG stack → RuntimeConfig bridge."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.rag_runtime_bridge import (
    apply_rag_for_environment,
    apply_rag_stack_to_runtime_config,
    resolve_rag_stack_for_environment,
)
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile, ContextProfile
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.rag.bootstrap.rag_stack_bootstrap import create_default_rag_stack
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _request() -> RuntimeRequest:
    return RuntimeRequest(
        tenant_id="tenant-rag",
        agent_id="echo",
        user_id="user-rag",
        session_id="session-rag",
        message="rag bridge probe",
    )


def test_resolve_rag_stack_returns_none_when_rag_disabled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={"context_profile": ContextProfile(enable_rag=False)},
    )
    assert resolve_rag_stack_for_environment(env) is None


def test_resolve_rag_stack_builds_managers_when_enabled() -> None:
    register_default_integrations()
    env = ApplicationEnvironmentProfile.lab_defaults()
    stack = resolve_rag_stack_for_environment(env, llm_adapter=FakeLLMAdapter())

    assert stack is not None
    assert stack.vectorstore_manager is not None
    assert stack.embedding_manager is not None
    assert stack.retrieval_service is not None


def test_apply_rag_stack_to_runtime_config() -> None:
    register_default_integrations()
    stack = create_default_rag_stack()
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False, enable_rag=True)

    apply_rag_stack_to_runtime_config(config, stack)

    assert config.vectorstore_manager is stack.vectorstore_manager
    assert config.retrieval_service is stack.retrieval_service
    assert config.rag_profile is not None


def test_materialize_runtime_config_wires_rag_managers_from_environment_wiring() -> None:
    register_default_integrations()
    settings = LabApplicationSettings.from_env()
    env = build_lab_environment_profile(settings)
    wiring = wire_application_environment(build_lab_manifest(settings), env)
    config = materialize_runtime_config(
        _request(),
        wiring.build_context,
        env,
        llm_adapter=FakeLLMAdapter(),
    )

    assert config.enable_rag is True
    assert config.vectorstore_manager is not None
    assert config.retrieval_service is not None


def test_apply_rag_for_environment_skips_when_disabled() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False, enable_rag=False)
    env = ApplicationEnvironmentProfile.lab_defaults()

    apply_rag_for_environment(config, env)

    assert config.vectorstore_manager is None
