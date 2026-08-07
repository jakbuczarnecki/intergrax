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
from intergrax.tools.providers.rag.scope import (
    resolve_tenant_scoped_vectorstore,
    vectorstore_tenant_id,
)
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile, ContextProfile
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.bootstrap.rag_stack_bootstrap import create_default_rag_stack
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def _request() -> RuntimeRequest:
    return RuntimeRequest(
        tenant_id="tenant-rag",
        agent_id="echo",
        user_id="user-rag",
        session_id="session-rag",
        message="rag bridge probe",
    )


def _request_for(tenant_id: str) -> RuntimeRequest:
    request = _request()
    request.tenant_id = tenant_id
    return request


def test_resolve_rag_stack_returns_none_when_rag_disabled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={"context_profile": ContextProfile(enable_rag=False)},
    )
    assert resolve_rag_stack_for_environment(env, tenant_id=None) is None


def test_resolve_rag_stack_builds_managers_when_enabled() -> None:
    register_default_integrations()
    env = ApplicationEnvironmentProfile.lab_defaults()
    stack = resolve_rag_stack_for_environment(
        env,
        tenant_id="tenant-rag",
        llm_adapter=FakeLLMAdapter(),
    )

    assert stack is not None
    assert stack.vectorstore_manager is not None
    assert stack.embedding_manager is not None
    assert stack.retrieval_service is not None


def test_apply_rag_stack_to_runtime_config() -> None:
    register_default_integrations()
    stack = create_default_rag_stack(tenant_id="tenant-rag")
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


def test_rag_enabled_host_wires_tenant_neutral_prerequisites() -> None:
    register_default_integrations()
    settings = LabApplicationSettings.from_env()
    env = build_lab_environment_profile(settings)
    wiring = wire_application_environment(
        build_lab_manifest(settings),
        env,
        conformance_check=False,
    )
    context = wiring.tool_wiring.wiring_context

    assert env.context_profile.enable_rag is True
    assert context.embedding_manager is not None
    assert context.rag_profile is not None
    assert context.vectorstore_manager is None
    assert context.integration_profile is not None

    tenant_manager = resolve_tenant_scoped_vectorstore(context, "tenant-lazy")

    assert tenant_manager is not None
    assert vectorstore_tenant_id(tenant_manager) == "tenant-lazy"
    assert context.extras["tenant_vectorstore_managers"]["tenant-lazy"] is tenant_manager


def test_runtime_requests_bind_rag_managers_to_separate_tenants() -> None:
    register_default_integrations()
    settings = LabApplicationSettings.from_env()
    env = build_lab_environment_profile(settings)
    wiring = wire_application_environment(build_lab_manifest(settings), env)

    config_a = materialize_runtime_config(
        _request_for("tenant-a"),
        wiring.build_context,
        env,
        llm_adapter=FakeLLMAdapter(),
    )
    config_b = materialize_runtime_config(
        _request_for("tenant-b"),
        wiring.build_context,
        env,
        llm_adapter=FakeLLMAdapter(),
    )

    assert config_a.vectorstore_manager is not config_b.vectorstore_manager
    assert config_a.vectorstore_manager._bound_scope.tenant_id == "tenant-a"  # type: ignore[attr-defined]
    assert config_b.vectorstore_manager._bound_scope.tenant_id == "tenant-b"  # type: ignore[attr-defined]
    assert config_a.vectorstore_manager._bound_scope.tenant_id != env.profile_id  # type: ignore[attr-defined]
    assert config_b.vectorstore_manager._bound_scope.tenant_id != env.profile_id  # type: ignore[attr-defined]


def test_apply_rag_for_environment_skips_when_disabled() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False, enable_rag=False)
    env = ApplicationEnvironmentProfile.lab_defaults()

    apply_rag_for_environment(config, env)

    assert config.vectorstore_manager is None


def test_rag_stack_uses_request_tenant_not_shared_host_profile() -> None:
    register_default_integrations()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="host-profile-x")

    tenant_a = resolve_rag_stack_for_environment(env, tenant_id="tenant-a")
    tenant_b = resolve_rag_stack_for_environment(env, tenant_id="tenant-b")

    assert tenant_a is not None
    assert tenant_b is not None
    assert tenant_a.vectorstore_manager is not tenant_b.vectorstore_manager
    assert tenant_a.vectorstore_manager._bound_scope.tenant_id == "tenant-a"  # type: ignore[attr-defined]
    assert tenant_b.vectorstore_manager._bound_scope.tenant_id == "tenant-b"  # type: ignore[attr-defined]
    assert tenant_a.vectorstore_manager._bound_scope.tenant_id != env.profile_id  # type: ignore[attr-defined]
    assert tenant_b.vectorstore_manager._bound_scope.tenant_id != env.profile_id  # type: ignore[attr-defined]


def test_rag_stack_missing_tenant_fails_closed() -> None:
    register_default_integrations()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="host-profile-x")
    env.integration_profile = IntegrationProfile()

    with pytest.raises(ValueError, match="explicit tenant_id"):
        resolve_rag_stack_for_environment(env, tenant_id=None)
