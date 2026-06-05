# © Artur Czarnecki. All rights reserved.

"""INT-1: IntegrationProfile → RuntimeConfig bridge."""

from __future__ import annotations

import pytest

from intergrax.agents.reference_harness import default_reference_harness
from intergrax.applications._shared.integration_runtime_bridge import (
    apply_integration_profile_to_runtime_config,
    apply_integration_profiles_from_build_context,
    apply_integration_profiles_from_environment,
)
from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.integrations.registry.catalog_manifests import SQLITE
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _request() -> RuntimeRequest:
    return RuntimeRequest(
        tenant_id="tenant-int",
        agent_id="echo",
        user_id="user-int",
        session_id="session-int",
        message="integration bridge probe",
    )


def test_apply_integration_profile_to_runtime_config() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    profile = IntegrationProfile(relational_store=SQLITE)

    apply_integration_profile_to_runtime_config(config, profile)

    assert config.integration_profile is profile
    assert config.integration_profile.slug_for_category("relational_store") == SQLITE.slug


def test_apply_integration_profiles_from_build_context_overrides_environment() -> None:
    env_profile = IntegrationProfile.lab()
    wired_profile = IntegrationProfile(relational_store=SQLITE)
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    apply_integration_profiles_from_environment(
        config,
        ApplicationEnvironmentProfile.lab_defaults().model_copy(
            update={"integration_profile": env_profile}
        ),
    )

    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    build_ctx = ApplicationBuildContext.for_manifest(
        build_lab_manifest(settings),
        settings=settings,
        integration_profile=wired_profile,
    )
    apply_integration_profiles_from_build_context(config, build_ctx)

    assert config.integration_profile is wired_profile


def test_materialize_runtime_config_includes_integration_profile() -> None:
    settings = LabApplicationSettings.from_env()
    env = build_lab_environment_profile(settings)
    wiring = wire_application_environment(build_lab_manifest(settings), env)
    config = materialize_runtime_config(
        _request(),
        wiring.build_context,
        env,
        llm_adapter=FakeLLMAdapter(),
    )

    assert config.integration_profile is wiring.build_context.integration_profile


def test_materialize_runtime_config_lab_harness_uses_environment_integration() -> None:
    env = build_lab_environment_profile(LabApplicationSettings.from_env())
    harness = default_reference_harness()
    config = materialize_runtime_config(
        _request(),
        harness,
        env,
        llm_adapter=FakeLLMAdapter(),
    )

    assert config.integration_profile is env.integration_profile
