# © Artur Czarnecki. All rights reserved.

"""ApplicationEnvironmentProfile and manifest wiring (Phase H-APP.1.8)."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import ApplicationManifest
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest_default
from intergrax.fastapi_core.config import ApiEnvironment
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry import presets

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_application_environment_profile_lab_defaults() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults()
    assert env.tool_profile.enabled
    assert env.modality_profile is not None


def test_manifest_environment_defaults() -> None:
    from intergrax.applications.contracts.manifest import ApplicationProfile

    lab_env = ApplicationManifest.environment_defaults(ApplicationProfile.LAB)
    product_env = ApplicationManifest.environment_defaults(ApplicationProfile.PRODUCT)
    assert lab_env.profile_id != product_env.profile_id


def test_lab_manifest_with_environment_wiring() -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    manifest = build_lab_manifest_default()
    env = build_lab_environment_profile(settings)
    wiring = wire_application_environment(manifest, env, settings=settings, conformance_check=False)
    assert wiring.tool_wiring.registry is not None
    assert wiring.build_context.environment is env


def test_lab_environment_profile_adaptive_observe_enabled_by_default() -> None:
    settings = LabApplicationSettings()
    env = build_lab_environment_profile(settings)
    assert env.adaptive_profile.enabled is True
    assert env.adaptive_profile.mode == "observe"
    assert env.adaptive_profile.debug_readonly_routes is True


def test_lab_environment_profile_adaptive_observe_disabled_via_settings() -> None:
    settings = LabApplicationSettings(adaptive_observe_enabled=False)
    env = build_lab_environment_profile(settings)
    assert env.adaptive_profile.enabled is False
    assert env.adaptive_profile.mode == "observe"


def test_resolve_llm_adapter_precedence() -> None:
    from intergrax.llm_adapters.registry.profile import LLMProfile

    env = ApplicationEnvironmentProfile.lab_defaults()
    adapter = resolve_llm_adapter(env)
    assert adapter is not None
    env_with_llm = env.model_copy(update={"llm_profile": LLMProfile.lab()})
    assert resolve_llm_adapter(env_with_llm) is not None


def test_harness_production_defaults_wires_catalog_stack() -> None:
    env = ApplicationEnvironmentProfile.harness_production_defaults()
    integration = env.integration_profile
    assert integration.slug_for_category(IntegrationCategory.RELATIONAL_STORE) == "postgresql"
    assert integration.slug_for_category(IntegrationCategory.VECTOR_STORE) == "pgvector"
    assert integration.slug_for_category(IntegrationCategory.SECRETS_STORE) == "doppler"
    assert integration.slug_for_category(IntegrationCategory.FEATURE_FLAG) == "unleash"
    assert integration.slug_for_category(IntegrationCategory.CI_CD) == "github_actions"
    assert env.identity_profile.require_api_key is True
    assert env.adaptive_profile.feature_flag_slug == "unleash"


def test_harness_production_stack_preset() -> None:
    profile = presets.harness_production_stack(secrets_slug="vault", enable_grafana_stack=True)
    assert profile.slug_for_category(IntegrationCategory.SECRETS_STORE) == "vault"
    assert profile.slug_for_category(IntegrationCategory.OBSERVABILITY_BACKEND) == "grafana"


def test_lab_environment_profile_prod_with_secrets_backend() -> None:
    settings = LabApplicationSettings(
        environment=ApiEnvironment.PROD,
        secrets_backend_slug="doppler",
        observability_grafana_stack=True,
        adaptive_feature_flag_slug="unleash",
    )
    env = build_lab_environment_profile(settings)
    assert env.integration_profile.slug_for_category(IntegrationCategory.SECRETS_STORE) == "doppler"
    assert env.integration_profile.slug_for_category(IntegrationCategory.FEATURE_FLAG) == "unleash"
    assert env.adaptive_profile.feature_flag_slug == "unleash"
    assert env.identity_profile.require_api_key is True


def test_lab_environment_profile_grafana_stack_in_dev() -> None:
    settings = LabApplicationSettings(observability_grafana_stack=True, otel_enabled=True)
    env = build_lab_environment_profile(settings)
    assert env.integration_profile.slug_for_category(IntegrationCategory.OBSERVABILITY_BACKEND) == "grafana"
