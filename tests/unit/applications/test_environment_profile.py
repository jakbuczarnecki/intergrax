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


def test_resolve_llm_adapter_precedence() -> None:
    from intergrax.llm_adapters.registry.profile import LLMProfile

    env = ApplicationEnvironmentProfile.lab_defaults()
    adapter = resolve_llm_adapter(env)
    assert adapter is not None
    env_with_llm = env.model_copy(update={"llm_profile": LLMProfile.lab()})
    assert resolve_llm_adapter(env_with_llm) is not None
