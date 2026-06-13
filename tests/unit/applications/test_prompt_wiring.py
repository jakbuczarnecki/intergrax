# © Artur Czarnecki. All rights reserved.

"""PE-2: Prompt registry wiring from Tier-3 environment."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications._shared.prompt_wiring import (
    DEFAULT_PROMPT_CATALOG,
    resolve_prompt_catalog_path,
    resolve_prompt_registry,
    resolve_prompt_registry_protocol,
)
from intergrax.applications._shared.runtime_config_bridge import build_runtime_context_from_environment
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PromptProfile,
)
from intergrax.prompts.registry.prompt_registry_protocol import PromptRegistryProtocol
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _request() -> RuntimeRequest:
    return RuntimeRequest(
        tenant_id="tenant-pe-wire",
        agent_id="echo",
        user_id="user-pe-wire",
        session_id="session-pe-wire",
        message="prompt wiring probe",
    )


def test_resolve_prompt_catalog_path_defaults_to_prompts() -> None:
    assert resolve_prompt_catalog_path(PromptProfile()) == DEFAULT_PROMPT_CATALOG


def test_resolve_prompt_catalog_path_uses_profile_override() -> None:
    custom = Path("custom/prompts")
    assert resolve_prompt_catalog_path(PromptProfile(catalog_path=custom)) == custom


def test_resolve_prompt_registry_returns_yaml_registry() -> None:
    registry = resolve_prompt_registry(PromptProfile(catalog_path=Path("prompts")))
    assert isinstance(registry, YamlPromptRegistry)


def test_resolve_prompt_registry_protocol_satisfies_protocol() -> None:
    registry = resolve_prompt_registry_protocol(PromptProfile(catalog_path=Path("prompts")))
    assert isinstance(registry, YamlPromptRegistry)


@pytest.mark.no_ci
def test_wire_application_environment_includes_prompt_registry() -> None:
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="pe.wire")
    env.prompt_profile = PromptProfile(catalog_path=Path("prompts"))
    wiring = wire_application_environment(build_lab_manifest(settings), env)

    assert wiring.prompt_registry is not None
    assert wiring.build_context.prompt_registry is wiring.prompt_registry


@pytest.mark.no_ci
def test_build_runtime_context_from_environment_injects_prompt_registry() -> None:
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="pe.ctx")
    env.prompt_profile = PromptProfile(catalog_path=Path("prompts"))
    manifest = build_lab_manifest(settings)
    wiring = wire_application_environment(manifest, env)

    ctx = build_runtime_context_from_environment(
        _request(),
        wiring.build_context,
        env,
        llm_adapter=FakeLLMAdapter(),
    )

    assert ctx.prompt_registry is not None
    assert isinstance(ctx.prompt_registry, YamlPromptRegistry)
