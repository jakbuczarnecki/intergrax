# © Artur Czarnecki. All rights reserved.

"""PE-1: PromptProfile → RuntimeConfig bridge."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.prompt_runtime_bridge import (
    apply_prompt_profile_to_runtime_config,
    apply_prompt_profiles_from_environment,
)
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PromptProfile,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def _request() -> RuntimeRequest:
    return RuntimeRequest(
        tenant_id="tenant-pe",
        agent_id="echo",
        user_id="user-pe",
        session_id="session-pe",
        message="prompt bridge probe",
    )


def test_apply_prompt_profile_sets_catalog_path_on_runtime_config() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
    catalog = Path("prompts")

    apply_prompt_profile_to_runtime_config(
        config,
        PromptProfile(catalog_path=catalog),
    )

    assert config.prompt_catalog_path == str(catalog)


def test_apply_prompt_profiles_from_environment() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="pe.bridge")
    env.prompt_profile = PromptProfile(catalog_path=Path("prompts"))
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)

    apply_prompt_profiles_from_environment(config, env)

    assert config.prompt_catalog_path == "prompts"


def test_materialize_runtime_config_wires_prompt_catalog_from_environment() -> None:
    from intergrax.applications._shared.environment_wiring import wire_application_environment
    from lab_application.host.settings import LabApplicationSettings
    from lab_application.manifest import build_lab_manifest

    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="pe.materialize")
    env.prompt_profile = PromptProfile(catalog_path=Path("prompts"))
    wiring = wire_application_environment(build_lab_manifest(settings), env)
    config = materialize_runtime_config(
        _request(),
        wiring.build_context,
        env,
        llm_adapter=FakeLLMAdapter(),
    )

    assert config.prompt_catalog_path == "prompts"
