# © Artur Czarnecki. All rights reserved.

"""TOOL-ENG-23 — tool_invocation_mode host profile bridge."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.catalog_runtime_bridge import (
    apply_tool_engine_settings_from_environment,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.config_types import ToolInvocationMode
from testing_support.builder import FakeLLMAdapter

pytestmark = pytest.mark.unit


def _config() -> RuntimeConfig:
    return RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)


def test_bridge_maps_tool_invocation_mode() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="lab.test").model_copy(
        update={"tool_invocation_mode": "bounded_react"},
    )
    config = apply_tool_engine_settings_from_environment(_config(), env)
    assert config.tool_invocation_mode == ToolInvocationMode.BOUNDED_REACT


def test_bridge_falls_back_on_invalid_invocation_mode() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="lab.test").model_copy(
        update={"tool_invocation_mode": "not_a_real_mode"},
    )
    config = apply_tool_engine_settings_from_environment(_config(), env)
    assert config.tool_invocation_mode == ToolInvocationMode.SINGLE_PASS
