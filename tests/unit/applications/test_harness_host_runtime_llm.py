# © Artur Czarnecki. All rights reserved.

"""Harness host runtime LLM ownership contract (R5/R6)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.harness_host_runtime_compat import resolve_harness_host_nexus_loop_legacy
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.applications._shared.nexus_factory import build_nexus_loop_from_environment
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    OrchestrationProfile,
)
from intergrax.llm_adapters.registry.registration_contract import LLMProviderNotConfiguredError
from intergrax.runtime.registry.agent_registry import AgentRegistry
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_build_harness_host_runtime_does_not_invoke_llm_resolution_without_requirement() -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    env = build_lab_environment_profile(settings)
    manifest = build_lab_manifest(settings)

    with patch(
        "intergrax.applications._shared.nexus_factory.resolve_environment_llm_adapter",
    ) as resolve_mock:
        runtime = build_harness_host_runtime(
            manifest,
            env,
            settings=settings,
            use_in_memory_trace=True,
        )

    resolve_mock.assert_not_called()
    assert resolve_harness_host_nexus_loop_legacy(runtime) is not None
    assert runtime.execution is not None
    assert runtime.env_wiring.build_context.tool_profile is not None


def test_engine_planner_without_provider_raises_at_orchestration_composition() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(
        profile_id="test.engine-no-llm",
    ).model_copy(
        update={"orchestration_profile": OrchestrationProfile(planner_kind="engine")},
    )
    with pytest.raises(LLMProviderNotConfiguredError, match="not explicitly configured"):
        build_nexus_loop_from_environment(AgentRegistry(), env=env)


def test_build_harness_host_runtime_default_planner_without_engine_kind() -> None:
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    env = build_lab_environment_profile(settings)
    manifest = build_lab_manifest(settings)

    runtime = build_harness_host_runtime(
        manifest,
        env,
        settings=settings,
        use_in_memory_trace=True,
    )

    assert resolve_harness_host_nexus_loop_legacy(runtime) is not None
    assert runtime.execution is not None
    assert runtime.env_wiring.build_context.tool_profile is not None
