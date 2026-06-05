# © Artur Czarnecki. All rights reserved.

"""TS-2: Harness host runtime passes resolved LLM adapter to Nexus factory."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.applications._shared.orchestration_wiring import EngineBackedNexusPlanner
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    OrchestrationProfile,
)
from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_build_harness_host_runtime_passes_llm_adapter_for_engine_planner() -> None:
    env = build_lab_environment_profile(LabApplicationSettings.from_env()).model_copy(
        update={
            "orchestration_profile": OrchestrationProfile(planner_kind="engine"),
        }
    )
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    manifest = build_lab_manifest(settings)
    fake_llm = FakeLLMAdapter()

    with patch(
        "intergrax.applications._shared.harness_host_runtime.resolve_llm_adapter",
        return_value=fake_llm,
    ) as resolve_mock:
        runtime = build_harness_host_runtime(
            manifest,
            env,
            settings=settings,
            use_in_memory_trace=True,
        )

    resolve_mock.assert_called_once_with(env)
    planner = runtime.nexus_loop._planner  # noqa: SLF001 — wiring verification
    assert isinstance(planner, EngineBackedNexusPlanner)


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

    assert runtime.nexus_loop is not None
    assert runtime.env_wiring.build_context.tool_profile is not None
