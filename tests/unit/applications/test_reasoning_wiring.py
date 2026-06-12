# © Artur Czarnecki. All rights reserved.

from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.reasoning_wiring import (
    resolve_engine_planner_prompt_config,
    resolve_planner_llm_adapter,
    resolve_planner_model_id,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.reasoning_profile import ReasoningProfile
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_resolve_planner_llm_adapter_uses_separate_profile() -> None:
    producer = FakeLLMAdapter(fixed_text="producer")
    planner_adapter = FakeLLMAdapter(fixed_text="planner")
    planner_profile = MagicMock()
    planner_profile.create_adapter.return_value = planner_adapter
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "reasoning_profile": ReasoningProfile.model_construct(
                planner_llm_profile=planner_profile,
                planner_llm_profile_id="planner-model",
            )
        }
    )
    resolved = resolve_planner_llm_adapter(env, producer_adapter=producer)
    assert resolved is planner_adapter
    planner_profile.create_adapter.assert_called_once()


def test_resolve_planner_llm_adapter_falls_back_to_producer() -> None:
    producer = FakeLLMAdapter(fixed_text="producer")
    env = ApplicationEnvironmentProfile.lab_defaults()
    assert resolve_planner_llm_adapter(env, producer_adapter=producer) is producer


def test_resolve_planner_model_id_prefers_explicit_id() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "reasoning_profile": ReasoningProfile(planner_llm_profile_id="custom-id"),
        }
    )
    assert resolve_planner_model_id(env) == "custom-id"


def test_resolve_engine_planner_prompt_config() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "reasoning_profile": ReasoningProfile(engine_planner_prompt_id="planner_replan_default"),
        }
    )
    config = resolve_engine_planner_prompt_config(env)
    assert config.prompt_id == "planner_replan_default"
