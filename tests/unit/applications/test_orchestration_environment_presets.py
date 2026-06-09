# © Artur Czarnecki. All rights reserved.

"""ORCH-CONFIG.6/7/8 environment presets."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.task_intake import apply_long_running_from_profile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_apply_long_running_from_profile_uses_orchestration_flag() -> None:
    base = ApplicationEnvironmentProfile.lab_defaults()
    env = base.model_copy(
        update={
            "orchestration_profile": base.orchestration_profile.model_copy(
                update={"long_running_enabled": True},
            ),
        },
    )
    task = Task(tenant_id="t", user_id="u", message="job", context=TaskContext())
    updated = apply_long_running_from_profile(task, env)
    assert updated.options.long_running.enabled is True


def test_strict_multi_agent_defaults_preset() -> None:
    env = ApplicationEnvironmentProfile.strict_multi_agent_defaults()
    assert env.execution_mode is ExecutionMode.STRICT
    assert env.orchestration_profile.merge_strategy == "structured_json"
    assert env.critic_profile.require_critic_on_completion is True
    assert env.critic_profile.semantic_judge_enabled is True


def test_reference_host_platform_defaults_engine_and_rules_classifier() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(
        profile_id="ref.test",
    ).with_reference_host_platform_defaults()
    assert env.orchestration_profile.planner_kind == "engine"
    assert env.orchestration_profile.classifier_kind == "rules"
    assert env.reliability_profile.long_running_scheduler_enabled is True


def test_reference_host_platform_defaults_multi_agent_critic() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(
        profile_id="ref.multi",
    ).with_reference_host_platform_defaults(multi_agent_critic=True)
    assert env.critic_profile.require_critic_on_completion is True
    assert env.critic_profile.semantic_judge_enabled is False
    assert env.orchestration_profile.merge_strategy == "structured_json"
    assert env.orchestration_profile.long_running_enabled is True


def test_swarm_exploration_defaults_parallel_cap() -> None:
    env = ApplicationEnvironmentProfile.swarm_exploration_defaults(max_parallel_nodes=24)
    assert env.orchestration_profile.max_parallel_nodes == 24
    assert env.orchestration_profile.max_inflight_nodes == 24
