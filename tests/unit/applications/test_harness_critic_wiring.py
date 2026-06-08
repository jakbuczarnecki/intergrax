# © Artur Czarnecki. All rights reserved.

"""CRIT-V-1: Critic profile runtime bridge tests."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.critic_runtime_bridge import (
    apply_critic_profile_to_runtime_config,
    apply_critic_profiles_from_environment,
    resolve_critic_wiring_options,
)
from intergrax.applications._shared.critic_wiring import wire_application_critic
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    CriticProfile,
    CriticVerificationScopes,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_resolve_critic_wiring_options_maps_profile_fields() -> None:
    profile = CriticProfile(
        semantic_judge_enabled=True,
        trajectory_eval_enabled=True,
        judge_threshold=0.85,
        require_critic_on_completion=True,
        evaluator_loop_max_iterations=4,
        critic_llm_profile_ref="llm.critic",
        default_rubric_ref="prompt.rubric.default",
        scopes=CriticVerificationScopes(
            node_partial=True,
            graph_final=True,
            uaep_step=False,
        ),
    )
    options = resolve_critic_wiring_options(profile)
    assert options.semantic_judge_enabled is True
    assert options.judge_threshold == 0.85
    assert options.verify_node_partial is True
    assert options.default_rubric_ref == "prompt.rubric.default"


def test_apply_critic_profile_to_runtime_config_sets_field() -> None:
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter())
    profile = CriticProfile(semantic_judge_enabled=True)
    updated = apply_critic_profile_to_runtime_config(config, profile)
    assert updated.critic_profile == profile


def test_apply_critic_profiles_from_environment_uses_env_profile() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="critic.env")
    env.critic_profile = CriticProfile(trajectory_eval_enabled=True)
    config = RuntimeConfig(llm_adapter=FakeLLMAdapter())
    updated = apply_critic_profiles_from_environment(config, env)
    assert updated.critic_profile is not None
    assert updated.critic_profile.trajectory_eval_enabled is True


def test_lab_defaults_keep_semantic_critic_disabled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="critic.lab")
    assert env.critic_profile.semantic_judge_enabled is False
    assert env.critic_profile.require_critic_on_completion is False


def test_wire_application_critic_builds_graph_hooks_for_graph_final() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="critic.wire")
    wiring = wire_application_critic(env)
    assert wiring.graph_hooks is not None
    assert wiring.graph_hooks.verify_graph_final is True
    assert "critic_governance" in wiring.domain_fragments


def test_materialize_runtime_config_applies_critic_profile() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="critic.materialize")
    env.critic_profile = CriticProfile(
        semantic_judge_enabled=True,
        scopes=CriticVerificationScopes(graph_final=True),
    )
    request = RuntimeRequest(
        message="verify me",
        tenant_id="t1",
        agent_id="echo",
        user_id="user-1",
        session_id="session-1",
    )
    manifest = build_lab_manifest(LabApplicationSettings.from_env())
    ctx = ApplicationBuildContext.for_manifest(manifest, environment=env)
    config = materialize_runtime_config(request, ctx, env)
    assert config.critic_profile is not None
    assert config.critic_profile.semantic_judge_enabled is True
