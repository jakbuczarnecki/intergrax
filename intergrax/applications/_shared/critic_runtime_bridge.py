# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile critic fields to runtime config (Phase CRIT-V-1.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    CriticProfile,
)
from intergrax.runtime.nexus.config import RuntimeConfig


@dataclass(frozen=True, slots=True)
class CriticWiringOptions:
    """Resolved critic wiring flags for Tier-3 hosts."""

    semantic_judge_enabled: bool
    trajectory_eval_enabled: bool
    judge_threshold: float
    require_critic_on_completion: bool
    evaluator_loop_max_iterations: int
    critic_llm_profile_ref: str | None
    default_rubric_ref: str | None
    verify_node_partial: bool
    verify_graph_final: bool
    verify_uaep_step: bool


def resolve_critic_wiring_options(profile: CriticProfile) -> CriticWiringOptions:
    """Translate ``CriticProfile`` into host wiring flags."""
    return CriticWiringOptions(
        semantic_judge_enabled=profile.semantic_judge_enabled,
        trajectory_eval_enabled=profile.trajectory_eval_enabled,
        judge_threshold=profile.judge_threshold,
        require_critic_on_completion=profile.require_critic_on_completion,
        evaluator_loop_max_iterations=profile.evaluator_loop_max_iterations,
        critic_llm_profile_ref=profile.critic_llm_profile_ref,
        default_rubric_ref=profile.default_rubric_ref,
        verify_node_partial=profile.scopes.node_partial,
        verify_graph_final=profile.scopes.graph_final,
        verify_uaep_step=profile.scopes.uaep_step,
    )


def apply_critic_profile_to_runtime_config(
    config: RuntimeConfig,
    profile: CriticProfile,
) -> RuntimeConfig:
    """Record critic posture on runtime config for downstream Nexus steps."""
    config.critic_profile = profile
    return config


def apply_critic_profiles_from_environment(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
) -> RuntimeConfig:
    """Apply environment-declared critic profile."""
    return apply_critic_profile_to_runtime_config(config, env.critic_profile)
