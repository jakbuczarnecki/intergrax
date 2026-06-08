# © Artur Czarnecki. All rights reserved.

"""Tier-3 critic wiring (Phase CRIT-V-6.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.applications._shared.critic_runtime_bridge import (
    CriticWiringOptions,
    resolve_critic_wiring_options,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    CriticProfile,
)
from intergrax.runtime.critic.critic_wiring import (
    CriticGraphHooks,
    CriticHookConfig,
    build_critic_graph_hooks,
)
from intergrax.runtime.critic.eval_tool_client import CriticEvalToolClient
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine


@dataclass(frozen=True, slots=True)
class ApplicationCriticWiring:
    """Resolved critic artifacts for a Tier-3 host."""

    profile: CriticProfile
    options: CriticWiringOptions
    hook_config: CriticHookConfig
    graph_hooks: CriticGraphHooks | None
    domain_fragments: dict[str, Any]


def wire_application_critic(
    env: ApplicationEnvironmentProfile,
    *,
    l1_client: CriticEvalToolClient | None = None,
    validation_engine: NexusValidationEngine | None = None,
) -> ApplicationCriticWiring:
    """Materialize critic graph hooks and policy fragments from environment profile."""
    profile = env.critic_profile
    options = resolve_critic_wiring_options(profile)
    hook_config = CriticHookConfig(
        verify_node_partial=options.verify_node_partial,
        verify_graph_final=options.verify_graph_final,
        semantic_judge_enabled=options.semantic_judge_enabled,
        trajectory_eval_enabled=options.trajectory_eval_enabled,
        judge_threshold=options.judge_threshold,
        default_rubric_ref=options.default_rubric_ref,
        require_critic_on_completion=options.require_critic_on_completion,
    )
    graph_hooks = build_critic_graph_hooks(
        config=hook_config,
        validation_engine=validation_engine,
        l1_client=l1_client,
    )
    return ApplicationCriticWiring(
        profile=profile,
        options=options,
        hook_config=hook_config,
        graph_hooks=graph_hooks,
        domain_fragments={
            "critic_governance": {
                "semantic_judge_enabled": profile.semantic_judge_enabled,
                "trajectory_eval_enabled": profile.trajectory_eval_enabled,
                "require_critic_on_completion": profile.require_critic_on_completion,
                "verify_node_partial": profile.scopes.node_partial,
                "verify_graph_final": profile.scopes.graph_final,
                "verify_uaep_step": profile.scopes.uaep_step,
                "judge_threshold": profile.judge_threshold,
                "evaluator_loop_max_iterations": profile.evaluator_loop_max_iterations,
                "default_rubric_ref": profile.default_rubric_ref,
                "critic_llm_profile_ref": profile.critic_llm_profile_ref,
            },
        },
    )


def apply_application_critic_wiring(
    nexus: NexusLoop,
    wiring: ApplicationCriticWiring,
) -> None:
    """Attach resolved critic hooks to an existing ``NexusLoop`` instance."""
    nexus.apply_critic_graph_hooks(wiring.graph_hooks)
