# © Artur Czarnecki. All rights reserved.

"""Tier-3 critic profile fragments (DS-MIG-02: inert legacy config only)."""

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
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine


@dataclass(frozen=True, slots=True)
class ApplicationCriticWiring:
    """Legacy critic profile artifacts retained for config parsing (DS-MIG-05 removal)."""

    profile: CriticProfile
    options: CriticWiringOptions
    domain_fragments: dict[str, Any]


def wire_application_critic(
    env: ApplicationEnvironmentProfile,
    *,
    l1_client: object | None = None,
    validation_engine: NexusValidationEngine | None = None,
) -> ApplicationCriticWiring:
    """Materialize legacy critic governance fragments without runtime authority."""
    _ = l1_client
    _ = validation_engine
    profile = env.critic_profile
    options = resolve_critic_wiring_options(profile)
    return ApplicationCriticWiring(
        profile=profile,
        options=options,
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
                "l2_human_required": profile.l2_human_required,
                "l2_borderline_margin": profile.l2_borderline_margin,
                "runtime_effect": "none",
                "scheduled_removal": "DS-MIG-05",
            },
        },
    )


def apply_application_critic_wiring(
    nexus: NexusLoop,
    wiring: ApplicationCriticWiring,
) -> None:
    """No-op: CriticOrchestrator retired from production authority (DS-MIG-02)."""
    _ = nexus
    _ = wiring
