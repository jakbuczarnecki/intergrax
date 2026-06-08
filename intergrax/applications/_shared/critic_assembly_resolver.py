# © Artur Czarnecki. All rights reserved.

"""Critic assembly validation for Tier-3 hosts (Phase CRIT-V-6.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intergrax.applications._shared.critic_wiring import ApplicationCriticWiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


@dataclass(frozen=True, slots=True)
class CriticAssemblyValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()


class CriticAssemblyError(ValueError):
    def __init__(self, errors: Sequence[str]) -> None:
        self.errors: tuple[str, ...] = tuple(errors)
        super().__init__("; ".join(self.errors))


def validate_critic_wiring(
    wiring: ApplicationCriticWiring,
    env: ApplicationEnvironmentProfile,
    *,
    l1_client: object | None = None,
) -> CriticAssemblyValidationResult:
    errors: list[str] = []
    profile = env.critic_profile
    options = wiring.options
    fragment = wiring.domain_fragments.get("critic_governance", {})

    if options.semantic_judge_enabled != profile.semantic_judge_enabled:
        errors.append("semantic_judge_enabled mismatch between wiring and critic_profile")
    if options.verify_node_partial != profile.scopes.node_partial:
        errors.append("verify_node_partial mismatch between wiring and critic_profile")
    if options.verify_graph_final != profile.scopes.graph_final:
        errors.append("verify_graph_final mismatch between wiring and critic_profile")

    if profile.semantic_judge_enabled and not profile.default_rubric_ref:
        errors.append("semantic_judge_enabled requires default_rubric_ref")

    if profile.semantic_judge_enabled or profile.trajectory_eval_enabled:
        if l1_client is None:
            errors.append("semantic or trajectory critic requires configured L1 eval tool client")
        elif wiring.graph_hooks is not None and not wiring.graph_hooks.orchestrator.l1_client_configured:
            errors.append("critic graph hooks require configured L1 eval tool client")

    if profile.require_critic_on_completion and not (
        profile.scopes.node_partial or profile.scopes.graph_final
    ):
        errors.append("require_critic_on_completion requires node_partial or graph_final scope")

    if profile.scopes.node_partial or profile.scopes.graph_final:
        if wiring.graph_hooks is None:
            errors.append("critic scopes enabled but graph_hooks missing")

    if fragment.get("semantic_judge_enabled") != profile.semantic_judge_enabled:
        errors.append("critic_governance fragment must match semantic_judge_enabled")

    if fragment.get("require_critic_on_completion") != profile.require_critic_on_completion:
        errors.append("critic_governance fragment must match require_critic_on_completion")

    return CriticAssemblyValidationResult(valid=not errors, errors=tuple(errors))


def assert_critic_assembly_valid(
    wiring: ApplicationCriticWiring,
    env: ApplicationEnvironmentProfile,
    *,
    l1_client: object | None = None,
) -> None:
    result = validate_critic_wiring(wiring, env, l1_client=l1_client)
    if not result.valid:
        raise CriticAssemblyError(result.errors)
