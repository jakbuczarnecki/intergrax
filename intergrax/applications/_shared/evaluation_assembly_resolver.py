# © Artur Czarnecki. All rights reserved.

"""Evaluation assembly validation for Tier-3 hosts (Phase EVAL-2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intergrax.applications._shared.evaluation_wiring import ApplicationEvaluationWiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


@dataclass(frozen=True, slots=True)
class EvaluationAssemblyValidationResult:
    """Outcome of evaluation assembly validation."""

    valid: bool
    errors: tuple[str, ...] = ()


class EvaluationAssemblyError(ValueError):
    """Raised when evaluation assembly validation fails."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors: tuple[str, ...] = tuple(errors)
        message = "; ".join(self.errors)
        super().__init__(message)


def validate_evaluation_wiring(
    wiring: ApplicationEvaluationWiring,
    env: ApplicationEnvironmentProfile,
) -> EvaluationAssemblyValidationResult:
    """Validate evaluation artifacts match environment profile requirements."""
    errors: list[str] = []
    profile = env.evaluation_profile
    options = wiring.options
    fragment = wiring.domain_fragments.get("evaluation_governance", {})

    if options.shadow_eval_enabled != profile.shadow_eval_enabled:
        errors.append("shadow_eval_enabled mismatch between wiring and evaluation_profile")

    if options.online_registry_enabled != profile.online_registry_enabled:
        errors.append("online_registry_enabled mismatch between wiring and evaluation_profile")

    if profile.online_registry_enabled and wiring.registry is None:
        errors.append("online_registry_enabled requires evaluation registry")

    if profile.shadow_eval_enabled and profile.online_registry_enabled and wiring.registry is None:
        errors.append("shadow_eval_enabled with online_registry_enabled requires evaluation registry")

    if profile.shadow_eval_enabled and wiring.governance_bridge is None:
        errors.append("shadow_eval_enabled requires governance_bridge")

    if profile.require_baseline_for_release and not profile.trend_comparison_enabled:
        errors.append("require_baseline_for_release requires trend_comparison_enabled")

    if profile.require_baseline_for_release and not profile.online_registry_enabled:
        errors.append("require_baseline_for_release requires online_registry_enabled")

    if fragment.get("shadow_eval_enabled") != profile.shadow_eval_enabled:
        errors.append("evaluation_governance domain fragment must match shadow_eval_enabled")

    if fragment.get("online_registry_enabled") != profile.online_registry_enabled:
        errors.append("evaluation_governance domain fragment must match online_registry_enabled")

    if fragment.get("require_baseline_for_release") != profile.require_baseline_for_release:
        errors.append("evaluation_governance domain fragment must match require_baseline_for_release")

    return EvaluationAssemblyValidationResult(valid=not errors, errors=tuple(errors))


def assert_evaluation_assembly_valid(
    wiring: ApplicationEvaluationWiring,
    env: ApplicationEnvironmentProfile,
) -> None:
    """Raise :class:`EvaluationAssemblyError` when evaluation validation fails."""
    result = validate_evaluation_wiring(wiring, env)
    if not result.valid:
        raise EvaluationAssemblyError(result.errors)
