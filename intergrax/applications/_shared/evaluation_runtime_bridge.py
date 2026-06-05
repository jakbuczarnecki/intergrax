# © Artur Czarnecki. All rights reserved.

"""Map ApplicationEnvironmentProfile evaluation fields to wiring options (Phase EVAL-1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    EvaluationProfile,
)
from intergrax.runtime.architecture.online_evaluation_registry import OnlineEvaluationRegistry
from intergrax.runtime.nexus.config import RuntimeConfig


@dataclass(frozen=True, slots=True)
class EvaluationWiringOptions:
    """Resolved evaluation wiring flags for Tier-3 hosts."""

    shadow_eval_enabled: bool
    online_registry_enabled: bool
    offline_eval_runner_enabled: bool
    trend_comparison_enabled: bool
    require_baseline_for_release: bool
    evaluation_assets_ref: str | None


def resolve_evaluation_wiring_options(profile: EvaluationProfile) -> EvaluationWiringOptions:
    """Translate ``EvaluationProfile`` into host wiring flags."""
    return EvaluationWiringOptions(
        shadow_eval_enabled=profile.shadow_eval_enabled,
        online_registry_enabled=profile.online_registry_enabled,
        offline_eval_runner_enabled=profile.offline_eval_runner_enabled,
        trend_comparison_enabled=profile.trend_comparison_enabled,
        require_baseline_for_release=profile.require_baseline_for_release,
        evaluation_assets_ref=profile.evaluation_assets_ref,
    )


def apply_evaluation_profile_to_runtime_config(
    config: RuntimeConfig,
    profile: EvaluationProfile,
    *,
    registry: OnlineEvaluationRegistry | None = None,
) -> RuntimeConfig:
    """Record evaluation posture and registry on runtime config for downstream Nexus steps."""
    config.evaluation_profile = profile
    config.evaluation_registry = registry
    return config


def apply_evaluation_profiles_from_environment(
    config: RuntimeConfig,
    env: ApplicationEnvironmentProfile,
    *,
    registry: OnlineEvaluationRegistry | None = None,
) -> RuntimeConfig:
    """Apply environment-declared evaluation profile."""
    return apply_evaluation_profile_to_runtime_config(
        config,
        env.evaluation_profile,
        registry=registry,
    )
