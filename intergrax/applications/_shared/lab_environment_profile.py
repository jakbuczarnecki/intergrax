# © Artur Czarnecki. All rights reserved.

"""Lab-specific ApplicationEnvironmentProfile builder (Phase H-APP.5.1)."""

from __future__ import annotations

from pathlib import Path

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from lab_application.host.settings import LabApplicationSettings

_LAB_POLICY_RULES = (
    Path(__file__).resolve().parents[3]
    / "applications"
    / "lab_application"
    / "policy"
    / "rules"
    / "harness_lab.yaml"
)


def build_lab_environment_profile(
    settings: LabApplicationSettings,
) -> ApplicationEnvironmentProfile:
    """Compose lab environment from settings flags (replaces ad-hoc wiring)."""
    env = ApplicationEnvironmentProfile.lab_defaults(
        profile_id="lab.harness",
        harness_tools=settings.harness,
    )
    env.identity_profile.require_api_key = settings.requires_harness_api_key
    env.observability_profile.otel_enabled = settings.otel_enabled
    env.reliability_profile.long_running_scheduler_enabled = settings.include_scheduler
    env.orchestration_profile.long_running_enabled = settings.include_scheduler
    env.features = env.features.model_copy(
        update={
            "debug_surface": True,
            "interaction_routes": settings.include_interaction_routes,
            "long_running_scheduler": settings.include_scheduler,
        }
    )
    if _LAB_POLICY_RULES.is_file():
        env.policy_rules = PolicyRulesProfile(rules_path=_LAB_POLICY_RULES)
    env.adaptive_profile = env.adaptive_profile.model_copy(
        update={
            "enabled": settings.adaptive_observe_enabled,
            "mode": "observe",
            "debug_readonly_routes": True,
        }
    )
    return env
