# © Artur Czarnecki. All rights reserved.

"""Lab-specific ApplicationEnvironmentProfile builder (Phase H-APP.5.1)."""

from __future__ import annotations

from pathlib import Path

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
from intergrax.fastapi_core.config import ApiEnvironment
from intergrax.integrations.registry import presets
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
    if settings.environment == ApiEnvironment.PROD and settings.secrets_backend_slug:
        env = ApplicationEnvironmentProfile.harness_production_defaults(
            profile_id="lab.harness.prod",
            harness_tools=settings.harness,
            secrets_slug=settings.secrets_backend_slug,
            enable_grafana_stack=settings.observability_grafana_stack,
        )
    else:
        env = ApplicationEnvironmentProfile.lab_defaults(
            profile_id="lab.harness",
            harness_tools=settings.harness,
        )
        if settings.observability_grafana_stack:
            env = env.model_copy(
                update={
                    "integration_profile": presets.observability_stack(
                        enable_otel=settings.otel_enabled,
                        enable_grafana_stack=True,
                    )
                }
            )
        elif settings.otel_enabled:
            env = env.model_copy(
                update={
                    "integration_profile": presets.observability_stack(
                        enable_otel=True,
                        enable_grafana_stack=False,
                    )
                }
            )

    env.identity_profile.require_api_key = settings.requires_harness_api_key
    env.observability_profile.otel_enabled = settings.otel_enabled
    env.reliability_profile.long_running_scheduler_enabled = settings.include_scheduler
    env.orchestration_profile = env.orchestration_profile.model_copy(
        update={
            "long_running_enabled": settings.include_scheduler,
            "emit_coordination_advisory": settings.harness,
        }
    )
    env.features = env.features.model_copy(
        update={
            "debug_surface": True,
            "interaction_routes": settings.include_interaction_routes,
            "long_running_scheduler": settings.include_scheduler,
        }
    )
    if _LAB_POLICY_RULES.is_file():
        env.policy_rules = PolicyRulesProfile(rules_path=_LAB_POLICY_RULES)

    adaptive_updates: dict[str, object] = {
        "enabled": settings.adaptive_observe_enabled,
        "mode": "observe",
        "debug_readonly_routes": True,
    }
    if settings.adaptive_feature_flag_slug:
        adaptive_updates["feature_flag_slug"] = settings.adaptive_feature_flag_slug
        env = env.model_copy(
            update={
                "integration_profile": env.integration_profile.model_copy(
                    update={"feature_flag": settings.adaptive_feature_flag_slug}
                )
            }
        )
    env.adaptive_profile = env.adaptive_profile.model_copy(update=adaptive_updates)
    from intergrax.applications._shared.context_presets import production_context_profile

    env = env.model_copy(
        update={
            "context_profile": production_context_profile().model_copy(
                update={"context_plugin_ids": ["intergrax.builtin"]}
            )
        }
    )
    env = env.model_copy(
        update={
            "tool_invocation_mode": settings.tool_invocation_mode,
        }
    )
    if settings.enable_llm_guardrails:
        from intergrax.applications.contracts.environment_profile import GuardrailProfile

        env = env.model_copy(
            update={
                "integration_profile": presets.harness_guardrail_stack(
                    primary="llm_guard",
                    semantic="presidio",
                ),
                "guardrail_profile": GuardrailProfile(
                    enabled=True,
                    scan_input=True,
                    scan_output=True,
                    secondary_slug="presidio",
                ),
            },
        )
    return env
