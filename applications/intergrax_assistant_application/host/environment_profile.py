# © Artur Czarnecki. All rights reserved.

"""Tier-3 environment profile for intergrax_assistant_application (Phase H-APP.5.5)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    OrchestrationProfile,
)
from intergrax.llm_adapters.registry.profile import llm_profile_from_env
from intergrax_assistant_application.host.settings import IntergraxAssistantApplicationSettings


def build_intergrax_assistant_environment_profile(
    settings: IntergraxAssistantApplicationSettings,
) -> ApplicationEnvironmentProfile:
    """Harness chat lab — full memory/RAG/tools with swappable LLM adapter."""
    llm = llm_profile_from_env(prefix=settings.llm_env_prefix)
    base = ApplicationEnvironmentProfile.lab_defaults(profile_id="intergrax_assistant.harness_lab")
    orchestration = OrchestrationProfile(
        planner_kind="engine" if settings.engine_planner else None,
        classifier_kind="default",
        long_running_enabled=True,
        max_delegation_depth=settings.max_delegation_depth,
        allow_dynamic_replan=settings.engine_planner,
        merge_strategy="last_wins",
    )
    return base.model_copy(
        update={
            "llm_profile": llm,
            "orchestration_profile": orchestration,
        }
    )
