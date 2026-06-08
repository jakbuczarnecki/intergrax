# © Artur Czarnecki. All rights reserved.

"""Resolve separate LLM adapter for critic judges (Phase CRIT-V-FOLLOWUP)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.registry.profile import LLMProfile


def resolve_critic_llm_adapter(
    env: ApplicationEnvironmentProfile,
    *,
    producer_adapter: LLMAdapter,
) -> LLMAdapter:
    """
    Resolve judge LLM with producer/critic separation.

    Precedence:
    1. ``CriticProfile.critic_llm_profile`` when set
    2. Producer adapter (when semantic judge disabled or no separate profile)
    """
    critic_profile = env.critic_profile
    separate: LLMProfile | None = critic_profile.critic_llm_profile
    if separate is not None:
        return separate.create_adapter()
    return producer_adapter
