# © Artur Czarnecki. All rights reserved.

"""Resolve separate LLM adapter for critic judges (Phase CRIT-V-FOLLOWUP)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.runtime.nexus.config import RuntimeConfig


def resolve_critic_llm_adapter(
    env: ApplicationEnvironmentProfile,
    *,
    producer_adapter: LLMAdapter,
    runtime_config: RuntimeConfig | None = None,
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
        adapter = separate.create_adapter()
    else:
        adapter = producer_adapter
    if runtime_config is not None:
        from intergrax.applications._shared.llm_routing_runtime_bridge import (
            maybe_wrap_secondary_routing_adapter,
        )

        wrapped = maybe_wrap_secondary_routing_adapter(adapter, env, runtime_config)
        if wrapped is not None:
            adapter = wrapped
    return adapter
