# © Artur Czarnecki. All rights reserved.

"""LLM adapter precedence: agent factory > environment > platform (Phase H-APP.1.6)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.registry.profile import LLMProfile, llm_profile_from_env


def resolve_llm_adapter(
    env: ApplicationEnvironmentProfile | None,
    agent_override: LLMAdapter | None = None,
) -> LLMAdapter:
    """
    Resolve LLM adapter with explicit precedence.

    1. ``agent_override`` when provided by Tier-2 factory
    2. ``env.llm_profile`` when set on environment
    3. Platform default from ``INTERGRAX_LLM_*`` env vars
    """
    if agent_override is not None:
        return agent_override
    if env is not None and env.llm_profile is not None:
        return env.llm_profile.create_adapter()
    return llm_profile_from_env().create_adapter()
