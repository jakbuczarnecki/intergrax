# © Artur Czarnecki. All rights reserved.

"""Resolve independent verifier LLM adapter for Decision semantic verification (DS-MIG-05)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.runtime.nexus.config import RuntimeConfig


def resolve_decision_verifier_llm_adapter(
    env: ApplicationEnvironmentProfile,
    *,
    producer_adapter: LLMAdapter,
    runtime_config: RuntimeConfig | None = None,
) -> LLMAdapter:
    """
    Resolve semantic verifier LLM with producer/verifier separation.

    Precedence:
    1. ``DecisionVerificationProfile.verifier_llm_profile`` when set
    2. Producer adapter when no explicit verifier profile is declared
    """
    verification = env.decision_profile.verification
    separate: LLMProfile | None = verification.verifier_llm_profile
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
