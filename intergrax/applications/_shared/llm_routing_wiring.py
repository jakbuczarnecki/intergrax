# © Artur Czarnecki. All rights reserved.

"""Live cost/latency/quality model routing on product hosts (AUDIT-IDEAL-6.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.registry.model_router import ModelRouter, ModelRoutingDecision
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.runtime.adaptive.routing_tuning_engine import RoutingTuningEngine
from intergrax.runtime.architecture.adaptive_governance import AdaptiveLoopKind
from intergrax.runtime.adaptive.bandit_state_store import InMemoryBanditStateStore


@dataclass(frozen=True, slots=True)
class LiveModelRoutingWiring:
    enabled: bool
    engine_id: str
    routing_decision: ModelRoutingDecision | None


def resolve_live_model_routing_wiring(
    env: ApplicationEnvironmentProfile,
) -> LiveModelRoutingWiring:
    """Wire AHI routing tuning with policy-driven model router on product hosts."""
    adaptive = env.adaptive_profile
    loop_enabled = AdaptiveLoopKind.ROUTING_TUNING in adaptive.enabled_loops
    is_product = env.application_profile is ApplicationProfile.PRODUCT
    if not is_product or not adaptive.enabled or not loop_enabled or not adaptive.live_model_routing_enabled:
        return LiveModelRoutingWiring(enabled=False, engine_id="routing_tuning", routing_decision=None)

    primary = env.llm_profile or LLMProfile.lab()
    fallbacks = tuple(primary.fallback_profiles)
    if not fallbacks and primary.model:
        fallbacks = (LLMProfile(provider=primary.provider, model=primary.model),)
    router = ModelRouter.from_profiles(
        primary,
        fallbacks=fallbacks,
        policy_route_hint="balanced",
    )
    engine = RoutingTuningEngine(InMemoryBanditStateStore())
    return LiveModelRoutingWiring(
        enabled=True,
        engine_id=engine.engine_id,
        routing_decision=router.resolve(),
    )
