# © Artur Czarnecki. All rights reserved.

"""Live cost/latency/quality model routing on product hosts (AUDIT-IDEAL-6.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.registry.model_router import ModelRouter, ModelRoutingDecision
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import RoutingContext
from intergrax.runtime.adaptive.routing_tuning_engine import RoutingTuningEngine
from intergrax.runtime.architecture.adaptive_governance import AdaptiveLoopKind
from intergrax.runtime.adaptive.bandit_state_store import InMemoryBanditStateStore

_BANDIT_ARM_HINTS: dict[str, str] = {
    "rag_tier_default": "balanced",
    "rag_tier_deep": "quality",
    "llm_route_balanced": "balanced",
}


@dataclass(frozen=True, slots=True)
class LiveModelRoutingWiring:
    enabled: bool
    engine_id: str
    routing_decision: ModelRoutingDecision | None


def _select_policy_hint_from_bandit(
    *,
    tenant_id: str,
    task_class: str,
    store: InMemoryBanditStateStore,
) -> str:
    arm_scores = {
        arm_id: store.sample_arm_score(
            tenant_id=tenant_id,
            task_class=task_class,
            arm_id=arm_id,
        )
        for arm_id in _BANDIT_ARM_HINTS
    }
    selected_arm = max(arm_scores, key=arm_scores.get)
    return _BANDIT_ARM_HINTS[selected_arm]


def resolve_live_model_routing_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    routing_context: RoutingContext | None = None,
) -> LiveModelRoutingWiring:
    """Wire AHI routing tuning with policy-driven model router on product hosts."""
    adaptive = env.adaptive_profile
    loop_enabled = AdaptiveLoopKind.ROUTING_TUNING in adaptive.enabled_loops
    is_product = env.application_profile is ApplicationProfile.PRODUCT
    if not is_product or not adaptive.enabled or not loop_enabled or not adaptive.live_model_routing_enabled:
        return LiveModelRoutingWiring(enabled=False, engine_id="routing_tuning", routing_decision=None)

    context = routing_context or RoutingContext()
    tenant_id = context.tenant_id or "default"
    task_class = context.task_class or "default"

    primary = env.llm_profile or LLMProfile.lab()
    fallbacks = tuple(primary.fallback_profiles)
    if not fallbacks and primary.model:
        fallbacks = (LLMProfile(provider=primary.provider, model=primary.model),)

    store = InMemoryBanditStateStore()
    policy_hint = _select_policy_hint_from_bandit(
        tenant_id=tenant_id,
        task_class=task_class,
        store=store,
    )
    router = ModelRouter.from_profiles(
        primary,
        fallbacks=fallbacks,
        policy_route_hint=policy_hint,
    )
    engine = RoutingTuningEngine(store)
    return LiveModelRoutingWiring(
        enabled=True,
        engine_id=engine.engine_id,
        routing_decision=router.resolve(),
    )
