# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition root for orchestration post-admission reliability (GR-10-R13)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from intergrax.contracts.orchestration_consequential_effect_reliability import (
    OrchestrationConsequentialEffectReliabilityPort,
)
from intergrax.contracts.provider_invocation_store import ProviderInvocationStore
from intergrax.runtime.governance.orchestration_consequential_effect_reliability_boundary import (
    ProviderInvocationOrchestrationConsequentialEffectReliabilityBoundary,
)


class OrchestrationConsequentialEffectReliabilityCompositionError(RuntimeError):
    """Fail closed when production orchestration reliability cannot be wired."""


def build_production_orchestration_consequential_effect_reliability_boundary(
    *,
    provider_invocation_store: ProviderInvocationStore,
    clock: Callable[[], datetime],
    tenant_id: str,
    production_mode: bool = True,
) -> OrchestrationConsequentialEffectReliabilityPort:
    if not production_mode:
        raise OrchestrationConsequentialEffectReliabilityCompositionError(
            "production orchestration reliability boundary requires production_mode=True",
        )
    if not provider_invocation_store.is_durable:
        raise OrchestrationConsequentialEffectReliabilityCompositionError(
            "production orchestration reliability requires durable ProviderInvocationStore",
        )
    return ProviderInvocationOrchestrationConsequentialEffectReliabilityBoundary(
        store=provider_invocation_store,
        clock=clock,
        tenant_id=tenant_id,
    )


__all__ = [
    "OrchestrationConsequentialEffectReliabilityCompositionError",
    "build_production_orchestration_consequential_effect_reliability_boundary",
]
