# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Strict production orchestration topology composition (GR-10-R13-R2)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.contracts.orchestration_consequential_effect_reliability import (
    OrchestrationConsequentialEffectReliabilityPort,
)
from intergrax.contracts.orchestration_topology import (
    OrchestrationSlotId,
    OrchestrationTopologySubmissionPort,
)
from intergrax.runtime.execution.orchestration_topology_submission import (
    build_production_orchestration_topology_submission_port,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.contracts.provider_invocation_store import ProviderInvocationStore
from intergrax.runtime.execution.orchestration_topology_slot_mse_enforcement import (
    OrchestrationTopologySlotEffectAuthorityOwner,
    OrchestrationTopologySlotMsePolicy,
    build_orchestration_topology_slot_mse_policy,
    default_consequential_topology_slot,
    default_resolve_orchestration_topology_slot_effect_authority_owner,
    default_topology_slot_enforcement_request,
)
from intergrax.runtime.governance.orchestration_consequential_effect_reliability_composition import (
    OrchestrationConsequentialEffectReliabilityCompositionError,
    build_production_orchestration_consequential_effect_reliability_boundary,
)
def build_orchestration_reliability_composition(
    *,
    provider_invocation_store: ProviderInvocationStore | None,
    clock: Callable[[], datetime],
    tenant_id: str,
    production_mode: bool = True,
) -> OrchestrationConsequentialEffectReliabilityPort:
    """Wire canonical Reliability boundary from a contract ``ProviderInvocationStore``."""
    if provider_invocation_store is None:
        raise OrchestrationConsequentialEffectReliabilityCompositionError(
            "strict production orchestration reliability requires ProviderInvocationStore",
        )
    return build_production_orchestration_consequential_effect_reliability_boundary(
        provider_invocation_store=provider_invocation_store,
        clock=clock,
        tenant_id=tenant_id,
        production_mode=production_mode,
    )


def build_strict_production_orchestration_topology_slot_mse_policy(
    *,
    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None,
    provider_invocation_store: ProviderInvocationStore | None,
    tenant_id: str,
    clock: Callable[[], datetime],
    build_enforcement_request: Callable[
        [OrchestrationSlotId, object],
        CollaborativeWorkEnforcementRequest,
    ]
    | None = None,
    is_consequential_slot: Callable[[OrchestrationSlotId, object], bool] | None = None,
    resolve_effect_authority_owner: Callable[
        [object],
        OrchestrationTopologySlotEffectAuthorityOwner | None,
    ]
    | None = None,
) -> OrchestrationTopologySlotMsePolicy:
    """Production topology MSE policy with mandatory durable Reliability port."""
    effect_reliability = build_orchestration_reliability_composition(
        provider_invocation_store=provider_invocation_store,
        clock=clock,
        tenant_id=tenant_id,
        production_mode=True,
    )
    return build_orchestration_topology_slot_mse_policy(
        meaningful_side_effect_authorization=meaningful_side_effect_authorization,
        production_mode=True,
        build_enforcement_request=build_enforcement_request
        or default_topology_slot_enforcement_request,
        is_consequential_slot=is_consequential_slot or default_consequential_topology_slot,
        resolve_effect_authority_owner=(
            resolve_effect_authority_owner
            or default_resolve_orchestration_topology_slot_effect_authority_owner
        ),
        effect_reliability=effect_reliability,
    )


def build_strict_production_orchestration_topology_submission_port(
    nexus_loop: NexusLoop,
    *,
    provider_invocation_store: ProviderInvocationStore | None,
    tenant_id: str,
    clock: Callable[[], datetime],
    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None,
) -> OrchestrationTopologySubmissionPort[object, object]:
    """Strict production topology submission with durable Reliability composition."""
    policy = build_strict_production_orchestration_topology_slot_mse_policy(
        meaningful_side_effect_authorization=meaningful_side_effect_authorization,
        provider_invocation_store=provider_invocation_store,
        tenant_id=tenant_id,
        clock=clock,
    )
    return build_production_orchestration_topology_submission_port(
        nexus_loop,
        slot_mse_policy=policy,
    )


__all__ = [
    "build_orchestration_reliability_composition",
    "build_strict_production_orchestration_topology_slot_mse_policy",
    "build_strict_production_orchestration_topology_submission_port",
]
