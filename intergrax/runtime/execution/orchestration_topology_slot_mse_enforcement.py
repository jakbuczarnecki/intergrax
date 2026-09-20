# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Mandatory MSE enforcement for canonical orchestration topology slot paths (GR-10-R9-R3/R4)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, Protocol, TypeVar, runtime_checkable

from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectKind
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.contracts.orchestration_consequential_effect_reliability import (
    OrchestrationConsequentialEffectReliabilityPort,
)
from intergrax.contracts.orchestration_topology import (
    OrchestrationSlotContinuationExecutor,
    OrchestrationSlotExecutor,
    OrchestrationSlotId,
)
from intergrax.runtime.nexus.orchestration.governed_consequential_operation import (
    GovernedOrchestrationSlotContinuationExecutor,
    GovernedOrchestrationSlotExecutor,
)
from intergrax.runtime.nexus.orchestration.orchestration_graph_meaningful_side_effect import (
    build_orchestration_graph_slot_enforcement_request,
    build_orchestration_graph_slot_meaningful_side_effect_request,
)

PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")


class OrchestrationTopologyReliabilityCompositionError(RuntimeError):
    """Fail closed when production orchestration topology omits post-admission reliability."""


class OrchestrationTopologySlotEffectAuthorityOwner(Enum):
    """Which canonical boundary owns orchestration slot physical effects (composition-internal)."""

    MSE = auto()
    PHYSICAL_DELEGATION = auto()


@runtime_checkable
class OrchestrationSlotEffectAuthoritySurface(Protocol):
    """Typed optional surface for slot effect-authority declaration (no attribute probing)."""

    @property
    def orchestration_slot_effect_authority_owner(
        self,
    ) -> OrchestrationTopologySlotEffectAuthorityOwner:
        ...


def default_resolve_orchestration_topology_slot_effect_authority_owner(
    executor: object,
) -> OrchestrationTopologySlotEffectAuthorityOwner | None:
    """Resolve declared slot effect authority from typed contract surface."""
    if not isinstance(executor, OrchestrationSlotEffectAuthoritySurface):
        return None
    owner = executor.orchestration_slot_effect_authority_owner
    if isinstance(owner, OrchestrationTopologySlotEffectAuthorityOwner):
        return owner
    return None


def orchestration_topology_slot_mse_delegated(
    executor: object,
    *,
    resolve_effect_authority_owner: Callable[
        [object],
        OrchestrationTopologySlotEffectAuthorityOwner | None,
    ],
) -> bool:
    """True when another canonical boundary already authorizes slot physical effects."""
    return (
        resolve_effect_authority_owner(executor)
        is OrchestrationTopologySlotEffectAuthorityOwner.PHYSICAL_DELEGATION
    )


def default_consequential_topology_slot(
    _slot_id: OrchestrationSlotId,
    _payload: object,
) -> bool:
    """Fail-closed: unknown slots are treated as consequential on production topology paths."""
    return True


def default_topology_slot_enforcement_request(
    slot_id: OrchestrationSlotId,
    payload: object,
) -> CollaborativeWorkEnforcementRequest:
    """Pure projection for custom topology slots without host-specific metadata."""
    operation_id = f"slot:{slot_id}"
    resource_scope = f"orchestration/topology/slot/{slot_id}"
    side_effect = build_orchestration_graph_slot_meaningful_side_effect_request(
        slot_id=slot_id,
        operation_id=operation_id,
        resource_scope=resource_scope,
        side_effect_scope_id=f"{resource_scope}:effect",
        kinds=(MeaningfulSideEffectKind.MUTATION,),
    )
    return build_orchestration_graph_slot_enforcement_request(
        slot_id=slot_id,
        side_effect=side_effect,
        operation_id=operation_id,
        resource_scope=resource_scope,
    )


@dataclass(frozen=True, slots=True)
class OrchestrationTopologySlotMsePolicy:
    """Composition-root policy for canonical topology slot MSE (not a public orchestration contract)."""

    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None
    production_mode: bool
    build_enforcement_request: Callable[
        [OrchestrationSlotId, object],
        CollaborativeWorkEnforcementRequest,
    ] = default_topology_slot_enforcement_request
    is_consequential_slot: Callable[[OrchestrationSlotId, object], bool] = (
        default_consequential_topology_slot
    )
    resolve_effect_authority_owner: Callable[
        [object],
        OrchestrationTopologySlotEffectAuthorityOwner | None,
    ] = default_resolve_orchestration_topology_slot_effect_authority_owner
    effect_reliability: OrchestrationConsequentialEffectReliabilityPort | None = None


def build_orchestration_topology_slot_mse_policy(
    *,
    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None,
    production_mode: bool,
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
    effect_reliability: OrchestrationConsequentialEffectReliabilityPort | None = None,
) -> OrchestrationTopologySlotMsePolicy:
    if production_mode and effect_reliability is None:
        raise OrchestrationTopologyReliabilityCompositionError(
            "production orchestration topology requires OrchestrationConsequentialEffectReliabilityPort",
        )
    return OrchestrationTopologySlotMsePolicy(
        meaningful_side_effect_authorization=meaningful_side_effect_authorization,
        production_mode=production_mode,
        build_enforcement_request=(
            build_enforcement_request or default_topology_slot_enforcement_request
        ),
        is_consequential_slot=is_consequential_slot or default_consequential_topology_slot,
        resolve_effect_authority_owner=(
            resolve_effect_authority_owner
            or default_resolve_orchestration_topology_slot_effect_authority_owner
        ),
        effect_reliability=effect_reliability,
    )


def _already_governed_slot_executor(
    executor: OrchestrationSlotExecutor[PayloadT, ResultT],
) -> bool:
    return isinstance(executor, GovernedOrchestrationSlotExecutor)


def _already_governed_continuation_executor(
    executor: OrchestrationSlotContinuationExecutor[PayloadT, ResultT],
) -> bool:
    return isinstance(executor, GovernedOrchestrationSlotContinuationExecutor)


def prepare_orchestration_topology_slot_executor(
    slot_executor: OrchestrationSlotExecutor[PayloadT, ResultT],
    *,
    policy: OrchestrationTopologySlotMsePolicy | None,
) -> OrchestrationSlotExecutor[PayloadT, ResultT]:
    if policy is None:
        return slot_executor
    if orchestration_topology_slot_mse_delegated(
        slot_executor,
        resolve_effect_authority_owner=policy.resolve_effect_authority_owner,
    ):
        return slot_executor
    if _already_governed_slot_executor(slot_executor):
        return slot_executor
    return GovernedOrchestrationSlotExecutor(
        inner=slot_executor,
        meaningful_side_effect_authorization=policy.meaningful_side_effect_authorization,
        production_mode=policy.production_mode,
        build_enforcement_request=policy.build_enforcement_request,
        is_consequential_slot=policy.is_consequential_slot,
        effect_reliability=policy.effect_reliability,
    )


def prepare_orchestration_topology_slot_continuation_executor(
    slot_continuation_executor: OrchestrationSlotContinuationExecutor[PayloadT, ResultT],
    *,
    policy: OrchestrationTopologySlotMsePolicy | None,
) -> OrchestrationSlotContinuationExecutor[PayloadT, ResultT]:
    if policy is None:
        return slot_continuation_executor
    if orchestration_topology_slot_mse_delegated(
        slot_continuation_executor,
        resolve_effect_authority_owner=policy.resolve_effect_authority_owner,
    ):
        return slot_continuation_executor
    if _already_governed_continuation_executor(slot_continuation_executor):
        return slot_continuation_executor
    return GovernedOrchestrationSlotContinuationExecutor(
        inner=slot_continuation_executor,
        meaningful_side_effect_authorization=policy.meaningful_side_effect_authorization,
        production_mode=policy.production_mode,
        build_enforcement_request=policy.build_enforcement_request,
        is_consequential_slot=policy.is_consequential_slot,
        effect_reliability=policy.effect_reliability,
    )


__all__ = [
    "OrchestrationSlotEffectAuthoritySurface",
    "OrchestrationTopologyReliabilityCompositionError",
    "OrchestrationTopologySlotEffectAuthorityOwner",
    "OrchestrationTopologySlotMsePolicy",
    "build_orchestration_topology_slot_mse_policy",
    "default_consequential_topology_slot",
    "default_resolve_orchestration_topology_slot_effect_authority_owner",
    "default_topology_slot_enforcement_request",
    "orchestration_topology_slot_mse_delegated",
    "prepare_orchestration_topology_slot_executor",
    "prepare_orchestration_topology_slot_continuation_executor",
]
