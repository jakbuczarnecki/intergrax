# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Authorize orchestration graph / non-tool consequential effects via canonical MSE port (GR-10-R9-R2)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
    MeaningfulSideEffectAuthorizationResult,
    assert_consistent_meaningful_side_effect_authorization_result,
)
from intergrax.contracts.orchestration_topology import (
    OrchestrationSlotExecutionError,
    OrchestrationSlotExecutor,
    OrchestrationSlotFailure,
    OrchestrationSlotId,
    OrchestrationSlotOutcome,
    OrchestrationSlotStatus,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.policy.side_effect_authorization_errors import (
    MeaningfulSideEffectAuthorizationRequiredError,
    SideEffectAuthorizationFailureReason,
)

PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")


class OrchestrationConsequentialEffectBlockedError(RuntimeError):
    """Raised when MSE denies or blocks a non-tool orchestration consequential effect."""


@dataclass(frozen=True, slots=True)
class GovernedOrchestrationSlotExecutor(Generic[PayloadT, ResultT]):
    """Wrap a slot executor with per-slot canonical MSE authorization before physical effect."""

    inner: OrchestrationSlotExecutor[PayloadT, ResultT]
    meaningful_side_effect_authorization: MeaningfulSideEffectAuthorizationPort | None
    production_mode: bool
    build_enforcement_request: Callable[
        [OrchestrationSlotId, PayloadT],
        CollaborativeWorkEnforcementRequest,
    ]
    source_agent_id: str = "platform.orchestration.graph_slot"

    async def execute_slot(
        self,
        *,
        slot_id: OrchestrationSlotId,
        payload: PayloadT,
    ) -> ResultT:
        enforcement_request = self.build_enforcement_request(slot_id, payload)
        try:
            authorize_orchestration_consequential_effect(
                self.meaningful_side_effect_authorization,
                enforcement_request=enforcement_request,
                production_mode=self.production_mode,
                source_agent_id=self.source_agent_id,
                source_step_id=str(slot_id),
            )
        except (
            OrchestrationConsequentialEffectBlockedError,
            MeaningfulSideEffectAuthorizationRequiredError,
        ) as exc:
            code = (
                "meaningful_side_effect_not_configured"
                if isinstance(exc, MeaningfulSideEffectAuthorizationRequiredError)
                else "meaningful_side_effect_blocked"
            )
            raise OrchestrationSlotExecutionError(code=code, message=str(exc)) from exc
        return await self.inner.execute_slot(slot_id=slot_id, payload=payload)


def authorize_orchestration_consequential_effect(
    boundary: MeaningfulSideEffectAuthorizationPort | None,
    *,
    enforcement_request: CollaborativeWorkEnforcementRequest,
    production_mode: bool,
    source_agent_id: str,
    source_step_id: str | None,
) -> MeaningfulSideEffectAuthorizationResult:
    """Fail-closed MSE gate before a non-tool orchestration physical effect."""
    if boundary is None:
        if production_mode:
            raise MeaningfulSideEffectAuthorizationRequiredError(
                run_id=str(enforcement_request.operation_id),
                agent_id=source_agent_id,
                tool_id=enforcement_request.resource_scope or enforcement_request.operation_id,
                reason=SideEffectAuthorizationFailureReason.NOT_CONFIGURED,
            )
        raise OrchestrationConsequentialEffectBlockedError(
            "meaningful_side_effect_authorization is required for orchestration consequential effects",
        )

    authorization = boundary.authorize(
        enforcement_request,
        source_agent_id=source_agent_id,
        source_step_id=source_step_id,
    )
    if not isinstance(authorization, MeaningfulSideEffectAuthorizationResult):
        raise MeaningfulSideEffectAuthorizationRequiredError(
            run_id=str(enforcement_request.operation_id),
            agent_id=source_agent_id,
            tool_id=enforcement_request.resource_scope or enforcement_request.operation_id,
            reason=SideEffectAuthorizationFailureReason.NOT_CONFIGURED,
        )
    try:
        assert_consistent_meaningful_side_effect_authorization_result(authorization)
    except ValueError:
        raise MeaningfulSideEffectAuthorizationRequiredError(
            run_id=str(enforcement_request.operation_id),
            agent_id=source_agent_id,
            tool_id=enforcement_request.resource_scope or enforcement_request.operation_id,
            reason=SideEffectAuthorizationFailureReason.NOT_CONFIGURED,
        ) from None

    decision = authorization.decision
    if decision.action in (
        PolicyAction.REQUIRE_HUMAN,
        PolicyAction.ESCALATE,
        PolicyAction.DENY,
        PolicyAction.MODIFY,
    ) or not authorization.permitted:
        raise OrchestrationConsequentialEffectBlockedError(
            f"orchestration consequential effect blocked: {decision.action.value}",
        )
    return authorization


async def execute_governed_orchestration_consequential_effect(
    boundary: MeaningfulSideEffectAuthorizationPort | None,
    *,
    enforcement_request: CollaborativeWorkEnforcementRequest,
    production_mode: bool,
    source_agent_id: str,
    source_step_id: str | None,
    effect: Callable[[], Awaitable[ResultT]],
) -> ResultT:
    """Authorize then run one async consequential orchestration effect (exactly once on ALLOW)."""
    authorize_orchestration_consequential_effect(
        boundary,
        enforcement_request=enforcement_request,
        production_mode=production_mode,
        source_agent_id=source_agent_id,
        source_step_id=source_step_id,
    )
    return await effect()


def orchestration_slot_outcome_from_blocked_effect(
    *,
    slot_id: OrchestrationSlotId,
    exc: BaseException,
) -> OrchestrationSlotOutcome[ResultT]:
    """Map a blocked consequential effect to a typed orchestration slot failure."""
    code = "meaningful_side_effect_blocked"
    if isinstance(exc, MeaningfulSideEffectAuthorizationRequiredError):
        code = "meaningful_side_effect_not_configured"
    return OrchestrationSlotOutcome(
        slot_id=slot_id,
        status=OrchestrationSlotStatus.FAILURE,
        failure=OrchestrationSlotFailure(
            code=code,
            message=str(exc),
        ),
    )


__all__ = [
    "GovernedOrchestrationSlotExecutor",
    "OrchestrationConsequentialEffectBlockedError",
    "authorize_orchestration_consequential_effect",
    "execute_governed_orchestration_consequential_effect",
    "orchestration_slot_outcome_from_blocked_effect",
]
