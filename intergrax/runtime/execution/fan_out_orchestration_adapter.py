# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical fan-out orchestration adapter (NPSC-5B/R4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    FanOutItem,
    FanOutItemFailure,
    FanOutItemId,
    FanOutItemOutcome,
    FanOutItemStatus,
    FanOutOrchestrationContractError,
    FanOutOrchestrationPort,
    FanOutRequest,
)
from intergrax.agent_distribution.multi_agent_coordination import (
    CoordinationCleanupError,
    CoordinationError,
    CoordinationFailureCode,
    MultiAgentCoordinationService,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.orchestration_topology import (
    OrchestrationResult,
    OrchestrationSchedulingPolicy,
    OrchestrationSlot,
    OrchestrationSlotId,
    OrchestrationSlotOutcome,
    OrchestrationSlotStatus,
    OrchestrationTopology,
    OrchestrationTopologySubmissionPort,
)
from intergrax.runtime.execution.orchestration_topology_submission import (
    build_orchestration_topology_host_task,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


def to_orchestration_slot_id(item_id: FanOutItemId) -> OrchestrationSlotId:
    """Project fan-out item identity to orchestration slot identity."""
    return OrchestrationSlotId(str(item_id))


def to_fan_out_item_id(slot_id: OrchestrationSlotId) -> FanOutItemId:
    """Project orchestration slot identity to fan-out item identity."""
    return FanOutItemId(str(slot_id))


@dataclass(frozen=True, slots=True)
class FanOutSlotPayload(Generic[RequestT]):
    """Immutable per-slot payload for fan-out orchestration."""

    item: FanOutItem[RequestT]


def project_fan_out_to_topology(
    request: FanOutRequest[RequestT],
) -> OrchestrationTopology[FanOutSlotPayload[RequestT]]:
    """Map a fan-out request to an independent-slot orchestration topology."""
    return OrchestrationTopology(
        slots=tuple(
            OrchestrationSlot(
                slot_id=to_orchestration_slot_id(item.item_id),
                payload=FanOutSlotPayload(item=item),
                depends_on=(),
            )
            for item in request.items
        ),
    )


def project_fan_out_scheduling_policy(
    request: FanOutRequest[object],
) -> OrchestrationSchedulingPolicy:
    """Map fan-out concurrency to orchestration scheduling policy."""
    return OrchestrationSchedulingPolicy(max_concurrency=request.max_concurrency)


def _coordination_failure_code_from_orchestration_code(code: str) -> CoordinationFailureCode:
    try:
        return CoordinationFailureCode(code)
    except ValueError:
        return CoordinationFailureCode.INVALID_COORDINATION


def map_orchestration_outcome_to_fan_out(
    slot_outcome: OrchestrationSlotOutcome[FanOutItemOutcome[ResultT]],
    *,
    expected_item_id: FanOutItemId,
) -> FanOutItemOutcome[ResultT]:
    """Project one orchestration slot outcome to a fan-out item outcome."""
    if slot_outcome.status is OrchestrationSlotStatus.SKIPPED:
        skip_message = "orchestration slot skipped"
        if slot_outcome.failure is not None:
            skip_message = slot_outcome.failure.message
        return FanOutItemOutcome(
            item_id=expected_item_id,
            status=FanOutItemStatus.FAILURE,
            failure=FanOutItemFailure(
                failure_code=CoordinationFailureCode.INVALID_COORDINATION,
                message=skip_message,
            ),
        )

    if slot_outcome.status is OrchestrationSlotStatus.FAILURE:
        failure = slot_outcome.failure
        code = failure.code if failure is not None else "orchestration_failure"
        message = failure.message if failure is not None else "orchestration slot failed"
        return FanOutItemOutcome(
            item_id=expected_item_id,
            status=FanOutItemStatus.FAILURE,
            failure=FanOutItemFailure(
                failure_code=_coordination_failure_code_from_orchestration_code(code),
                message=message,
            ),
        )

    if slot_outcome.result is None:
        raise FanOutOrchestrationContractError(
            f"missing fan-out outcome payload for item_id: {expected_item_id}",
        )

    embedded = slot_outcome.result
    if embedded.item_id != expected_item_id:
        raise FanOutOrchestrationContractError(
            "orchestration slot outcome item_id does not match topology slot",
        )
    return embedded


def map_orchestration_result_to_fan_out_outcomes(
    request: FanOutRequest[RequestT],
    result: OrchestrationResult[FanOutItemOutcome[ResultT]],
) -> tuple[FanOutItemOutcome[ResultT], ...]:
    """Project orchestration aggregate result to fan-out item outcomes."""
    if len(result.outcomes) != len(request.items):
        raise FanOutOrchestrationContractError(
            "orchestration outcome count must match fan-out request item count",
        )
    return tuple(
        map_orchestration_outcome_to_fan_out(
            slot_outcome,
            expected_item_id=item.item_id,
        )
        for item, slot_outcome in zip(request.items, result.outcomes, strict=True)
    )


@dataclass(frozen=True, slots=True)
class FanOutCoordinationSlotExecutor(Generic[RequestT, ResultT]):
    """Execute one fan-out slot through canonical multi-agent coordination."""

    coordination: MultiAgentCoordinationService[RequestT, ResultT]
    principal: RequestIdentity

    async def execute_slot(
        self,
        *,
        slot_id: OrchestrationSlotId,
        payload: FanOutSlotPayload[RequestT],
    ) -> FanOutItemOutcome[ResultT]:
        item = payload.item
        item_id = to_fan_out_item_id(slot_id)
        if item.item_id != item_id:
            raise FanOutOrchestrationContractError(
                "fan-out slot payload item_id does not match orchestration slot_id",
            )
        try:
            coordination_result = await self.coordination.coordinate(
                item.request,
                delegation=item.delegation,
                principal=self.principal,
            )
        except CoordinationCleanupError as exc:
            return FanOutItemOutcome(
                item_id=item_id,
                status=FanOutItemStatus.FAILURE,
                failure=FanOutItemFailure(
                    failure_code=CoordinationFailureCode.LEASE_RELEASE_FAILED,
                    message=str(exc),
                    partial_result=exc.result,
                ),
            )
        except CoordinationError as exc:
            return FanOutItemOutcome(
                item_id=item_id,
                status=FanOutItemStatus.FAILURE,
                failure=FanOutItemFailure(
                    failure_code=exc.failure_code,
                    message=str(exc),
                ),
            )
        return FanOutItemOutcome(
            item_id=item_id,
            status=FanOutItemStatus.SUCCESS,
            result=coordination_result,
        )


def _build_fan_out_host_task(
    request: FanOutRequest[RequestT],
    principal: RequestIdentity,
):
    user_id = principal.user_id if principal.user_id is not None else "unknown"
    return build_orchestration_topology_host_task(
        tenant_id=principal.tenant_id,
        user_id=user_id,
        task_id=str(request.items[0].request.task_scope_id),
    )


@dataclass(frozen=True, slots=True)
class CanonicalFanOutOrchestrationAdapter(Generic[RequestT, ResultT]):
    """FanOutOrchestrationPort consumer wired to canonical topology submission."""

    topology_submission: OrchestrationTopologySubmissionPort[
        FanOutSlotPayload[RequestT],
        FanOutItemOutcome[ResultT],
    ]
    coordination: MultiAgentCoordinationService[RequestT, ResultT]

    async def orchestrate_fan_out(
        self,
        request: FanOutRequest[RequestT],
        *,
        principal: RequestIdentity,
    ) -> tuple[FanOutItemOutcome[ResultT], ...]:
        topology = project_fan_out_to_topology(request)
        scheduling_policy = project_fan_out_scheduling_policy(request)
        slot_executor = FanOutCoordinationSlotExecutor(
            coordination=self.coordination,
            principal=principal,
        )
        host_task = _build_fan_out_host_task(request, principal)
        governed = ActiveGovernedExecutionTask()
        token = governed.bind(host_task)
        try:
            orchestration_result = await self.topology_submission.submit(
                topology,
                scheduling_policy,
                slot_executor,
            )
        finally:
            governed.reset(token)
        return map_orchestration_result_to_fan_out_outcomes(
            request,
            orchestration_result,
        )


def build_fan_out_orchestration_port(
    topology_submission: OrchestrationTopologySubmissionPort[
        FanOutSlotPayload[RequestT],
        FanOutItemOutcome[ResultT],
    ],
    coordination: MultiAgentCoordinationService[RequestT, ResultT],
) -> FanOutOrchestrationPort[RequestT, ResultT]:
    """Composition-root factory for canonical fan-out orchestration."""
    return CanonicalFanOutOrchestrationAdapter(
        topology_submission=topology_submission,
        coordination=coordination,
    )


__all__ = [
    "CanonicalFanOutOrchestrationAdapter",
    "FanOutCoordinationSlotExecutor",
    "FanOutSlotPayload",
    "build_fan_out_orchestration_port",
    "map_orchestration_outcome_to_fan_out",
    "map_orchestration_result_to_fan_out_outcomes",
    "project_fan_out_scheduling_policy",
    "project_fan_out_to_topology",
    "to_fan_out_item_id",
    "to_orchestration_slot_id",
]
