# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical partial fan-out recovery orchestration (NPSC-5E/R3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, cast
from uuid import uuid4

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    FanOutItemId,
    FanOutItemOutcome,
    FanOutItemStatus,
    FanOutRequest,
    FanOutResult,
    validate_fan_out_request,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.execution_identity import ExecutionId
from intergrax.contracts.execution_retry import ExecutionFailureKind
from intergrax.contracts.orchestration_topology import (
    OrchestrationSlotFailure,
    OrchestrationSlotId,
    OrchestrationSlotOutcome,
    OrchestrationSlotRecoveryRequest,
    OrchestrationSlotStatus,
    OrchestrationTopologyExecutionId,
)
from intergrax.contracts.partial_recovery import (
    PartialRecoveryError,
    PartialRecoveryErrorCode,
    PartialRecoveryRequest,
    SlotRecoveryDisposition,
    SlotRecoveryPolicyAction,
    SlotRecoveryPolicyRequest,
    evaluate_slot_recovery_policy,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.execution.fan_out_orchestration_adapter import (
    CanonicalFanOutOrchestrationAdapter,
    FanOutCoordinationSlotExecutor,
    FanOutSlotPayload,
    map_orchestration_result_to_fan_out_outcomes,
    project_fan_out_scheduling_policy,
    project_fan_out_to_topology,
    to_orchestration_slot_id,
)
from intergrax.runtime.execution.orchestration_topology_submission import (
    CanonicalOrchestrationTopologySubmissionPort,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)
from intergrax.runtime.long_running.checkpoint_revision import StaleCheckpointWriteError
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.execution.fan_out_orchestration_adapter import _build_fan_out_host_task
from intergrax.runtime.long_running.topology_recovery_snapshot import (
    TopologyRecoverySnapshot,
    capture_topology_recovery_snapshot,
)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class PartialRecoveryResult(Generic[ResultT]):
    """Typed partial recovery outcome with preserved sibling successes."""

    fan_out_result: FanOutResult[ResultT]
    recovered_slot_ids: tuple[FanOutItemId, ...]
    preserved_slot_ids: tuple[FanOutItemId, ...]
    checkpoint_revision: int


def _failure_kind_from_snapshot(
    snapshot: TopologyRecoverySnapshot,
    slot_id: OrchestrationSlotId,
) -> ExecutionFailureKind | None:
    for slot in snapshot.slots:
        if slot.slot_id != str(slot_id):
            continue
        if slot.outcome is None or slot.outcome.failure_code is None:
            return None
        code = slot.outcome.failure_code
        if code == "child_execution_failed":
            return ExecutionFailureKind.RETRYABLE_TRANSIENT
        if code in {"governance_denied"}:
            return ExecutionFailureKind.GOVERNANCE_DENIED
        if code in {"authority_scope_mismatch"}:
            return ExecutionFailureKind.AUTHORITY_DENIED
        if code in {"governance_requires_human"}:
            return ExecutionFailureKind.UNKNOWN
        return ExecutionFailureKind.NON_RETRYABLE_PERMANENT
    return None


def _slot_disposition(
    snapshot: TopologyRecoverySnapshot,
    slot_id: OrchestrationSlotId,
) -> SlotRecoveryDisposition | None:
    for slot in snapshot.slots:
        if slot.slot_id == str(slot_id):
            return slot.disposition
    return None


def _outcomes_from_fan_out_result(
    request: FanOutRequest[RequestT],
    result: FanOutResult[ResultT],
) -> dict[OrchestrationSlotId, OrchestrationSlotOutcome[FanOutItemOutcome[ResultT]]]:
    topology = project_fan_out_to_topology(request)
    outcomes: dict[OrchestrationSlotId, OrchestrationSlotOutcome[FanOutItemOutcome[ResultT]]] = {}
    for item, fan_out_outcome in zip(request.items, result.items, strict=True):
        slot_id = to_orchestration_slot_id(item.item_id)
        if fan_out_outcome.status is FanOutItemStatus.SUCCESS:
            outcomes[slot_id] = OrchestrationSlotOutcome(
                slot_id=slot_id,
                status=OrchestrationSlotStatus.SUCCESS,
                result=fan_out_outcome,
            )
        else:
            failure = fan_out_outcome.failure
            code = str(failure.failure_code) if failure is not None else "fan_out_failure"
            message = failure.message if failure is not None else "fan-out slot failed"
            outcomes[slot_id] = OrchestrationSlotOutcome(
                slot_id=slot_id,
                status=OrchestrationSlotStatus.FAILURE,
                failure=OrchestrationSlotFailure(code=code, message=message),
            )
    if len(outcomes) != len(topology.slots):
        raise PartialRecoveryError(
            "fan-out outcome cardinality mismatch",
            code=PartialRecoveryErrorCode.WRONG_SLOT,
        )
    return outcomes


@dataclass(slots=True)
class FanOutPartialRecoveryService(Generic[RequestT, ResultT]):
    """Recover exact failed fan-out slots while preserving successful siblings."""

    adapter: CanonicalFanOutOrchestrationAdapter[RequestT, ResultT]
    submission_port: CanonicalOrchestrationTopologySubmissionPort[
        FanOutSlotPayload[RequestT],
        FanOutItemOutcome[ResultT],
    ]
    checkpoint_store: TaskCheckpointPersistence

    def validate_recovery_request(
        self,
        request: PartialRecoveryRequest,
        *,
        fan_out_request: FanOutRequest[RequestT],
        checkpoint: TaskCheckpoint,
        root_execution_id: ExecutionId,
        tenant_id: str,
        policy_decision: PolicyDecision | None = None,
        parent_cancelled: bool = False,
    ) -> TopologyRecoverySnapshot:
        if checkpoint.tenant_id != tenant_id:
            raise PartialRecoveryError(
                "tenant mismatch for partial recovery",
                code=PartialRecoveryErrorCode.TENANT_MISMATCH,
            )
        if checkpoint.revision != request.source_checkpoint_revision:
            raise PartialRecoveryError(
                "checkpoint revision mismatch",
                code=PartialRecoveryErrorCode.WRONG_REVISION,
            )
        if checkpoint.runtime is None or checkpoint.runtime.topology_recovery is None:
            raise PartialRecoveryError(
                "checkpoint missing topology recovery snapshot",
                code=PartialRecoveryErrorCode.STALE_CHECKPOINT,
            )
        snapshot = checkpoint.runtime.topology_recovery
        snapshot.validate_canonical()
        if snapshot.topology_execution_id != str(request.topology_execution_id):
            raise PartialRecoveryError(
                "topology execution id mismatch",
                code=PartialRecoveryErrorCode.WRONG_TOPOLOGY,
            )
        if str(snapshot.fan_out_id) != str(fan_out_request.fan_out_id):
            raise PartialRecoveryError(
                "fan_out_id mismatch",
                code=PartialRecoveryErrorCode.WRONG_TOPOLOGY,
            )
        if checkpoint.runtime.attempt_id != request.source_attempt_id:
            raise PartialRecoveryError(
                "attempt id mismatch",
                code=PartialRecoveryErrorCode.STALE_CHECKPOINT,
            )
        if checkpoint.runtime.execution_tree.entries[0].execution_id != root_execution_id:
            raise PartialRecoveryError(
                "root execution id mismatch",
                code=PartialRecoveryErrorCode.WRONG_ROOT,
            )
        if str(request.slot_id) not in snapshot.slot_order:
            raise PartialRecoveryError(
                "slot not present in topology recovery snapshot",
                code=PartialRecoveryErrorCode.WRONG_SLOT,
            )
        if policy_decision is not None and policy_decision.action is PolicyAction.DENY:
            raise PartialRecoveryError(
                "governance denied partial recovery",
                code=PartialRecoveryErrorCode.GOVERNANCE_DENIED,
            )
        if parent_cancelled:
            raise PartialRecoveryError(
                "parent execution cancelled",
                code=PartialRecoveryErrorCode.PARENT_CANCELLED,
            )
        return snapshot

    def _ensure_execution_record(
        self,
        *,
        fan_out_request: FanOutRequest[RequestT],
        principal: RequestIdentity,
        snapshot: TopologyRecoverySnapshot,
        partial_result: FanOutResult[ResultT],
        execution_id: OrchestrationTopologyExecutionId,
    ) -> None:
        host_task = _build_fan_out_host_task(fan_out_request, principal)
        topology = project_fan_out_to_topology(fan_out_request)
        payloads = {
            to_orchestration_slot_id(item.item_id): FanOutSlotPayload(item=item)
            for item in fan_out_request.items
        }
        outcomes = _outcomes_from_fan_out_result(fan_out_request, partial_result)
        continuable = {
            to_orchestration_slot_id(outcome.item_id)
            for outcome in partial_result.items
            if outcome.status is FanOutItemStatus.FAILURE
            and outcome.failure is not None
            and outcome.failure.continuation is not None
        }
        self.submission_port.restore_execution_record(
            execution_id,
            topology=topology,
            scheduling_policy=project_fan_out_scheduling_policy(
                cast("FanOutRequest[object]", fan_out_request),
            ),
            host_task=host_task,
            payloads=payloads,
            outcomes=outcomes,
            continuable_slots=continuable,
        )

    async def recover_failed_slot(
        self,
        request: PartialRecoveryRequest,
        *,
        fan_out_request: FanOutRequest[RequestT],
        partial_result: FanOutResult[ResultT],
        principal: RequestIdentity,
        checkpoint: TaskCheckpoint,
        root_execution_id: ExecutionId,
        correlation_id: str,
        tenant_id: str,
        policy_decision: PolicyDecision | None = None,
        parent_cancelled: bool = False,
    ) -> PartialRecoveryResult[ResultT]:
        snapshot = self.validate_recovery_request(
            request,
            fan_out_request=fan_out_request,
            checkpoint=checkpoint,
            root_execution_id=root_execution_id,
            tenant_id=tenant_id,
            policy_decision=policy_decision,
            parent_cancelled=parent_cancelled,
        )
        disposition = _slot_disposition(snapshot, request.slot_id)
        if disposition is SlotRecoveryDisposition.SUCCEEDED:
            return PartialRecoveryResult(
                fan_out_result=partial_result,
                recovered_slot_ids=(),
                preserved_slot_ids=tuple(
                    FanOutItemId(slot_id) for slot_id in snapshot.slot_order
                ),
                checkpoint_revision=checkpoint.revision or 1,
            )
        policy = evaluate_slot_recovery_policy(
            SlotRecoveryPolicyRequest(
                disposition=disposition or SlotRecoveryDisposition.UNKNOWN_UNSAFE,
                failure_kind=_failure_kind_from_snapshot(snapshot, request.slot_id),
                waiting_for_human=disposition is SlotRecoveryDisposition.WAITING_FOR_HUMAN,
                parent_cancelled=parent_cancelled,
            ),
        )
        if policy.action is SlotRecoveryPolicyAction.WAIT:
            raise PartialRecoveryError(
                policy.reason,
                code=PartialRecoveryErrorCode.REQUIRE_HUMAN,
            )
        if policy.action is not SlotRecoveryPolicyAction.RECOVER:
            raise PartialRecoveryError(
                policy.reason,
                code=PartialRecoveryErrorCode.SLOT_NOT_RECOVERABLE,
            )

        execution_id = OrchestrationTopologyExecutionId(snapshot.topology_execution_id)
        self._ensure_execution_record(
            fan_out_request=fan_out_request,
            principal=principal,
            snapshot=snapshot,
            partial_result=partial_result,
            execution_id=execution_id,
        )
        host_task = _build_fan_out_host_task(fan_out_request, principal)
        slot_executor = FanOutCoordinationSlotExecutor(
            coordination=self.adapter.coordination,
            principal=principal,
        )
        governed = ActiveGovernedExecutionTask()
        token = governed.bind(host_task)
        try:
            orchestration_result = await self.submission_port.recover_failed_slot(
                OrchestrationSlotRecoveryRequest(
                    execution_id=execution_id,
                    slot_id=request.slot_id,
                    correlation_id=correlation_id,
                    source_checkpoint_revision=request.source_checkpoint_revision,
                ),
                slot_executor=slot_executor,
            )
        finally:
            governed.reset(token)

        recovered_outcomes = map_orchestration_result_to_fan_out_outcomes(
            fan_out_request,
            orchestration_result,
        )
        merged_items = list(partial_result.items)
        for index, item in enumerate(fan_out_request.items):
            if to_orchestration_slot_id(item.item_id) == request.slot_id:
                merged_items[index] = recovered_outcomes[index]
        validate_fan_out_request(cast("FanOutRequest[object]", fan_out_request))
        merged = FanOutResult(
            fan_out_id=fan_out_request.fan_out_id,
            items=tuple(merged_items),
        )
        updated_snapshot = capture_topology_recovery_snapshot(
            request=cast("FanOutRequest[object]", fan_out_request),
            result=merged,
            topology_execution_id=execution_id,
        )
        assert checkpoint.runtime is not None
        updated_runtime = checkpoint.runtime.model_copy(
            update={"topology_recovery": updated_snapshot},
        )
        updated_checkpoint = checkpoint.model_copy(
            update={
                "checkpoint_id": f"ckpt_{uuid4().hex[:16]}",
                "runtime": updated_runtime,
                "progress_message": "partial fan-out recovery committed",
            },
        )
        try:
            saved = self.checkpoint_store.save(
                updated_checkpoint,
                expected_revision=checkpoint.revision,
            )
        except StaleCheckpointWriteError as exc:
            raise PartialRecoveryError(
                str(exc),
                code=PartialRecoveryErrorCode.STALE_CHECKPOINT,
            ) from exc

        preserved = tuple(
            FanOutItemId(slot_id)
            for slot_id in snapshot.slot_order
            if slot_id != str(request.slot_id)
            and _slot_disposition(snapshot, OrchestrationSlotId(slot_id))
            is SlotRecoveryDisposition.SUCCEEDED
        )
        return PartialRecoveryResult(
            fan_out_result=merged,
            recovered_slot_ids=(FanOutItemId(str(request.slot_id)),),
            preserved_slot_ids=preserved,
            checkpoint_revision=saved.revision or request.source_checkpoint_revision,
        )


__all__ = [
    "FanOutPartialRecoveryService",
    "PartialRecoveryResult",
]
