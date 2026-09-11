# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable topology recovery snapshot for partial fan-out resume (NPSC-5E/R3)."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    FanOutItemId,
    FanOutItemOutcome,
    FanOutItemStatus,
    FanOutRequest,
    FanOutResult,
)
from intergrax.contracts.orchestration_topology import OrchestrationTopologyExecutionId
from intergrax.contracts.partial_recovery import SlotRecoveryDisposition

ResultT = TypeVar("ResultT")

CANONICAL_TOPOLOGY_RECOVERY_SCHEMA_VERSION = "topology_recovery.v1"


class FanOutItemOutcomeSnapshot(BaseModel):
    """Provider-neutral serialized fan-out item outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str
    status: str
    result_text: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    has_governed_continuation: bool = False


class SlotRecoverySnapshot(BaseModel):
    """Per-slot durable recovery state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot_id: str
    disposition: SlotRecoveryDisposition
    outcome: FanOutItemOutcomeSnapshot | None = None


class TopologyRecoverySnapshot(BaseModel):
    """Immutable partial fan-out topology recovery state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = CANONICAL_TOPOLOGY_RECOVERY_SCHEMA_VERSION
    fan_out_id: str
    topology_execution_id: str
    slot_order: tuple[str, ...]
    slots: tuple[SlotRecoverySnapshot, ...]
    max_concurrency: int = Field(ge=1, le=64)

    def validate_canonical(self) -> None:
        if self.schema_version != CANONICAL_TOPOLOGY_RECOVERY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported topology recovery schema_version: {self.schema_version!r}"
            )
        if len(self.slot_order) != len(self.slots):
            raise ValueError("slot_order length must match slots length")
        slot_ids = {slot.slot_id for slot in self.slots}
        if len(slot_ids) != len(self.slots):
            raise ValueError("duplicate slot_id in topology recovery snapshot")
        for slot_id in self.slot_order:
            if slot_id not in slot_ids:
                raise ValueError(f"slot_order references unknown slot_id: {slot_id!r}")


def _disposition_from_outcome(
    outcome: FanOutItemOutcome[ResultT],
) -> SlotRecoveryDisposition:
    if outcome.status is FanOutItemStatus.SUCCESS:
        return SlotRecoveryDisposition.SUCCEEDED
    if outcome.failure is None:
        return SlotRecoveryDisposition.UNKNOWN_UNSAFE
    if outcome.failure.continuation is not None:
        return SlotRecoveryDisposition.WAITING_FOR_HUMAN
    return SlotRecoveryDisposition.FAILED


def _result_payload_text(result_payload: object) -> str:
    try:
        text = result_payload.text  # type: ignore[attr-defined]
    except AttributeError:
        return str(result_payload)
    return str(text)


def _outcome_to_snapshot(
    outcome: FanOutItemOutcome[ResultT],
) -> FanOutItemOutcomeSnapshot:
    result_text = None
    if outcome.result is not None and outcome.result.result is not None:
        result_text = _result_payload_text(outcome.result.result)
    failure_code = None
    failure_message = None
    has_continuation = False
    if outcome.failure is not None:
        failure_code = str(outcome.failure.failure_code)
        failure_message = outcome.failure.message
        has_continuation = outcome.failure.continuation is not None
    return FanOutItemOutcomeSnapshot(
        item_id=str(outcome.item_id),
        status=str(outcome.status.value),
        result_text=result_text,
        failure_code=failure_code,
        failure_message=failure_message,
        has_governed_continuation=has_continuation,
    )


def capture_topology_recovery_snapshot(
    *,
    request: FanOutRequest[object],
    result: FanOutResult[ResultT],
    topology_execution_id: OrchestrationTopologyExecutionId,
) -> TopologyRecoverySnapshot:
    """Capture durable partial fan-out state from a completed fan-out invocation."""
    if str(result.fan_out_id) != str(request.fan_out_id):
        raise ValueError("fan_out_id mismatch between request and result")
    if len(result.items) != len(request.items):
        raise ValueError("fan-out result cardinality mismatch")
    slot_order = tuple(str(item.item_id) for item in request.items)
    slots = tuple(
        SlotRecoverySnapshot(
            slot_id=str(outcome.item_id),
            disposition=_disposition_from_outcome(outcome),
            outcome=_outcome_to_snapshot(outcome),
        )
        for outcome in result.items
    )
    snapshot = TopologyRecoverySnapshot(
        fan_out_id=str(request.fan_out_id),
        topology_execution_id=str(topology_execution_id),
        slot_order=slot_order,
        slots=slots,
        max_concurrency=request.max_concurrency,
    )
    snapshot.validate_canonical()
    return snapshot


def failed_slot_ids(snapshot: TopologyRecoverySnapshot) -> tuple[FanOutItemId, ...]:
    """Return item ids for slots that are failed and not waiting for human."""
    return tuple(
        FanOutItemId(slot.slot_id)
        for slot in snapshot.slots
        if slot.disposition is SlotRecoveryDisposition.FAILED
        or slot.disposition is SlotRecoveryDisposition.INTERRUPTED
    )


def successful_slot_ids(snapshot: TopologyRecoverySnapshot) -> tuple[FanOutItemId, ...]:
    return tuple(
        FanOutItemId(slot.slot_id)
        for slot in snapshot.slots
        if slot.disposition is SlotRecoveryDisposition.SUCCEEDED
    )


__all__ = [
    "CANONICAL_TOPOLOGY_RECOVERY_SCHEMA_VERSION",
    "FanOutItemOutcomeSnapshot",
    "SlotRecoverySnapshot",
    "TopologyRecoverySnapshot",
    "capture_topology_recovery_snapshot",
    "failed_slot_ids",
    "successful_slot_ids",
]
