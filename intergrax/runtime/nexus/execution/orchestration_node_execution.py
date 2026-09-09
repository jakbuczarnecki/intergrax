# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Generic orchestration work-node execution port for canonical GraphExecutor scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Mapping, Protocol, TypeVar

from intergrax.contracts.orchestration_topology import (
    OrchestrationSlotExecutor,
    OrchestrationSlotFailure,
    OrchestrationSlotId,
    OrchestrationSlotOutcome,
    OrchestrationSlotStatus,
)

PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")


class OrchestrationNodeExecutionPort(Protocol[PayloadT, ResultT]):
    """Execute one orchestration work node through canonical child execution."""

    async def execute_node(
        self,
        *,
        slot_id: OrchestrationSlotId,
    ) -> OrchestrationSlotOutcome[ResultT]:
        ...


@dataclass(frozen=True, slots=True)
class BoundOrchestrationNodeExecution(Generic[PayloadT, ResultT]):
    """Slot executor bound to an immutable payload map for one topology submission."""

    payloads: Mapping[OrchestrationSlotId, PayloadT]
    slot_executor: OrchestrationSlotExecutor[PayloadT, ResultT]

    async def execute_node(
        self,
        *,
        slot_id: OrchestrationSlotId,
    ) -> OrchestrationSlotOutcome[ResultT]:
        bound_payload = self.payloads[slot_id]
        try:
            result = await self.slot_executor.execute_slot(
                slot_id=slot_id,
                payload=bound_payload,
            )
        except Exception as exc:
            return OrchestrationSlotOutcome(
                slot_id=slot_id,
                status=OrchestrationSlotStatus.FAILURE,
                failure=OrchestrationSlotFailure(
                    code=type(exc).__name__,
                    message=str(exc),
                ),
            )
        return OrchestrationSlotOutcome(
            slot_id=slot_id,
            status=OrchestrationSlotStatus.SUCCESS,
            result=result,
        )


def bind_orchestration_node_execution(
    *,
    payloads: Mapping[OrchestrationSlotId, PayloadT],
    slot_executor: OrchestrationSlotExecutor[PayloadT, ResultT],
) -> BoundOrchestrationNodeExecution[PayloadT, ResultT]:
    return BoundOrchestrationNodeExecution(
        payloads=payloads,
        slot_executor=slot_executor,
    )
