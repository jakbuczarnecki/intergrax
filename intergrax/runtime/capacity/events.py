# © Artur Czarnecki. All rights reserved.

"""Capacity runtime event helpers (ECP-2.4 / ECP-3.4 / ECP-4.3)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.capacity.contracts import CapacitySignal, ScalingAction, ScalingActionPlan
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType

PublishFn = Callable[[RuntimeEvent], None]


def publish_capacity_signal_collected(
    publish: PublishFn,
    signal: CapacitySignal,
    *,
    tenant_id: str = "harness",
) -> None:
    publish(
        RuntimeEvent(
            event_type=RuntimeEventType.CAPACITY_SIGNAL_COLLECTED,
            tenant_id=tenant_id,
            task_id=signal.signal_id,
            run_id=signal.signal_id,
            phase=ExecutionPhase.STEP_EXECUTION,
            payload={
                "target": signal.target.value,
                "metric_name": signal.metric_name,
                "value": signal.value,
            },
        )
    )


def publish_scale_evaluated(
    publish: PublishFn,
    plan: ScalingActionPlan,
    *,
    tenant_id: str = "harness",
    run_id: str = "capacity-eval",
) -> None:
    publish(
        RuntimeEvent(
            event_type=RuntimeEventType.SCALE_EVALUATED,
            tenant_id=tenant_id,
            task_id=run_id,
            run_id=run_id,
            phase=ExecutionPhase.STEP_EXECUTION,
            payload={
                "evaluation_status": plan.evaluation_status,
                "action_count": len(plan.actions),
                "actions": [action.model_dump(mode="json") for action in plan.actions],
            },
        )
    )


def publish_scale_applied(
    publish: PublishFn,
    action: ScalingAction,
    *,
    tenant_id: str = "harness",
    run_id: str = "capacity-apply",
) -> None:
    publish(
        RuntimeEvent(
            event_type=RuntimeEventType.SCALE_APPLIED,
            tenant_id=tenant_id,
            task_id=run_id,
            run_id=run_id,
            phase=ExecutionPhase.STEP_EXECUTION,
            payload=action.model_dump(mode="json"),
        )
    )


def publish_scale_failed(
    publish: PublishFn,
    action: ScalingAction,
    *,
    reason: str,
    tenant_id: str = "harness",
    run_id: str = "capacity-apply",
) -> None:
    publish(
        RuntimeEvent(
            event_type=RuntimeEventType.SCALE_FAILED,
            tenant_id=tenant_id,
            task_id=run_id,
            run_id=run_id,
            phase=ExecutionPhase.STEP_EXECUTION,
            payload={
                **action.model_dump(mode="json"),
                "reason": reason,
            },
        )
    )
