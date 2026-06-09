# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.capacity.action_gate import CapacityActionGate
from intergrax.runtime.capacity.collector import CapacitySignalCollector
from intergrax.runtime.capacity.contracts import (
    CapacitySignal,
    ScalingAction,
    ScalingActionKind,
    ScalingPolicy,
    ScalingRule,
    ScalingTarget,
)
from intergrax.runtime.capacity.evaluator import ScalingEvaluator
from intergrax.runtime.capacity.provisioner import ScalingProvisioner
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.hooks.hook_point import HookPoint

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_capacity_signal_emits_dedicated_event_type() -> None:
    events: list = []

    def _publish(event) -> None:
        events.append(event)

    collector = CapacitySignalCollector(publish=_publish)
    collector.collect(backpressure_rate=1.0)
    assert events
    assert events[0].event_type is RuntimeEventType.CAPACITY_SIGNAL_COLLECTED


def test_scale_evaluated_event_emitted() -> None:
    events: list = []
    policy = ScalingPolicy(
        enabled=True,
        rules=[
            ScalingRule(
                rule_id="q",
                target=ScalingTarget.CELERY_POOL,
                metric_name="queue_depth",
                scale_up_threshold=5.0,
                scale_down_threshold=1.0,
                action_kind=ScalingActionKind.SCALE_CELERY_WORKERS,
            )
        ],
    )
    evaluator = ScalingEvaluator(
        policy,
        publish=lambda event: events.append(event),
    )
    evaluator.evaluate(
        [
            CapacitySignal(
                target=ScalingTarget.CELERY_POOL,
                metric_name="queue_depth",
                value=8.0,
            )
        ]
    )
    assert any(event.event_type is RuntimeEventType.SCALE_EVALUATED for event in events)


def test_capacity_action_gate_denies_scale_up() -> None:
    gate = CapacityActionGate(
        before_action=lambda action, _point: action.delta <= 0,
    )
    provisioner = ScalingProvisioner(action_gate=gate)
    denied = provisioner.apply(
        ScalingAction(
            kind=ScalingActionKind.SCALE_CELERY_WORKERS,
            target=ScalingTarget.CELERY_POOL,
            delta=1,
        )
    )
    assert denied is False
    assert provisioner.failures


def test_scale_failed_event_on_denied_action() -> None:
    events: list = []
    gate = CapacityActionGate(before_action=lambda _action, _point: False)
    provisioner = ScalingProvisioner(
        action_gate=gate,
        publish=lambda event: events.append(event),
    )
    provisioner.apply(
        ScalingAction(
            kind=ScalingActionKind.SCALE_CELERY_WORKERS,
            target=ScalingTarget.CELERY_POOL,
            delta=1,
        )
    )
    assert any(event.event_type is RuntimeEventType.SCALE_FAILED for event in events)
    assert HookPoint.BEFORE_CAPACITY_ACTION.value == "before_capacity_action"
