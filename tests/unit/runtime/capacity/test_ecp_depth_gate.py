# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.scaling_wiring import wire_application_scaling
from intergrax.runtime.capacity import ScalingTarget
from intergrax.runtime.capacity.ahi_bridge import scaling_action_from_ahi_proposal
from intergrax.runtime.capacity.collector import CapacitySignalCollector
from intergrax.runtime.capacity.contracts import (
    CapacitySignal,
    ScalingAction,
    ScalingActionKind,
    ScalingPolicy,
    ScalingRule,
)
from intergrax.runtime.capacity.evaluator import ScalingEvaluator
from intergrax.runtime.capacity.metrics import export_capacity_metrics, record_scale_action
from intergrax.runtime.capacity.provisioner import ScalingProvisioner

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ecp_package_imports() -> None:
    from intergrax.runtime import capacity  # noqa: F401

    assert ScalingTarget.NEXUS_HOST.value == "nexus_host"


def test_scaling_profile_on_environment() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults()
    assert env.scaling_profile.policy.enabled is False


def test_scaling_evaluator_rule_cooldown_blocks_repeat_action() -> None:
    policy = ScalingPolicy(
        enabled=True,
        rules=[
            ScalingRule(
                rule_id="q",
                target=ScalingTarget.CELERY_POOL,
                metric_name="queue_depth",
                scale_up_threshold=10.0,
                scale_down_threshold=2.0,
                action_kind=ScalingActionKind.SCALE_CELERY_WORKERS,
                cooldown_seconds=300,
            )
        ],
    )
    evaluator = ScalingEvaluator(policy)
    signals = [
        CapacitySignal(
            target=ScalingTarget.CELERY_POOL,
            metric_name="queue_depth",
            value=12.0,
        )
    ]
    first = evaluator.evaluate(signals)
    assert first.evaluation_status == "planned"
    assert len(first.actions) == 1

    second = evaluator.evaluate(signals)
    assert second.evaluation_status == "noop"
    assert not second.actions


def test_scaling_evaluator_cooldown_is_per_rule() -> None:
    policy = ScalingPolicy(
        enabled=True,
        rules=[
            ScalingRule(
                rule_id="q1",
                target=ScalingTarget.CELERY_POOL,
                metric_name="queue_depth",
                scale_up_threshold=10.0,
                scale_down_threshold=2.0,
                action_kind=ScalingActionKind.SCALE_CELERY_WORKERS,
                cooldown_seconds=300,
            ),
            ScalingRule(
                rule_id="q2",
                target=ScalingTarget.NEXUS_HOST,
                metric_name="graph_backpressure_rate",
                scale_up_threshold=1.0,
                scale_down_threshold=0.0,
                action_kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
                cooldown_seconds=300,
            ),
        ],
    )
    evaluator = ScalingEvaluator(policy)
    celery_signal = CapacitySignal(
        target=ScalingTarget.CELERY_POOL,
        metric_name="queue_depth",
        value=12.0,
    )
    backpressure_signal = CapacitySignal(
        target=ScalingTarget.NEXUS_HOST,
        metric_name="graph_backpressure_rate",
        value=2.0,
    )

    first = evaluator.evaluate([celery_signal])
    assert first.evaluation_status == "planned"
    assert len(first.actions) == 1

    second = evaluator.evaluate([celery_signal, backpressure_signal])
    assert second.evaluation_status == "planned"
    assert len(second.actions) == 1
    assert second.actions[0].target is ScalingTarget.NEXUS_HOST


def test_scaling_evaluator_hysteresis() -> None:
    policy = ScalingPolicy(
        enabled=True,
        rules=[
            ScalingRule(
                rule_id="q",
                target=ScalingTarget.CELERY_POOL,
                metric_name="queue_depth",
                scale_up_threshold=10.0,
                scale_down_threshold=2.0,
                action_kind=ScalingActionKind.SCALE_CELERY_WORKERS,
            )
        ],
    )
    evaluator = ScalingEvaluator(policy)
    signals = list(CapacitySignalCollector().collect(backpressure_rate=0.0))
    signals.append(
        CapacitySignal(
            target=ScalingTarget.CELERY_POOL,
            metric_name="queue_depth",
            value=12.0,
        )
    )
    plan = evaluator.evaluate(signals)
    assert plan.evaluation_status == "planned"
    assert plan.actions


def test_kubernetes_provisioner_scale() -> None:
    class _Client:
        replicas = 2

        def get_replicas(self, deployment: str, *, namespace: str) -> int:
            return self.replicas

        def scale_workload(self, deployment: str, *, replicas: int, namespace: str) -> int:
            self.replicas = replicas
            return replicas

        def health(self) -> bool:
            return True

    from intergrax.integrations._shared.p5.clients import KubernetesCloudPlatform

    k8s = KubernetesCloudPlatform(_Client(), namespace="lab")
    provisioner = ScalingProvisioner(kubernetes=k8s)
    from intergrax.runtime.capacity.contracts import ScalingAction

    ok = provisioner.apply(
        ScalingAction(
            kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
            target=ScalingTarget.NEXUS_HOST,
            delta=1,
        ),
        deployment="nexus-host",
    )
    assert ok is True
    assert k8s.get_replicas(deployment="nexus-host") == 3


def test_scaling_wiring_noop_when_disabled() -> None:
    wiring = wire_application_scaling(ApplicationEnvironmentProfile.lab_defaults())
    assert wiring.scheduler is None
    assert wiring.event_bridge is None
    assert wiring.approval_queue is None


async def _publish_backpressure(bus, event) -> None:
    await bus.publish(event)


def test_capacity_event_bridge_records_backpressure() -> None:
    import asyncio

    from intergrax.contracts.execution_phase import ExecutionPhase
    from intergrax.runtime.capacity.event_bridge import CapacityEventBridge
    from intergrax.runtime.events.event_bus import RuntimeEventBus
    from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType

    collector = CapacitySignalCollector()
    bus = RuntimeEventBus(record_history=False)
    bridge = CapacityEventBridge(collector, bus)
    bridge.attach()
    asyncio.run(
        _publish_backpressure(
            bus,
            RuntimeEvent(
                event_type=RuntimeEventType.GRAPH_BACKPRESSURE,
                tenant_id="t1",
                task_id="task-1",
                run_id="run-1",
                phase=ExecutionPhase.STEP_EXECUTION,
            ),
        )
    )
    signals = collector.collect()
    assert signals[0].value >= 1.0
    bridge.detach()


def test_capacity_approval_queue_flow() -> None:
    import asyncio

    from intergrax.runtime.capacity.approval_queue import CapacityApprovalQueue
    from intergrax.runtime.capacity.governance import approve_capacity_plan
    from intergrax.runtime.capacity.scheduler import CapacityScheduler
    from intergrax.runtime.events.runtime_event import RuntimeEventType

    events: list = []
    policy = ScalingPolicy(
        enabled=True,
        require_hitl_for_scale_up=True,
        rules=[
            ScalingRule(
                rule_id="bp",
                target=ScalingTarget.NEXUS_HOST,
                metric_name="graph_backpressure_rate",
                scale_up_threshold=1.0,
                scale_down_threshold=0.0,
                action_kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
            )
        ],
    )

    class _K8s:
        replicas = 2

        def get_replicas(self, *, deployment: str) -> int:
            return self.replicas

        def scale_workload(self, *, deployment: str, replicas: int) -> int:
            self.replicas = replicas
            return replicas

    queue = CapacityApprovalQueue()
    collector = CapacitySignalCollector()
    provisioner = ScalingProvisioner(kubernetes=_K8s())
    scheduler = CapacityScheduler(
        collector=collector,
        evaluator=ScalingEvaluator(policy),
        provisioner=provisioner,
        approval_queue=queue,
        publish=lambda event: events.append(event),
    )
    collector.record_backpressure()
    asyncio.run(scheduler.tick())
    assert queue.list_pending()
    assert any(
        event.event_type is RuntimeEventType.DOMAIN_SIGNAL
        and event.event_kind == "platform.capacity.scale_requested"
        for event in events
    )
    pending_id = queue.list_pending()[0].plan_id
    approve_capacity_plan(queue, pending_id)
    asyncio.run(scheduler.tick())
    assert provisioner.applied


def test_scheduler_skips_hitl_required_plan() -> None:
    import asyncio

    from intergrax.runtime.capacity.scheduler import CapacityScheduler

    policy = ScalingPolicy(
        enabled=True,
        require_hitl_for_scale_up=True,
        rules=[
            ScalingRule(
                rule_id="bp",
                target=ScalingTarget.NEXUS_HOST,
                metric_name="graph_backpressure_rate",
                scale_up_threshold=1.0,
                scale_down_threshold=0.0,
                action_kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
            )
        ],
    )
    collector = CapacitySignalCollector()
    evaluator = ScalingEvaluator(policy)
    provisioner = ScalingProvisioner()
    scheduler = CapacityScheduler(
        collector=collector,
        evaluator=evaluator,
        provisioner=provisioner,
    )
    collector.record_backpressure()
    asyncio.run(scheduler.tick())
    assert not provisioner.applied


def test_celery_provisioner_scale() -> None:
    from intergrax.runtime.capacity.production_adapters import CeleryProductionAdapter

    celery = CeleryProductionAdapter(worker_count=2)
    provisioner = ScalingProvisioner(celery=celery)
    ok = provisioner.apply(
        ScalingAction(
            kind=ScalingActionKind.SCALE_CELERY_WORKERS,
            target=ScalingTarget.CELERY_POOL,
            delta=1,
        )
    )
    assert ok is True
    assert celery.worker_count == 3


def test_ceiling_provisioner_raise() -> None:
    from intergrax.runtime.capacity.ceiling_patcher import BoundedOrchestrationCeilingPatcher

    patcher = BoundedOrchestrationCeilingPatcher(max_inflight_nodes=10, max_raise_percent=20)
    provisioner = ScalingProvisioner(ceiling_patcher=patcher)
    ok = provisioner.apply(
        ScalingAction(
            kind=ScalingActionKind.RAISE_ORCHESTRATION_CEILING,
            target=ScalingTarget.ORCHESTRATION_CEILING,
            delta=1,
        )
    )
    assert ok is True
    assert patcher.max_inflight_nodes == 11


def test_ahi_bridge_action() -> None:
    action = scaling_action_from_ahi_proposal(ceiling_delta=2, reason="approved")
    assert action.delta == 2


def test_pending_queue_depth_provider() -> None:
    from intergrax.distributed.contracts.kv_store import DistributedKVStore
    from intergrax.queueing.contracts.task_queue import TaskStatus
    from intergrax.queueing.task_index import record_task_index
    from intergrax.runtime.capacity.queue_depth import pending_queue_depth

    class _KV(DistributedKVStore):
        def __init__(self) -> None:
            self._data: dict[tuple[str, str], bytes] = {}

        def get(self, tenant_id: str, key: str) -> bytes | None:
            return self._data.get((tenant_id, key))

        def set(
            self,
            tenant_id: str,
            key: str,
            value: bytes,
            *,
            ttl_seconds: int | None = None,
        ) -> None:
            _ = ttl_seconds
            self._data[(tenant_id, key)] = value

        def delete(self, tenant_id: str, key: str) -> None:
            self._data.pop((tenant_id, key), None)

        def compare_and_set(
            self,
            tenant_id: str,
            key: str,
            expected: bytes | None,
            new_value: bytes,
            *,
            ttl_seconds: int | None = None,
        ) -> bool:
            current = self.get(tenant_id, key)
            if current != expected:
                return False
            self.set(tenant_id, key, new_value, ttl_seconds=ttl_seconds)
            return True

    kv = _KV()
    record_task_index(
        kv,
        tenant_id="t1",
        task_id="task-1",
        task_name="demo",
        provider="celery",
        status=TaskStatus.PENDING,
    )
    assert pending_queue_depth(kv, "t1", provider="celery") == 1.0


def test_resolve_kubernetes_backend_in_memory_by_default(monkeypatch) -> None:
    from intergrax.runtime.capacity.production_adapters import resolve_kubernetes_backend

    monkeypatch.delenv("INTERGRAX_KUBERNETES_URL", raising=False)
    _backend, kind = resolve_kubernetes_backend()
    assert kind == "in_memory"


def test_capacity_metrics_export() -> None:
    record_scale_action(target="nexus_host")
    metrics = export_capacity_metrics()
    assert metrics["harness_scale_actions_total"] >= 1.0
