# © Artur Czarnecki. All rights reserved.

import asyncio

import pytest

from intergrax.runtime.capacity.collector import CapacitySignalCollector
from intergrax.runtime.capacity.contracts import (
    ScalingActionKind,
    ScalingPolicy,
    ScalingRule,
    ScalingTarget,
)
from intergrax.runtime.capacity.evaluator import ScalingEvaluator
from intergrax.runtime.capacity.provisioner import ScalingProvisioner
from intergrax.runtime.capacity.scheduler import CapacityScheduler

pytestmark = [pytest.mark.integration, pytest.mark.gate]


class _MockKubernetes:
    def __init__(self) -> None:
        self.replicas = 2

    def get_replicas(self, *, deployment: str) -> int:
        return self.replicas

    def scale_workload(self, *, deployment: str, replicas: int) -> int:
        self.replicas = replicas
        return replicas


@pytest.mark.asyncio
async def test_sustained_backpressure_scales_k8s_deployment() -> None:
    policy = ScalingPolicy(
        enabled=True,
        require_hitl_for_scale_up=False,
        rules=[
            ScalingRule(
                rule_id="bp",
                target=ScalingTarget.NEXUS_HOST,
                metric_name="graph_backpressure_rate",
                scale_up_threshold=2.0,
                scale_down_threshold=0.0,
                action_kind=ScalingActionKind.SCALE_K8S_DEPLOYMENT,
                delta=1,
            )
        ],
    )
    collector = CapacitySignalCollector()
    collector.record_backpressure()
    collector.record_backpressure()
    collector.record_backpressure()
    kubernetes = _MockKubernetes()
    scheduler = CapacityScheduler(
        collector=collector,
        evaluator=ScalingEvaluator(policy),
        provisioner=ScalingProvisioner(kubernetes=kubernetes),
    )
    await scheduler.tick()
    assert kubernetes.replicas == 3
