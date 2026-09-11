# © Artur Czarnecki. All rights reserved.

"""Reusable multi-agent coordination qualification harness builders."""

from __future__ import annotations

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    BoundedMultiAgentFanOutService,
)
from intergrax.agent_distribution.delegated_subtasks import (
    DelegatedSubtaskLifecyclePlan,
    DelegationId,
)
from intergrax.agent_distribution.multi_agent_coordination import (
    CoordinationId,
    CoordinationRequest,
    MultiAgentCoordinationService,
)
from intergrax.agent_distribution.task_capability_resolution import (
    build_task_capability_resolution_request,
    unresolved_agent_distribution_capability_need,
)
from intergrax.agent_distribution.task_scoped_agents import TaskScopedAgentLeaseId
from intergrax.contracts.execution_identity import (
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.runtime.execution.boundary import ExecutionIdentityBinding
from intergrax.runtime.execution.fan_out_orchestration_adapter import (
    build_fan_out_orchestration_port,
)
from intergrax.runtime.execution.orchestration_topology_submission import (
    build_orchestration_topology_submission_port,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from testing_support.agent_distribution.delegated_subtask_qualification_harness import (
    DelegatedSubtaskQualificationHarness,
    OcrQualificationRequest,
    OcrQualificationResult,
    build_delegated_subtask_qualification_harness,
)
from testing_support.agent_distribution.task_scoped_agent_qualification_harness import (
    QUALIFICATION_APPLICATION_ID,
    QUALIFICATION_ENVIRONMENT_ID,
    qualification_task_acquire_request,
)


def qualification_root_execution_identity_binding() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def build_multi_agent_coordination_qualification_service(
    harness: DelegatedSubtaskQualificationHarness,
) -> MultiAgentCoordinationService[OcrQualificationRequest, OcrQualificationResult]:
    return MultiAgentCoordinationService(
        delegated_subtasks=harness.service,
    )


def qualification_coordination_request(
    *,
    task_scope: TaskId,
    coordination_id: str = "coordination-1",
    delegation_id: str = "delegation-1",
    lease_id: str = "lease-delegate-1",
    task_kind: str = "document.ocr",
) -> CoordinationRequest:
    return CoordinationRequest(
        coordination_id=CoordinationId(coordination_id),
        delegation_id=delegation_id,
        task_scope_id=task_scope,
        application_id=QUALIFICATION_APPLICATION_ID,
        application_environment_id=QUALIFICATION_ENVIRONMENT_ID,
        lease_id=TaskScopedAgentLeaseId(lease_id),
        capability_need=unresolved_agent_distribution_capability_need(
            build_task_capability_resolution_request(task_kind=task_kind),
        ),
    )


class FanOutQualificationAcquisitionPlanFactory:
    def __init__(self, **kwargs: object) -> None:
        self._kwargs = kwargs
        self._harness: DelegatedSubtaskQualificationHarness | None = None

    def bind_harness(self, harness: DelegatedSubtaskQualificationHarness) -> None:
        self._harness = harness

    def build_acquisition_plan(
        self,
        *,
        delegation_id: DelegationId,
        task_scope_id: TaskId,
        application_id: str,
        application_environment_id: str,
        lease_id: TaskScopedAgentLeaseId,
        selected_identity: object,
    ) -> DelegatedSubtaskLifecyclePlan:
        del delegation_id, application_id, application_environment_id
        prior_revision_id = None
        pointer_revision = 0
        if self._harness is not None:
            serving = self._harness.stack.stack.service.inspect_serving(
                application_id=QUALIFICATION_APPLICATION_ID,
                application_environment_id=QUALIFICATION_ENVIRONMENT_ID,
            )
            prior_revision_id = serving.traffic_serving_revision_id
            pointer_revision = serving.serving_pointer_revision
        revision_id = f"rev-{lease_id}"
        return DelegatedSubtaskLifecyclePlan(
            acquisition_request=qualification_task_acquire_request(
                str(lease_id),
                task_scope_id,
                revision_id,
                identity=selected_identity,
                prior_revision_id=prior_revision_id,
                pointer_revision=pointer_revision,
                **self._kwargs,
            ),
        )


def build_fan_out_delegated_subtask_qualification_harness(
    *,
    candidates: tuple[object, ...],
    specialist_delegate: object | None = None,
    capability_resolver: object | None = None,
    physical_delegation_governance: object | None = None,
) -> DelegatedSubtaskQualificationHarness:
    factory = FanOutQualificationAcquisitionPlanFactory()
    harness = build_delegated_subtask_qualification_harness(
        candidates=candidates,
        specialist_delegate=specialist_delegate,
        acquisition_plan_factory=factory,
        capability_resolver=capability_resolver,
        physical_delegation_governance=physical_delegation_governance,
    )
    factory.bind_harness(harness)
    return harness


def build_bounded_multi_agent_fan_out_qualification_service(
    harness: DelegatedSubtaskQualificationHarness,
) -> BoundedMultiAgentFanOutService[OcrQualificationRequest, OcrQualificationResult]:
    coordination = build_multi_agent_coordination_qualification_service(harness)
    nexus_loop = NexusLoop(AgentRegistry())
    submission_port = build_orchestration_topology_submission_port(nexus_loop)
    orchestration = build_fan_out_orchestration_port(
        submission_port,
        coordination,
    )
    return BoundedMultiAgentFanOutService(orchestration=orchestration)
