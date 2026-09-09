# © Artur Czarnecki. All rights reserved.

"""Reusable NPSC-5C/R3 decision-driven coordination execution qualification helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, fields, replace

from intergrax.agent_distribution.coordination_intent import CoordinationContributionId
from intergrax.agent_distribution.coordination_intent_executor import (
    CoordinationContributionBinding,
    CoordinationIntentBinding,
    CoordinationIntentExecutor,
)
from intergrax.agent_distribution.decision_coordination_projection import (
    coordination_intent_id_from_decision_identity,
    project_authoritative_accepted_decision_coordination,
)
from intergrax.agent_distribution.task_capability_resolution import (
    TaskCapabilityResolutionContractError,
    TaskCapabilityResolutionRequest,
    TaskCapabilityResolutionResult,
    TaskCapabilityResolverId,
)
from intergrax.agent_distribution.task_scoped_agents import TaskScopedAgentLeaseId
from intergrax.contracts.decision_coordination import (
    DecisionCapabilityRequirement,
    DecisionCoordinationContribution,
    DecisionCoordinationSemantic,
    DecisionCoordinationShape,
    decision_coordination_artifact,
    validate_decision_capability_id,
    validate_decision_contribution_id,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
    validate_decision_id,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    decision_lineage_ref,
    decision_version_lineage,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    require_active_execution_id,
)
from tests.unit.agent_distribution.test_bounded_multi_agent_fanout import (
    _build_fan_out_service,
    build_fan_out_harness,
)
from tests.unit.agent_distribution.test_delegated_subtasks import (
    OcrRequest,
    OcrResult,
    _APP,
    _ENV,
    build_delegated_harness,
)
from tests.unit.agent_distribution.test_multi_agent_coordination import (
    _build_coordination_service,
)


class FailingTaskCapabilityResolver:
    """Resolver that fails when invoked — proves pre-resolved decision capabilities."""

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def resolver_id(self) -> TaskCapabilityResolverId:
        return TaskCapabilityResolverId(value="failing.npsc5c-r3")

    def resolve(
        self,
        request: TaskCapabilityResolutionRequest,
    ) -> TaskCapabilityResolutionResult:
        self.call_count += 1
        raise TaskCapabilityResolutionContractError(
            "resolver must not run for decision-derived resolved capability need",
        )


def decision_identity(
    *,
    decision_id: str | None = None,
    version: DecisionVersion | None = None,
) -> DecisionIdentity:
    resolved_decision_id = (
        mint_decision_id()
        if decision_id is None
        else validate_decision_id(decision_id)
    )
    return DecisionIdentity(
        decision_id=resolved_decision_id,
        version=version or initial_decision_version(),
        scope=DecisionScope(namespace="coordination", subject="npsc5c-r3"),
        tenant_id="tenant-a",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )


def decision_capability(capability_id: str = "document.ocr") -> DecisionCapabilityRequirement:
    return DecisionCapabilityRequirement(
        capability_id=validate_decision_capability_id(capability_id),
    )


def decision_contribution(
    contribution_id: str,
    *,
    capability_id: str = "document.ocr",
    document_ref: str | None = None,
) -> DecisionCoordinationContribution[OcrRequest]:
    resolved_ref = document_ref or contribution_id
    return DecisionCoordinationContribution(
        contribution_id=validate_decision_contribution_id(contribution_id),
        capability_requirement=decision_capability(capability_id),
        payload=OcrRequest(document_ref=resolved_ref),
    )


def accepted_decision(
    shape: DecisionCoordinationShape,
    contributions: tuple[DecisionCoordinationContribution[OcrRequest], ...],
    *,
    identity: DecisionIdentity | None = None,
) -> AuthoritativeAcceptedDecision[DecisionCoordinationSemantic[OcrRequest]]:
    resolved_identity = identity or decision_identity()
    semantic = DecisionCoordinationSemantic(shape=shape, contributions=contributions)
    return AuthoritativeAcceptedDecision(
        identity=resolved_identity,
        artifact=decision_coordination_artifact(semantic),
        lineage=decision_version_lineage(
            current=decision_lineage_ref(resolved_identity.version),
        ),
    )


def next_version_accepted_decision(
    prior: AuthoritativeAcceptedDecision[DecisionCoordinationSemantic[OcrRequest]],
) -> AuthoritativeAcceptedDecision[DecisionCoordinationSemantic[OcrRequest]]:
    next_identity = DecisionIdentity(
        decision_id=prior.identity.decision_id,
        version=next_decision_version(prior.identity.version),
        scope=prior.identity.scope,
        tenant_id=prior.identity.tenant_id,
        execution=prior.identity.execution,
    )
    return AuthoritativeAcceptedDecision(
        identity=next_identity,
        artifact=prior.artifact,
        lineage=decision_version_lineage(
            current=decision_lineage_ref(next_identity.version),
            parents=(decision_lineage_ref(prior.identity.version),),
        ),
    )


def coordination_binding(
    task_scope,
    contribution_lease_pairs: tuple[tuple[str, str], ...],
) -> CoordinationIntentBinding:
    return CoordinationIntentBinding(
        task_scope_id=task_scope,
        application_id=_APP,
        application_environment_id=_ENV,
        contribution_bindings=tuple(
            CoordinationContributionBinding(
                contribution_id=CoordinationContributionId(contribution_id),
                lease_id=TaskScopedAgentLeaseId(lease_id),
            )
            for contribution_id, lease_id in contribution_lease_pairs
        ),
    )


@dataclass(frozen=True, slots=True)
class DecisionCoordinationExecutorFixture:
    harness: object
    executor: CoordinationIntentExecutor[OcrRequest, OcrResult]
    failing_resolver: FailingTaskCapabilityResolver
    specialist_child_ids: list[str] | None = None


def _wrap_lineage_specialist(
    delegate,
    specialist_child_ids: list[str],
):
    class _LineageCapturingDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            specialist_child_ids.append(require_active_execution_id())
            return await delegate.execute(request)

    return _LineageCapturingDelegate()


def build_decision_coordination_executor_fixture(
    *,
    candidates,
    specialist_delegate=None,
    fan_out: bool = True,
    track_lineage: bool = False,
) -> DecisionCoordinationExecutorFixture:
    failing_resolver = FailingTaskCapabilityResolver()
    specialist_child_ids: list[str] = []
    resolved_delegate = specialist_delegate
    if track_lineage and specialist_delegate is not None:
        resolved_delegate = _wrap_lineage_specialist(
            specialist_delegate,
            specialist_child_ids,
        )
    harness_builder = build_fan_out_harness if fan_out else build_delegated_harness
    harness = harness_builder(
        candidates=candidates,
        specialist_delegate=resolved_delegate,
        capability_resolver=failing_resolver,
    )
    fan_out_service = _build_fan_out_service(harness)
    coordination = _build_coordination_service(harness)
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=fan_out_service,
    )
    return DecisionCoordinationExecutorFixture(
        harness=harness,
        executor=executor,
        failing_resolver=failing_resolver,
        specialist_child_ids=specialist_child_ids if track_lineage else None,
    )


def project_accepted_decision(
    accepted: AuthoritativeAcceptedDecision[DecisionCoordinationSemantic[OcrRequest]],
):
    return project_authoritative_accepted_decision_coordination(accepted)


def expected_intent_id(identity: DecisionIdentity):
    return coordination_intent_id_from_decision_identity(identity)


def snapshot_accepted_decision(
    accepted: AuthoritativeAcceptedDecision[DecisionCoordinationSemantic[OcrRequest]],
) -> AuthoritativeAcceptedDecision[DecisionCoordinationSemantic[OcrRequest]]:
    return replace(accepted)


def decision_has_no_physical_agent_fields(
    accepted: AuthoritativeAcceptedDecision[DecisionCoordinationSemantic[OcrRequest]],
) -> None:
    forbidden = frozenset(
        {
            "agent_id",
            "agent_instance_id",
            "lease_id",
            "execution_id",
            "execution_strategy",
        },
    )
    for field in fields(accepted):
        assert field.name not in forbidden
    for field in fields(accepted.artifact):
        assert field.name not in forbidden
    semantic = accepted.artifact.content
    for field in fields(semantic):
        assert field.name not in forbidden
    for contribution in semantic.contributions:
        for field in fields(contribution):
            assert field.name not in forbidden
        for field in fields(contribution.capability_requirement):
            assert field.name not in forbidden


class GatedFanOutDelegate:
    """Deterministic fan-out completion ordering via explicit release gates."""

    def __init__(
        self,
        *,
        start_gates: dict[str, asyncio.Event],
        completion_order: list[str],
        all_waiting: asyncio.Event,
        waiting_count: list[int],
        expected_waiters: int,
    ) -> None:
        self._start_gates = start_gates
        self._completion_order = completion_order
        self._all_waiting = all_waiting
        self._waiting_count = waiting_count
        self._expected_waiters = expected_waiters

    async def execute(self, request: OcrRequest) -> OcrResult:
        self._waiting_count[0] += 1
        if self._waiting_count[0] >= self._expected_waiters:
            self._all_waiting.set()
        await self._start_gates[request.document_ref].wait()
        self._completion_order.append(request.document_ref)
        return OcrResult(text=f"ocr:{request.document_ref}")
