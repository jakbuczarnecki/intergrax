# © Artur Czarnecki. All rights reserved.

"""NPSC-5D/R2 — physical delegation governance boundary tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.agent_distribution.agent_selection import (
    AgentSelectionContext,
    AgentSelectionRequest,
    AgentSelectionStrategy,
    DeterministicIdentitySelectionStrategy,
    require_selected_identity,
)
from intergrax.agent_distribution.delegated_subtasks import (
    DelegatedSubtaskDelegate,
    DelegatedSubtaskGovernanceDenied,
    DelegatedSubtaskGovernanceRequiresHuman,
    DelegatedSubtaskInvocation,
    DelegatedSubtaskService,
    SpecialistInvocationPort,
)
from intergrax.agent_distribution.bounded_multi_agent_fanout import FanOutItemFailure
from intergrax.agent_distribution.multi_agent_coordination import (
    CoordinationDelegation,
    CoordinationFailureCode,
    GovernanceDeniedError,
    GovernanceRequiresHumanError,
)
from intergrax.agent_distribution.task_capability_resolution import (
    resolved_agent_distribution_capability_need,
)
from intergrax.agent_distribution.capability_matching import build_agent_capability_requirement
from intergrax.contracts.physical_delegation_governance import (
    PhysicalDelegationGovernancePort,
    PhysicalDelegationGovernanceRequest,
    PhysicalDelegationGovernanceResult,
    PhysicalDelegationGovernedContinuation,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.runtime.execution.boundary import ExecutionBoundary
from intergrax.runtime.governance.physical_delegation_governance import (
    DenyingPhysicalDelegationGovernance,
    PhysicalDelegationGovernanceBoundary,
    RequireHumanPhysicalDelegationGovernance,
    RuntimePhysicalDelegationGovernance,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from testing_support.agent_distribution.coordination_governance import (
    allowing_physical_delegation_governance,
    denying_physical_delegation_governance,
    require_human_physical_delegation_governance,
    unavailable_physical_delegation_governance,
)
from tests.unit.agent_distribution.test_delegated_subtasks import (
    OcrRequest,
    OcrResult,
    _LEGAL_PACKAGE,
    _OCR_PACKAGE,
    _delegated_request,
    _discovery_candidate,
    _root_identity,
    _run_delegation,
    admin_test_principal,
    build_delegated_harness,
)
from tests.unit.agent_distribution.test_multi_agent_coordination import (
    _build_coordination_service,
    _coordination_request,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DELEGATED_SOURCE = _REPO_ROOT / "intergrax" / "agent_distribution" / "delegated_subtasks.py"


class _CountingSelector(AgentSelectionStrategy):
    def __init__(self, inner: AgentSelectionStrategy) -> None:
        self._inner = inner
        self.call_count = 0
        self.last_request: AgentSelectionRequest | None = None

    def select(self, request: AgentSelectionRequest):
        self.call_count += 1
        self.last_request = request
        return self._inner.select(request)


class _CountingTaskScopedAgents:
    def __init__(self, inner) -> None:
        self._inner = inner
        self.acquire_count = 0
        self.release_count = 0

    def acquire(self, *args, **kwargs):
        self.acquire_count += 1
        return self._inner.acquire(*args, **kwargs)

    def release(self, *args, **kwargs):
        self.release_count += 1
        return self._inner.release(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class _CountingSpecialistInvocation(SpecialistInvocationPort[OcrRequest, OcrResult]):
    def __init__(self, inner: SpecialistInvocationPort[OcrRequest, OcrResult]) -> None:
        self._inner = inner
        self.call_count = 0

    def resolve_delegate(self, **kwargs) -> DelegatedSubtaskDelegate[OcrRequest, OcrResult]:
        self.call_count += 1
        return self._inner.resolve_delegate(**kwargs)


class _CountingChildExecution:
    def __init__(self, inner) -> None:
        self._inner = inner
        self.call_count = 0

    async def execute_child(self, **kwargs):
        self.call_count += 1
        return await self._inner.execute_child(**kwargs)


class _RecordingGovernance(PhysicalDelegationGovernancePort):
    def __init__(self, inner: PhysicalDelegationGovernancePort) -> None:
        self._inner = inner
        self.call_count = 0
        self.last_request: PhysicalDelegationGovernanceRequest | None = None

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PhysicalDelegationGovernanceResult:
        self.call_count += 1
        self.last_request = request
        return self._inner.evaluate(request)


class _PackageDenyingGovernance(PhysicalDelegationGovernancePort):
    def __init__(self, *, denied_package_id: str) -> None:
        self._denied_package_id = denied_package_id

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PhysicalDelegationGovernanceResult:
        if (
            request.selected_identity.distribution_package_id
            == self._denied_package_id
        ):
            return DenyingPhysicalDelegationGovernance().evaluate(request)
        return allowing_physical_delegation_governance().evaluate(request)


class _DelegationDenyingGovernance(PhysicalDelegationGovernancePort):
    def __init__(self, *, denied_delegation_id: str) -> None:
        self._denied_delegation_id = denied_delegation_id

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PhysicalDelegationGovernanceResult:
        if request.delegation_id == self._denied_delegation_id:
            return DenyingPhysicalDelegationGovernance().evaluate(request)
        return allowing_physical_delegation_governance().evaluate(request)


class _DelegationRequireHumanGovernance(PhysicalDelegationGovernancePort):
    def __init__(self, *, require_human_delegation_id: str) -> None:
        self._require_human_delegation_id = require_human_delegation_id

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PhysicalDelegationGovernanceResult:
        if request.delegation_id == self._require_human_delegation_id:
            return require_human_physical_delegation_governance().evaluate(request)
        return allowing_physical_delegation_governance().evaluate(request)


class _ModifyGovernance(PhysicalDelegationGovernancePort):
    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PhysicalDelegationGovernanceResult:
        boundary = PhysicalDelegationGovernanceBoundary(
            evaluator=_StaticEvaluator(
                PolicyDecision(
                    action=PolicyAction.MODIFY,
                    reason="modify_requested",
                    policy_rule_id="test.physical_delegation.modify",
                ),
            ),
        )
        return boundary.evaluate(request)


class _StaticEvaluator:
    def __init__(self, decision: PolicyDecision) -> None:
        self._decision = decision

    def evaluate(self, request: PhysicalDelegationGovernanceRequest) -> PolicyDecision:
        del request
        return self._decision


def _build_instrumented_harness(
    *,
    candidates: tuple,
    governance: PhysicalDelegationGovernancePort,
    selector: AgentSelectionStrategy | None = None,
):
    harness = build_delegated_harness(
        candidates=candidates,
        physical_delegation_governance=governance,
    )
    counting_selector = _CountingSelector(
        selector or DeterministicIdentitySelectionStrategy(),
    )
    counting_task_scoped = _CountingTaskScopedAgents(harness.task_scoped)
    counting_specialist = _CountingSpecialistInvocation(
        harness.service._specialist_invocation,
    )
    counting_child = _CountingChildExecution(harness.service._child_execution)
    recording_governance = _RecordingGovernance(governance)
    service = DelegatedSubtaskService(
        capability_resolver=harness.service._capability_resolver,
        discovery=harness.service._discovery,
        matcher=harness.service._matcher,
        selector=counting_selector,
        task_scoped_agents=counting_task_scoped,
        task_scope_authority=harness.task_scope_authority,
        acquisition_plan_factory=harness.service._acquisition_plan_factory,
        release_plan_factory=harness.service._release_plan_factory,
        specialist_invocation=counting_specialist,
        child_execution=counting_child,
        physical_delegation_governance=recording_governance,
    )
    harness.service = service
    return harness, counting_selector, counting_task_scoped, counting_specialist, counting_child, recording_governance


@pytest.mark.asyncio
async def test_single_allow_acquires_and_executes_once() -> None:
    harness, _, task_scoped, specialist, child, _ = _build_instrumented_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        governance=allowing_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    result = await _run_delegation(
        harness,
        task_scope=task_scope,
    )
    assert result.result.text == "ocr:doc-1"
    assert task_scoped.acquire_count == 1
    assert specialist.call_count == 1
    assert child.call_count == 1


@pytest.mark.asyncio
async def test_single_deny_blocks_before_acquisition() -> None:
    harness, _, task_scoped, specialist, child, governance = _build_instrumented_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        governance=denying_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    with pytest.raises(DelegatedSubtaskGovernanceDenied):
        await _run_delegation(
            harness,
            task_scope=task_scope,
        )
    assert governance.call_count == 1
    assert governance.last_request is not None
    assert (
        governance.last_request.selected_identity.distribution_package_id
        == _OCR_PACKAGE
    )
    assert task_scoped.acquire_count == 0
    assert specialist.call_count == 0
    assert child.call_count == 0
    assert task_scoped.release_count == 0


@pytest.mark.asyncio
async def test_no_fallback_selection_after_deny() -> None:
    harness, selector, task_scoped, specialist, child, governance = _build_instrumented_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
            _discovery_candidate(_LEGAL_PACKAGE, capability_ids=("document.ocr",)),
        ),
        governance=denying_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    with pytest.raises(DelegatedSubtaskGovernanceDenied):
        await _run_delegation(
            harness,
            task_scope=task_scope,
        )
    assert selector.call_count == 1
    assert governance.last_request is not None
    selected = require_selected_identity(selector._inner.select(selector.last_request))
    from intergrax.agent_distribution.physical_delegation_governance_adapter import (
        project_physical_delegation_selected_identity,
    )

    assert governance.last_request.selected_identity == project_physical_delegation_selected_identity(
        selected,
    )
    assert governance.last_request.selected_identity.distribution_package_id == _OCR_PACKAGE
    assert task_scoped.acquire_count == 0
    assert specialist.call_count == 0
    assert child.call_count == 0


@pytest.mark.asyncio
async def test_governance_request_binds_capability_requirement() -> None:
    harness, _, _, _, _, governance = _build_instrumented_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        governance=allowing_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    await _run_delegation(
        harness,
        task_scope=task_scope,
    )
    assert governance.last_request is not None
    from intergrax.agent_distribution.physical_delegation_governance_adapter import (
        project_physical_delegation_capability_requirement,
    )

    assert governance.last_request.capability_requirement == (
        project_physical_delegation_capability_requirement(
            build_agent_capability_requirement(required=("document.ocr",)),
        )
    )


@pytest.mark.asyncio
async def test_pre_resolved_capability_does_not_rerun_resolver() -> None:
    from tests.unit.agent_distribution.test_delegated_subtasks import _FailingResolver

    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        capability_resolver=_FailingResolver(),
        physical_delegation_governance=allowing_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    root = _root_identity()
    captured: list[object] = []

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            delegated_request = _delegated_request(task_scope=task_scope).model_copy(
                update={
                    "capability_need": resolved_agent_distribution_capability_need(
                        build_agent_capability_requirement(required=("document.ocr",)),
                    ),
                },
            )
            result = await harness.service.execute(
                delegated_request,
                invocation=DelegatedSubtaskInvocation(payload=request),
                principal=admin_test_principal(),
            )
            captured.append(result)
            return result.result

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="doc-1"))
    assert captured
    assert captured[0].result.text == "ocr:doc-1"


@pytest.mark.asyncio
async def test_policy_unavailable_fail_closed() -> None:
    harness, _, task_scoped, specialist, child, _ = _build_instrumented_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        governance=unavailable_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    with pytest.raises(DelegatedSubtaskGovernanceDenied):
        await _run_delegation(
            harness,
            task_scope=task_scope,
        )
    assert task_scoped.acquire_count == 0
    assert specialist.call_count == 0
    assert child.call_count == 0


@pytest.mark.asyncio
async def test_modify_fail_closed_without_acquisition() -> None:
    harness, _, task_scoped, specialist, child, _ = _build_instrumented_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        governance=_ModifyGovernance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    with pytest.raises(DelegatedSubtaskGovernanceDenied):
        await _run_delegation(
            harness,
            task_scope=task_scope,
        )
    assert task_scoped.acquire_count == 0
    assert specialist.call_count == 0
    assert child.call_count == 0


@pytest.mark.asyncio
async def test_require_human_blocks_before_acquisition() -> None:
    harness, _, task_scoped, specialist, child, _ = _build_instrumented_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        governance=require_human_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    with pytest.raises(DelegatedSubtaskGovernanceRequiresHuman) as exc_info:
        await _run_delegation(
            harness,
            task_scope=task_scope,
        )
    assert exc_info.value.result.requires_governed_continuation is True
    assert task_scoped.acquire_count == 0
    assert specialist.call_count == 0
    assert child.call_count == 0


@pytest.mark.asyncio
async def test_coordination_maps_governance_denied() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        physical_delegation_governance=denying_physical_delegation_governance(),
    )
    coordination = _build_coordination_service(harness)
    task_scope = harness.task_scope_authority.task_scope_id
    root = _root_identity()

    class _Root:
        async def execute(self, payload: OcrRequest) -> OcrResult:
            return (
                await coordination.coordinate(
                    _coordination_request(task_scope=task_scope),
                    delegation=CoordinationDelegation(payload=payload),
                    principal=admin_test_principal(),
                )
            ).result

    with pytest.raises(GovernanceDeniedError) as exc_info:
        await ExecutionBoundary[OcrRequest, OcrResult](
            _Root(),
            identity=root,
        ).execute(OcrRequest(document_ref="doc-1"))
    assert exc_info.value.failure_code is CoordinationFailureCode.GOVERNANCE_DENIED


@pytest.mark.asyncio
async def test_coordination_maps_governance_requires_human() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        physical_delegation_governance=require_human_physical_delegation_governance(),
    )
    coordination = _build_coordination_service(harness)
    task_scope = harness.task_scope_authority.task_scope_id
    root = _root_identity()

    class _Root:
        async def execute(self, payload: OcrRequest) -> OcrResult:
            return (
                await coordination.coordinate(
                    _coordination_request(task_scope=task_scope),
                    delegation=CoordinationDelegation(payload=payload),
                    principal=admin_test_principal(),
                )
            ).result

    with pytest.raises(GovernanceRequiresHumanError) as exc_info:
        await ExecutionBoundary[OcrRequest, OcrResult](
            _Root(),
            identity=root,
        ).execute(OcrRequest(document_ref="doc-1"))
    assert exc_info.value.failure_code is CoordinationFailureCode.GOVERNANCE_REQUIRES_HUMAN
    continuation = exc_info.value.continuation
    assert isinstance(continuation, PhysicalDelegationGovernedContinuation)
    assert continuation.selected_identity.distribution_package_id == _OCR_PACKAGE
    assert continuation.governance_result.requires_governed_continuation is True
    assert continuation.governance_result.evidence.request_digest.startswith("sha256:")


@pytest.mark.asyncio
async def test_coordination_continuation_no_cause_walk() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        physical_delegation_governance=require_human_physical_delegation_governance(),
    )
    coordination = _build_coordination_service(harness)
    task_scope = harness.task_scope_authority.task_scope_id
    root = _root_identity()

    class _Root:
        async def execute(self, payload: OcrRequest) -> OcrResult:
            return (
                await coordination.coordinate(
                    _coordination_request(task_scope=task_scope),
                    delegation=CoordinationDelegation(payload=payload),
                    principal=admin_test_principal(),
                )
            ).result

    with pytest.raises(GovernanceRequiresHumanError) as exc_info:
        await ExecutionBoundary[OcrRequest, OcrResult](
            _Root(),
            identity=root,
        ).execute(OcrRequest(document_ref="doc-1"))
    outer = exc_info.value
    assert outer.__cause__ is not None
    assert not isinstance(outer.__cause__, PhysicalDelegationGovernedContinuation)
    assert outer.continuation.delegation_id == outer.continuation.governance_result.evidence.delegation_id


@pytest.mark.asyncio
async def test_require_human_preserves_exact_selected_identity() -> None:
    harness, selector, task_scoped, specialist, child, governance = _build_instrumented_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
            _discovery_candidate(_LEGAL_PACKAGE, capability_ids=("document.ocr",)),
        ),
        governance=require_human_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    with pytest.raises(DelegatedSubtaskGovernanceRequiresHuman) as exc_info:
        await _run_delegation(
            harness,
            task_scope=task_scope,
        )
    from intergrax.agent_distribution.physical_delegation_governance_adapter import (
        project_physical_delegation_selected_identity,
    )

    selected = require_selected_identity(selector._inner.select(selector.last_request))
    expected_identity = project_physical_delegation_selected_identity(selected)
    assert governance.last_request is not None
    assert exc_info.value.continuation.selected_identity == expected_identity
    assert exc_info.value.continuation.selected_identity.distribution_package_id == _OCR_PACKAGE
    assert selector.call_count == 1
    assert task_scoped.acquire_count == 0
    assert specialist.call_count == 0
    assert child.call_count == 0


@pytest.mark.asyncio
async def test_deny_has_no_continuation_on_coordination_boundary() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        physical_delegation_governance=denying_physical_delegation_governance(),
    )
    coordination = _build_coordination_service(harness)
    task_scope = harness.task_scope_authority.task_scope_id
    root = _root_identity()

    class _Root:
        async def execute(self, payload: OcrRequest) -> OcrResult:
            return (
                await coordination.coordinate(
                    _coordination_request(task_scope=task_scope),
                    delegation=CoordinationDelegation(payload=payload),
                    principal=admin_test_principal(),
                )
            ).result

    with pytest.raises(GovernanceDeniedError) as exc_info:
        await ExecutionBoundary[OcrRequest, OcrResult](
            _Root(),
            identity=root,
        ).execute(OcrRequest(document_ref="doc-1"))
    assert exc_info.value.failure_code is CoordinationFailureCode.GOVERNANCE_DENIED
    assert not hasattr(exc_info.value, "continuation")


def test_fan_out_item_failure_rejects_inconsistent_contract() -> None:
    from intergrax.contracts.physical_delegation_governance import (
        PhysicalDelegationCapabilityRequirement,
        PhysicalDelegationGovernanceRequest,
        PhysicalDelegationSelectedIdentity,
        build_physical_delegation_governed_continuation,
    )

    with pytest.raises(ValueError, match="requires continuation payload"):
        FanOutItemFailure(
            failure_code=CoordinationFailureCode.GOVERNANCE_REQUIRES_HUMAN,
            message="requires human",
        )
    governance_request = PhysicalDelegationGovernanceRequest(
        delegation_id="delegation-1",
        task_scope_id="task-scope-1",
        application_id="app-a",
        application_environment_id="env-a",
        principal=admin_test_principal(),
        capability_requirement=PhysicalDelegationCapabilityRequirement(
            required_capability_ids=("document.ocr",),
        ),
        selected_identity=PhysicalDelegationSelectedIdentity(
            catalog_source_id="builtin-1",
            provider_kind="builtin",
            distribution_package_id=_OCR_PACKAGE,
            package_version="1.0.0",
            package_digest="sha256:abc",
        ),
    )
    governance_result = RequireHumanPhysicalDelegationGovernance().evaluate(
        governance_request,
    )
    continuation = build_physical_delegation_governed_continuation(
        request=governance_request,
        governance_result=governance_result,
    )
    with pytest.raises(ValueError, match="only allowed for GOVERNANCE_REQUIRES_HUMAN"):
        FanOutItemFailure(
            failure_code=CoordinationFailureCode.GOVERNANCE_DENIED,
            message="denied",
            continuation=continuation,
        )


@pytest.mark.asyncio
async def test_fan_out_mixed_allow_require_human_allow_preserves_siblings() -> None:
    from intergrax.agent_distribution.bounded_multi_agent_fanout import (
        FanOutId,
        FanOutItemId,
        FanOutItemStatus,
        FanOutRequest,
    )
    from tests.unit.agent_distribution.test_bounded_multi_agent_fanout import (
        _FanOutAcquisitionPlanFactory,
        _fan_out_item,
        _run_fan_out,
    )

    factory = _FanOutAcquisitionPlanFactory()
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
            _discovery_candidate(_LEGAL_PACKAGE, capability_ids=("document.ocr",)),
        ),
        acquisition_plan_factory=factory,
        physical_delegation_governance=_DelegationRequireHumanGovernance(
            require_human_delegation_id="delegation-b",
        ),
    )
    factory.bind_harness(harness)
    task_scope = harness.task_scope_authority.task_scope_id
    result = await _run_fan_out(
        harness,
        task_scope=task_scope,
        items=(
            _fan_out_item(
                item_id="a",
                task_scope=task_scope,
                coordination_id="coord-a",
                delegation_id="delegation-a",
                lease_id="lease-a",
                document_ref="doc-a",
            ),
            _fan_out_item(
                item_id="b",
                task_scope=task_scope,
                coordination_id="coord-b",
                delegation_id="delegation-b",
                lease_id="lease-b",
                document_ref="doc-b",
            ),
            _fan_out_item(
                item_id="c",
                task_scope=task_scope,
                coordination_id="coord-c",
                delegation_id="delegation-c",
                lease_id="lease-c",
                document_ref="doc-c",
            ),
        ),
        fan_out_id="fan-out-require-human",
        max_concurrency=3,
    )
    assert len(result.items) == 3
    assert [item.item_id for item in result.items] == [
        FanOutItemId("a"),
        FanOutItemId("b"),
        FanOutItemId("c"),
    ]
    assert result.items[0].status is FanOutItemStatus.SUCCESS
    assert result.items[1].status is FanOutItemStatus.FAILURE
    assert result.items[1].failure is not None
    assert (
        result.items[1].failure.failure_code
        is CoordinationFailureCode.GOVERNANCE_REQUIRES_HUMAN
    )
    continuation = result.items[1].failure.continuation
    assert continuation is not None
    assert continuation.delegation_id == "delegation-b"
    assert continuation.governance_result.requires_governed_continuation is True
    assert result.items[2].status is FanOutItemStatus.SUCCESS


@pytest.mark.asyncio
async def test_decision_backed_coordination_preserves_r2_continuation() -> None:
    from intergrax.agent_distribution.bounded_multi_agent_fanout import (
        BoundedMultiAgentFanOutService,
    )
    from intergrax.agent_distribution.coordination_intent import (
        CoordinationContribution,
        CoordinationContributionId,
        CoordinationExecutionMode,
        CoordinationIntent,
        CoordinationIntentId,
    )
    from intergrax.agent_distribution.coordination_intent_executor import (
        CoordinationIntentExecutor,
    )
    from intergrax.agent_distribution.task_capability_resolution import (
        build_task_capability_resolution_request,
        unresolved_agent_distribution_capability_need,
    )
    from intergrax.contracts.delegation_authority import ParentExecutionAuthority
    from testing_support.agent_distribution.coordination_governance import (
        allowing_coordination_governance,
        bound_governed_host_task,
    )
    from tests.unit.agent_distribution.test_coordination_intent_executor import (
        _StaticOrchestrationPort,
        _binding,
    )

    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        physical_delegation_governance=require_human_physical_delegation_governance(),
    )
    coordination = _build_coordination_service(harness)
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=BoundedMultiAgentFanOutService(orchestration=_StaticOrchestrationPort(())),
        governance=allowing_coordination_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    harness.task_scope_authority.task_scope_id = task_scope
    root = _root_identity()
    captured: list[GovernanceRequiresHumanError] = []

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            del request
            with bound_governed_host_task():
                try:
                    await executor.execute(
                        CoordinationIntent(
                            intent_id=CoordinationIntentId("intent-r2-h1"),
                            mode=CoordinationExecutionMode.SINGLE,
                            contributions=(
                                CoordinationContribution(
                                    contribution_id=CoordinationContributionId("contrib-a"),
                                    payload=OcrRequest(document_ref="doc-r2-h1"),
                                    capability_need=unresolved_agent_distribution_capability_need(
                                        build_task_capability_resolution_request(
                                            task_kind="document.ocr",
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        binding=_binding(task_scope, pairs=(("contrib-a", "lease-a"),)),
                        principal=admin_test_principal(),
                    )
                except GovernanceRequiresHumanError as exc:
                    captured.append(exc)
                    raise
            raise AssertionError("expected GovernanceRequiresHumanError")

    with pytest.raises(GovernanceRequiresHumanError):
        await ExecutionBoundary[OcrRequest, OcrResult](
            RootDelegate(),
            identity=root,
            authority=ParentExecutionAuthority.unrestricted_root(),
        ).execute(OcrRequest(document_ref="doc-r2-h1"))
    assert captured
    continuation = captured[0].continuation
    assert continuation.delegation_id
    assert continuation.selected_identity.distribution_package_id == _OCR_PACKAGE


def test_architecture_gate_selection_before_governance_before_acquire() -> None:
    source = _DELEGATED_SOURCE.read_text(encoding="utf-8-sig")
    source_lines = source.splitlines()
    select_idx = next(
        i for i, line in enumerate(source_lines) if "self._selector.select" in line
    )
    governance_idx = next(
        i
        for i, line in enumerate(source_lines)
        if "_enforce_physical_delegation_governance" in line
    )
    acquire_idx = next(
        i for i, line in enumerate(source_lines) if "self._task_scoped_agents.acquire" in line
    )
    assert select_idx < governance_idx < acquire_idx


@pytest.mark.asyncio
async def test_fan_out_mixed_allow_deny_preserves_sibling_isolation() -> None:
    from intergrax.agent_distribution.bounded_multi_agent_fanout import (
        FanOutId,
        FanOutItemStatus,
        FanOutRequest,
    )
    from tests.unit.agent_distribution.test_bounded_multi_agent_fanout import (
        _FanOutAcquisitionPlanFactory,
        _fan_out_item,
        _run_fan_out,
    )

    factory = _FanOutAcquisitionPlanFactory()
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
            _discovery_candidate(_LEGAL_PACKAGE, capability_ids=("document.ocr",)),
        ),
        acquisition_plan_factory=factory,
        physical_delegation_governance=_DelegationDenyingGovernance(
            denied_delegation_id="delegation-b",
        ),
    )
    factory.bind_harness(harness)
    task_scope = harness.task_scope_authority.task_scope_id
    result = await _run_fan_out(
        harness,
        task_scope=task_scope,
        items=(
            _fan_out_item(
                item_id="a",
                task_scope=task_scope,
                coordination_id="coord-a",
                delegation_id="delegation-a",
                lease_id="lease-a",
                document_ref="doc-a",
            ),
            _fan_out_item(
                item_id="b",
                task_scope=task_scope,
                coordination_id="coord-b",
                delegation_id="delegation-b",
                lease_id="lease-b",
                document_ref="doc-b",
            ),
            _fan_out_item(
                item_id="c",
                task_scope=task_scope,
                coordination_id="coord-c",
                delegation_id="delegation-c",
                lease_id="lease-c",
                document_ref="doc-c",
            ),
        ),
        fan_out_id="fan-out-mixed",
        max_concurrency=3,
    )
    assert len(result.items) == 3
    assert result.items[0].status is FanOutItemStatus.SUCCESS
    assert result.items[1].status is FanOutItemStatus.FAILURE
    assert result.items[1].failure is not None
    assert result.items[1].failure.failure_code is CoordinationFailureCode.GOVERNANCE_DENIED
    assert result.items[2].status is FanOutItemStatus.SUCCESS


@pytest.mark.asyncio
async def test_runtime_evaluator_unconfigured_fail_closed() -> None:
    governance = RuntimePhysicalDelegationGovernance(policy_engine=RuntimePolicyEngine())
    harness, _, task_scoped, _, _, _ = _build_instrumented_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        governance=governance,
    )
    task_scope = harness.task_scope_authority.task_scope_id
    with pytest.raises(DelegatedSubtaskGovernanceDenied):
        await _run_delegation(
            harness,
            task_scope=task_scope,
        )
    assert task_scoped.acquire_count == 0
