# © Artur Czarnecki. All rights reserved.

"""NPSC-5D/R2 Final — cross-layer physical delegation governance qualification."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

import pytest

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    BoundedMultiAgentFanOutService,
    FanOutItemFailure,
    FanOutItemStatus,
)
from intergrax.agent_distribution.capability_matching import build_agent_capability_requirement
from intergrax.agent_distribution.coordination_intent import CoordinationExecutionMode
from intergrax.agent_distribution.coordination_intent_executor import (
    CoordinationGovernanceDenied,
    CoordinationIntentExecutor,
)
from intergrax.agent_distribution.errors import AgentPackageTrustError
from intergrax.agent_distribution.delegated_subtasks import (
    DelegatedSubtaskGovernanceDenied,
    DelegatedSubtaskGovernanceRequiresHuman,
    DelegatedSubtaskInvocation,
    DelegatedSubtaskLifecyclePlan,
    DelegatedSubtaskResult,
    DelegatedSubtaskService,
    DelegationId,
)
from intergrax.agent_distribution.physical_delegation_governance_adapter import (
    build_physical_delegation_governance_request,
)
from intergrax.agent_distribution.multi_agent_coordination import (
    CoordinationDelegation,
    CoordinationFailureCode,
    GovernanceRequiresHumanError,
)
from intergrax.agent_distribution.task_capability_resolution import (
    resolved_agent_distribution_capability_need,
)
from intergrax.contracts.control_plane_mutation import GovernanceEvaluationPoint
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.physical_delegation_governance import (
    PhysicalDelegationGovernancePort,
    PhysicalDelegationGovernanceRequest,
    PhysicalDelegationGovernanceResult,
    PhysicalDelegationGovernedContinuation,
    build_physical_delegation_governed_continuation,
    physical_delegation_governance_request_digest,
)
from intergrax.agent_distribution.task_scoped_agents import TaskScopedAgentLeaseId
from intergrax.runtime.execution.boundary import ExecutionBoundary
from intergrax.runtime.governance.physical_delegation_governance import (
    RequireHumanPhysicalDelegationGovernance,
)
from testing_support.agent_distribution.coordination_governance import (
    allowing_physical_delegation_governance,
    bound_governed_host_task,
    denying_coordination_governance,
    denying_physical_delegation_governance,
    require_human_physical_delegation_governance,
)
from testing_support.agent_distribution.decision_coordination_qualification import (
    accepted_decision,
    build_decision_coordination_executor_fixture,
    decision_contribution,
)
from intergrax.contracts.decision_coordination import DecisionCoordinationShape
from intergrax.agent_distribution.decision_coordination_projection import (
    project_authoritative_accepted_decision_coordination,
)
from tests.unit.agent_distribution.test_bounded_multi_agent_fanout import (
    _FanOutAcquisitionPlanFactory,
    _fan_out_item,
    _run_fan_out,
)
from tests.unit.agent_distribution.test_coordination_intent import _single_intent
from tests.unit.agent_distribution.test_coordination_intent_executor import (
    _StaticOrchestrationPort,
    _binding,
)
from tests.unit.agent_distribution.test_delegated_subtasks import (
    OcrRequest,
    OcrResult,
    _FailingResolver,
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
from tests.unit.agent_distribution.test_delegated_subtasks import _APP, _ENV
from tests.unit.agent_distribution.test_physical_delegation_governance_boundary import (
    _CountingChildExecution,
    _CountingSelector,
    _CountingSpecialistInvocation,
    _CountingTaskScopedAgents,
    _DelegationDenyingGovernance,
    _DelegationRequireHumanGovernance,
    _build_instrumented_harness,
)
from tests.unit.agent_distribution.test_task_scoped_agents import (
    _task_acquire_request,
    _trust,
)
from tests.unit.runtime.architecture.test_npsc5d_r1_final_qualification import (
    _execute_decision_backed_intent,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

_FORBIDDEN_ENGINE_NAMES = (
    "PhysicalDelegationGovernanceEngine",
    "MultiAgentDelegationPolicyEngine",
)

_R2_CONTRACT_MODULES = (
    _REPO_ROOT / "intergrax" / "contracts" / "physical_delegation_governance.py",
    _REPO_ROOT / "intergrax" / "runtime" / "governance" / "physical_delegation_governance.py",
    _REPO_ROOT
    / "intergrax"
    / "agent_distribution"
    / "physical_delegation_governance_adapter.py",
)

_CONTINUATION_MODULES = (
    _REPO_ROOT / "intergrax" / "agent_distribution" / "multi_agent_coordination.py",
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "fan_out_orchestration_adapter.py",
)

_GOVERNANCE_RUNTIME = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "governance"
    / "physical_delegation_governance.py"
)  # noqa: RUF100 — single path, not a tuple

_DELEGATED_SOURCE = _REPO_ROOT / "intergrax" / "agent_distribution" / "delegated_subtasks.py"
_ORCHESTRATION_ADAPTER = (
    _REPO_ROOT / "intergrax" / "runtime" / "execution" / "fan_out_orchestration_adapter.py"
)


class _RecordingPhysicalGovernance(PhysicalDelegationGovernancePort):
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


class _UntrustedAcquisitionPlanFactory:
    def __init__(self, *, revision_id: str = "rev-delegate-untrusted") -> None:
        self._revision_id = revision_id
        self._harness = None

    def bind_harness(self, harness) -> None:
        self._harness = harness

    def build_acquisition_plan(
        self,
        *,
        delegation_id: DelegationId,
        task_scope_id,
        application_id: str,
        application_environment_id: str,
        lease_id: TaskScopedAgentLeaseId,
        selected_identity,
    ) -> DelegatedSubtaskLifecyclePlan:
        del delegation_id, application_id, application_environment_id
        request = _task_acquire_request(
            str(lease_id),
            task_scope_id,
            self._revision_id,
            identity=selected_identity,
        )
        bad_trust = _trust().model_copy(
            update={"package_digest": "sha256:" + ("f" * 64)},
        )
        install = request.acquisition_request.install.model_copy(
            update={"trust_record": bad_trust},
        )
        acquisition = request.acquisition_request.model_copy(
            update={"install": install},
        )
        return DelegatedSubtaskLifecyclePlan(
            acquisition_request=request.model_copy(
                update={"acquisition_request": acquisition},
            ),
        )


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.gate
def test_npsc5d_r2_no_second_governance_engine() -> None:
    violations: list[str] = []
    for path in (_REPO_ROOT / "intergrax").rglob("*.py"):
        if "build" in path.parts or ".tmp" in path.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        for name in _FORBIDDEN_ENGINE_NAMES:
            if name in source:
                violations.append(f"{path.relative_to(_REPO_ROOT)}:{name}")
    assert violations == []


@pytest.mark.gate
def test_npsc5d_r2_gate_selection_before_governance_before_acquire() -> None:
    source_lines = _DELEGATED_SOURCE.read_text(encoding="utf-8-sig").splitlines()
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


@pytest.mark.gate
def test_npsc5d_r2_governance_runtime_no_acquire_or_child() -> None:
    source = _GOVERNANCE_RUNTIME.read_text(encoding="utf-8-sig")
    assert "TaskScopedAgentService" not in source
    assert ".acquire(" not in source
    assert "ChildExecution" not in source
    assert "ChildExecutionPort" not in source


@pytest.mark.gate
def test_npsc5d_r2_contract_modules_no_nexus_imports() -> None:
    forbidden: list[str] = []
    for path in _R2_CONTRACT_MODULES:
        for module in sorted(_module_imports(path)):
            if "intergrax.runtime.nexus" in module or "intergrax.runtime.execution.nexus" in module:
                forbidden.append(f"{path.name}:{module}")
    assert forbidden == []


@pytest.mark.gate
def test_npsc5d_r2_continuation_modules_no_reselection() -> None:
    for path in _CONTINUATION_MODULES:
        source = path.read_text(encoding="utf-8-sig")
        assert "AgentSelectionStrategy" not in source
        assert "self._selector.select" not in source


@pytest.mark.gate
def test_npsc5d_r2_orchestration_adapter_projection_only() -> None:
    source = _ORCHESTRATION_ADAPTER.read_text(encoding="utf-8-sig")
    assert "PhysicalDelegationGovernance" not in source
    assert "GovernanceRequiresHumanError" in source
    assert "continuation" in source


def _sample_governance_request(**updates) -> PhysicalDelegationGovernanceRequest:
    candidate = _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",))
    request = build_physical_delegation_governance_request(
        delegation_id="delegation-1",
        task_scope_id="task-scope-1",
        application_id=_APP,
        application_environment_id=_ENV,
        capability_requirement=build_agent_capability_requirement(
            required=("document.ocr",),
        ),
        selected_identity=candidate.identity,
        principal=admin_test_principal(),
    )
    if updates:
        return request.model_copy(update=updates)
    return request


@pytest.mark.gate
def test_npsc5d_r2_evaluation_point_distinct_from_r1() -> None:
    request = _sample_governance_request()
    assert request.evaluation_point is GovernanceEvaluationPoint.MULTI_AGENT_DELEGATION
    assert (
        request.evaluation_point
        is not GovernanceEvaluationPoint.MULTI_AGENT_COORDINATION
    )


@pytest.mark.gate
def test_npsc5d_r2_request_digest_binds_typed_facts() -> None:
    request_a = _sample_governance_request()
    request_b = _sample_governance_request(delegation_id="delegation-2")
    digest_a = physical_delegation_governance_request_digest(request_a)
    digest_b = physical_delegation_governance_request_digest(request_b)
    assert digest_a.startswith("sha256:")
    assert digest_a != digest_b


@pytest.mark.asyncio
async def test_npsc5d_r2_single_allow_acquires_and_executes() -> None:
    harness, _, task_scoped, specialist, child, governance = _build_instrumented_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        governance=allowing_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    result = cast(
        DelegatedSubtaskResult[OcrResult],
        await _run_delegation(harness, task_scope=task_scope),
    )
    assert result.result.text == "ocr:doc-1"
    assert governance.call_count == 1
    assert task_scoped.acquire_count == 1
    assert specialist.call_count == 1
    assert child.call_count == 1


@pytest.mark.asyncio
async def test_npsc5d_r2_single_deny_zero_downstream_effects() -> None:
    harness, selector, task_scoped, specialist, child, governance = _build_instrumented_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
            _discovery_candidate(_LEGAL_PACKAGE, capability_ids=("document.ocr",)),
        ),
        governance=denying_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    with pytest.raises(DelegatedSubtaskGovernanceDenied):
        await _run_delegation(harness, task_scope=task_scope)
    assert selector.call_count == 1
    assert governance.call_count == 1
    assert task_scoped.acquire_count == 0
    assert specialist.call_count == 0
    assert child.call_count == 0
    assert task_scoped.release_count == 0


@pytest.mark.asyncio
async def test_npsc5d_r2_single_require_human_zero_downstream_effects() -> None:
    harness, selector, task_scoped, specialist, child, _ = _build_instrumented_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
            _discovery_candidate(_LEGAL_PACKAGE, capability_ids=("document.ocr",)),
        ),
        governance=require_human_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    with pytest.raises(DelegatedSubtaskGovernanceRequiresHuman) as exc_info:
        await _run_delegation(harness, task_scope=task_scope)
    continuation = exc_info.value.continuation
    assert isinstance(continuation, PhysicalDelegationGovernedContinuation)
    assert continuation.selected_identity.distribution_package_id == _OCR_PACKAGE
    assert continuation.governance_result.permitted is False
    assert continuation.governance_result.requires_governed_continuation is True
    assert selector.call_count == 1
    assert task_scoped.acquire_count == 0
    assert specialist.call_count == 0
    assert child.call_count == 0


@pytest.mark.asyncio
async def test_npsc5d_r2_no_fallback_after_deny() -> None:
    harness, selector, task_scoped, specialist, child, governance = _build_instrumented_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
            _discovery_candidate(_LEGAL_PACKAGE, capability_ids=("document.ocr",)),
        ),
        governance=denying_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    with pytest.raises(DelegatedSubtaskGovernanceDenied):
        await _run_delegation(harness, task_scope=task_scope)
    assert selector.call_count == 1
    assert governance.last_request is not None
    assert governance.last_request.selected_identity.distribution_package_id == _OCR_PACKAGE
    assert task_scoped.acquire_count == 0
    assert specialist.call_count == 0
    assert child.call_count == 0


@pytest.mark.asyncio
async def test_npsc5d_r2_fan_out_mixed_deny_preserves_cardinality_and_order() -> None:
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
        fan_out_id="fan-out-mixed-deny",
        max_concurrency=3,
    )
    assert len(result.items) == 3
    assert [item.item_id for item in result.items] == ["a", "b", "c"]
    assert result.items[0].status is FanOutItemStatus.SUCCESS
    assert result.items[1].status is FanOutItemStatus.FAILURE
    assert result.items[1].failure is not None
    assert result.items[1].failure.failure_code is CoordinationFailureCode.GOVERNANCE_DENIED
    assert result.items[1].failure.continuation is None
    assert result.items[2].status is FanOutItemStatus.SUCCESS


@pytest.mark.asyncio
async def test_npsc5d_r2_fan_out_mixed_require_human_preserves_siblings() -> None:
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
    assert result.items[0].status is FanOutItemStatus.SUCCESS
    assert result.items[1].status is FanOutItemStatus.FAILURE
    assert result.items[1].failure is not None
    assert (
        result.items[1].failure.failure_code
        is CoordinationFailureCode.GOVERNANCE_REQUIRES_HUMAN
    )
    continuation = result.items[1].failure.continuation
    assert isinstance(continuation, PhysicalDelegationGovernedContinuation)
    assert continuation.delegation_id == "delegation-b"
    assert result.items[2].status is FanOutItemStatus.SUCCESS


@pytest.mark.gate
def test_npsc5d_r2_fan_out_contract_invariant() -> None:
    governance_request = _sample_governance_request()
    governance_result = RequireHumanPhysicalDelegationGovernance().evaluate(
        governance_request,
    )
    continuation = build_physical_delegation_governed_continuation(
        request=governance_request,
        governance_result=governance_result,
    )
    with pytest.raises(ValueError, match="requires continuation payload"):
        FanOutItemFailure(
            failure_code=CoordinationFailureCode.GOVERNANCE_REQUIRES_HUMAN,
            message="requires human",
        )
    with pytest.raises(ValueError, match="only allowed for GOVERNANCE_REQUIRES_HUMAN"):
        FanOutItemFailure(
            failure_code=CoordinationFailureCode.GOVERNANCE_DENIED,
            message="denied",
            continuation=continuation,
        )


@pytest.mark.asyncio
async def test_npsc5d_r2_r1_deny_short_circuits_r2() -> None:
    recording = _RecordingPhysicalGovernance(
        allowing_physical_delegation_governance(),
    )
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        physical_delegation_governance=recording,
    )
    counting_selector = _CountingSelector(harness.service._selector)
    counting_task_scoped = _CountingTaskScopedAgents(harness.task_scoped)
    counting_specialist = _CountingSpecialistInvocation(
        harness.service._specialist_invocation,
    )
    counting_child = _CountingChildExecution(harness.service._child_execution)
    harness.service = DelegatedSubtaskService(
        capability_resolver=harness.service._capability_resolver,
        discovery=harness.service._discovery,
        matcher=harness.service._matcher,
        selector=counting_selector,
        task_scoped_agents=cast(Any, counting_task_scoped),
        task_scope_authority=harness.task_scope_authority,
        acquisition_plan_factory=harness.service._acquisition_plan_factory,
        release_plan_factory=harness.service._release_plan_factory,
        specialist_invocation=counting_specialist,
        child_execution=counting_child,
        physical_delegation_governance=recording,
    )
    coordination = _build_coordination_service(harness)
    executor = CoordinationIntentExecutor(
        coordination=cast(Any, coordination),
        fan_out=BoundedMultiAgentFanOutService(
            orchestration=_StaticOrchestrationPort(()),
        ),
        governance=denying_coordination_governance(),
    )
    task_scope = mint_task_id()
    intent = _single_intent("contrib-a")
    binding = _binding(task_scope, pairs=(("contrib-a", "lease-a"),))
    with bound_governed_host_task():
        with pytest.raises(CoordinationGovernanceDenied):
            await executor.execute(
                intent,
                binding=binding,
                principal=admin_test_principal(),
            )
    assert recording.call_count == 0
    assert counting_selector.call_count == 0
    assert counting_task_scoped.acquire_count == 0
    assert counting_specialist.call_count == 0
    assert counting_child.call_count == 0


@pytest.mark.asyncio
async def test_npsc5d_r2_allow_ac3_deny_blocks_child() -> None:
    factory = _UntrustedAcquisitionPlanFactory()
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        acquisition_plan_factory=factory,
        physical_delegation_governance=allowing_physical_delegation_governance(),
    )
    factory.bind_harness(harness)
    counting_child = _CountingChildExecution(harness.service._child_execution)
    harness.service._child_execution = counting_child
    task_scope = harness.task_scope_authority.task_scope_id
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            with pytest.raises(AgentPackageTrustError):
                await harness.service.execute(
                    _delegated_request(task_scope=task_scope),
                    invocation=DelegatedSubtaskInvocation(payload=request),
                    principal=admin_test_principal(),
                )
            return OcrResult(text="blocked")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="ac3-deny"))
    assert counting_child.call_count == 0


@pytest.mark.asyncio
async def test_npsc5d_r2_decision_backed_path() -> None:
    fixture = build_decision_coordination_executor_fixture(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        fan_out=False,
    )
    decision = accepted_decision(
        DecisionCoordinationShape.SINGLE,
        (decision_contribution("contrib-a"),),
    )
    intent = project_authoritative_accepted_decision_coordination(decision)
    task_scope = mint_task_id()
    binding = _binding(task_scope, pairs=(("contrib-a", "lease-a"),))
    result = await _execute_decision_backed_intent(fixture, intent, binding)
    assert result.mode is CoordinationExecutionMode.SINGLE


@pytest.mark.asyncio
async def test_npsc5d_r2_deterministic_producer_path() -> None:
    harness, selector, task_scoped, specialist, child, governance = _build_instrumented_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
            _discovery_candidate(_LEGAL_PACKAGE, capability_ids=("document.ocr",)),
        ),
        governance=allowing_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    result = cast(
        DelegatedSubtaskResult[OcrResult],
        await _run_delegation(harness, task_scope=task_scope),
    )
    assert result.selected_identity.package.distribution_package_id == _OCR_PACKAGE
    assert selector.call_count == 1
    assert governance.call_count == 1
    assert task_scoped.acquire_count == 1
    assert specialist.call_count == 1
    assert child.call_count == 1


@pytest.mark.asyncio
async def test_npsc5d_r2_pre_resolved_capability_resolver_calls_zero() -> None:
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
    ).execute(OcrRequest(document_ref="pre-resolved"))
    assert captured


@pytest.mark.asyncio
async def test_npsc5d_r2_coordination_require_human_maps_continuation() -> None:
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
        async def execute(self, request: OcrRequest) -> OcrResult:
            return (
                await coordination.coordinate(
                    _coordination_request(task_scope=task_scope),
                    delegation=CoordinationDelegation(payload=request),
                    principal=admin_test_principal(),
                )
            ).result

    with pytest.raises(GovernanceRequiresHumanError) as exc_info:
        await ExecutionBoundary[OcrRequest, OcrResult](
            _Root(),
            identity=root,
        ).execute(OcrRequest(document_ref="doc-1"))
    continuation = exc_info.value.continuation
    assert continuation.delegation_id == continuation.governance_result.evidence.delegation_id
    assert (
        continuation.selected_identity.distribution_package_id
        == continuation.governance_result.evidence.selected_package_id
    )


@pytest.mark.asyncio
async def test_npsc5d_r2_allow_path_lease_cleanup_regression() -> None:
    harness, _, task_scoped, _, _, _ = _build_instrumented_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        governance=allowing_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    await _run_delegation(harness, task_scope=task_scope)
    assert task_scoped.acquire_count == 1
    assert task_scoped.release_count == 1
