# © Artur Czarnecki. All rights reserved.

"""NPSC-5D/R3 Final — canonical HITL governed continuation qualification & freeze."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

import pytest

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    BoundedMultiAgentFanOutService,
    FanOutItemStatus,
)
from intergrax.agent_distribution.delegated_subtasks import (
    DelegatedSelectionProvenanceKind,
    DelegatedSubtaskContinuationGrantError,
    DelegatedSubtaskContractError,
    DelegatedSubtaskGovernanceDenied,
    DelegatedSubtaskInvocation,
    DelegatedSubtaskResult,
    DelegationId,
)
from intergrax.agent_distribution.task_scoped_agents import TaskScopedAgentLeaseId, TaskScopeId
from intergrax.agent_distribution.multi_agent_coordination import (
    CoordinationFailureCode,
    GovernanceRequiresHumanError,
)
from intergrax.agent_distribution.task_scoped_agents import TaskScopedAgentError
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.orchestration_topology import (
    OrchestrationSlotContinuationError,
    OrchestrationSlotContinuationRequest,
    OrchestrationSlotId,
    OrchestrationTopologyExecutionId,
)
from intergrax.contracts.physical_delegation_governance import (
    PhysicalDelegationContinuationApprovalGrant,
    PhysicalDelegationGovernancePort,
    PhysicalDelegationGovernanceRequest,
    PhysicalDelegationGovernanceResult,
    PhysicalDelegationGovernedContinuation,
    physical_delegation_governed_continuation_digest,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary
from intergrax.runtime.execution.fan_out_orchestration_adapter import (
    CanonicalFanOutOrchestrationAdapter,
)
from intergrax.runtime.governance.physical_delegation_governance import (
    DenyingPhysicalDelegationGovernance,
    RequireHumanPhysicalDelegationGovernance,
)
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.human.pause import HumanApprovalResolutionError, HumanPauseCoordinator
from intergrax.runtime.human.physical_delegation_continuation_grant import (
    PhysicalDelegationContinuationGrantCoordinator,
    PhysicalDelegationContinuationGrantError,
    matches_current_physical_delegation_requirement,
)
from intergrax.runtime.human.physical_delegation_governed_continuation_bridge import (
    apply_physical_delegation_governed_continuation_pause,
)
from testing_support.agent_distribution.coordination_governance import (
    allowing_physical_delegation_governance,
    require_human_physical_delegation_governance,
)
from tests.unit.agent_distribution.test_bounded_multi_agent_fanout import (
    _fan_out_item,
)
from tests.unit.agent_distribution.test_delegated_subtasks import (
    OcrRequest,
    OcrResult,
    _LEGAL_PACKAGE,
    _OCR_PACKAGE,
    _delegated_request,
    _discovery_candidate,
    _root_identity,
    admin_test_principal,
    build_delegated_harness,
)
from tests.unit.agent_distribution.test_physical_delegation_governance_boundary import (
    _DelegationRequireHumanGovernance,
    _build_instrumented_harness,
)
from tests.unit.runtime.architecture.test_npsc5d_r3_governed_continuation import (
    APPROVER,
    RUN_ID,
    SOURCE_AGENT,
    _approve_physical_continuation,
    _bound_task_scope_execution,
    _build_governed_fan_out_stack,
    _continue_governed_delegation,
    _require_human_continuation,
    _run_governed_fan_out,
    _run_governed_fan_out_resume,
)
from intergrax.contracts.governed_continuation_grant import GovernedContinuationApprovalGrant

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

_FORBIDDEN_RUNTIME_NAMES = (
    "PhysicalDelegationRuntime",
    "ContinuationRuntime",
    "FanOutResumeRuntime",
)

_RESUME_SOURCE_PATHS = (
    _REPO_ROOT / "intergrax" / "agent_distribution" / "delegated_subtasks.py",
    _REPO_ROOT / "intergrax" / "agent_distribution" / "multi_agent_coordination.py",
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "fan_out_orchestration_adapter.py",
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "orchestration_topology_submission.py",
)

_NEXUS_CORE_PATHS = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "execution" / "graph_executor.py",
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "nexus_loop.py",
)

_FORBIDDEN_NEXUS_PHYSICAL_IMPORTS = (
    "PhysicalDelegationGovernedContinuation",
    "AgentDiscoveryCandidateIdentity",
    "AgentCapabilityRequirement",
)


class _MultiRequireHumanGovernance(PhysicalDelegationGovernancePort):
    def __init__(self, *, delegation_ids: frozenset[str]) -> None:
        self._delegation_ids = delegation_ids

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PhysicalDelegationGovernanceResult:
        if request.delegation_id in self._delegation_ids:
            return require_human_physical_delegation_governance().evaluate(request)
        return allowing_physical_delegation_governance().evaluate(request)


class _FlipDenyGovernance(PhysicalDelegationGovernancePort):
    def __init__(self) -> None:
        self._calls = 0

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PhysicalDelegationGovernanceResult:
        self._calls += 1
        if self._calls == 1:
            return RequireHumanPhysicalDelegationGovernance().evaluate(request)
        return DenyingPhysicalDelegationGovernance().evaluate(request)


class _StalePolicyRuleGovernance(PhysicalDelegationGovernancePort):
    def __init__(self) -> None:
        self._calls = 0

    def evaluate(
        self,
        request: PhysicalDelegationGovernanceRequest,
    ) -> PhysicalDelegationGovernanceResult:
        self._calls += 1
        if self._calls == 1:
            return RequireHumanPhysicalDelegationGovernance().evaluate(request)
        result = RequireHumanPhysicalDelegationGovernance().evaluate(request)
        decision = result.decision.model_copy(update={"policy_rule_id": "rule-changed"})
        evidence = result.evidence.model_copy(update={"policy_rule_id": "rule-changed"})
        return result.model_copy(update={"decision": decision, "evidence": evidence})


async def _typed_continuation(harness, *, task_scope) -> PhysicalDelegationGovernedContinuation:
    return cast(
        PhysicalDelegationGovernedContinuation,
        await _require_human_continuation(harness, task_scope=task_scope),
    )


def _function_source(path: Path, function_name: str) -> str:
    module_source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(module_source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == function_name:
            lines = module_source.splitlines()
            start = node.lineno - 1
            end = node.end_lineno or node.lineno
            return "\n".join(lines[start:end])
    raise AssertionError(f"{function_name} not found in {path}")


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.gate
def test_npsc5d_r3_gate_resume_path_no_synthetic_selection() -> None:
    source = _function_source(
        _RESUME_SOURCE_PATHS[0],
        "continue_governed_delegation",
    )
    assert "AgentSelectionDecision(" not in source
    assert "self._selector.select" not in source
    assert "self._discovery" not in source
    assert "self._matcher" not in source


@pytest.mark.gate
def test_npsc5d_r3_gate_resume_path_no_whole_fanout_or_topology_submit() -> None:
    adapter_source = _function_source(
        _RESUME_SOURCE_PATHS[2],
        "continue_governed_fan_out_slot",
    )
    assert "BoundedMultiAgentFanOutService" not in adapter_source
    assert ".orchestrate_fan_out(" not in adapter_source
    assert "topology_submission.submit(" not in adapter_source

    topology_source = _function_source(
        _RESUME_SOURCE_PATHS[3],
        "continue_slot",
    )
    assert "topology_submission.submit(" not in topology_source


@pytest.mark.gate
def test_npsc5d_r3_gate_nexus_core_no_physical_delegation_knowledge() -> None:
    violations: list[str] = []
    for path in _NEXUS_CORE_PATHS:
        source = path.read_text(encoding="utf-8-sig")
        for name in _FORBIDDEN_NEXUS_PHYSICAL_IMPORTS:
            if name in source:
                violations.append(f"{path.name}:{name}")
    assert violations == []


@pytest.mark.gate
def test_npsc5d_r3_gate_agent_distribution_no_new_scheduler_or_runtime() -> None:
    ad_root = _REPO_ROOT / "intergrax" / "agent_distribution"
    scheduler_tokens = ("Semaphore(", "ThreadPoolExecutor", "asyncio.TaskGroup")
    violations: list[str] = []
    for path in ad_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8-sig")
        for name in _FORBIDDEN_RUNTIME_NAMES:
            if name in source:
                violations.append(f"{path.relative_to(_REPO_ROOT)}:{name}")
        if any(token in source for token in scheduler_tokens):
            if "continue_governed" in source or "physical_delegation" in path.name:
                violations.append(f"{path.relative_to(_REPO_ROOT)}:scheduler")
    assert violations == []


@pytest.mark.gate
def test_npsc5d_r3_gate_physical_grant_distinct_from_side_effect_grant() -> None:
    physical_fields = set(PhysicalDelegationContinuationApprovalGrant.model_fields)
    side_effect_fields = set(GovernedContinuationApprovalGrant.model_fields)
    assert "side_effect_scope_id" not in physical_fields
    assert "side_effect_scope_id" in side_effect_fields
    assert physical_fields.isdisjoint({"side_effect_scope_id"})


@pytest.mark.asyncio
@pytest.mark.gate
async def test_npsc5d_r3_gate_result_contract_invariant() -> None:
    harness, _, _, _, _, _ = _build_instrumented_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        governance=require_human_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    continuation = await _typed_continuation(harness, task_scope=task_scope)
    with pytest.raises(DelegatedSubtaskContractError, match="SELECTED provenance requires"):
        DelegatedSubtaskResult(
            delegation_id=DelegationId("delegation-1"),
            task_scope_id=TaskScopeId(str(task_scope)),
            capability_resolution=None,
            capability_requirement=cast(Any, continuation.capability_requirement),
            match_results=(),
            selection_decision=None,
            selection_provenance_kind=DelegatedSelectionProvenanceKind.SELECTED,
            selected_identity=cast(Any, continuation.selected_identity),
            lease_id=TaskScopedAgentLeaseId("lease-1"),
            application_binding_id="binding-1",
            acquisition_result=cast(Any, object()),
            release_result=cast(Any, object()),
            result=OcrResult(text="x"),
        )


@pytest.mark.asyncio
async def test_npsc5d_r3_single_hitl_e2e() -> None:
    harness, selector, task_scoped, specialist, child, _ = _build_instrumented_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
            _discovery_candidate(_LEGAL_PACKAGE, capability_ids=("document.ocr",)),
        ),
        governance=_DelegationRequireHumanGovernance(
            require_human_delegation_id="delegation-1",
        ),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    continuation = await _typed_continuation(harness, task_scope=task_scope)
    assert selector.call_count == 1

    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        grant = _approve_physical_continuation(task, continuation)
        result = cast(
            DelegatedSubtaskResult[OcrResult],
            await _continue_governed_delegation(
                harness,
                task_scope=task_scope,
                task=task,
                continuation=continuation,
                grant=grant,
            ),
        )

    assert result.result.text == "ocr:doc-1"
    assert (
        result.selection_provenance_kind
        is DelegatedSelectionProvenanceKind.PRESERVED_GOVERNED_CONTINUATION
    )
    assert result.selection_decision is None
    assert selector.call_count == 1
    assert task_scoped.acquire_count == 1
    assert specialist.call_count == 1
    assert child.call_count == 1
    assert task_scoped.release_count == 1


@pytest.mark.asyncio
async def test_npsc5d_r3_single_replay_matrix_blocked() -> None:
    harness, _, task_scoped, _, _, _ = _build_instrumented_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        governance=require_human_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    continuation = await _typed_continuation(harness, task_scope=task_scope)

    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        apply_physical_delegation_governed_continuation_pause(
            task,
            continuation,
            source_agent_id=SOURCE_AGENT,
            run_id=RUN_ID,
        )
        pause_record = task.runtime.governance.pause_record
        human_request = task.runtime.governance.human_request
        assert pause_record is not None
        assert human_request is not None

        with pytest.raises(HumanApprovalResolutionError, match="pause_id mismatch"):
            HumanPauseCoordinator.resolve_human_response(
                task,
                HumanResponseVerdict.APPROVE,
                approver=APPROVER,
                pause_id="wrong-pause",
                human_request_id=human_request.request_id,
                run_id=RUN_ID,
            )

        with pytest.raises(HumanApprovalResolutionError, match="human_request_id mismatch"):
            HumanPauseCoordinator.resolve_human_response(
                task,
                HumanResponseVerdict.APPROVE,
                approver=APPROVER,
                pause_id=pause_record.pause_id,
                human_request_id="wrong-human-request",
                run_id=RUN_ID,
            )

        HumanPauseCoordinator.resolve_human_response(
            task,
            HumanResponseVerdict.APPROVE,
            approver=APPROVER,
            pause_id=pause_record.pause_id,
            human_request_id=human_request.request_id,
            run_id="wrong-run",
        )
        with pytest.raises(PhysicalDelegationContinuationGrantError, match="run_id mismatch"):
            PhysicalDelegationContinuationGrantCoordinator.create_grant_from_approval(task)

        task.runtime.governance.hitl_resolution = None
        HumanPauseCoordinator.resolve_human_response(
            task,
            HumanResponseVerdict.APPROVE,
            approver=APPROVER,
            pause_id=pause_record.pause_id,
            human_request_id=human_request.request_id,
            run_id=RUN_ID,
        )
        grant = PhysicalDelegationContinuationGrantCoordinator.create_grant_from_approval(task)
        assert grant is not None

        wrong_delegation = continuation.model_copy(update={"delegation_id": "delegation-other"})
        with pytest.raises(DelegatedSubtaskContinuationGrantError):
            await harness.service.continue_governed_delegation(
                _delegated_request(task_scope=task_scope),
                invocation=DelegatedSubtaskInvocation(payload=OcrRequest(document_ref="doc-1")),
                continuation=wrong_delegation,
                principal=admin_test_principal(),
                task=task,
                expected_grant_id=grant.grant_id,
            )

        wrong_specialist = continuation.model_copy(
            update={
                "selected_identity": continuation.selected_identity.model_copy(
                    update={"distribution_package_id": _LEGAL_PACKAGE},
                ),
            },
        )
        with pytest.raises(DelegatedSubtaskContinuationGrantError):
            await harness.service.continue_governed_delegation(
                _delegated_request(task_scope=task_scope),
                invocation=DelegatedSubtaskInvocation(payload=OcrRequest(document_ref="doc-1")),
                continuation=wrong_specialist,
                principal=admin_test_principal(),
                task=task,
                expected_grant_id=grant.grant_id,
            )

        wrong_capability = continuation.model_copy(
            update={
                "capability_requirement": continuation.capability_requirement.model_copy(
                    update={"required_capability_ids": ("document.legal",)},
                ),
            },
        )
        with pytest.raises(DelegatedSubtaskContinuationGrantError):
            await harness.service.continue_governed_delegation(
                _delegated_request(task_scope=task_scope),
                invocation=DelegatedSubtaskInvocation(payload=OcrRequest(document_ref="doc-1")),
                continuation=wrong_capability,
                principal=admin_test_principal(),
                task=task,
                expected_grant_id=grant.grant_id,
            )

        stale_grant = grant.model_copy(update={"governance_request_digest": "sha256:" + ("a" * 64)})
        task.runtime.governance.physical_delegation_continuation_grant = stale_grant
        with pytest.raises(DelegatedSubtaskContinuationGrantError):
            await harness.service.continue_governed_delegation(
                _delegated_request(task_scope=task_scope),
                invocation=DelegatedSubtaskInvocation(payload=OcrRequest(document_ref="doc-1")),
                continuation=continuation,
                principal=admin_test_principal(),
                task=task,
                expected_grant_id=grant.grant_id,
            )

    assert task_scoped.acquire_count == 0


@pytest.mark.asyncio
async def test_npsc5d_r3_single_escalate_no_acquire() -> None:
    harness, _, task_scoped, specialist, child, _ = _build_instrumented_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        governance=require_human_physical_delegation_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    continuation = await _typed_continuation(harness, task_scope=task_scope)
    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        apply_physical_delegation_governed_continuation_pause(
            task,
            continuation,
            source_agent_id=SOURCE_AGENT,
            run_id=RUN_ID,
        )
        pause_record = task.runtime.governance.pause_record
        human_request = task.runtime.governance.human_request
        assert pause_record is not None
        assert human_request is not None
        HumanPauseCoordinator.resolve_human_response(
            task,
            HumanResponseVerdict.ESCALATE,
            approver=APPROVER,
            pause_id=pause_record.pause_id,
            human_request_id=human_request.request_id,
            run_id=RUN_ID,
        )
        assert (
            PhysicalDelegationContinuationGrantCoordinator.create_grant_from_approval(task)
            is None
        )
    assert task_scoped.acquire_count == 0
    assert specialist.call_count == 0
    assert child.call_count == 0


@pytest.mark.asyncio
async def test_npsc5d_r3_single_policy_freshness_and_stale_requirement() -> None:
    harness, _, task_scoped, _, _, _ = _build_instrumented_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        governance=_FlipDenyGovernance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    continuation = await _typed_continuation(harness, task_scope=task_scope)
    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        grant = _approve_physical_continuation(task, continuation)
        with pytest.raises(DelegatedSubtaskGovernanceDenied):
            await _continue_governed_delegation(
                harness,
                task_scope=task_scope,
                task=task,
                continuation=continuation,
                grant=grant,
            )
    assert task_scoped.acquire_count == 0

    harness_stale, _, task_scoped_stale, _, _, _ = _build_instrumented_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        governance=_StalePolicyRuleGovernance(),
    )
    task_scope_stale = harness_stale.task_scope_authority.task_scope_id
    continuation_stale = await _typed_continuation(
        harness_stale,
        task_scope=task_scope_stale,
    )
    with _bound_task_scope_execution(task_id=str(task_scope_stale)) as task:
        grant_stale = _approve_physical_continuation(task, continuation_stale)
        stored = task.runtime.governance.physical_delegation_continuation_grant
        assert stored is not None
        stale_current = continuation_stale.governance_result.model_copy(
            update={
                "decision": continuation_stale.governance_result.decision.model_copy(
                    update={"policy_rule_id": "rule-changed"},
                ),
                "evidence": continuation_stale.governance_result.evidence.model_copy(
                    update={"policy_rule_id": "rule-changed"},
                ),
            },
        )
        assert not matches_current_physical_delegation_requirement(
            stored,
            continuation=continuation_stale,
            current_result=stale_current,
        )
        with pytest.raises(DelegatedSubtaskGovernanceDenied):
            await harness_stale.service.continue_governed_delegation(
                _delegated_request(task_scope=task_scope_stale),
                invocation=DelegatedSubtaskInvocation(payload=OcrRequest(document_ref="doc-1")),
                continuation=continuation_stale,
                principal=admin_test_principal(),
                task=task,
                expected_grant_id=grant_stale.grant_id,
            )
    assert task_scoped_stale.acquire_count == 0


@pytest.mark.asyncio
async def test_npsc5d_r3_fan_out_hitl_e2e_with_sibling_counters() -> None:
    candidates = (
        _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        _discovery_candidate(_LEGAL_PACKAGE, capability_ids=("document.ocr",)),
    )
    harness, selector, task_scoped, _, child, adapter, fan_out = _build_governed_fan_out_stack(
        candidates,
    )
    task_scope = mint_task_id()
    items = (
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
    )
    initial = await _run_governed_fan_out(
        harness,
        fan_out,
        task_scope=task_scope,
        items=items,
    )
    assert len(initial.items) == 3
    assert [item.item_id for item in initial.items] == [entry.item_id for entry in items]
    assert child.call_count == 2
    assert selector.call_count == 3

    continuation = initial.items[1].failure.continuation
    assert continuation is not None
    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        grant = _approve_physical_continuation(task, continuation)
        resumed = await _run_governed_fan_out_resume(
            harness,
            adapter,
            task_scope=task_scope,
            items=items,
            item_id=items[1].item_id,
            continuation=continuation,
            grant=grant,
            task=task,
        )

    assert len(resumed) == 3
    assert resumed[0].result is not None and resumed[0].result.result.text == "ocr:doc-a"
    assert resumed[1].result is not None and resumed[1].result.result.text == "ocr:doc-b"
    assert resumed[2].result is not None and resumed[2].result.result.text == "ocr:doc-c"
    assert child.call_count == 3
    assert selector.call_count == 3
    assert task_scoped.acquire_count == 3


@pytest.mark.asyncio
async def test_npsc5d_r3_fan_out_wrong_topology_blocked() -> None:
    candidates = (_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),)
    harness, _, _, _, _, adapter, fan_out = _build_governed_fan_out_stack(candidates)
    task_scope = mint_task_id()
    items = (
        _fan_out_item(
            item_id="a",
            task_scope=task_scope,
            coordination_id="coord-a",
            delegation_id="delegation-a",
            lease_id="lease-a",
        ),
        _fan_out_item(
            item_id="b",
            task_scope=task_scope,
            coordination_id="coord-b",
            delegation_id="delegation-b",
            lease_id="lease-b",
        ),
        _fan_out_item(
            item_id="c",
            task_scope=task_scope,
            coordination_id="coord-c",
            delegation_id="delegation-c",
            lease_id="lease-c",
        ),
    )
    initial = await _run_governed_fan_out(
        harness,
        fan_out,
        task_scope=task_scope,
        items=items,
        fan_out_id="fan-out-t1",
    )
    continuation = initial.items[1].failure.continuation
    assert continuation is not None
    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        _approve_physical_continuation(task, continuation)
        canonical_adapter = cast(CanonicalFanOutOrchestrationAdapter[OcrRequest, OcrResult], adapter)
        with pytest.raises(OrchestrationSlotContinuationError, match="unknown orchestration"):
            await canonical_adapter.topology_continuation.continue_slot(
                OrchestrationSlotContinuationRequest(
                    execution_id=OrchestrationTopologyExecutionId("topology-missing"),
                    slot_id=OrchestrationSlotId("b"),
                    correlation_id="wrong-topology",
                ),
                slot_continuation_executor=cast(Any, object()),
            )


@pytest.mark.asyncio
async def test_npsc5d_r3_fan_out_reject_preserves_siblings() -> None:
    candidates = (_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),)
    harness, _, task_scoped, _, child, adapter, fan_out = _build_governed_fan_out_stack(
        candidates,
    )
    task_scope = mint_task_id()
    items = (
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
    )
    initial = await _run_governed_fan_out(
        harness,
        fan_out,
        task_scope=task_scope,
        items=items,
    )
    continuation = initial.items[1].failure.continuation
    assert continuation is not None
    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        apply_physical_delegation_governed_continuation_pause(
            task,
            continuation,
            source_agent_id=SOURCE_AGENT,
            run_id=RUN_ID,
        )
        pause_record = task.runtime.governance.pause_record
        human_request = task.runtime.governance.human_request
        assert pause_record is not None
        assert human_request is not None
        HumanPauseCoordinator.resolve_human_response(
            task,
            HumanResponseVerdict.REJECT,
            approver=APPROVER,
            pause_id=pause_record.pause_id,
            human_request_id=human_request.request_id,
            run_id=RUN_ID,
        )
        assert (
            PhysicalDelegationContinuationGrantCoordinator.create_grant_from_approval(task)
            is None
        )
    assert child.call_count == 2
    assert task_scoped.acquire_count == 2


@pytest.mark.asyncio
async def test_npsc5d_r3_fan_out_ac3_deny_preserves_siblings() -> None:
    candidates = (_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),)
    harness, _, task_scoped, _, child, adapter, fan_out = _build_governed_fan_out_stack(
        candidates,
    )

    task_scope = mint_task_id()
    items = (
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
    )
    initial = await _run_governed_fan_out(
        harness,
        fan_out,
        task_scope=task_scope,
        items=items,
    )
    continuation = initial.items[1].failure.continuation
    assert continuation is not None

    def _deny_acquire(*args, **kwargs):
        raise TaskScopedAgentError("ac-3 deny")

    task_scoped.acquire = _deny_acquire  # type: ignore[method-assign]

    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        grant = _approve_physical_continuation(task, continuation)
        resumed = await _run_governed_fan_out_resume(
            harness,
            adapter,
            task_scope=task_scope,
            items=items,
            item_id=items[1].item_id,
            continuation=continuation,
            grant=grant,
            task=task,
        )
    assert resumed[0].status is FanOutItemStatus.SUCCESS
    assert resumed[1].status is FanOutItemStatus.FAILURE
    assert resumed[2].status is FanOutItemStatus.SUCCESS
    assert child.call_count == 2
    assert task_scoped.acquire_count == 2


@pytest.mark.asyncio
async def test_npsc5d_r3_multiple_blocked_slot_identity_independent() -> None:
    harness = build_delegated_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        physical_delegation_governance=_MultiRequireHumanGovernance(
            delegation_ids=frozenset({"delegation-a", "delegation-b"}),
        ),
    )
    from tests.unit.agent_distribution.test_multi_agent_coordination import (
        _build_coordination_service,
    )

    coordination_service = _build_coordination_service(harness)
    fan_out = BoundedMultiAgentFanOutService(
        orchestration=cast(Any, object()),
    )
    del fan_out
    from intergrax.runtime.execution.fan_out_orchestration_adapter import (
        build_fan_out_orchestration_port,
    )
    from intergrax.runtime.execution.orchestration_topology_submission import (
        build_orchestration_topology_submission_port,
    )
    from intergrax.runtime.nexus.nexus_loop import NexusLoop
    from intergrax.runtime.registry.agent_registry import AgentRegistry

    nexus_loop = NexusLoop(AgentRegistry())
    topology_port = cast(
        Any,
        build_orchestration_topology_submission_port(nexus_loop),
    )
    adapter = build_fan_out_orchestration_port(topology_port, coordination_service)
    fan_out_service = BoundedMultiAgentFanOutService(orchestration=adapter)

    task_scope = mint_task_id()
    items = (
        _fan_out_item(
            item_id="a",
            task_scope=task_scope,
            coordination_id="coord-a",
            delegation_id="delegation-a",
            lease_id="lease-a",
        ),
        _fan_out_item(
            item_id="b",
            task_scope=task_scope,
            coordination_id="coord-b",
            delegation_id="delegation-b",
            lease_id="lease-b",
        ),
        _fan_out_item(
            item_id="c",
            task_scope=task_scope,
            coordination_id="coord-c",
            delegation_id="delegation-c",
            lease_id="lease-c",
            document_ref="doc-c",
        ),
    )
    result = await _run_governed_fan_out(
        harness,
        fan_out_service,
        task_scope=task_scope,
        items=items,
    )
    cont_a = result.items[0].failure.continuation
    cont_b = result.items[1].failure.continuation
    assert cont_a is not None
    assert cont_b is not None
    assert cont_a.delegation_id == "delegation-a"
    assert cont_b.delegation_id == "delegation-b"
    assert cont_a.delegation_id != cont_b.delegation_id
    assert physical_delegation_governed_continuation_digest(cont_a) != (
        physical_delegation_governed_continuation_digest(cont_b)
    )


@pytest.mark.asyncio
async def test_npsc5d_r3_one_approval_does_not_authorize_other_slot() -> None:
    candidates = (_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),)
    harness = build_delegated_harness(
        candidates=candidates,
        physical_delegation_governance=_MultiRequireHumanGovernance(
            delegation_ids=frozenset({"delegation-a", "delegation-b"}),
        ),
    )
    from tests.unit.agent_distribution.test_multi_agent_coordination import (
        _build_coordination_service,
    )
    from intergrax.runtime.execution.fan_out_orchestration_adapter import (
        build_fan_out_orchestration_port,
    )
    from intergrax.runtime.execution.orchestration_topology_submission import (
        build_orchestration_topology_submission_port,
    )
    from intergrax.runtime.nexus.nexus_loop import NexusLoop
    from intergrax.runtime.registry.agent_registry import AgentRegistry

    coordination = _build_coordination_service(harness)
    nexus_loop = NexusLoop(AgentRegistry())
    topology_port = cast(
        Any,
        build_orchestration_topology_submission_port(nexus_loop),
    )
    adapter = build_fan_out_orchestration_port(topology_port, coordination)
    fan_out = BoundedMultiAgentFanOutService(orchestration=adapter)

    task_scope = mint_task_id()
    items = (
        _fan_out_item(
            item_id="a",
            task_scope=task_scope,
            coordination_id="coord-a",
            delegation_id="delegation-a",
            lease_id="lease-a",
        ),
        _fan_out_item(
            item_id="b",
            task_scope=task_scope,
            coordination_id="coord-b",
            delegation_id="delegation-b",
            lease_id="lease-b",
        ),
        _fan_out_item(
            item_id="c",
            task_scope=task_scope,
            coordination_id="coord-c",
            delegation_id="delegation-c",
            lease_id="lease-c",
            document_ref="doc-c",
        ),
    )
    initial = await _run_governed_fan_out(
        harness,
        fan_out,
        task_scope=task_scope,
        items=items,
    )
    cont_a = initial.items[0].failure.continuation
    cont_b = initial.items[1].failure.continuation
    assert cont_a is not None and cont_b is not None
    with _bound_task_scope_execution(task_id=str(task_scope)) as task:
        grant_a = _approve_physical_continuation(task, cont_a)
        resumed = await _run_governed_fan_out_resume(
            harness,
            adapter,
            task_scope=task_scope,
            items=items,
            item_id=items[1].item_id,
            continuation=cont_b,
            grant=grant_a,
            task=task,
            correlation_id="wrong-slot-grant",
        )
        assert resumed[1].status is FanOutItemStatus.FAILURE
        assert resumed[1].failure is not None
        assert resumed[1].failure.failure_code is CoordinationFailureCode.INVALID_COORDINATION
        assert "grant" in resumed[1].failure.message.lower()


@pytest.mark.asyncio
async def test_npsc5d_r3_decision_backed_path_uses_same_r3_mechanics() -> None:
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
    from testing_support.agent_distribution.coordination_governance import (
        allowing_coordination_governance,
        bound_governed_host_task,
    )
    from tests.unit.agent_distribution.test_coordination_intent_executor import (
        _StaticOrchestrationPort,
        _binding,
    )
    from tests.unit.agent_distribution.test_multi_agent_coordination import (
        _build_coordination_service,
    )

    harness = build_delegated_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        physical_delegation_governance=require_human_physical_delegation_governance(),
    )
    coordination = _build_coordination_service(harness)
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=BoundedMultiAgentFanOutService(orchestration=_StaticOrchestrationPort(())),
        governance=allowing_coordination_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    root = _root_identity()
    captured: list[GovernanceRequiresHumanError] = []

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            del request
            with bound_governed_host_task():
                try:
                    await executor.execute(
                        CoordinationIntent(
                            intent_id=CoordinationIntentId("intent-decision"),
                            mode=CoordinationExecutionMode.SINGLE,
                            contributions=(
                                CoordinationContribution(
                                    contribution_id=CoordinationContributionId("contrib-a"),
                                    payload=OcrRequest(document_ref="doc-decision"),
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
        ).execute(OcrRequest(document_ref="decision-backed"))
    assert captured
    continuation = captured[0].continuation
    assert isinstance(continuation, PhysicalDelegationGovernedContinuation)


@pytest.mark.gate
def test_npsc5d_r3_grant_binding_fields_present() -> None:
    fields = PhysicalDelegationContinuationApprovalGrant.model_fields
    required = {
        "continuation_digest",
        "delegation_id",
        "task_scope_id",
        "run_id",
        "selected_identity",
        "capability_requirement",
        "governance_request_digest",
        "pause_id",
        "human_request_id",
    }
    assert required.issubset(fields.keys())
