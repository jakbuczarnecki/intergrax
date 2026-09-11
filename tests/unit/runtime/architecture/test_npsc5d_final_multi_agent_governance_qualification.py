# © Artur Czarnecki. All rights reserved.

"""NPSC-5D Final — unified multi-agent governance plane qualification & freeze."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from typing import cast

import pytest

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    BoundedMultiAgentFanOutService,
    FanOutItemStatus,
)
from intergrax.agent_distribution.coordination_intent import (
    CoordinationContribution,
    CoordinationContributionId,
    CoordinationExecutionMode,
    CoordinationIntent,
    CoordinationIntentId,
)
from intergrax.agent_distribution.coordination_intent_executor import (
    CoordinationGovernanceDenied,
    CoordinationIntentExecutor,
    CoordinationIntentResult,
)
from intergrax.agent_distribution.delegated_subtasks import (
    DelegatedSelectionProvenanceKind,
    DelegatedSubtaskResult,
)
from intergrax.agent_distribution.multi_agent_coordination import GovernanceRequiresHumanError
from intergrax.agent_distribution.task_capability_resolution import (
    build_task_capability_resolution_request,
    unresolved_agent_distribution_capability_need,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.physical_delegation_governance import (
    PhysicalDelegationGovernedContinuation,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.execution.boundary import ExecutionBoundary
from testing_support.agent_distribution.coordination_governance import (
    allowing_coordination_governance,
    allowing_physical_delegation_governance,
    bound_governed_host_task,
    denying_coordination_governance,
    require_human_physical_delegation_governance,
)
from tests.unit.agent_distribution.test_bounded_multi_agent_fanout import (
    _fan_out_item,
)
from tests.unit.agent_distribution.test_coordination_intent_executor import (
    _StaticOrchestrationPort,
    _binding,
)
from tests.unit.agent_distribution.test_delegated_subtasks import (
    OcrRequest,
    OcrResult,
    _LEGAL_PACKAGE,
    _OCR_PACKAGE,
    _discovery_candidate,
    admin_test_principal,
    build_delegated_harness,
)
from tests.unit.agent_distribution.test_multi_agent_coordination import (
    _build_coordination_service,
    _root_identity,
)
from tests.unit.agent_distribution.test_physical_delegation_governance_boundary import (
    _DelegationRequireHumanGovernance,
    _build_instrumented_harness,
)
from tests.unit.runtime.architecture.test_npsc5d_r3_final_qualification import (
    _typed_continuation,
)
from tests.unit.runtime.architecture.test_npsc5d_r3_governed_continuation import (
    _approve_physical_continuation,
    _bound_task_scope_execution,
    _build_governed_fan_out_stack,
    _continue_governed_delegation,
    _run_governed_fan_out,
    _run_governed_fan_out_resume,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

R1_FREEZE_SHA = "d90c4c67010c00d4e24be021a87a123c6de1cb25"
R2_FREEZE_SHA = "7fef5f942cc2697f080edc14521e23792331a7a5"
R3_IMPLEMENTATION_SHA = "36c2be6a03411e117250c56c4519d759682b8032"
R3_H1_SHA = "39e923494bac55f89858471072fd809c311923a8"
R3_FREEZE_SHA = "0f336ef804c1801ac323beeab1fc97a34feb5477"

_CANONICAL_PROVENANCE = (
    R1_FREEZE_SHA,
    R2_FREEZE_SHA,
    R3_IMPLEMENTATION_SHA,
    R3_H1_SHA,
    R3_FREEZE_SHA,
)

_FORBIDDEN_ENGINE_NAMES = (
    "MultiAgentGovernanceEngine",
    "PhysicalDelegationPolicyEngine",
    "MultiAgentPolicyEngine",
    "CoordinationGovernanceEngine",
    "NpscGovernanceRuntime",
    "MultiAgentAuthorityEngine",
    "MultiAgentPolicyComposer",
    "PhysicalDelegationGovernanceEngine",
    "MultiAgentDelegationPolicyEngine",
    "AgentGovernanceRuntime",
    "NpscPolicyEngine",
    "DelegationPolicyEngine",
)

_FORBIDDEN_HITL_RUNTIME_NAMES = (
    "PhysicalDelegationRuntime",
    "ContinuationRuntime",
    "FanOutResumeRuntime",
)

_SECURITY_CONTINUATION_PATHS = (
    _REPO_ROOT / "intergrax" / "agent_distribution" / "delegated_subtasks.py",
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "human"
    / "physical_delegation_continuation_grant.py",
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "human"
    / "physical_delegation_governed_continuation_bridge.py",
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "execution"
    / "fan_out_orchestration_adapter.py",
)

_REFLECTION_PATTERN = re.compile(r"\b(getattr|setattr|hasattr)\(")

_FROZEN_REGRESSION_MODULES = (
    "tests.unit.runtime.architecture.test_npsc5a_multi_agent_coordination_gate",
    "tests.unit.runtime.architecture.test_npsc5a_coordination_delegation_e2e",
    "tests.unit.runtime.architecture.test_npsc5b_final_production_fanout_fanin_qualification",
    "tests.unit.runtime.architecture.test_npsc5c_coordination_intent_gate",
    "tests.unit.runtime.architecture.test_npsc5c_decision_execution_e2e",
    "tests.unit.runtime.architecture.test_npsc5d_r1_final_qualification",
    "tests.unit.runtime.architecture.test_npsc5d_r2_final_qualification",
    "tests.unit.runtime.architecture.test_npsc5d_r3_final_qualification",
    "tests.unit.runtime.architecture.test_npsc5d_governance_gate",
)


def _git_object_exists(sha: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=_REPO_ROOT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


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


@pytest.mark.gate
def test_npsc5d_final_canonical_provenance_commits_exist() -> None:
    missing = [sha for sha in _CANONICAL_PROVENANCE if not _git_object_exists(sha)]
    assert missing == [], f"missing canonical provenance commits: {missing}"


@pytest.mark.gate
def test_npsc5d_final_one_governance_plane_no_duplicate_engines() -> None:
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
def test_npsc5d_final_no_duplicate_hitl_runtime() -> None:
    violations: list[str] = []
    for path in (_REPO_ROOT / "intergrax").rglob("*.py"):
        if "build" in path.parts or ".tmp" in path.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        for name in _FORBIDDEN_HITL_RUNTIME_NAMES:
            if name in source:
                violations.append(f"{path.relative_to(_REPO_ROOT)}:{name}")
    assert violations == []


@pytest.mark.gate
def test_npsc5d_final_cross_layer_gate_ordering() -> None:
    intent_executor = (
        _REPO_ROOT / "intergrax" / "agent_distribution" / "coordination_intent_executor.py"
    ).read_text(encoding="utf-8-sig")
    governance_idx = intent_executor.index("_enforce_governance")
    coordinate_idx = intent_executor.index("await self._coordination.coordinate")
    fan_out_idx = intent_executor.index("await self._fan_out.fan_out")
    assert governance_idx < coordinate_idx
    assert governance_idx < fan_out_idx

    delegated_lines = (
        _REPO_ROOT / "intergrax" / "agent_distribution" / "delegated_subtasks.py"
    ).read_text(encoding="utf-8-sig").splitlines()
    select_idx = next(i for i, line in enumerate(delegated_lines) if "self._selector.select" in line)
    physical_idx = next(
        i for i, line in enumerate(delegated_lines) if "_enforce_physical_delegation_governance" in line
    )
    acquire_idx = next(
        i for i, line in enumerate(delegated_lines) if "self._task_scoped_agents.acquire" in line
    )
    assert select_idx < physical_idx < acquire_idx


@pytest.mark.gate
def test_npsc5d_final_resume_path_static_invariants() -> None:
    resume_source = _function_source(
        _SECURITY_CONTINUATION_PATHS[0],
        "continue_governed_delegation",
    )
    assert "AgentSelectionDecision(" not in resume_source
    assert "self._selector.select" not in resume_source
    assert "self._discovery" not in resume_source
    assert "self._matcher" not in resume_source

    adapter_source = _function_source(
        _SECURITY_CONTINUATION_PATHS[3],
        "continue_governed_fan_out_slot",
    )
    assert "BoundedMultiAgentFanOutService" not in adapter_source
    assert ".orchestrate_fan_out(" not in adapter_source
    assert "topology_submission.submit(" not in adapter_source


@pytest.mark.gate
def test_npsc5d_final_security_continuation_paths_no_reflection() -> None:
    violations: list[str] = []
    for path in _SECURITY_CONTINUATION_PATHS:
        source = path.read_text(encoding="utf-8-sig")
        for match in _REFLECTION_PATTERN.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            violations.append(f"{path.relative_to(_REPO_ROOT)}:{line}:{match.group(1)}")
    assert violations == []


@pytest.mark.gate
def test_npsc5d_final_identity_authority_policy_execution_separation() -> None:
    principal = RequestIdentity(
        tenant_id="tenant-a",
        user_id="admin-user",
        auth_subject="subject-acting",
    )
    policy = PolicyDecision(
        action=PolicyAction.ALLOW,
        reason="coordination_allowed",
        policy_rule_id="test.coordination.allow",
    )
    assert type(principal).__name__ != type(policy).__name__
    assert PolicyAction.ALLOW is policy.action
    assert principal.tenant_id == "tenant-a"


@pytest.mark.gate
@pytest.mark.parametrize("module_name", _FROZEN_REGRESSION_MODULES)
def test_npsc5d_final_frozen_regression_module_exists(module_name: str) -> None:
    import importlib

    importlib.import_module(module_name)


@pytest.mark.asyncio
@pytest.mark.gate
async def test_npsc5d_final_cross_layer_single_allow_e2e() -> None:
    harness = build_delegated_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        physical_delegation_governance=allowing_physical_delegation_governance(),
    )
    coordination = _build_coordination_service(harness)
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=BoundedMultiAgentFanOutService(orchestration=_StaticOrchestrationPort(())),
        governance=allowing_coordination_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    root = _root_identity()
    captured: list[CoordinationIntentResult] = []

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            del request
            with bound_governed_host_task():
                captured.append(
                    await executor.execute(
                        CoordinationIntent(
                            intent_id=CoordinationIntentId("intent-final-allow"),
                            mode=CoordinationExecutionMode.SINGLE,
                            contributions=(
                                CoordinationContribution(
                                    contribution_id=CoordinationContributionId("contrib-a"),
                                    payload=OcrRequest(document_ref="doc-final"),
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
                    ),
                )
            return OcrResult(text="root-done")

    result = await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="final-allow"))
    assert result.text == "root-done"
    assert captured
    assert captured[0].mode is CoordinationExecutionMode.SINGLE
    assert captured[0].single is not None
    assert captured[0].single.coordination.result.text == "ocr:doc-final"


@pytest.mark.asyncio
@pytest.mark.gate
async def test_npsc5d_final_cross_layer_r1_deny_short_circuits_plane() -> None:
    harness = build_delegated_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        physical_delegation_governance=allowing_physical_delegation_governance(),
    )
    coordination = _build_coordination_service(harness)
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=BoundedMultiAgentFanOutService(orchestration=_StaticOrchestrationPort(())),
        governance=denying_coordination_governance(),
    )
    task_scope = harness.task_scope_authority.task_scope_id
    root = _root_identity()

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            del request
            with bound_governed_host_task():
                with pytest.raises(CoordinationGovernanceDenied):
                    await executor.execute(
                        CoordinationIntent(
                            intent_id=CoordinationIntentId("intent-final-deny"),
                            mode=CoordinationExecutionMode.SINGLE,
                            contributions=(
                                CoordinationContribution(
                                    contribution_id=CoordinationContributionId("contrib-a"),
                                    payload=OcrRequest(document_ref="doc-deny"),
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
            raise AssertionError("R1 DENY must short-circuit")

    with pytest.raises(AssertionError, match="R1 DENY must short-circuit"):
        await ExecutionBoundary[OcrRequest, OcrResult](
            RootDelegate(),
            identity=root,
            authority=ParentExecutionAuthority.unrestricted_root(),
        ).execute(OcrRequest(document_ref="final-deny"))


@pytest.mark.asyncio
@pytest.mark.gate
async def test_npsc5d_final_cross_layer_r2_require_human_r3_resume_e2e() -> None:
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
    assert isinstance(continuation, PhysicalDelegationGovernedContinuation)
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


@pytest.mark.asyncio
@pytest.mark.gate
async def test_npsc5d_final_cross_layer_fan_out_exact_slot_continuation() -> None:
    candidates = (
        _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        _discovery_candidate(_LEGAL_PACKAGE, capability_ids=("document.ocr",)),
    )
    harness, _, task_scoped, specialist, child, adapter, fan_out = _build_governed_fan_out_stack(
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
    assert initial.items[0].status is FanOutItemStatus.SUCCESS
    assert initial.items[1].status is FanOutItemStatus.FAILURE
    assert initial.items[1].failure is not None
    assert initial.items[2].status is FanOutItemStatus.SUCCESS
    assert child.call_count == 2

    continuation = initial.items[1].failure.continuation
    assert isinstance(continuation, PhysicalDelegationGovernedContinuation)

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
            correlation_id="final-fanout-slot-b",
        )

    assert resumed[0].result is not None and resumed[0].result.result.text == "ocr:doc-a"
    assert resumed[1].result is not None and resumed[1].result.result.text == "ocr:doc-b"
    assert resumed[2].result is not None and resumed[2].result.result.text == "ocr:doc-c"
    assert child.call_count == 3
    assert task_scoped.acquire_count == 3
    assert specialist.call_count == 3


@pytest.mark.asyncio
@pytest.mark.gate
async def test_npsc5d_final_r1_require_human_distinct_from_r2_continuation() -> None:
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
                            intent_id=CoordinationIntentId("intent-r2-hitl"),
                            mode=CoordinationExecutionMode.SINGLE,
                            contributions=(
                                CoordinationContribution(
                                    contribution_id=CoordinationContributionId("contrib-a"),
                                    payload=OcrRequest(document_ref="doc-r2"),
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
            raise AssertionError("expected R2 REQUIRE_HUMAN")

    with pytest.raises(GovernanceRequiresHumanError):
        await ExecutionBoundary[OcrRequest, OcrResult](
            RootDelegate(),
            identity=root,
            authority=ParentExecutionAuthority.unrestricted_root(),
        ).execute(OcrRequest(document_ref="r2-hitl"))
    assert captured
    continuation = captured[0].continuation
    assert isinstance(continuation, PhysicalDelegationGovernedContinuation)
    assert continuation.selected_identity.distribution_package_id == _OCR_PACKAGE
