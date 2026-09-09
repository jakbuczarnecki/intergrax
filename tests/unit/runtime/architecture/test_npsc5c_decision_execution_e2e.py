# © Artur Czarnecki. All rights reserved.

"""NPSC-5C/R3 — Decision → NPSC → Execution cross-system E2E qualification."""

from __future__ import annotations

import ast
import asyncio
from dataclasses import fields
from pathlib import Path

import pytest

from intergrax.agent_distribution.bounded_multi_agent_fanout import FanOutItemId, FanOutItemStatus
from intergrax.agent_distribution.coordination_intent import (
    CoordinationContributionId,
    CoordinationExecutionMode,
    CoordinationIntentId,
)
from intergrax.agent_distribution.delegated_subtasks import DelegatedSubtaskDelegate
from intergrax.agent_distribution.multi_agent_coordination import CoordinationFailureCode
from intergrax.agent_distribution.task_capability_resolution import (
    AgentDistributionCapabilityNeedKind,
)
from intergrax.contracts.decision_coordination import DecisionCoordinationShape
from intergrax.contracts.decision_identity import DecisionId, initial_decision_version
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    ExecutionId,
    mint_task_id,
    peek_active_parent_execution_id,
    require_active_execution_id,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from testing_support.agent_distribution.coordination_governance import bound_governed_host_task
from testing_support.agent_distribution.decision_coordination_qualification import (
    GatedFanOutDelegate,
    accepted_decision,
    build_decision_coordination_executor_fixture,
    coordination_binding,
    decision_contribution,
    decision_has_no_physical_agent_fields,
    decision_identity,
    expected_intent_id,
    next_version_accepted_decision,
    project_accepted_decision,
    snapshot_accepted_decision,
)
from tests.unit.agent_distribution.test_delegated_subtasks import (
    OcrRequest,
    OcrResult,
    _OCR_PACKAGE,
    _discovery_candidate,
    admin_test_principal,
)
from tests.unit.agent_distribution.test_multi_agent_coordination import _root_identity

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_E2E_PATH = Path(__file__).resolve()

_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())

_FORBIDDEN_CALL_ATTRS = frozenset({"coordinate", "fan_out"})
_FORBIDDEN_CONSTRUCTORS = frozenset(
    {
        "MultiAgentCoordinationService",
        "BoundedMultiAgentFanOutService",
        "DelegatedSubtaskService",
        "NexusLoop",
    },
)
_RAW_PROJECTION_API = "project_decision_coordination_artifact"
_RAW_PROJECTION_CONTEXT = "DecisionCoordinationProjectionContext"

_REQUIRED_ENTRYPOINTS = (
    "project_authoritative_accepted_decision_coordination",
    "CoordinationIntentExecutor",
    "AuthoritativeAcceptedDecision",
)


class _LineageOcrDelegate(DelegatedSubtaskDelegate[OcrRequest, OcrResult]):
    def __init__(self) -> None:
        self.child_execution_id: ExecutionId | None = None
        self.child_parent_execution_id: ExecutionId | None = None

    async def execute(self, request: OcrRequest) -> OcrResult:
        self.child_execution_id = require_active_execution_id()
        self.child_parent_execution_id = peek_active_parent_execution_id()
        return OcrResult(text=f"ocr:{request.document_ref}")


class _PartialFailureDelegate:
    async def execute(self, request: OcrRequest) -> OcrResult:
        if request.document_ref == "contrib-b":
            raise RuntimeError("expected specialist failure")
        return OcrResult(text=f"ocr:{request.document_ref}")


async def _execute_projected_intent(
    fixture,
    accepted,
    *,
    contribution_lease_pairs: tuple[tuple[str, str], ...],
    binding_order: tuple[tuple[str, str], ...] | None = None,
    bind_budget: bool = False,
):
    intent = project_accepted_decision(accepted)
    task_scope = mint_task_id()
    fixture.harness.task_scope_authority.task_scope_id = task_scope
    binding = coordination_binding(task_scope, binding_order or contribution_lease_pairs)
    root = _root_identity()
    captured = []

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            del request
            budget_token = None
            if bind_budget:
                budget_token = bind_root_execution_budget(
                    execution_id=require_active_execution_id(),
                    ledger=_UNLIMITED_LEDGER,
                )
            try:
                with bound_governed_host_task():
                    result = await fixture.executor.execute(
                        intent,
                        binding=binding,
                        principal=admin_test_principal(),
                    )
            finally:
                if budget_token is not None:
                    reset_active_execution_budget(budget_token)
            captured.append(result)
            return OcrResult(text="root-done")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="root"))
    assert captured
    return intent, captured[0], root


@pytest.mark.gate
def test_npsc5c_r3_qualification_harness_uses_canonical_entrypoints() -> None:
    source = _E2E_PATH.read_text(encoding="utf-8-sig")
    for symbol in _REQUIRED_ENTRYPOINTS:
        assert symbol in source
    tree = ast.parse(source, filename=str(_E2E_PATH))
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "coordinate",
            "fan_out",
        }:
            violations.append(f"{_E2E_PATH.name}:{node.lineno}:.{node.func.attr}(")
    assert violations == [], (
        "R3 qualification must not call lower-level coordination services directly:\n"
        + "\n".join(violations)
    )


def _forbidden_runtime_calls_in_module(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    violations: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name.startswith("test_npsc5c_r3_qualification_"):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if isinstance(child.func, ast.Attribute):
                if child.func.attr in _FORBIDDEN_CALL_ATTRS:
                    violations.append(f"{path.name}:{child.lineno}:.{child.func.attr}(")
            elif isinstance(child.func, ast.Name):
                if child.func.id in _FORBIDDEN_CONSTRUCTORS:
                    violations.append(f"{path.name}:{child.lineno}:{child.func.id}(")
    return violations


@pytest.mark.gate
def test_npsc5c_r3_qualification_harness_forbids_lower_layer_construction() -> None:
    violations = _forbidden_runtime_calls_in_module(_E2E_PATH)
    assert violations == [], (
        "R3 qualification harness must not construct forbidden lower-layer owners:\n"
        + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc5c_r3_projection_entrypoint_is_public() -> None:
    source = _E2E_PATH.read_text(encoding="utf-8-sig")
    assert f"{_RAW_PROJECTION_API}(" not in source
    assert f"{_RAW_PROJECTION_CONTEXT}(" not in source


@pytest.mark.asyncio
async def test_npsc5c_r3_single_decision_to_execution_e2e() -> None:
    specialist = _LineageOcrDelegate()
    fixture = build_decision_coordination_executor_fixture(
        candidates=(
            _discovery_candidate("ocr-primary", capability_ids=("document.ocr",)),
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=specialist,
        fan_out=False,
    )
    accepted = accepted_decision(
        DecisionCoordinationShape.SINGLE,
        (decision_contribution("contrib-a", document_ref="doc-single"),),
    )
    before = snapshot_accepted_decision(accepted)
    decision_has_no_physical_agent_fields(accepted)

    intent, result, root = await _execute_projected_intent(
        fixture,
        accepted,
        contribution_lease_pairs=(("contrib-a", "lease-a"),),
    )

    assert intent.mode is CoordinationExecutionMode.SINGLE
    assert intent.requested_max_concurrency is None
    assert intent.intent_id == expected_intent_id(accepted.identity)
    assert (
        intent.contributions[0].capability_need.kind
        is AgentDistributionCapabilityNeedKind.RESOLVED_REQUIREMENT
    )
    assert fixture.failing_resolver.call_count == 0
    assert result.single is not None
    assert result.single.contribution_id == CoordinationContributionId("contrib-a")
    assert result.single.coordination.result.text == "ocr:doc-single"
    assert result.single.coordination.delegated.selection_decision.outcome.value == "selected"
    assert specialist.child_execution_id is not None
    assert specialist.child_parent_execution_id == root.execution_id
    assert specialist.child_execution_id != root.execution_id
    assert snapshot_accepted_decision(accepted) == before


@pytest.mark.asyncio
async def test_npsc5c_r3_fan_out_decision_to_execution_e2e() -> None:
    completion_order: list[str] = []
    start_gates = {
        "contrib-c": asyncio.Event(),
        "contrib-a": asyncio.Event(),
        "contrib-b": asyncio.Event(),
    }
    all_waiting = asyncio.Event()
    waiting_count = [0]
    fixture = build_decision_coordination_executor_fixture(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=GatedFanOutDelegate(
            start_gates=start_gates,
            completion_order=completion_order,
            all_waiting=all_waiting,
            waiting_count=waiting_count,
            expected_waiters=3,
        ),
        track_lineage=True,
    )
    accepted = accepted_decision(
        DecisionCoordinationShape.FAN_OUT,
        (
            decision_contribution("contrib-c", document_ref="contrib-c"),
            decision_contribution("contrib-a", document_ref="contrib-a"),
            decision_contribution("contrib-b", document_ref="contrib-b"),
        ),
    )

    async def _release_completion_gates() -> None:
        await asyncio.wait_for(all_waiting.wait(), timeout=5.0)
        start_gates["contrib-b"].set()
        await asyncio.sleep(0)
        start_gates["contrib-c"].set()
        await asyncio.sleep(0)
        start_gates["contrib-a"].set()

    release_task = asyncio.create_task(_release_completion_gates())
    try:
        intent, result, root = await _execute_projected_intent(
            fixture,
            accepted,
            contribution_lease_pairs=(
                ("contrib-c", "lease-c"),
                ("contrib-a", "lease-a"),
                ("contrib-b", "lease-b"),
            ),
            bind_budget=True,
        )
    finally:
        await release_task

    assert intent.mode is CoordinationExecutionMode.FAN_OUT
    assert intent.requested_max_concurrency is None
    assert fixture.failing_resolver.call_count == 0
    assert result.fan_out is not None
    assert [item.item_id for item in result.fan_out.fan_out.items] == [
        FanOutItemId("contrib-c"),
        FanOutItemId("contrib-a"),
        FanOutItemId("contrib-b"),
    ]
    assert completion_order == ["contrib-b", "contrib-c", "contrib-a"]
    assert completion_order != ["contrib-c", "contrib-a", "contrib-b"]
    assert all(
        item.status is FanOutItemStatus.SUCCESS
        for item in result.fan_out.fan_out.items
    )
    assert fixture.specialist_child_ids is not None
    assert len(fixture.specialist_child_ids) == 3
    assert root.execution_id not in fixture.specialist_child_ids


@pytest.mark.asyncio
async def test_npsc5c_r3_fan_out_partial_failure_cross_system() -> None:
    fixture = build_decision_coordination_executor_fixture(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
        specialist_delegate=_PartialFailureDelegate(),
    )
    accepted = accepted_decision(
        DecisionCoordinationShape.FAN_OUT,
        (
            decision_contribution("contrib-a", document_ref="contrib-a"),
            decision_contribution("contrib-b", document_ref="contrib-b"),
            decision_contribution("contrib-c", document_ref="contrib-c"),
        ),
    )
    _, result, _ = await _execute_projected_intent(
        fixture,
        accepted,
        contribution_lease_pairs=(
            ("contrib-a", "lease-a"),
            ("contrib-b", "lease-b"),
            ("contrib-c", "lease-c"),
        ),
        bind_budget=True,
    )
    assert result.fan_out is not None
    items = result.fan_out.fan_out.items
    assert len(items) == 3
    assert items[0].status is FanOutItemStatus.SUCCESS
    assert items[1].status is FanOutItemStatus.FAILURE
    assert items[1].failure is not None
    assert items[1].failure.failure_code is CoordinationFailureCode.CHILD_EXECUTION_FAILED
    assert items[2].status is FanOutItemStatus.SUCCESS
    assert items[0].item_id == FanOutItemId("contrib-a")
    assert items[1].item_id == FanOutItemId("contrib-b")
    assert items[2].item_id == FanOutItemId("contrib-c")


@pytest.mark.asyncio
async def test_npsc5c_r3_reordered_bindings_resolve_by_contribution_id() -> None:
    fixture = build_decision_coordination_executor_fixture(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    accepted = accepted_decision(
        DecisionCoordinationShape.FAN_OUT,
        (
            decision_contribution("contrib-a", document_ref="contrib-a"),
            decision_contribution("contrib-b", document_ref="contrib-b"),
            decision_contribution("contrib-c", document_ref="contrib-c"),
        ),
    )
    _, result, _ = await _execute_projected_intent(
        fixture,
        accepted,
        contribution_lease_pairs=(
            ("contrib-a", "lease-a"),
            ("contrib-b", "lease-b"),
            ("contrib-c", "lease-c"),
        ),
        binding_order=(
            ("contrib-c", "lease-c"),
            ("contrib-a", "lease-a"),
            ("contrib-b", "lease-b"),
        ),
        bind_budget=True,
    )
    assert result.fan_out is not None
    assert [item.item_id for item in result.fan_out.fan_out.items] == [
        FanOutItemId("contrib-a"),
        FanOutItemId("contrib-b"),
        FanOutItemId("contrib-c"),
    ]


def test_npsc5c_r3_decision_identity_maps_to_intent_identity() -> None:
    fixed_id = DecisionId(f"decision_{'d' * 32}")
    identity = decision_identity(
        decision_id=str(fixed_id),
        version=initial_decision_version(),
    )
    accepted_v1 = accepted_decision(
        DecisionCoordinationShape.SINGLE,
        (decision_contribution("contrib-a"),),
        identity=identity,
    )
    accepted_v2 = next_version_accepted_decision(accepted_v1)
    intent_v1 = project_accepted_decision(accepted_v1)
    intent_v2 = project_accepted_decision(accepted_v2)
    assert intent_v1.intent_id == CoordinationIntentId(f"coordination_intent:{fixed_id}:v1")
    assert intent_v2.intent_id == CoordinationIntentId(f"coordination_intent:{fixed_id}:v2")
    replay = project_accepted_decision(accepted_v1)
    assert replay.intent_id == intent_v1.intent_id
    assert replay.contributions[0].contribution_id == intent_v1.contributions[0].contribution_id
    assert replay.contributions[0].payload == intent_v1.contributions[0].payload


def test_npsc5c_r3_decision_artifact_has_no_runtime_ownership_fields() -> None:
    accepted = accepted_decision(
        DecisionCoordinationShape.FAN_OUT,
        (
            decision_contribution("contrib-a"),
            decision_contribution("contrib-b"),
        ),
    )
    decision_has_no_physical_agent_fields(accepted)
    forbidden_names = {
        "agent_id",
        "agent_instance_id",
        "lease_id",
        "execution_id",
        "execution_strategy",
        "requested_max_concurrency",
    }
    for field in fields(accepted.artifact.content):
        assert field.name not in forbidden_names
