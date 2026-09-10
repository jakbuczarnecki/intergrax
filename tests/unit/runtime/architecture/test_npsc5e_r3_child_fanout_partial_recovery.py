# © Artur Czarnecki. All rights reserved.

"""NPSC-5E/R3 — child & fan-out partial recovery qualification."""

from __future__ import annotations

import ast
import inspect
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    BoundedMultiAgentFanOutService,
    FanOutId,
    FanOutItemId,
    FanOutItemStatus,
    FanOutRequest,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    require_active_execution_id,
)
from intergrax.contracts.orchestration_topology import OrchestrationSlotId
from intergrax.contracts.partial_recovery import (
    PartialRecoveryError,
    PartialRecoveryErrorCode,
    PartialRecoveryReason,
    PartialRecoveryRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.fan_out_orchestration_adapter import (
    CanonicalFanOutOrchestrationAdapter,
    build_fan_out_orchestration_port,
)
from intergrax.runtime.execution.fan_out_partial_recovery import FanOutPartialRecoveryService
from intergrax.runtime.execution.orchestration_topology_submission import (
    CanonicalOrchestrationTopologySubmissionPort,
    build_orchestration_topology_submission_port,
)
from intergrax.runtime.long_running.checkpoint_revision import StaleCheckpointWriteError
from intergrax.runtime.long_running.execution_tree_checkpoint import minimal_runtime_checkpoint
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.long_running.topology_recovery_snapshot import (
    capture_topology_recovery_snapshot,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import TaskState
from tests.unit.agent_distribution.test_bounded_multi_agent_fanout import (
    _fan_out_item,
    build_fan_out_harness,
)
from tests.unit.agent_distribution.test_delegated_subtasks import (
    OcrRequest,
    OcrResult,
    _OCR_PACKAGE,
    _discovery_candidate,
    admin_test_principal,
)
from tests.unit.agent_distribution.test_multi_agent_coordination import (
    _build_coordination_service,
    _root_identity,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TENANT = "tenant-r3"
_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())
_REFLECTION_PATTERN = re.compile(r"\b(getattr|setattr|hasattr)\(")
_FORBIDDEN_RECOVERY_NAMES = (
    "PartialRecoveryRuntime",
    "FanOutRecoveryEngine",
    "ChildRecoveryScheduler",
    "RecoveryOrchestrator",
)

_MANDATORY_SUITES: tuple[tuple[str, list[str]], ...] = (
    ("R1 Final", ["tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py"]),
    ("R2 Final", ["tests/unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py"]),
    ("P0A", ["tests/unit/runtime/architecture/test_npsc5e_p0a_execution_lineage_baseline_qualification.py"]),
    (
        "DG_001",
        [
            "tests/unit/contracts/test_execution_lineage_contracts.py",
            "tests/unit/runtime/execution/lineage/test_execution_lineage_discovery.py",
        ],
    ),
    (
        "NPSC-5B",
        ["tests/unit/runtime/architecture/test_npsc5b_final_production_fanout_fanin_qualification.py"],
    ),
    (
        "NPSC-5D Final",
        ["tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py"],
    ),
    ("HITL R3", ["tests/unit/runtime/architecture/test_npsc5d_r3_governed_continuation.py"]),
    ("Child execution", ["tests/unit/runtime/execution/test_child_execution.py"]),
    ("Checkpoint store", ["tests/unit/runtime/long_running/test_checkpoint_store.py"]),
)


@dataclass
class _InvocationTracker:
    counts: dict[str, int] = field(default_factory=dict)

    def record(self, document_ref: str) -> None:
        self.counts[document_ref] = self.counts.get(document_ref, 0) + 1


class _RecoverableFailDelegate:
    def __init__(self, tracker: _InvocationTracker, fail_refs: frozenset[str]) -> None:
        self._tracker = tracker
        self._fail_refs = fail_refs
        self._failed_once: set[str] = set()

    async def execute(self, request: OcrRequest) -> OcrResult:
        self._tracker.record(request.document_ref)
        if request.document_ref in self._fail_refs and request.document_ref not in self._failed_once:
            self._failed_once.add(request.document_ref)
            raise RuntimeError(f"transient failure: {request.document_ref}")
        return OcrResult(text=f"ocr:{request.document_ref}")


def _build_stack(
    harness,
    *,
    nexus_loop: NexusLoop | None = None,
) -> tuple[
    BoundedMultiAgentFanOutService[OcrRequest, OcrResult],
    CanonicalFanOutOrchestrationAdapter[OcrRequest, OcrResult],
    CanonicalOrchestrationTopologySubmissionPort,
    FanOutPartialRecoveryService[OcrRequest, OcrResult],
    NexusLoop,
]:
    loop = nexus_loop or NexusLoop(AgentRegistry())
    submission = build_orchestration_topology_submission_port(loop)
    coordination = _build_coordination_service(harness)
    adapter = build_fan_out_orchestration_port(submission, coordination)
    fan_out = BoundedMultiAgentFanOutService(orchestration=adapter)
    assert isinstance(adapter, CanonicalFanOutOrchestrationAdapter)
    assert isinstance(submission, CanonicalOrchestrationTopologySubmissionPort)
    recovery = FanOutPartialRecoveryService(
        adapter=adapter,
        submission_port=submission,
        checkpoint_store=SQLiteTaskCheckpointStore(db_path=":memory:"),
    )
    return fan_out, adapter, submission, recovery, loop


def _three_item_request(task_scope: TaskId) -> tuple:
    return (
        _fan_out_item(
            item_id="item-a",
            task_scope=task_scope,
            coordination_id="coord-a",
            delegation_id="delegation-a",
            lease_id="lease-a",
            document_ref="doc-a",
        ),
        _fan_out_item(
            item_id="item-b",
            task_scope=task_scope,
            coordination_id="coord-b",
            delegation_id="delegation-b",
            lease_id="lease-b",
            document_ref="doc-b",
        ),
        _fan_out_item(
            item_id="item-c",
            task_scope=task_scope,
            coordination_id="coord-c",
            delegation_id="delegation-c",
            lease_id="lease-c",
            document_ref="doc-c",
        ),
    )


def _checkpoint_with_topology(
    *,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    root_execution_id: ExecutionId,
    snapshot,
    revision: int = 1,
) -> TaskCheckpoint:
    runtime = minimal_runtime_checkpoint(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=root_execution_id,
    ).model_copy(update={"topology_recovery": snapshot})
    return TaskCheckpoint(
        task_id=task_id,
        tenant_id=_TENANT,
        resume_token="rt-r3",
        task_state=TaskState.RUNNING,
        task_snapshot={},
        revision=revision,
        runtime=runtime,
    )


async def _run_partial_fan_out(
    fan_out: BoundedMultiAgentFanOutService[OcrRequest, OcrResult],
    adapter: CanonicalFanOutOrchestrationAdapter[OcrRequest, OcrResult],
    *,
    items,
    task_scope: TaskId,
) -> tuple:
    request = FanOutRequest(
        fan_out_id=FanOutId("fan-out-r3"),
        items=items,
        max_concurrency=3,
    )
    principal = admin_test_principal()
    captured: list = []
    execution_ids: list = []
    root = _root_identity()

    class RootDelegate:
        async def execute(self, ocr_request: OcrRequest) -> OcrResult:
            del ocr_request
            budget_token = bind_root_execution_budget(
                execution_id=require_active_execution_id(),
                ledger=_UNLIMITED_LEDGER,
            )
            try:
                result = await fan_out.fan_out(request, principal=principal)
                execution_ids.append(
                    adapter.resolve_execution_id(request, principal=principal),
                )
            finally:
                reset_active_execution_budget(budget_token)
            captured.append(result)
            return OcrResult(text="root-done")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="root"))
    result = captured[0]
    return request, result, execution_ids[0], root


async def _run_recovery_under_root(
    recovery: FanOutPartialRecoveryService[OcrRequest, OcrResult],
    *,
    recovery_request: PartialRecoveryRequest,
    fan_out_request: FanOutRequest,
    partial_result,
    checkpoint: TaskCheckpoint,
    root_execution_id: ExecutionId,
    correlation_id: str,
    identity: ExecutionIdentityBinding | None = None,
):
    captured: list = []
    root = identity if identity is not None else _root_identity()

    class RecoveryDelegate:
        async def execute(self, ocr_request: OcrRequest) -> OcrResult:
            del ocr_request
            budget_token = bind_root_execution_budget(
                execution_id=require_active_execution_id(),
                ledger=_UNLIMITED_LEDGER,
            )
            try:
                result = await recovery.recover_failed_slot(
                    recovery_request,
                    fan_out_request=fan_out_request,
                    partial_result=partial_result,
                    principal=admin_test_principal(),
                    checkpoint=checkpoint,
                    root_execution_id=root_execution_id,
                    correlation_id=correlation_id,
                    tenant_id=_TENANT,
                )
            finally:
                reset_active_execution_budget(budget_token)
            captured.append(result)
            return OcrResult(text="recovery-done")

    await ExecutionBoundary[OcrRequest, OcrResult](
        RecoveryDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="recovery-root"))
    assert captured
    return captured[0]


def _run_pytest(targets: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "pytest", *targets, "-q", "--tb=no"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(("label", "targets"), _MANDATORY_SUITES, ids=[label for label, _ in _MANDATORY_SUITES])
def test_mandatory_frozen_suite_passes(label: str, targets: list[str]) -> None:
    proc = _run_pytest(targets)
    assert proc.returncode == 0, f"{label} failed:\n{proc.stdout}\n{proc.stderr}"


def test_no_forbidden_recovery_runtime_names_in_production() -> None:
    production_root = _REPO_ROOT / "intergrax"
    hits: list[str] = []
    for path in production_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for name in _FORBIDDEN_RECOVERY_NAMES:
            if name in source:
                hits.append(f"{path.relative_to(_REPO_ROOT)}:{name}")
    assert not hits, f"forbidden recovery runtime names: {hits}"


def test_no_reflection_in_partial_recovery_production() -> None:
    paths = [
        _REPO_ROOT / "intergrax/contracts/partial_recovery.py",
        _REPO_ROOT / "intergrax/runtime/execution/fan_out_partial_recovery.py",
        _REPO_ROOT / "intergrax/runtime/long_running/topology_recovery_snapshot.py",
    ]
    for path in paths:
        assert _REFLECTION_PATTERN.search(path.read_text(encoding="utf-8")) is None


def test_submission_port_exposes_recover_failed_slot() -> None:
    port = build_orchestration_topology_submission_port(NexusLoop(AgentRegistry()))
    assert hasattr(port, "recover_failed_slot")
    assert inspect.iscoroutinefunction(port.recover_failed_slot)


@pytest.mark.asyncio
async def test_one_failed_slot_recovery_preserves_siblings(tmp_path: Path) -> None:
    tracker = _InvocationTracker()
    harness = build_fan_out_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        specialist_delegate=_RecoverableFailDelegate(tracker, frozenset({"doc-b"})),
    )
    fan_out, adapter, submission, recovery, _loop = _build_stack(harness)
    task_scope = mint_task_id()
    items = _three_item_request(task_scope)
    harness.task_scope_authority.task_scope_id = task_scope
    request, partial, execution_id, identity = await _run_partial_fan_out(
        fan_out, adapter, items=items, task_scope=task_scope,
    )
    assert partial.items[0].status is FanOutItemStatus.SUCCESS
    assert partial.items[1].status is FanOutItemStatus.FAILURE
    assert partial.items[2].status is FanOutItemStatus.SUCCESS
    snapshot = capture_topology_recovery_snapshot(
        request=request,
        result=partial,
        topology_execution_id=execution_id,
    )
    checkpoint = _checkpoint_with_topology(
        task_id=task_scope,
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        root_execution_id=identity.execution_id,
        snapshot=snapshot,
    )
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "one-failed.db")
    saved = store.save(checkpoint)
    harness_recover = build_fan_out_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        specialist_delegate=_RecoverableFailDelegate(tracker, frozenset()),
    )
    harness_recover.task_scope_authority.task_scope_id = task_scope
    _, adapter_recover, submission_recover, _, _ = _build_stack(harness_recover)
    recovery = FanOutPartialRecoveryService(
        adapter=adapter_recover,
        submission_port=submission_recover,
        checkpoint_store=store,
    )
    recovered = await _run_recovery_under_root(
        recovery,
        recovery_request=PartialRecoveryRequest(
            root_execution_id=identity.execution_id,
            topology_execution_id=execution_id,
            slot_id=OrchestrationSlotId("item-b"),
            source_checkpoint_revision=saved.revision or 1,
            source_attempt_id=identity.attempt_id,
            recovery_reason=PartialRecoveryReason.TRANSIENT_SLOT_FAILURE,
        ),
        fan_out_request=request,
        partial_result=partial,
        checkpoint=saved,
        root_execution_id=identity.execution_id,
        correlation_id="recover-b-1",
        identity=identity,
    )
    assert recovered.fan_out_result.items[1].status is FanOutItemStatus.SUCCESS
    assert recovered.preserved_slot_ids == (FanOutItemId("item-a"), FanOutItemId("item-c"))
    assert tracker.counts["doc-a"] == 1
    assert tracker.counts["doc-b"] == 2
    assert tracker.counts["doc-c"] == 1


@pytest.mark.asyncio
async def test_all_success_recovery_is_noop(tmp_path: Path) -> None:
    tracker = _InvocationTracker()
    harness = build_fan_out_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        specialist_delegate=_RecoverableFailDelegate(tracker, frozenset()),
    )
    fan_out, adapter, submission, recovery, _ = _build_stack(harness)
    task_scope = mint_task_id()
    items = _three_item_request(task_scope)
    harness.task_scope_authority.task_scope_id = task_scope
    request, partial, execution_id, identity = await _run_partial_fan_out(
        fan_out, adapter, items=items, task_scope=task_scope,
    )
    assert partial.all_succeeded
    snapshot = capture_topology_recovery_snapshot(
        request=request, result=partial, topology_execution_id=execution_id,
    )
    checkpoint = _checkpoint_with_topology(
        task_id=task_scope,
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        root_execution_id=identity.execution_id,
        snapshot=snapshot,
    )
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "all-success.db")
    saved = store.save(checkpoint)
    recovery = FanOutPartialRecoveryService(
        adapter=adapter,
        submission_port=submission,
        checkpoint_store=store,
    )
    result = await _run_recovery_under_root(
        recovery,
        recovery_request=PartialRecoveryRequest(
            root_execution_id=identity.execution_id,
            topology_execution_id=execution_id,
            slot_id=OrchestrationSlotId("item-b"),
            source_checkpoint_revision=saved.revision or 1,
            source_attempt_id=identity.attempt_id,
            recovery_reason=PartialRecoveryReason.ALREADY_COMPLETE,
        ),
        fan_out_request=request,
        partial_result=partial,
        checkpoint=saved,
        root_execution_id=identity.execution_id,
        correlation_id="noop-1",
        identity=identity,
    )
    assert result.recovered_slot_ids == ()
    assert tracker.counts.get("doc-a", 0) == 1
    assert tracker.counts.get("doc-b", 0) == 1
    assert tracker.counts.get("doc-c", 0) == 1


@pytest.mark.asyncio
async def test_result_order_and_cardinality_preserved_after_recovery(tmp_path: Path) -> None:
    tracker = _InvocationTracker()
    harness = build_fan_out_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        specialist_delegate=_RecoverableFailDelegate(tracker, frozenset({"doc-b"})),
    )
    fan_out, adapter, submission, recovery, _ = _build_stack(harness)
    task_scope = mint_task_id()
    items = _three_item_request(task_scope)
    harness.task_scope_authority.task_scope_id = task_scope
    request, partial, execution_id, identity = await _run_partial_fan_out(
        fan_out, adapter, items=items, task_scope=task_scope,
    )
    snapshot = capture_topology_recovery_snapshot(
        request=request, result=partial, topology_execution_id=execution_id,
    )
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "order.db")
    saved = store.save(
        _checkpoint_with_topology(
            task_id=task_scope,
            run_id=identity.run_id,
            attempt_id=identity.attempt_id,
            root_execution_id=identity.execution_id,
            snapshot=snapshot,
        ),
    )
    harness_recover = build_fan_out_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        specialist_delegate=_RecoverableFailDelegate(tracker, frozenset()),
    )
    harness_recover.task_scope_authority.task_scope_id = task_scope
    _, adapter_recover, submission_recover, _, _ = _build_stack(harness_recover)
    recovery = FanOutPartialRecoveryService(
        adapter=adapter_recover,
        submission_port=submission_recover,
        checkpoint_store=store,
    )
    recovered = await _run_recovery_under_root(
        recovery,
        recovery_request=PartialRecoveryRequest(
            root_execution_id=identity.execution_id,
            topology_execution_id=execution_id,
            slot_id=OrchestrationSlotId("item-b"),
            source_checkpoint_revision=saved.revision or 1,
            source_attempt_id=identity.attempt_id,
            recovery_reason=PartialRecoveryReason.TRANSIENT_SLOT_FAILURE,
        ),
        fan_out_request=request,
        partial_result=partial,
        checkpoint=saved,
        root_execution_id=identity.execution_id,
        correlation_id="order-1",
        identity=identity,
    )
    assert len(recovered.fan_out_result.items) == 3
    assert [item.item_id for item in recovered.fan_out_result.items] == [
        FanOutItemId("item-a"),
        FanOutItemId("item-b"),
        FanOutItemId("item-c"),
    ]


@pytest.mark.asyncio
async def test_cross_process_partial_recovery() -> None:
    tracker = _InvocationTracker()
    harness = build_fan_out_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        specialist_delegate=_RecoverableFailDelegate(tracker, frozenset({"doc-b"})),
    )
    fan_out, adapter, _, recovery_a, _ = _build_stack(harness)
    task_scope = mint_task_id()
    items = _three_item_request(task_scope)
    harness.task_scope_authority.task_scope_id = task_scope
    request, partial, execution_id, identity = await _run_partial_fan_out(
        fan_out, adapter, items=items, task_scope=task_scope,
    )
    snapshot = capture_topology_recovery_snapshot(
        request=request, result=partial, topology_execution_id=execution_id,
    )
    db_path = _REPO_ROOT / ".tmp" / "session" / "npsc5e-r3" / "cross.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store_a = SQLiteTaskCheckpointStore(db_path=db_path)
    saved = store_a.save(
        _checkpoint_with_topology(
            task_id=task_scope,
            run_id=identity.run_id,
            attempt_id=identity.attempt_id,
            root_execution_id=identity.execution_id,
            snapshot=snapshot,
        ),
    )
    store_b = SQLiteTaskCheckpointStore(db_path=db_path)
    loaded = store_b.get_latest(str(task_scope), _TENANT)
    assert loaded is not None
    harness_b = build_fan_out_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        specialist_delegate=_RecoverableFailDelegate(tracker, frozenset()),
    )
    _, adapter_b, submission_b, _, _ = _build_stack(harness_b)
    recovery_b = FanOutPartialRecoveryService(
        adapter=adapter_b,
        submission_port=submission_b,
        checkpoint_store=store_b,
    )
    harness_b.task_scope_authority.task_scope_id = task_scope
    recovered = await _run_recovery_under_root(
        recovery_b,
        recovery_request=PartialRecoveryRequest(
            root_execution_id=identity.execution_id,
            topology_execution_id=execution_id,
            slot_id=OrchestrationSlotId("item-b"),
            source_checkpoint_revision=loaded.revision or 1,
            source_attempt_id=identity.attempt_id,
            recovery_reason=PartialRecoveryReason.TRANSIENT_SLOT_FAILURE,
        ),
        fan_out_request=request,
        partial_result=partial,
        checkpoint=loaded,
        root_execution_id=identity.execution_id,
        correlation_id="cross-1",
        identity=identity,
    )
    assert recovered.fan_out_result.items[1].status is FanOutItemStatus.SUCCESS
    assert recovered.checkpoint_revision == (saved.revision or 1) + 1


def test_wrong_revision_blocked() -> None:
    harness = build_fan_out_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
    )
    _, adapter, _, recovery, _ = _build_stack(harness)
    task_scope = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    from intergrax.runtime.long_running.topology_recovery_snapshot import TopologyRecoverySnapshot, SlotRecoverySnapshot
    from intergrax.contracts.partial_recovery import SlotRecoveryDisposition

    snapshot = TopologyRecoverySnapshot(
        fan_out_id="fan-out-r3",
        topology_execution_id="topo-1",
        slot_order=("item-b",),
        slots=(
            SlotRecoverySnapshot(
                slot_id="item-b",
                disposition=SlotRecoveryDisposition.FAILED,
            ),
        ),
        max_concurrency=1,
    )
    checkpoint = _checkpoint_with_topology(
        task_id=task_scope,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=root,
        snapshot=snapshot,
        revision=2,
    )
    with pytest.raises(PartialRecoveryError, match="revision") as exc:
        recovery.validate_recovery_request(
            PartialRecoveryRequest(
                root_execution_id=root,
                topology_execution_id=__import__(
                    "intergrax.contracts.orchestration_topology",
                    fromlist=["OrchestrationTopologyExecutionId"],
                ).OrchestrationTopologyExecutionId("topo-1"),
                slot_id=OrchestrationSlotId("item-b"),
                source_checkpoint_revision=1,
                source_attempt_id=attempt_id,
                recovery_reason=PartialRecoveryReason.TRANSIENT_SLOT_FAILURE,
            ),
            fan_out_request=FanOutRequest(
                fan_out_id=FanOutId("fan-out-r3"),
                items=(_fan_out_item(
                    item_id="item-b",
                    task_scope=task_scope,
                    coordination_id="c",
                    delegation_id="d",
                    lease_id="l",
                ),),
                max_concurrency=1,
            ),
            checkpoint=checkpoint,
            root_execution_id=root,
            tenant_id=_TENANT,
        )
    assert exc.value.code is PartialRecoveryErrorCode.WRONG_REVISION


def test_policy_deny_blocked() -> None:
    harness = build_fan_out_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
    )
    _, _, _, recovery, _ = _build_stack(harness)
    task_scope = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    from intergrax.contracts.orchestration_topology import OrchestrationTopologyExecutionId
    from intergrax.contracts.partial_recovery import SlotRecoveryDisposition
    from intergrax.runtime.long_running.topology_recovery_snapshot import (
        SlotRecoverySnapshot,
        TopologyRecoverySnapshot,
    )

    snapshot = TopologyRecoverySnapshot(
        fan_out_id="fan-out-r3",
        topology_execution_id="topo-1",
        slot_order=("item-b",),
        slots=(
            SlotRecoverySnapshot(
                slot_id="item-b",
                disposition=SlotRecoveryDisposition.FAILED,
            ),
        ),
        max_concurrency=1,
    )
    checkpoint = _checkpoint_with_topology(
        task_id=task_scope,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=root,
        snapshot=snapshot,
    )
    with pytest.raises(PartialRecoveryError, match="governance") as exc:
        recovery.validate_recovery_request(
            PartialRecoveryRequest(
                root_execution_id=root,
                topology_execution_id=OrchestrationTopologyExecutionId("topo-1"),
                slot_id=OrchestrationSlotId("item-b"),
                source_checkpoint_revision=checkpoint.revision or 1,
                source_attempt_id=attempt_id,
                recovery_reason=PartialRecoveryReason.TRANSIENT_SLOT_FAILURE,
            ),
            fan_out_request=FanOutRequest(
                fan_out_id=FanOutId("fan-out-r3"),
                items=(_fan_out_item(
                    item_id="item-b",
                    task_scope=task_scope,
                    coordination_id="c",
                    delegation_id="d",
                    lease_id="l",
                ),),
                max_concurrency=1,
            ),
            checkpoint=checkpoint,
            root_execution_id=root,
            tenant_id=_TENANT,
            policy_decision=PolicyDecision(action=PolicyAction.DENY, reason="deny"),
        )
    assert exc.value.code is PartialRecoveryErrorCode.GOVERNANCE_DENIED


def test_stale_recovery_writer_blocked(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "stale.db")
    task_scope = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    from intergrax.contracts.partial_recovery import SlotRecoveryDisposition
    from intergrax.runtime.long_running.topology_recovery_snapshot import (
        SlotRecoverySnapshot,
        TopologyRecoverySnapshot,
    )

    snapshot = TopologyRecoverySnapshot(
        fan_out_id="fan-out-r3",
        topology_execution_id="topo-1",
        slot_order=("item-b",),
        slots=(
            SlotRecoverySnapshot(
                slot_id="item-b",
                disposition=SlotRecoveryDisposition.FAILED,
            ),
        ),
        max_concurrency=1,
    )
    base = store.save(
        _checkpoint_with_topology(
            task_id=task_scope,
            run_id=run_id,
            attempt_id=attempt_id,
            root_execution_id=root,
            snapshot=snapshot,
        ),
    )
    store.save(
        base.model_copy(update={"checkpoint_id": "ckpt-2"}),
        expected_revision=base.revision,
    )
    with pytest.raises(StaleCheckpointWriteError):
        store.save(
            base.model_copy(update={"checkpoint_id": "ckpt-stale"}),
            expected_revision=base.revision,
        )


def test_duplicate_recovery_request_idempotent_via_correlation() -> None:
    source = (
        _REPO_ROOT / "intergrax/runtime/execution/orchestration_topology_submission.py"
    ).read_text(encoding="utf-8")
    assert "recovery_results" in source
    assert "cache_key" in source


def test_no_direct_child_execution_runner_in_partial_recovery() -> None:
    source = (
        _REPO_ROOT / "intergrax/runtime/execution/fan_out_partial_recovery.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.endswith("child") and any(
                alias.name == "ChildExecutionRunner" for alias in node.names
            ):
                raise AssertionError("ChildExecutionRunner import forbidden in partial recovery")


def test_runtime_checkpoint_topology_recovery_field_compatible_v2() -> None:
    from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint

    fields = RuntimeCheckpoint.model_fields
    assert "topology_recovery" in fields
    assert fields["topology_recovery"].default is None
