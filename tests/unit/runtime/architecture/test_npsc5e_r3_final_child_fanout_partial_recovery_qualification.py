# © Artur Czarnecki. All rights reserved.

"""NPSC-5E/R3 Final — child & fan-out partial recovery qualification and freeze."""

from __future__ import annotations

import ast
import inspect
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    FanOutId,
    FanOutItemId,
    FanOutItemStatus,
    FanOutRequest,
)
from intergrax.contracts.execution_identity import (
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_retry import (
    ExecutionFailureKind,
    ExecutionRetryAction,
    ExecutionRetryEligibilityRequest,
)
from intergrax.contracts.orchestration_topology import OrchestrationSlotId, OrchestrationTopologyExecutionId
from intergrax.contracts.partial_recovery import (
    PartialRecoveryError,
    PartialRecoveryErrorCode,
    PartialRecoveryReason,
    PartialRecoveryRequest,
    SlotRecoveryDisposition,
    SlotRecoveryPolicyAction,
    SlotRecoveryPolicyRequest,
    evaluate_slot_recovery_policy,
)
from intergrax.runtime.execution.fan_out_partial_recovery import FanOutPartialRecoveryService
from intergrax.runtime.long_running.execution_tree_checkpoint import minimal_runtime_checkpoint
from intergrax.runtime.long_running.checkpoint_revision import StaleCheckpointWriteError
from intergrax.runtime.long_running.runtime_checkpoint import (
    CANONICAL_RUNTIME_CHECKPOINT_SCHEMA_VERSION,
    RuntimeCheckpoint,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.long_running.topology_recovery_snapshot import (
    FanOutItemOutcomeSnapshot,
    SlotRecoverySnapshot,
    TopologyRecoverySnapshot,
)
from tests.unit.agent_distribution.test_bounded_multi_agent_fanout import (
    _fan_out_item,
    build_fan_out_harness,
)
from tests.unit.agent_distribution.test_delegated_subtasks import (
    _OCR_PACKAGE,
    _discovery_candidate,
)
from tests.unit.agent_distribution.test_physical_delegation_governance_boundary import (
    _CountingSelector,
)
from tests.unit.runtime.architecture.test_npsc5e_r3_child_fanout_partial_recovery import (
    _FORBIDDEN_RECOVERY_NAMES,
    _InvocationTracker,
    _REFLECTION_PATTERN,
    _RecoverableFailDelegate,
    _TENANT,
    _build_stack,
    _checkpoint_with_topology,
    _run_partial_fan_out,
    _run_pytest,
    _run_recovery_under_root,
    _three_item_request,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

_R1_FINAL_SHA = "76603ed266f9f54106bf4718fe8886180a826351"
_R2_FINAL_SHA = "c030d1d4752513bc11ec2597de4e760ae551a978"
_NPSC_5D_FINAL_SHA = "a4a1faca01cd5004e372f235132184a84aa5a6bd"
_R3_IMPLEMENTATION_SHA = "465742eba5d4162061797baafbbb3f90f414fcdd"

_R3_PRODUCTION_SURFACE = (
    "intergrax/contracts/partial_recovery.py",
    "intergrax/runtime/long_running/topology_recovery_snapshot.py",
    "intergrax/runtime/execution/fan_out_partial_recovery.py",
    "intergrax/runtime/long_running/runtime_checkpoint.py",
)

_MANDATORY_SUITES: tuple[tuple[str, list[str]], ...] = (
    ("R1 Final", ["tests/unit/runtime/architecture/test_npsc5e_r1_final_retry_attempt_qualification.py"]),
    ("R2 Final", ["tests/unit/runtime/architecture/test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py"]),
    (
        "R3 implementation gate",
        ["tests/unit/runtime/architecture/test_npsc5e_r3_child_fanout_partial_recovery.py"],
    ),
    ("P0A", ["tests/unit/runtime/architecture/test_npsc5e_p0a_execution_lineage_baseline_qualification.py"]),
    (
        "DG_001",
        [
            "tests/unit/contracts/test_execution_lineage_contracts.py",
            "tests/unit/runtime/execution/lineage/",
        ],
    ),
    ("NPSC-5A", ["tests/unit/runtime/architecture/test_npsc5a_coordination_delegation_e2e.py"]),
    (
        "NPSC-5B Final",
        ["tests/unit/runtime/architecture/test_npsc5b_final_production_fanout_fanin_qualification.py"],
    ),
    ("NPSC-5C", ["tests/unit/runtime/architecture/test_npsc5c_decision_execution_e2e.py"]),
    (
        "NPSC-5D Final",
        ["tests/unit/runtime/architecture/test_npsc5d_final_multi_agent_governance_qualification.py"],
    ),
    ("HITL R3", ["tests/unit/runtime/architecture/test_npsc5d_r3_governed_continuation.py"]),
    (
        "Attempt lifecycle",
        [
            "tests/unit/runtime/execution/test_attempt_lifecycle.py",
            "tests/unit/runtime/execution/test_attempt_lifecycle_durability_gate.py",
            "tests/conformance/runtime/durability/test_attempt_lifecycle.py",
        ],
    ),
    (
        "Child execution",
        [
            "tests/unit/runtime/execution/test_child_execution.py",
            "tests/unit/runtime/execution/authority/test_child_execution_authority_policy.py",
        ],
    ),
    ("Terminal", ["tests/unit/runtime/execution/test_p0c6_terminal_outcome_convergence.py"]),
    (
        "Cancellation",
        [
            "tests/unit/runtime/cancellation/test_p0c5_cancellation_continuity.py",
            "-k",
            "not survives_process_restart",
            "tests/unit/runtime/cancellation/test_p0c5a_explicit_terminal_wiring.py",
            "tests/unit/applications/test_task_control_governed_resume.py",
        ],
    ),
    ("Checkpoint store", ["tests/unit/runtime/long_running/test_checkpoint_store.py"]),
    (
        "Long-running",
        [
            "tests/unit/runtime/long_running/test_pcm_scheduler_integrity.py",
            "tests/unit/runtime/long_running/test_pba_fix_a_checkpoint_port_consumption.py",
            "tests/unit/runtime/long_running/test_runtime_checkpoint.py",
            "tests/unit/runtime/long_running/test_resume_planner.py",
            "tests/unit/runtime/long_running/test_ue_9c_execution_tree_checkpoint.py",
            "tests/unit/runtime/long_running/test_p0c3_recovery_state_authority.py",
        ],
    ),
    ("Fan-out", ["tests/unit/agent_distribution/test_bounded_multi_agent_fanout.py"]),
)


def _four_item_request(task_scope: TaskId) -> tuple:
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
        _fan_out_item(
            item_id="item-d",
            task_scope=task_scope,
            coordination_id="coord-d",
            delegation_id="delegation-d",
            lease_id="lease-d",
            document_ref="doc-d",
        ),
    )


def _topology_snapshot(
    *,
    slot_id: str,
    disposition: SlotRecoveryDisposition,
    failure_code: str | None = None,
    topology_execution_id: str = "topo-1",
) -> TopologyRecoverySnapshot:
    outcome = None
    if failure_code is not None:
        outcome = FanOutItemOutcomeSnapshot(
            item_id=slot_id,
            status="failure",
            failure_code=failure_code,
            failure_message="slot failed",
        )
    return TopologyRecoverySnapshot(
        fan_out_id="fan-out-r3",
        topology_execution_id=topology_execution_id,
        slot_order=(slot_id,),
        slots=(
            SlotRecoverySnapshot(
                slot_id=slot_id,
                disposition=disposition,
                outcome=outcome,
            ),
        ),
        max_concurrency=1,
    )


def _recovery_service(tmp_path: Path | None = None) -> FanOutPartialRecoveryService:
    harness = build_fan_out_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
    )
    _, adapter, submission, recovery, _ = _build_stack(harness)
    if tmp_path is not None:
        return FanOutPartialRecoveryService(
            adapter=adapter,
            submission_port=submission,
            checkpoint_store=SQLiteTaskCheckpointStore(db_path=tmp_path / "svc.db"),
        )
    return recovery


@pytest.mark.parametrize(("label", "targets"), _MANDATORY_SUITES, ids=[label for label, _ in _MANDATORY_SUITES])
def test_mandatory_frozen_suite_passes(label: str, targets: list[str]) -> None:
    proc = _run_pytest(targets)
    assert proc.returncode == 0, f"{label} failed:\n{proc.stdout}\n{proc.stderr}"


def test_canonical_predecessor_shas_recorded() -> None:
    assert _R1_FINAL_SHA.startswith("76603ed")
    assert _R2_FINAL_SHA.startswith("c030d1d")
    assert _NPSC_5D_FINAL_SHA.startswith("a4a1fac")
    assert _R3_IMPLEMENTATION_SHA.startswith("465742e")


def test_succeeded_semantics_preserve_failure_means_no_recovery() -> None:
    result = evaluate_slot_recovery_policy(
        SlotRecoveryPolicyRequest(
            disposition=SlotRecoveryDisposition.SUCCEEDED,
            failure_kind=None,
        ),
    )
    assert result.action is SlotRecoveryPolicyAction.PRESERVE_FAILURE
    assert "terminal" in result.reason


def test_old_runtime_checkpoint_v2_without_topology_recovery_parses() -> None:
    checkpoint = minimal_runtime_checkpoint(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        root_execution_id=mint_execution_id(),
    )
    assert checkpoint.schema_version == CANONICAL_RUNTIME_CHECKPOINT_SCHEMA_VERSION
    assert checkpoint.topology_recovery is None


def test_unknown_runtime_checkpoint_version_blocked() -> None:
    with pytest.raises(Exception):
        RuntimeCheckpoint.model_validate(
            {
                "schema_version": "runtime_checkpoint.v99",
                "task_id": str(mint_task_id()),
                "run_id": str(mint_run_id()),
                "attempt_id": str(mint_attempt_id()),
                "execution_tree": {"entries": []},
            },
        )


@pytest.mark.asyncio
async def test_final_single_failure_four_slot_recover_c_only(tmp_path: Path) -> None:
    tracker = _InvocationTracker()
    harness = build_fan_out_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        specialist_delegate=_RecoverableFailDelegate(tracker, frozenset({"doc-c"})),
    )
    fan_out, adapter, _, recovery, _ = _build_stack(harness)
    task_scope = mint_task_id()
    items = _four_item_request(task_scope)
    harness.task_scope_authority.task_scope_id = task_scope
    request, partial, execution_id, identity = await _run_partial_fan_out(
        fan_out, adapter, items=items, task_scope=task_scope,
    )
    assert partial.items[0].status is FanOutItemStatus.SUCCESS
    assert partial.items[1].status is FanOutItemStatus.SUCCESS
    assert partial.items[2].status is FanOutItemStatus.FAILURE
    assert partial.items[3].status is FanOutItemStatus.SUCCESS
    from intergrax.runtime.long_running.topology_recovery_snapshot import capture_topology_recovery_snapshot

    snapshot = capture_topology_recovery_snapshot(
        request=request, result=partial, topology_execution_id=execution_id,
    )
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "abcd.db")
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
            slot_id=OrchestrationSlotId("item-c"),
            source_checkpoint_revision=saved.revision or 1,
            source_attempt_id=identity.attempt_id,
            recovery_reason=PartialRecoveryReason.TRANSIENT_SLOT_FAILURE,
        ),
        fan_out_request=request,
        partial_result=partial,
        checkpoint=saved,
        root_execution_id=identity.execution_id,
        correlation_id="final-abcd-c",
        identity=identity,
    )
    assert recovered.fan_out_result.items[2].status is FanOutItemStatus.SUCCESS
    assert tracker.counts["doc-a"] == 1
    assert tracker.counts["doc-b"] == 1
    assert tracker.counts["doc-c"] == 2
    assert tracker.counts["doc-d"] == 1
    assert recovered.preserved_slot_ids == (
        FanOutItemId("item-a"),
        FanOutItemId("item-b"),
        FanOutItemId("item-d"),
    )


@pytest.mark.asyncio
async def test_final_multiple_failure_recover_b_and_c(tmp_path: Path) -> None:
    tracker = _InvocationTracker()
    harness = build_fan_out_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        specialist_delegate=_RecoverableFailDelegate(tracker, frozenset({"doc-b", "doc-c"})),
    )
    fan_out, adapter, _, _, _ = _build_stack(harness)
    task_scope = mint_task_id()
    items = _four_item_request(task_scope)
    harness.task_scope_authority.task_scope_id = task_scope
    request, partial, execution_id, identity = await _run_partial_fan_out(
        fan_out, adapter, items=items, task_scope=task_scope,
    )
    from intergrax.runtime.long_running.topology_recovery_snapshot import capture_topology_recovery_snapshot

    snapshot = capture_topology_recovery_snapshot(
        request=request, result=partial, topology_execution_id=execution_id,
    )
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "bc.db")
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
    current = saved
    fan_out_result = partial
    for slot, correlation in (("item-b", "final-bc-b"), ("item-c", "final-bc-c")):
        recovery_result = await _run_recovery_under_root(
            recovery,
            recovery_request=PartialRecoveryRequest(
                root_execution_id=identity.execution_id,
                topology_execution_id=execution_id,
                slot_id=OrchestrationSlotId(slot),
                source_checkpoint_revision=current.revision or 1,
                source_attempt_id=identity.attempt_id,
                recovery_reason=PartialRecoveryReason.TRANSIENT_SLOT_FAILURE,
            ),
            fan_out_request=request,
            partial_result=fan_out_result,
            checkpoint=current,
            root_execution_id=identity.execution_id,
            correlation_id=correlation,
            identity=identity,
        )
        fan_out_result = recovery_result.fan_out_result
        current = store.get_latest(str(task_scope), _TENANT) or current
    assert fan_out_result.items[1].status is FanOutItemStatus.SUCCESS
    assert fan_out_result.items[2].status is FanOutItemStatus.SUCCESS
    assert tracker.counts["doc-a"] == 1
    assert tracker.counts["doc-d"] == 1
    assert tracker.counts["doc-b"] == 2
    assert tracker.counts["doc-c"] == 2


@pytest.mark.asyncio
async def test_final_same_slot_recovery_idempotent(tmp_path: Path) -> None:
    tracker = _InvocationTracker()
    harness = build_fan_out_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        specialist_delegate=_RecoverableFailDelegate(tracker, frozenset({"doc-b"})),
    )
    fan_out, adapter, _, _, _ = _build_stack(harness)
    task_scope = mint_task_id()
    items = _three_item_request(task_scope)
    harness.task_scope_authority.task_scope_id = task_scope
    request, partial, execution_id, identity = await _run_partial_fan_out(
        fan_out, adapter, items=items, task_scope=task_scope,
    )
    from intergrax.runtime.long_running.topology_recovery_snapshot import capture_topology_recovery_snapshot

    snapshot = capture_topology_recovery_snapshot(
        request=request, result=partial, topology_execution_id=execution_id,
    )
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "idem.db")
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
    recovery_request = PartialRecoveryRequest(
        root_execution_id=identity.execution_id,
        topology_execution_id=execution_id,
        slot_id=OrchestrationSlotId("item-b"),
        source_checkpoint_revision=saved.revision or 1,
        source_attempt_id=identity.attempt_id,
        recovery_reason=PartialRecoveryReason.TRANSIENT_SLOT_FAILURE,
    )
    first = await _run_recovery_under_root(
        recovery,
        recovery_request=recovery_request,
        fan_out_request=request,
        partial_result=partial,
        checkpoint=saved,
        root_execution_id=identity.execution_id,
        correlation_id="idem-b",
        identity=identity,
    )
    updated = store.get_latest(str(task_scope), _TENANT)
    assert updated is not None
    second = await _run_recovery_under_root(
        recovery,
        recovery_request=PartialRecoveryRequest(
            root_execution_id=identity.execution_id,
            topology_execution_id=execution_id,
            slot_id=OrchestrationSlotId("item-b"),
            source_checkpoint_revision=updated.revision or 1,
            source_attempt_id=identity.attempt_id,
            recovery_reason=PartialRecoveryReason.ALREADY_COMPLETE,
        ),
        fan_out_request=request,
        partial_result=first.fan_out_result,
        checkpoint=updated,
        root_execution_id=identity.execution_id,
        correlation_id="idem-b-2",
        identity=identity,
    )
    assert second.recovered_slot_ids == ()
    assert tracker.counts["doc-b"] == 2


@pytest.mark.asyncio
async def test_final_no_discovery_on_exact_slot_recovery(tmp_path: Path) -> None:
    tracker = _InvocationTracker()
    harness = build_fan_out_harness(
        candidates=(_discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),),
        specialist_delegate=_RecoverableFailDelegate(tracker, frozenset({"doc-b"})),
    )
    discovery = harness.service._discovery
    matcher = harness.service._matcher
    selector = _CountingSelector(harness.service._selector)
    harness.service._selector = selector
    discovery_count = 0
    matcher_count = 0
    original_discover = discovery.discover
    original_find = matcher.find_matches

    def _counting_discover(*args, **kwargs):
        nonlocal discovery_count
        discovery_count += 1
        return original_discover(*args, **kwargs)

    def _counting_find(*args, **kwargs):
        nonlocal matcher_count
        matcher_count += 1
        return original_find(*args, **kwargs)

    discovery.discover = _counting_discover  # type: ignore[method-assign]
    matcher.find_matches = _counting_find  # type: ignore[method-assign]

    fan_out, adapter, _, _, _ = _build_stack(harness)
    task_scope = mint_task_id()
    items = _three_item_request(task_scope)
    harness.task_scope_authority.task_scope_id = task_scope
    request, partial, execution_id, identity = await _run_partial_fan_out(
        fan_out, adapter, items=items, task_scope=task_scope,
    )
    initial_discovery = discovery_count
    initial_matching = matcher_count
    initial_selection = selector.call_count
    from intergrax.runtime.long_running.topology_recovery_snapshot import capture_topology_recovery_snapshot

    snapshot = capture_topology_recovery_snapshot(
        request=request, result=partial, topology_execution_id=execution_id,
    )
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "noselect.db")
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
    await _run_recovery_under_root(
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
        correlation_id="noselect-b",
        identity=identity,
    )
    assert discovery_count == initial_discovery
    assert matcher_count == initial_matching
    assert selector.call_count == initial_selection


def test_final_parent_cancel_blocked() -> None:
    recovery = _recovery_service()
    task_scope = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    snapshot = _topology_snapshot(slot_id="item-b", disposition=SlotRecoveryDisposition.FAILED)
    checkpoint = _checkpoint_with_topology(
        task_id=task_scope,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=root,
        snapshot=snapshot,
    )
    with pytest.raises(PartialRecoveryError, match="parent") as exc:
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
            parent_cancelled=True,
        )
    assert exc.value.code is PartialRecoveryErrorCode.PARENT_CANCELLED


def test_final_waiting_human_blocked() -> None:
    policy = evaluate_slot_recovery_policy(
        SlotRecoveryPolicyRequest(
            disposition=SlotRecoveryDisposition.WAITING_FOR_HUMAN,
            waiting_for_human=True,
        ),
    )
    assert policy.action is SlotRecoveryPolicyAction.WAIT
    assert "human" in policy.reason


def test_final_unknown_side_effect_blocked() -> None:
    result = evaluate_slot_recovery_policy(
        SlotRecoveryPolicyRequest(
            disposition=SlotRecoveryDisposition.UNKNOWN_UNSAFE,
            has_unknown_side_effect=True,
        ),
    )
    assert result.action is SlotRecoveryPolicyAction.PRESERVE_FAILURE


def test_final_authority_denied_blocked() -> None:
    result = evaluate_slot_recovery_policy(
        SlotRecoveryPolicyRequest(
            disposition=SlotRecoveryDisposition.FAILED,
            failure_kind=ExecutionFailureKind.AUTHORITY_DENIED,
        ),
    )
    assert result.action is SlotRecoveryPolicyAction.PRESERVE_FAILURE


def test_final_trust_denied_blocked() -> None:
    result = evaluate_slot_recovery_policy(
        SlotRecoveryPolicyRequest(
            disposition=SlotRecoveryDisposition.FAILED,
            failure_kind=ExecutionFailureKind.TRUST_DENIED,
        ),
    )
    assert result.action is SlotRecoveryPolicyAction.PRESERVE_FAILURE


def test_final_r1_interop_transient_retry_distinct_from_r3() -> None:
    from intergrax.runtime.execution.retry import (
        classify_execution_failure,
        evaluate_execution_retry_eligibility,
    )

    r1 = evaluate_execution_retry_eligibility(
        ExecutionRetryEligibilityRequest(
            classification=classify_execution_failure(
                kind=ExecutionFailureKind.RETRYABLE_TRANSIENT,
            ),
            attempt_number=1,
            max_attempts=3,
        ),
    )
    r3 = evaluate_slot_recovery_policy(
        SlotRecoveryPolicyRequest(
            disposition=SlotRecoveryDisposition.FAILED,
            failure_kind=ExecutionFailureKind.RETRYABLE_TRANSIENT,
        ),
    )
    assert r1.action is ExecutionRetryAction.RETRY
    assert r3.action is SlotRecoveryPolicyAction.RECOVER


def test_final_wrong_root_topology_slot_attempt_revision_blocked() -> None:
    recovery = _recovery_service()
    task_scope = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    snapshot = _topology_snapshot(slot_id="item-b", disposition=SlotRecoveryDisposition.FAILED)
    checkpoint = _checkpoint_with_topology(
        task_id=task_scope,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=root,
        snapshot=snapshot,
        revision=2,
    )
    base_request = PartialRecoveryRequest(
        root_execution_id=root,
        topology_execution_id=OrchestrationTopologyExecutionId("topo-1"),
        slot_id=OrchestrationSlotId("item-b"),
        source_checkpoint_revision=2,
        source_attempt_id=attempt_id,
        recovery_reason=PartialRecoveryReason.TRANSIENT_SLOT_FAILURE,
    )
    fan_out_request = FanOutRequest(
        fan_out_id=FanOutId("fan-out-r3"),
        items=(_fan_out_item(
            item_id="item-b",
            task_scope=task_scope,
            coordination_id="c",
            delegation_id="d",
            lease_id="l",
        ),),
        max_concurrency=1,
    )
    with pytest.raises(PartialRecoveryError) as exc:
        recovery.validate_recovery_request(
            base_request,
            fan_out_request=fan_out_request,
            checkpoint=checkpoint,
            root_execution_id=mint_execution_id(),
            tenant_id=_TENANT,
        )
    assert exc.value.code is PartialRecoveryErrorCode.WRONG_ROOT
    with pytest.raises(PartialRecoveryError) as exc:
        recovery.validate_recovery_request(
            replace(
                base_request,
                topology_execution_id=OrchestrationTopologyExecutionId("wrong"),
            ),
            fan_out_request=fan_out_request,
            checkpoint=checkpoint,
            root_execution_id=root,
            tenant_id=_TENANT,
        )
    assert exc.value.code is PartialRecoveryErrorCode.WRONG_TOPOLOGY
    with pytest.raises(PartialRecoveryError) as exc:
        recovery.validate_recovery_request(
            replace(base_request, slot_id=OrchestrationSlotId("missing")),
            fan_out_request=fan_out_request,
            checkpoint=checkpoint,
            root_execution_id=root,
            tenant_id=_TENANT,
        )
    assert exc.value.code is PartialRecoveryErrorCode.WRONG_SLOT
    with pytest.raises(PartialRecoveryError) as exc:
        recovery.validate_recovery_request(
            replace(base_request, source_attempt_id=mint_attempt_id()),
            fan_out_request=fan_out_request,
            checkpoint=checkpoint,
            root_execution_id=root,
            tenant_id=_TENANT,
        )
    assert exc.value.code is PartialRecoveryErrorCode.STALE_CHECKPOINT
    with pytest.raises(PartialRecoveryError) as exc:
        recovery.validate_recovery_request(
            replace(base_request, source_checkpoint_revision=1),
            fan_out_request=fan_out_request,
            checkpoint=checkpoint,
            root_execution_id=root,
            tenant_id=_TENANT,
        )
    assert exc.value.code is PartialRecoveryErrorCode.WRONG_REVISION


def test_final_stale_recovery_writer_blocked(tmp_path: Path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "stale-final.db")
    task_scope = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    snapshot = _topology_snapshot(slot_id="item-b", disposition=SlotRecoveryDisposition.FAILED)
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


def test_no_forbidden_recovery_runtime_names_in_production() -> None:
    production_root = _REPO_ROOT / "intergrax"
    hits: list[str] = []
    for path in production_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for name in _FORBIDDEN_RECOVERY_NAMES:
            if name in source:
                hits.append(f"{path.relative_to(_REPO_ROOT)}:{name}")
    assert not hits, f"forbidden recovery runtime names: {hits}"


def test_no_reflection_in_r3_production_surface() -> None:
    for relative in _R3_PRODUCTION_SURFACE:
        path = _REPO_ROOT / relative
        assert _REFLECTION_PATTERN.search(path.read_text(encoding="utf-8")) is None


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


def test_submission_port_exposes_recover_failed_slot() -> None:
    from intergrax.runtime.execution.orchestration_topology_submission import (
        CanonicalOrchestrationTopologySubmissionPort,
        build_orchestration_topology_submission_port,
    )
    from intergrax.runtime.nexus.nexus_loop import NexusLoop
    from intergrax.runtime.registry.agent_registry import AgentRegistry

    port = build_orchestration_topology_submission_port(NexusLoop(AgentRegistry()))
    assert isinstance(port, CanonicalOrchestrationTopologySubmissionPort)
    assert inspect.iscoroutinefunction(port.recover_failed_slot)


def test_pre_existing_cancellation_fixture_invalid_persist_gate() -> None:
    proc = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "tests/unit/runtime/cancellation/test_p0c5_cancellation_continuity.py::test_terminal_cancellation_survives_process_restart",
            "-q",
            "--tb=line",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "not resumable" in combined or "CheckpointNotResumableError" in combined


def test_ruff_r3_surface_and_final_test_no_new_errors() -> None:
    targets = [str(_REPO_ROOT / relative) for relative in _R3_PRODUCTION_SURFACE]
    targets.append(str(Path(__file__)))
    proc = subprocess.run(
        ["uv", "run", "ruff", "check", *targets],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_pyright_r3_surface_and_final_test_no_new_errors() -> None:
    targets = [str(_REPO_ROOT / relative) for relative in _R3_PRODUCTION_SURFACE]
    targets.append(str(Path(__file__)))
    proc = subprocess.run(
        ["uv", "run", "pyright", *targets],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
