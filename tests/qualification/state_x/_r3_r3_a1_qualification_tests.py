# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R3-A1 — Human Decision evidence vs execution authority boundary."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationLookup,
    ExecutionContinuationResolutionCommand,
    ExecutionPauseRequest,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.codecraft.ownership import (
    CodeCraftSessionOwnership,
    codecraft_exec_hitl_notes,
    resolve_codecraft_exec_authorization,
)
from intergrax.runtime.human.models import HumanResponseVerdict, build_human_decision_record
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.human.persistence_contract import InMemoryHumanDecisionPersistence
from intergrax.runtime.human.persistence_errors import HumanDecisionPersistenceValidationError
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.runtime.nexus.orchestration.human_response import (
    HumanDecisionPersistenceError,
    persist_human_decision,
)
from intergrax.tools.registry.wiring import ToolWiringContext
from tests.qualification.state_x._r3_r3_support import (
    SHARED_TASK,
    TENANT_A,
    TENANT_B,
    human_decision_store_factories,
    sample_record,
)
from tests.unit.runtime.human.test_gr5_r3_r1_canonical_first_atomic_projection import (
    _ATTEMPT,
    _EXECUTION,
    _HR,
    _PAUSE,
    _RUN,
    _RecordingContinuationPort,
    _waiting_setup,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OWNERSHIP = _REPO_ROOT / "intergrax/runtime/codecraft/ownership.py"
_INTAKE = _REPO_ROOT / "intergrax/runtime/nexus/orchestration/intake_runner.py"


def test_r3_r3_a1_q01_persisted_approve_cannot_authorize_codecraft_execution(tmp_path: Path) -> None:
    store = SQLiteHumanDecisionStore(db_path=tmp_path / "a1-q01.db")
    notes = codecraft_exec_hitl_notes("craft-a1")
    record = build_human_decision_record(
        task_id=SHARED_TASK,
        tenant_id=TENANT_A,
        approver=local_development_approver_evidence(tenant_id=TENANT_A, actor_id="u"),
        verdict=HumanResponseVerdict.APPROVE,
        response_text="ok",
        run_id="run-a1",
        notes=notes,
    ).model_copy(update={"decision_id": "hdec-a1-q01"})
    store.record(record)
    ctx = ToolWiringContext(human_decision_store=store)
    profile = CodeCraftProfile(mode="supervised", require_hitl_before_exec=True)
    ownership = CodeCraftSessionOwnership(tenant_id=TENANT_A, task_id=SHARED_TASK, run_id="run-a1")
    auth = resolve_codecraft_exec_authorization(
        ctx,
        profile=profile,
        ownership=ownership,
        craft_id="craft-a1",
    )
    assert auth.authorized is False


@pytest.mark.parametrize("store_factory", human_decision_store_factories())
def test_r3_r3_a1_q02_persistence_boundary_rejects_post_construction_mismatch(
    store_factory,
    tmp_path: Path,
) -> None:
    store = store_factory(tmp_path)
    valid = sample_record(decision_id="hdec-a1-q02-valid")
    invalid = valid.model_copy(
        update={"approver": local_development_approver_evidence(tenant_id=TENANT_B, actor_id="x")}
    )
    with pytest.raises(HumanDecisionPersistenceValidationError):
        store.record(invalid)
    assert store.get_decision("hdec-a1-q02-valid", TENANT_A) is None


@pytest.mark.parametrize("store_factory", human_decision_store_factories())
def test_r3_r3_a1_q03_invalid_record_cannot_mutate_provider_state(
    store_factory,
    tmp_path: Path,
) -> None:
    store = store_factory(tmp_path)
    baseline = sample_record(decision_id="hdec-a1-q03-base")
    store.record(baseline)
    before_list = store.list_for_task(SHARED_TASK, TENANT_A)
    invalid = baseline.model_copy(
        update={
            "decision_id": "hdec-a1-q03-invalid",
            "tenant_id": TENANT_B,
        }
    )
    with pytest.raises(HumanDecisionPersistenceValidationError):
        store.record(invalid)
    assert store.list_for_task(SHARED_TASK, TENANT_A) == before_list
    assert store.get_decision("hdec-a1-q03-invalid", TENANT_B) is None


@pytest.mark.parametrize(
    "verdict",
    [
        HumanResponseVerdict.APPROVE,
        HumanResponseVerdict.REJECT,
        HumanResponseVerdict.ESCALATE,
    ],
)
def test_r3_r3_a1_q04_q05_q06_canonical_resolution_precedes_persistence(
    verdict: HumanResponseVerdict,
) -> None:
    task, port, _waiting = _waiting_setup()
    recording = _RecordingContinuationPort(port)
    events: list[str] = []

    class _TrackingStore(InMemoryHumanDecisionPersistence):
        def record(self, record):  # noqa: ANN001
            events.append("persist")
            return super().record(record)

    store = _TrackingStore()
    approver = local_development_approver_evidence(tenant_id=task.tenant_id, actor_id="op")
    HumanPauseCoordinator.resolve_human_response_and_apply_canonical(
        task,
        verdict,
        approver=approver,
        continuation=recording,
        pause_id=_PAUSE,
        human_request_id=_HR,
        run_id=str(_RUN),
        attempt_id=str(_ATTEMPT),
        execution_id=str(_EXECUTION),
    )
    assert "apply_resolution" in recording.events
    events.append("canonical")
    persist_human_decision(task, verdict, human_store=store)
    assert events.index("canonical") < events.index("persist")


def test_r3_r3_a1_q07_canonical_resolution_failure_persists_nothing() -> None:
    task, port, _waiting = _waiting_setup()
    store = InMemoryHumanDecisionPersistence()

    class _FailPort:
        def get_pending(self, lookup: ExecutionContinuationLookup) -> PendingExecutionContinuation:
            return port.get_pending(lookup)

        def apply_resolution(
            self,
            command: ExecutionContinuationResolutionCommand,
        ) -> PendingExecutionContinuation:
            raise ExecutionContinuationError(
                "stale",
                code=ExecutionContinuationErrorCode.STALE_REVISION,
            )

        def request_pause(self, request: ExecutionPauseRequest) -> PendingExecutionContinuation:
            return port.request_pause(request)

        def resume(self, command):  # noqa: ANN001
            return port.resume(command)

    approver = local_development_approver_evidence(tenant_id=task.tenant_id, actor_id="op")
    with pytest.raises(ExecutionContinuationError):
        HumanPauseCoordinator.resolve_human_response_and_apply_canonical(
            task,
            HumanResponseVerdict.APPROVE,
            approver=approver,
            continuation=_FailPort(),
            pause_id=_PAUSE,
            human_request_id=_HR,
            run_id=str(_RUN),
            attempt_id=str(_ATTEMPT),
            execution_id=str(_EXECUTION),
        )
    with pytest.raises(HumanDecisionPersistenceError):
        persist_human_decision(task, HumanResponseVerdict.APPROVE, human_store=store)
    assert store.summarize_queue(task.tenant_id) == {}


def test_r3_r3_a1_q08_evidence_persistence_failure_is_not_swallowed() -> None:
    task, port, _waiting = _waiting_setup()
    store = InMemoryHumanDecisionPersistence()
    approver = local_development_approver_evidence(tenant_id=task.tenant_id, actor_id="op")
    HumanPauseCoordinator.resolve_human_response_and_apply_canonical(
        task,
        HumanResponseVerdict.APPROVE,
        approver=approver,
        continuation=port,
        pause_id=_PAUSE,
        human_request_id=_HR,
        run_id=str(_RUN),
        attempt_id=str(_ATTEMPT),
        execution_id=str(_EXECUTION),
    )

    def boom(record):  # noqa: ANN001
        raise HumanDecisionPersistenceValidationError(
            "forced",
            decision_id=record.decision_id,
            tenant_id=record.tenant_id,
        )

    with patch.object(store, "record", side_effect=boom):
        with pytest.raises(HumanDecisionPersistenceValidationError):
            persist_human_decision(task, HumanResponseVerdict.APPROVE, human_store=store)


def test_r3_r3_a1_q14_ownership_has_no_evidence_derived_allow_path() -> None:
    tree = ast.parse(_OWNERSHIP.read_text(encoding="utf-8"))
    resolve_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_codecraft_exec_authorization"
    )
    source_segment = ast.get_source_segment(_OWNERSHIP.read_text(encoding="utf-8"), resolve_fn) or ""
    lowered = "\n".join(
        line for line in source_segment.splitlines() if not line.strip().startswith('"""')
    )
    assert "HumanDecisionRecord" not in lowered
    assert "HumanResponseVerdict" not in lowered
    assert "human_decision_store" not in lowered
    assert "list_for_task" not in lowered


def test_r3_r3_a1_q18_provider_parity_still_parametrized() -> None:
    assert len(tuple(human_decision_store_factories())) == 2


def test_r3_r3_a1_q20_codecraft_execution_inventory_complete() -> None:
    patterns = (
        "CodeCraftOrchestrator",
        "WiringCodeCraftBoundCapabilityExecution",
        "codecraft_run(",
    )
    hits: set[str] = set()
    for path in (_REPO_ROOT / "intergrax").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(p in text for p in patterns):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if "runtime/codecraft" in rel or "tools/providers/codecraft" in rel:
                hits.add(rel)
    assert "intergrax/runtime/codecraft/orchestrator.py" in hits
    assert "intergrax/runtime/codecraft/wiring_bound_capability_execution.py" in hits
    assert "intergrax/tools/providers/codecraft/service.py" in hits


def test_r3_r3_a1_q16_intake_runner_orders_canonical_before_persist_for_approve() -> None:
    text = _INTAKE.read_text(encoding="utf-8")
    approve_idx = text.index("if verdict == HumanResponseVerdict.APPROVE:")
    segment = text[approve_idx : approve_idx + 8000]
    resolve_idx = segment.index("resolve_human_response_and_apply_canonical")
    persist_idx = segment.index("self.hitl.persist_human_decision")
    assert resolve_idx < persist_idx


def test_r3_r3_a1_q13_autonomous_profile_regression() -> None:
    profile = CodeCraftProfile(mode="autonomous", require_hitl_before_exec=False)
    auth = resolve_codecraft_exec_authorization(
        ToolWiringContext(),
        profile=profile,
        ownership=CodeCraftSessionOwnership(tenant_id=TENANT_A, task_id=SHARED_TASK, run_id="run-a"),
        craft_id="craft-auto",
    )
    assert auth.authorized is True


def test_r3_r3_a1_q15_no_hitl_approved_execution_bypass_in_codecraft_production() -> None:
    orchestrator = (_REPO_ROOT / "intergrax/runtime/codecraft/orchestrator.py").read_text(
        encoding="utf-8"
    )
    service = (_REPO_ROOT / "intergrax/tools/providers/codecraft/service.py").read_text(
        encoding="utf-8"
    )
    assert "if session.hitl_approved" not in orchestrator
    assert "if params.hitl_approved" not in service
    assert "hitl_approved" not in _OWNERSHIP.read_text(encoding="utf-8")


def test_r3_r3_a1_q17_hitl_tool_service_is_evidence_query_only() -> None:
    path = _REPO_ROOT / "intergrax/tools/providers/hitl/service.py"
    text = path.read_text(encoding="utf-8")
    assert "ExecutionContinuationPort" not in text
    assert "create_grant" not in text.lower()
    assert "resume" not in text or "human_decision_store_not_configured" in text


def test_r3_r3_a1_validate_helper_single_owner() -> None:
    path = _REPO_ROOT / "intergrax/runtime/human/persistence_validation.py"
    assert path.is_file()
    contract = (_REPO_ROOT / "intergrax/runtime/human/persistence_contract.py").read_text(
        encoding="utf-8"
    )
    store = (_REPO_ROOT / "intergrax/runtime/human/store.py").read_text(encoding="utf-8")
    assert "validate_human_decision_for_persistence" in contract
    assert "validate_human_decision_for_persistence" in store
    assert contract.count("validate_human_decision_for_persistence") == 2
    assert store.count("validate_human_decision_for_persistence") == 2
