# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R3 mechanical qualification (SX-F12 Human Decision / HITL Persistence)."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.runtime.codecraft.ownership import (
    CodeCraftSessionOwnership,
    codecraft_exec_hitl_notes,
    resolve_codecraft_exec_authorization,
)
from intergrax.runtime.human.models import HumanDecisionRecord, HumanResponseVerdict, build_human_decision_record
from intergrax.runtime.human.persistence_contract import (
    HumanDecisionPersistence,
    InMemoryHumanDecisionPersistence,
)
from intergrax.runtime.human.persistence_errors import (
    HumanDecisionApproverProvenanceError,
    HumanDecisionPersistenceConflictError,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.runtime.persistence.sqlite_composition import create_sqlite_human_decision_store
from intergrax.tools.registry.wiring import ToolWiringContext
from tests.qualification.state_x._r3_r3_support import (
    DECISION_ID,
    SHARED_TASK,
    TENANT_A,
    TENANT_B,
    HumanDecisionStoreFactory,
    human_decision_store_factories,
    sample_record,
    sqlite_human_decision_store,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PERSISTENCE_CONTRACT = _REPO_ROOT / "intergrax/runtime/human/persistence_contract.py"
_OWNERSHIP = _REPO_ROOT / "intergrax/runtime/codecraft/ownership.py"
_HUMAN_RESPONSE = _REPO_ROOT / "intergrax/runtime/nexus/orchestration/human_response.py"


def test_r3_r3_q01_closed_world_owner_provider_inventory() -> None:
    contract_text = _PERSISTENCE_CONTRACT.read_text(encoding="utf-8")
    assert "class HumanDecisionPersistence" in contract_text
    assert "class InMemoryHumanDecisionPersistence" in contract_text
    store_text = (_REPO_ROOT / "intergrax/runtime/human/store.py").read_text(encoding="utf-8")
    assert "class SQLiteHumanDecisionStore" in store_text
    composition = (_REPO_ROOT / "intergrax/runtime/persistence/sqlite_composition.py").read_text(
        encoding="utf-8"
    )
    assert "create_sqlite_human_decision_store" in composition


@pytest.mark.parametrize("store_factory", human_decision_store_factories())
def test_r3_r3_q02_provider_contract_instance(
    store_factory: HumanDecisionStoreFactory,
    tmp_path: Path,
) -> None:
    store = store_factory(tmp_path)
    assert isinstance(store, HumanDecisionPersistence)


@pytest.mark.parametrize("store_factory", human_decision_store_factories())
def test_r3_r3_q03_duplicate_decision_id_rejected(
    store_factory: HumanDecisionStoreFactory,
    tmp_path: Path,
) -> None:
    store = store_factory(tmp_path)
    first = sample_record()
    second = sample_record(verdict=HumanResponseVerdict.REJECT, notes="overwrite-attempt")
    store.record(first)
    with pytest.raises(HumanDecisionPersistenceConflictError):
        store.record(second)


@pytest.mark.parametrize("store_factory", human_decision_store_factories())
def test_r3_r3_q04_duplicate_cannot_overwrite_canonical_record(
    store_factory: HumanDecisionStoreFactory,
    tmp_path: Path,
) -> None:
    store = store_factory(tmp_path)
    first = sample_record()
    store.record(first)
    with pytest.raises(HumanDecisionPersistenceConflictError):
        store.record(sample_record(verdict=HumanResponseVerdict.REJECT))
    loaded = store.get_decision(DECISION_ID, TENANT_A)
    assert loaded is not None
    assert loaded.verdict is HumanResponseVerdict.APPROVE


@pytest.mark.parametrize("store_factory", human_decision_store_factories())
def test_r3_r3_q05_wrong_tenant_get_denied(
    store_factory: HumanDecisionStoreFactory,
    tmp_path: Path,
) -> None:
    store = store_factory(tmp_path)
    store.record(sample_record())
    assert store.get_decision(DECISION_ID, TENANT_B) is None


@pytest.mark.parametrize("store_factory", human_decision_store_factories())
def test_r3_r3_q06_same_task_cross_tenant_list_isolation(
    store_factory: HumanDecisionStoreFactory,
    tmp_path: Path,
) -> None:
    store = store_factory(tmp_path)
    store.record(sample_record(decision_id="hdec-a", tenant_id=TENANT_A))
    store.record(sample_record(decision_id="hdec-b", tenant_id=TENANT_B))
    listed_a = store.list_for_task(SHARED_TASK, TENANT_A)
    listed_b = store.list_for_task(SHARED_TASK, TENANT_B)
    assert len(listed_a) == 1 and listed_a[0].tenant_id == TENANT_A
    assert len(listed_b) == 1 and listed_b[0].tenant_id == TENANT_B


@pytest.mark.parametrize("store_factory", human_decision_store_factories())
def test_r3_r3_q07_escalation_list_tenant_isolation(
    store_factory: HumanDecisionStoreFactory,
    tmp_path: Path,
) -> None:
    store = store_factory(tmp_path)
    store.record(
        sample_record(
            decision_id="hdec-esc-a",
            tenant_id=TENANT_A,
            verdict=HumanResponseVerdict.ESCALATE,
        )
    )
    store.record(
        sample_record(
            decision_id="hdec-esc-b",
            tenant_id=TENANT_B,
            verdict=HumanResponseVerdict.ESCALATE,
        )
    )
    assert len(store.list_escalations(TENANT_A)) == 1
    assert store.list_escalations(TENANT_A)[0].tenant_id == TENANT_A
    assert all(item.tenant_id == TENANT_B for item in store.list_escalations(TENANT_B))


@pytest.mark.parametrize("store_factory", human_decision_store_factories())
def test_r3_r3_q08_summary_tenant_isolation(
    store_factory: HumanDecisionStoreFactory,
    tmp_path: Path,
) -> None:
    store = store_factory(tmp_path)
    store.record(sample_record(decision_id="hdec-sum-a", tenant_id=TENANT_A))
    store.record(
        sample_record(
            decision_id="hdec-sum-b",
            tenant_id=TENANT_B,
            verdict=HumanResponseVerdict.REJECT,
        )
    )
    counts_a = store.summarize_queue(TENANT_A)
    counts_b = store.summarize_queue(TENANT_B)
    assert counts_a.get(HumanResponseVerdict.APPROVE.value) == 1
    assert HumanResponseVerdict.REJECT.value not in counts_a
    assert counts_b.get(HumanResponseVerdict.REJECT.value) == 1
    assert HumanResponseVerdict.APPROVE.value not in counts_b


def test_r3_r3_q09_approver_tenant_mismatch_fail_closed() -> None:
    with pytest.raises(ValidationError):
        build_human_decision_record(
            task_id=SHARED_TASK,
            tenant_id=TENANT_A,
            approver=local_development_approver_evidence(tenant_id=TENANT_B, actor_id="x"),
            verdict=HumanResponseVerdict.APPROVE,
            response_text="x",
        )


def test_r3_r3_q10_deterministic_ordering_parity(tmp_path: Path) -> None:
    memory = InMemoryHumanDecisionPersistence()
    sqlite = sqlite_human_decision_store(tmp_path)
    ts = "2026-01-01T00:00:00+00:00"
    expected = ["hdec-order-a", "hdec-order-z"]
    for store in (memory, sqlite):
        store.record(sample_record(decision_id="hdec-order-z", created_at_utc=ts))
        store.record(sample_record(decision_id="hdec-order-a", created_at_utc=ts))
        ordered = [r.decision_id for r in store.list_for_task(SHARED_TASK, TENANT_A)]
        assert ordered == expected


def test_r3_r3_q11_sqlite_restart_round_trip(tmp_path: Path) -> None:
    db = tmp_path / "restart.db"
    store_1 = SQLiteHumanDecisionStore(db_path=db)
    record = sample_record(decision_id="hdec-restart")
    store_1.record(record)
    store_1.close()
    store_2 = SQLiteHumanDecisionStore(db_path=db)
    loaded = store_2.get_decision("hdec-restart", TENANT_A)
    assert loaded is not None
    assert loaded.decision_id == record.decision_id
    assert loaded.task_id == record.task_id
    assert loaded.tenant_id == record.tenant_id
    assert loaded.human_request_id == record.human_request_id
    assert loaded.verdict == record.verdict
    assert loaded.approver == record.approver
    assert loaded.run_id == record.run_id
    assert loaded.created_at_utc == record.created_at_utc


def test_r3_r3_q12_missing_approver_provenance_fail_closed(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "missing.db"
    SQLiteHumanDecisionStore(db_path=db).close()
    conn = sqlite3.connect(db)
    conn.execute(
        """
        INSERT INTO human_decisions (
            decision_id, task_id, tenant_id, user_id, human_request_id,
            verdict, response_text, escalation_level, escalation_target,
            agent_id, run_id, notes, created_at_utc, approver_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "hdec-missing",
            SHARED_TASK,
            TENANT_A,
            "",
            "",
            HumanResponseVerdict.APPROVE.value,
            "",
            0,
            None,
            None,
            None,
            "",
            "2026-01-01T00:00:00+00:00",
            None,
        ),
    )
    conn.commit()
    conn.close()
    store = SQLiteHumanDecisionStore(db_path=db)
    with pytest.raises(HumanDecisionApproverProvenanceError):
        store.get_decision("hdec-missing", TENANT_A)


def test_r3_r3_q13_corrupt_approver_provenance_fail_closed(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "corrupt.db"
    SQLiteHumanDecisionStore(db_path=db).close()
    conn = sqlite3.connect(db)
    conn.execute(
        """
        INSERT INTO human_decisions (
            decision_id, task_id, tenant_id, user_id, human_request_id,
            verdict, response_text, escalation_level, escalation_target,
            agent_id, run_id, notes, created_at_utc, approver_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "hdec-corrupt",
            SHARED_TASK,
            TENANT_A,
            "",
            "",
            HumanResponseVerdict.APPROVE.value,
            "",
            0,
            None,
            None,
            None,
            "",
            "2026-01-01T00:00:00+00:00",
            "{not-json",
        ),
    )
    conn.commit()
    conn.close()
    store = SQLiteHumanDecisionStore(db_path=db)
    with pytest.raises(HumanDecisionApproverProvenanceError):
        store.get_decision("hdec-corrupt", TENANT_A)


def test_r3_r3_q14_wrong_tenant_approver_provenance_fail_closed(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "mismatch.db"
    SQLiteHumanDecisionStore(db_path=db).close()
    conn = sqlite3.connect(db)
    bad = local_development_approver_evidence(tenant_id=TENANT_B, actor_id="x")
    conn.execute(
        """
        INSERT INTO human_decisions (
            decision_id, task_id, tenant_id, user_id, human_request_id,
            verdict, response_text, escalation_level, escalation_target,
            agent_id, run_id, notes, created_at_utc, approver_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "hdec-bad-approver",
            SHARED_TASK,
            TENANT_A,
            "",
            "",
            HumanResponseVerdict.APPROVE.value,
            "",
            0,
            None,
            None,
            None,
            "",
            "2026-01-01T00:00:00+00:00",
            bad.model_dump_json(),
        ),
    )
    conn.commit()
    conn.close()
    store = SQLiteHumanDecisionStore(db_path=db)
    with pytest.raises(HumanDecisionApproverProvenanceError):
        store.get_decision("hdec-bad-approver", TENANT_A)


def test_r3_r3_q15_human_persistence_cannot_mint_continuation_authority() -> None:
    source = inspect.getsource(HumanDecisionPersistence)
    assert "ExecutionContinuationPort" not in source
    contract_path = _PERSISTENCE_CONTRACT.read_text(encoding="utf-8")
    assert "ExecutionContinuationPort" not in contract_path


def test_r3_r3_q16_canonical_resolution_precedes_evidence_persistence() -> None:
    tree = ast.parse(_HUMAN_RESPONSE.read_text(encoding="utf-8"))
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "persist_human_decision"
    )
    assert any(
        isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_approver_from_resolution"
        for n in ast.walk(fn)
    )
    assert any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "record"
        for n in ast.walk(fn)
    )


def test_r3_r3_q17_persisted_evidence_cannot_authorize_codecraft_execution(tmp_path: Path) -> None:
    """Regression: persisted APPROVE must not mint CodeCraft execution permission (A1 boundary)."""
    store = SQLiteHumanDecisionStore(db_path=tmp_path / "codecraft.db")
    notes = codecraft_exec_hitl_notes("craft-r3r3")
    record = build_human_decision_record(
        task_id=SHARED_TASK,
        tenant_id=TENANT_A,
        approver=local_development_approver_evidence(tenant_id=TENANT_A, actor_id="u"),
        verdict=HumanResponseVerdict.APPROVE,
        response_text="ok",
        run_id="run-cc",
        notes=notes,
    ).model_copy(update={"decision_id": "hdec-cc-auth"})
    store.record(record)
    ctx = ToolWiringContext(human_decision_store=store)
    profile = CodeCraftProfile(mode="supervised", require_hitl_before_exec=True)
    ownership = CodeCraftSessionOwnership(tenant_id=TENANT_A, task_id=SHARED_TASK, run_id="run-cc")
    auth = resolve_codecraft_exec_authorization(
        ctx,
        profile=profile,
        ownership=ownership,
        craft_id="craft-r3r3",
    )
    assert auth.authorized is False
    assert auth.pending_hitl is True


def test_r3_r3_q18_composition_uses_human_decision_persistence_contract(tmp_path: Path) -> None:
    store = create_sqlite_human_decision_store(db_path=tmp_path / "composed.db")
    assert isinstance(store, HumanDecisionPersistence)


def test_r3_r3_q19_no_duplicate_production_human_decision_truth_store() -> None:
    hits = list(_REPO_ROOT.glob("intergrax/**/*.py"))
    canonical = 0
    for path in hits:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "class SQLiteHumanDecisionStore" in text or "class InMemoryHumanDecisionPersistence" in text:
            canonical += 1
    assert canonical == 2


@pytest.mark.parametrize("store_factory", human_decision_store_factories())
def test_r3_r3_q20_global_decision_id_collision_fails_closed_without_overwrite(
    store_factory: HumanDecisionStoreFactory,
    tmp_path: Path,
) -> None:
    store = store_factory(tmp_path)
    store.record(sample_record(decision_id=DECISION_ID, tenant_id=TENANT_A))
    with pytest.raises(HumanDecisionPersistenceConflictError):
        store.record(sample_record(decision_id=DECISION_ID, tenant_id=TENANT_B))
    assert store.get_decision(DECISION_ID, TENANT_A) is not None
    assert store.get_decision(DECISION_ID, TENANT_B) is None
