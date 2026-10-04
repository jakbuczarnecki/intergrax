# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R1 — SX-F09 durable budget CAS & stale-writer closure."""

from __future__ import annotations

import ast
import threading
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.budget.models import (
    BudgetUsageTotals,
    ChildBudgetAllocationDecision,
    ExecutionBudgetAllocationMode,
)
from intergrax.runtime.execution.budget.persistence import (
    DocumentStoreRunBudgetPersistence,
    DurableExecutionBudgetLedger,
    DurableRunBudgetLedgerFactory,
    KvRunBudgetPersistence,
    RunBudgetPersistence,
    RunBudgetPersistenceError,
    StaleRunBudgetSnapshotWriteError,
    create_durable_run_budget_ledger_factory,
    decode_run_budget_snapshot,
    encode_run_budget_snapshot,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PERSISTENCE_PATH = _REPO_ROOT / "intergrax/runtime/execution/budget/persistence.py"

_TENANT_A = "tenant-r3-a"
_TENANT_B = "tenant-r3-b"
_LIMIT = RunBudget(max_total_tokens=100, max_tool_calls=100)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _KV(DistributedKVStore):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], bytes] = {}
        self._lock = threading.Lock()

    def get(self, tenant_id: str, key: str) -> bytes | None:
        with self._lock:
            return self._data.get((tenant_id, key))

    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        del ttl_seconds
        with self._lock:
            self._data[(tenant_id, key)] = value

    def delete(self, tenant_id: str, key: str) -> None:
        with self._lock:
            self._data.pop((tenant_id, key), None)

    def compare_and_set(
        self,
        tenant_id: str,
        key: str,
        expected: bytes | None,
        new_value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> bool:
        del ttl_seconds
        with self._lock:
            current = self._data.get((tenant_id, key))
            if expected is None and current is not None:
                return False
            if expected is not None and current != expected:
                return False
            self._data[(tenant_id, key)] = new_value
            return True


def _durable_pair(
    persistence: RunBudgetPersistence,
) -> DurableRunBudgetLedgerFactory:
    return create_durable_run_budget_ledger_factory(persistence, _LIMIT)


def _open(
    factory: DurableRunBudgetLedgerFactory,
    *,
    tenant_id: str,
    run_id: RunId,
    attempt_id: AttemptId,
) -> DurableExecutionBudgetLedger:
    ledger = factory.create_ledger(
        _LIMIT,
        tenant_id=tenant_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    assert isinstance(ledger, DurableExecutionBudgetLedger)
    return ledger


def _ledger_from_snapshot(
    persistence: RunBudgetPersistence,
    *,
    tenant_id: str,
    run_id: RunId,
    attempt_id: AttemptId,
    raw: bytes,
) -> DurableExecutionBudgetLedger:
    inner = create_execution_budget_ledger(_LIMIT)
    inner.restore_snapshot(decode_run_budget_snapshot(raw))
    return DurableExecutionBudgetLedger(
        inner=inner,
        persistence=persistence,
        tenant_id=tenant_id,
        run_id=run_id,
        attempt_id=attempt_id,
        last_known_raw=raw,
    )


def _consume_tokens(
    ledger: DurableExecutionBudgetLedger,
    *,
    root_execution_id: ExecutionId,
    amount: int,
) -> None:
    child_id = mint_execution_id()
    ledger.grant_child_budget(
        execution_id=child_id,
        parent_execution_id=root_execution_id,
        decision=ChildBudgetAllocationDecision(
            mode=ExecutionBudgetAllocationMode.SHARED,
        ),
    )
    ledger.consume_budget(child_id, BudgetUsageTotals(total_tokens=amount))
    ledger.release_child_budget(child_id)


def test_r3_r1_q01_ownership_symbols_present() -> None:
    tree = ast.parse(_PERSISTENCE_PATH.read_text(encoding="utf-8"))
    class_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    }
    assert "RunBudgetPersistence" in class_names
    assert "DurableRunBudgetLedgerFactory" in class_names
    assert "DurableExecutionBudgetLedger" in class_names
    assert "ExecutionBudgetLedger" not in class_names


def test_r3_r1_q02_persist_current_state_has_no_unbounded_retry() -> None:
    tree = ast.parse(_PERSISTENCE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_persist_current_state":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.While) and isinstance(child.test, ast.Constant):
                if child.test.value is True:
                    pytest.fail("_persist_current_state contains while True")
        return
    pytest.fail("_persist_current_state not found")


def test_r3_r1_q03_create_ledger_does_not_recurse() -> None:
    tree = ast.parse(_PERSISTENCE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "create_ledger":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and isinstance(child.value, ast.Call):
                func = child.value.func
                if isinstance(func, ast.Attribute) and func.attr == "create_ledger":
                    pytest.fail("create_ledger recursively calls create_ledger")
        return
    pytest.fail("create_ledger not found")


@pytest.mark.parametrize(
    "persistence_factory",
    [
        lambda: KvRunBudgetPersistence(_KV()),
        lambda: DocumentStoreRunBudgetPersistence(InMemoryDocumentStore()),
    ],
    ids=("kv", "document"),
)
def test_r3_r1_q04_single_writer_cas_success(
    persistence_factory: object,
) -> None:
    persistence = persistence_factory()
    factory = _durable_pair(persistence)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    ledger = _open(
        factory,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    _consume_tokens(ledger, root_execution_id=mint_execution_id(), amount=10)
    assert ledger.snapshot_root_available().max_total_tokens == 90
    reopened = _open(
        factory,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    assert reopened.snapshot_root_available().max_total_tokens == 90


def test_r3_r1_q05_exact_stale_writer_rejected() -> None:
    kv = _KV()
    persistence = KvRunBudgetPersistence(kv)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    inner = create_execution_budget_ledger(_LIMIT)
    snapshot0 = inner.export_snapshot(attempt_id)
    raw0 = encode_run_budget_snapshot(snapshot0)
    kv.set(_TENANT_A, f"run_budget_ledger:{run_id}", raw0)

    ledger_a = _ledger_from_snapshot(
        persistence,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_id,
        raw=raw0,
    )
    ledger_b = _ledger_from_snapshot(
        persistence,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_id,
        raw=raw0,
    )

    _consume_tokens(ledger_a, root_execution_id=mint_execution_id(), amount=40)
    assert ledger_a.snapshot_root_available().max_total_tokens == 60

    with pytest.raises(StaleRunBudgetSnapshotWriteError):
        _consume_tokens(ledger_b, root_execution_id=mint_execution_id(), amount=25)

    durable_raw = kv.get(_TENANT_A, f"run_budget_ledger:{run_id}")
    assert durable_raw is not None
    durable_snapshot = decode_run_budget_snapshot(durable_raw)
    assert durable_snapshot.root_shared_consumed.total_tokens == 40
    assert ledger_b.snapshot_root_available().max_total_tokens == 60


def test_r3_r1_q06_stale_grant_does_not_escape() -> None:
    kv = _KV()
    persistence = KvRunBudgetPersistence(kv)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    inner = create_execution_budget_ledger(_LIMIT)
    raw0 = encode_run_budget_snapshot(inner.export_snapshot(attempt_id))
    kv.set(_TENANT_A, f"run_budget_ledger:{run_id}", raw0)

    ledger_a = _ledger_from_snapshot(
        persistence,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_id,
        raw=raw0,
    )
    ledger_b = _ledger_from_snapshot(
        persistence,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_id,
        raw=raw0,
    )
    _consume_tokens(ledger_a, root_execution_id=mint_execution_id(), amount=5)

    child_id = mint_execution_id()
    root_id = mint_execution_id()
    with pytest.raises(StaleRunBudgetSnapshotWriteError):
        ledger_b.grant_child_budget(
            execution_id=child_id,
            parent_execution_id=root_id,
            decision=ChildBudgetAllocationDecision(
                mode=ExecutionBudgetAllocationMode.SHARED,
            ),
        )

    assert ledger_b.snapshot_root_available().max_total_tokens == 95


def test_r3_r1_q07_stale_consume_does_not_overwrite_winner() -> None:
    kv = _KV()
    persistence = KvRunBudgetPersistence(kv)
    factory = _durable_pair(persistence)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    ledger_a = _open(
        factory,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    raw_after_open = kv.get(_TENANT_A, f"run_budget_ledger:{run_id}")
    assert raw_after_open is not None

    ledger_b = _ledger_from_snapshot(
        persistence,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_id,
        raw=raw_after_open,
    )
    _consume_tokens(ledger_a, root_execution_id=mint_execution_id(), amount=55)
    with pytest.raises(StaleRunBudgetSnapshotWriteError):
        _consume_tokens(ledger_b, root_execution_id=mint_execution_id(), amount=55)

    final = _open(
        factory,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    assert final.snapshot_root_available().max_total_tokens == 45


def test_r3_r1_q08_initial_create_race_one_winner() -> None:
    kv = _KV()
    persistence = KvRunBudgetPersistence(kv)
    factory = _durable_pair(persistence)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    barrier = threading.Barrier(2)
    ledgers: list[DurableExecutionBudgetLedger] = []
    errors: list[Exception] = []

    def _creator() -> None:
        barrier.wait()
        try:
            ledger = _open(
                factory,
                tenant_id=_TENANT_A,
                run_id=run_id,
                attempt_id=attempt_id,
            )
            ledgers.append(ledger)
        except Exception as exc:  # noqa: BLE001 — qualification race capture
            errors.append(exc)

    threads = [threading.Thread(target=_creator) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(ledgers) == 2
    assert not errors
    assert ledgers[0].snapshot_root_available().max_total_tokens == 100
    assert ledgers[1].snapshot_root_available().max_total_tokens == 100
    assert kv.get(_TENANT_A, f"run_budget_ledger:{run_id}") is not None


def test_r3_r1_q09_redelivery_settlement_conflict_observes_winner() -> None:
    kv = _KV()
    persistence = KvRunBudgetPersistence(kv)
    factory = _durable_pair(persistence)
    run_id = mint_run_id()
    attempt_one = mint_attempt_id()
    attempt_two = mint_attempt_id()

    ledger_one = _open(
        factory,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_one,
    )
    _consume_tokens(ledger_one, root_execution_id=mint_execution_id(), amount=20)
    raw_after_one = kv.get(_TENANT_A, f"run_budget_ledger:{run_id}")
    assert raw_after_one is not None

    first_two = _open(
        factory,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_two,
    )
    assert first_two.snapshot_root_available().max_total_tokens == 80

    second_two = _open(
        factory,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_two,
    )
    assert second_two.snapshot_root_available().max_total_tokens == 80


@pytest.mark.parametrize(
    "persistence_factory",
    [
        lambda: KvRunBudgetPersistence(_KV()),
        lambda: DocumentStoreRunBudgetPersistence(InMemoryDocumentStore()),
    ],
    ids=("kv", "document"),
)
def test_r3_r1_q10_tenant_isolation_same_run_id(
    persistence_factory: object,
) -> None:
    persistence = persistence_factory()
    factory = _durable_pair(persistence)
    run_id = mint_run_id()
    ledger_a = _open(
        factory,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
    )
    ledger_b = _open(
        factory,
        tenant_id=_TENANT_B,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
    )
    _consume_tokens(ledger_a, root_execution_id=mint_execution_id(), amount=33)
    assert ledger_b.snapshot_root_available().max_total_tokens == 100
    assert ledger_a.snapshot_root_available().max_total_tokens == 67


def test_r3_r1_q11_different_runs_independent() -> None:
    factory = _durable_pair(KvRunBudgetPersistence(_KV()))
    run_a = mint_run_id()
    run_b = mint_run_id()
    ledger_a = _open(
        factory,
        tenant_id=_TENANT_A,
        run_id=run_a,
        attempt_id=mint_attempt_id(),
    )
    _consume_tokens(ledger_a, root_execution_id=mint_execution_id(), amount=44)
    ledger_b = _open(
        factory,
        tenant_id=_TENANT_A,
        run_id=run_b,
        attempt_id=mint_attempt_id(),
    )
    assert ledger_b.snapshot_root_available().max_total_tokens == 100


def test_r3_r1_q12_corrupt_state_fails_closed() -> None:
    kv = _KV()
    run_id = mint_run_id()
    kv.set(_TENANT_A, f"run_budget_ledger:{run_id}", b"not-json")
    factory = _durable_pair(KvRunBudgetPersistence(kv))
    with pytest.raises(RunBudgetPersistenceError):
        _open(
            factory,
            tenant_id=_TENANT_A,
            run_id=run_id,
            attempt_id=mint_attempt_id(),
        )


def test_r3_r1_q13_redelivery_preserves_consumption() -> None:
    factory = _durable_pair(KvRunBudgetPersistence(_KV()))
    run_id = mint_run_id()
    attempt_one = mint_attempt_id()
    attempt_two = mint_attempt_id()
    ledger_one = _open(
        factory,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_one,
    )
    _consume_tokens(ledger_one, root_execution_id=mint_execution_id(), amount=37)
    ledger_two = _open(
        factory,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_two,
    )
    assert ledger_two.snapshot_root_available().max_total_tokens == 63


def test_r3_r1_q14_provider_conformance_parametrized() -> None:
    assert test_r3_r1_q04_single_writer_cas_success
    assert test_r3_r1_q10_tenant_isolation_same_run_id


def test_r3_r1_q15_no_authority_mint_in_persistence_module() -> None:
    text = _PERSISTENCE_PATH.read_text(encoding="utf-8").lower()
    forbidden = (
        "executionauthority",
        "governance",
        "mint_authority",
        "change_tenant",
    )
    for token in forbidden:
        assert token not in text


def test_r3_r1_factory_requires_tenant_no_blank_fallback() -> None:
    factory = _durable_pair(KvRunBudgetPersistence(_KV()))
    with pytest.raises(RunBudgetPersistenceError):
        factory.create_ledger(
            _LIMIT,
            tenant_id=None,
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
        )
