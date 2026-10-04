# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R1 — SX-F09 durable budget CAS & stale-writer closure."""

from __future__ import annotations

import ast
import threading
from collections.abc import Callable
from dataclasses import dataclass
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
from intergrax.runtime.execution.budget.snapshot import RunBudgetLedgerSnapshot
from intergrax.runtime.nexus.budget.budget_models import RunBudget

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PERSISTENCE_PATH = _REPO_ROOT / "intergrax/runtime/execution/budget/persistence.py"

_TENANT_A = "tenant-r3-a"
_TENANT_B = "tenant-r3-b"
_LIMIT = RunBudget(max_total_tokens=100, max_tool_calls=100)

type PersistenceFactory = Callable[[], RunBudgetPersistence]

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@dataclass
class _RedeliveryRaceCounters:
    redelivery_cas_calls: int = 0
    redelivery_cas_failed: int = 0
    post_conflict_load_calls: int = 0


class _RedeliveryRacePersistence(RunBudgetPersistence):
    """Deterministic orchestration for redelivery settlement CAS races."""

    def __init__(
        self,
        inner: RunBudgetPersistence,
        *,
        race_expected_raw: bytes,
        race_inject_winner_raw: bytes,
        race_settled_attempt_id: AttemptId,
        counters: _RedeliveryRaceCounters,
    ) -> None:
        self._inner = inner
        self._race_expected_raw = race_expected_raw
        self._race_inject_winner_raw = race_inject_winner_raw
        self._race_settled_attempt_id = race_settled_attempt_id
        self._counters = counters
        self._race_armed = True
        self._awaiting_post_conflict_load = False

    def load_snapshot(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
    ) -> bytes | None:
        raw = self._inner.load_snapshot(tenant_id=tenant_id, run_id=run_id)
        if self._awaiting_post_conflict_load:
            self._counters.post_conflict_load_calls += 1
            self._awaiting_post_conflict_load = False
        return raw

    def compare_and_swap_snapshot(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
        expected: bytes | None,
        snapshot: RunBudgetLedgerSnapshot,
    ) -> bool:
        is_redelivery_cas = (
            self._race_armed
            and expected == self._race_expected_raw
            and snapshot.attempt_id == self._race_settled_attempt_id
        )
        if not is_redelivery_cas:
            return self._inner.compare_and_swap_snapshot(
                tenant_id=tenant_id,
                run_id=run_id,
                expected=expected,
                snapshot=snapshot,
            )
        self._counters.redelivery_cas_calls += 1
        winner_snapshot = decode_run_budget_snapshot(self._race_inject_winner_raw)
        injected = self._inner.compare_and_swap_snapshot(
            tenant_id=tenant_id,
            run_id=run_id,
            expected=self._race_expected_raw,
            snapshot=winner_snapshot,
        )
        assert injected
        result = self._inner.compare_and_swap_snapshot(
            tenant_id=tenant_id,
            run_id=run_id,
            expected=expected,
            snapshot=snapshot,
        )
        assert not result
        self._counters.redelivery_cas_failed += 1
        self._race_armed = False
        self._awaiting_post_conflict_load = True
        return False


_CANONICAL_PERSISTENCE_FACTORIES: tuple[PersistenceFactory, ...] = (
    lambda: KvRunBudgetPersistence(_KV()),
    lambda: DocumentStoreRunBudgetPersistence(InMemoryDocumentStore()),
)


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


def _seed_durable_raw(
    persistence: RunBudgetPersistence,
    *,
    tenant_id: str,
    run_id: RunId,
    raw: bytes,
) -> None:
    snapshot = decode_run_budget_snapshot(raw)
    created = persistence.compare_and_swap_snapshot(
        tenant_id=tenant_id,
        run_id=run_id,
        expected=None,
        snapshot=snapshot,
    )
    assert created


def _canonical_redelivery_winner_raw(
    *,
    raw_sa: bytes,
    winner_attempt_id: AttemptId,
    extra_consume: int,
) -> bytes:
    helper = KvRunBudgetPersistence(_KV())
    run_id = mint_run_id()
    snapshot_sa = decode_run_budget_snapshot(raw_sa)
    assert helper.compare_and_swap_snapshot(
        tenant_id=_TENANT_A,
        run_id=run_id,
        expected=None,
        snapshot=snapshot_sa,
    )
    factory = _durable_pair(helper)
    ledger = _open(
        factory,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=winner_attempt_id,
    )
    if extra_consume:
        _consume_tokens(
            ledger,
            root_execution_id=mint_execution_id(),
            amount=extra_consume,
        )
    raw_winner = helper.load_snapshot(tenant_id=_TENANT_A, run_id=run_id)
    assert raw_winner is not None
    return raw_winner


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
    _CANONICAL_PERSISTENCE_FACTORIES,
    ids=("kv", "document"),
)
def test_r3_r1_q04_single_writer_cas_success(
    persistence_factory: PersistenceFactory,
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


@pytest.mark.parametrize(
    "persistence_factory",
    _CANONICAL_PERSISTENCE_FACTORIES,
    ids=("kv", "document"),
)
def test_r3_r1_q05_exact_stale_writer_rejected(
    persistence_factory: PersistenceFactory,
) -> None:
    persistence = persistence_factory()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    inner = create_execution_budget_ledger(_LIMIT)
    raw0 = encode_run_budget_snapshot(inner.export_snapshot(attempt_id))
    _seed_durable_raw(
        persistence,
        tenant_id=_TENANT_A,
        run_id=run_id,
        raw=raw0,
    )

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

    durable_raw = persistence.load_snapshot(tenant_id=_TENANT_A, run_id=run_id)
    assert durable_raw is not None
    durable_snapshot = decode_run_budget_snapshot(durable_raw)
    assert durable_snapshot.root_shared_consumed.total_tokens == 40
    assert ledger_b.snapshot_root_available().max_total_tokens == 60
    factory = _durable_pair(persistence)
    reopened = _open(
        factory,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    assert reopened.snapshot_root_available().max_total_tokens == 60


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


def test_r3_r1_q09_redelivery_settlement_cas_conflict_same_attempt_winner() -> None:
    inner_persistence = KvRunBudgetPersistence(_KV())
    run_id = mint_run_id()
    attempt_one = mint_attempt_id()
    attempt_two = mint_attempt_id()
    base_factory = _durable_pair(inner_persistence)
    ledger_one = _open(
        base_factory,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_one,
    )
    _consume_tokens(ledger_one, root_execution_id=mint_execution_id(), amount=20)
    raw_sa = inner_persistence.load_snapshot(tenant_id=_TENANT_A, run_id=run_id)
    assert raw_sa is not None

    winner_raw = _canonical_redelivery_winner_raw(
        raw_sa=raw_sa,
        winner_attempt_id=attempt_two,
        extra_consume=15,
    )
    winner_snapshot = decode_run_budget_snapshot(winner_raw)
    assert winner_snapshot.attempt_id == attempt_two
    assert winner_snapshot.root_shared_consumed.total_tokens == 35

    loser_candidate_snapshot = decode_run_budget_snapshot(raw_sa)
    inner = create_execution_budget_ledger(loser_candidate_snapshot.root_limits)
    inner.restore_snapshot(loser_candidate_snapshot)
    inner.prepare_for_attempt_redelivery()
    loser_candidate = inner.export_snapshot(attempt_two)
    assert loser_candidate.root_shared_consumed.total_tokens == 20

    counters = _RedeliveryRaceCounters()
    race_persistence = _RedeliveryRacePersistence(
        inner_persistence,
        race_expected_raw=raw_sa,
        race_inject_winner_raw=winner_raw,
        race_settled_attempt_id=attempt_two,
        counters=counters,
    )
    race_factory = _durable_pair(race_persistence)
    returned = _open(
        race_factory,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_two,
    )

    assert counters.redelivery_cas_calls == 1
    assert counters.redelivery_cas_failed == 1
    assert counters.post_conflict_load_calls == 1
    assert returned.snapshot_root_available().max_total_tokens == 65
    assert (
        returned.snapshot_root_available().max_total_tokens
        != 100 - loser_candidate.root_shared_consumed.total_tokens
    )

    durable_raw = inner_persistence.load_snapshot(tenant_id=_TENANT_A, run_id=run_id)
    assert durable_raw is not None
    assert durable_raw == winner_raw
    durable_snapshot = decode_run_budget_snapshot(durable_raw)
    assert durable_snapshot.root_shared_consumed.total_tokens == 35
    assert durable_snapshot.root_permanent_consumed.total_tokens == 0

    fresh = _open(
        base_factory,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_two,
    )
    assert fresh.snapshot_root_available().max_total_tokens == 65


def test_r3_r1_q09_redelivery_settlement_cas_conflict_different_attempt_stale() -> None:
    inner_persistence = KvRunBudgetPersistence(_KV())
    run_id = mint_run_id()
    attempt_one = mint_attempt_id()
    attempt_two = mint_attempt_id()
    attempt_three = mint_attempt_id()
    base_factory = _durable_pair(inner_persistence)
    ledger_one = _open(
        base_factory,
        tenant_id=_TENANT_A,
        run_id=run_id,
        attempt_id=attempt_one,
    )
    _consume_tokens(ledger_one, root_execution_id=mint_execution_id(), amount=20)
    raw_sa = inner_persistence.load_snapshot(tenant_id=_TENANT_A, run_id=run_id)
    assert raw_sa is not None

    winner_raw = _canonical_redelivery_winner_raw(
        raw_sa=raw_sa,
        winner_attempt_id=attempt_three,
        extra_consume=0,
    )
    assert decode_run_budget_snapshot(winner_raw).attempt_id == attempt_three

    counters = _RedeliveryRaceCounters()
    race_persistence = _RedeliveryRacePersistence(
        inner_persistence,
        race_expected_raw=raw_sa,
        race_inject_winner_raw=winner_raw,
        race_settled_attempt_id=attempt_two,
        counters=counters,
    )
    race_factory = _durable_pair(race_persistence)
    with pytest.raises(StaleRunBudgetSnapshotWriteError):
        _open(
            race_factory,
            tenant_id=_TENANT_A,
            run_id=run_id,
            attempt_id=attempt_two,
        )

    assert counters.redelivery_cas_calls == 1
    assert counters.redelivery_cas_failed == 1
    assert counters.post_conflict_load_calls == 1
    durable_raw = inner_persistence.load_snapshot(tenant_id=_TENANT_A, run_id=run_id)
    assert durable_raw == winner_raw


@pytest.mark.parametrize(
    "persistence_factory",
    _CANONICAL_PERSISTENCE_FACTORIES,
    ids=("kv", "document"),
)
def test_r3_r1_q10_tenant_isolation_same_run_id(
    persistence_factory: PersistenceFactory,
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
