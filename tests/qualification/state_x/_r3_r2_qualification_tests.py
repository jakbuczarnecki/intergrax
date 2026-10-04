# © Artur Czarnecki. All rights reserved.

"""STATE-X-R3-R2 qualification tests (imported into parent R3 test module)."""

from __future__ import annotations

import ast
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from intergrax.agents.persistence.compensation_queue_store import (
    CompensationJobStatus,
    InMemoryCompensationQueueStore,
    SQLiteCompensationQueueStore,
)
from intergrax.agents.persistence.compensation_queue_worker import drain_pending_compensation_jobs
from intergrax.agents.persistence.compensation_side_effect_input import (
    compensation_side_effect_input_from_job,
)
from intergrax.agents.persistence.declarative_tool_executor import (
    CallableDeclarativeToolInvoker,
    DeclarativeToolInvokeResult,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import AttemptId, RunId, mint_attempt_id, mint_run_id
from intergrax.contracts.idempotency_store import (
    ClaimOutcome,
    IdempotencyOperationConflictError,
    InvocationClaim,
    InvocationStatus,
    InvocationUncertaintyError,
    PreEffectSuspendedWorkRecoveryAuthority,
)
from intergrax.runtime.nexus.agents.catalog_declarative_invoker import CatalogDeclarativeToolInvoker
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import IdempotencyPreEffectCoordinator
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.runtime.tools.operation_identity import compute_invocation_operation_identity
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from tests.qualification.state_x._r3_r2_support import (
    CompensationQueueStoreFactory,
    IdempotencyStorePathFactory,
    _TENANT_A,
    _TENANT_B,
    _WIRING_PATH,
    _build_admitted_compensation_execution,
    _build_canonical_compensation_production_execution,
    _execute_compensation_with_lab_governance,
    _compensation_queue_store_factories,
    _inventory_by_id,
    _local_idempotency_factories,
    _paired_compensation_idempotency_factories,
    _redis_store_or_skip,
    _sample_compensation_job,
)
from tests.qualification.state_x.inventory import SemanticOwnershipRole
from tests.unit.agents.persistence.compensation_execution_test_support import (
    RecordingExecutionBoundDeclarativeToolInvoker,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_SIDE_EFFECT_PRODUCTION_SCAN_ROOTS = (
    "intergrax/runtime/nexus/tools",
    "intergrax/runtime/tools",
    "intergrax/agents/persistence",
    "intergrax/runtime/execution",
)


class _In(BaseModel):
    value: int = 1


class _Out(BaseModel):
    result: int = 1


def _side_effect_contract(tool_id: str = "tool.side") -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description="side effect",
        input_schema=_In,
        output_schema=_Out,
        error_mapping={},
        side_effects=True,
    )


def test_r3_r2_q01_closed_world_inventory() -> None:
    f10 = _inventory_by_id()["SX-F10"]
    assert "RedisIdempotencyStore" in f10.implementation_symbols
    assert "intergrax/distributed/providers/redis_idempotency_store.py" in f10.production_paths
    assert "InMemoryIdempotencyStore" in f10.implementation_symbols
    assert "SQLiteIdempotencyStore" in f10.implementation_symbols
    f11 = _inventory_by_id()["SX-F11"]
    assert "InMemoryCompensationQueueStore" in f11.implementation_symbols
    assert "SQLiteCompensationQueueStore" in f11.implementation_symbols


def test_r3_r2_q02_exactly_one_ownership() -> None:
    f10 = _inventory_by_id()["SX-F10"]
    assert f10.semantic_ownership_role == SemanticOwnershipRole.CANONICAL_OWNER
    assert tuple(ref.symbol for ref in f10.contract_references) == ("IdempotencyStore",)
    f11 = _inventory_by_id()["SX-F11"]
    assert f11.semantic_ownership_role == SemanticOwnershipRole.CANONICAL_OWNER
    assert tuple(ref.symbol for ref in f11.contract_references) == ("CompensationQueueStore",)


def test_r3_r2_q03_compensation_authority_required() -> None:
    tree = ast.parse(_WIRING_PATH.read_text(encoding="utf-8"), filename=str(_WIRING_PATH))
    fn: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_compensation_side_effect_execution":
            fn = node
            break
    assert fn is not None
    kw = {arg.arg: default for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults)}
    assert "authority" in kw
    assert kw["authority"] is None


def test_r3_r2_q04_no_unrestricted_fallback_in_wiring() -> None:
    text = _WIRING_PATH.read_text(encoding="utf-8")
    assert "authority or ParentExecutionAuthority.unrestricted_root()" not in text
    assert "unrestricted_root()" not in text


def test_r3_r2_q05_atomic_claim_inmemory() -> None:
    store = InMemoryIdempotencyStore()
    results: list[ClaimOutcome] = []
    barrier = threading.Barrier(2)

    def racer(owner: str) -> None:
        barrier.wait()
        outcome = store.claim(_TENANT_A, "race-key", owner, lease_seconds=30)
        results.append(outcome.outcome)

    t1 = threading.Thread(target=racer, args=("owner-a",))
    t2 = threading.Thread(target=racer, args=("owner-b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert results.count(ClaimOutcome.ACQUIRED) == 1
    assert results.count(ClaimOutcome.BLOCKED_ACTIVE) == 1


@pytest.mark.parametrize("store_factory", _local_idempotency_factories())
def test_r3_r2_q05b_atomic_claim_blocked_sqlite(store_factory: object) -> None:
    store = store_factory()
    first = store.claim(_TENANT_A, "sqlite-block", "owner-a", lease_seconds=30)
    second = store.claim(_TENANT_A, "sqlite-block", "owner-b", lease_seconds=30)
    assert first.outcome == ClaimOutcome.ACQUIRED
    assert second.outcome == ClaimOutcome.BLOCKED_ACTIVE


@pytest.mark.parametrize("store_factory", _local_idempotency_factories())
def test_r3_r2_q06_expired_pre_effect_safe_reclaim(store_factory: object) -> None:
    store = store_factory()
    contract = _side_effect_contract()
    from intergrax.contracts.execution_identity import mint_run_id

    request = ToolExecutionRequest(
        run_id=str(mint_run_id()),
        step_id="s1",
        tool_id=contract.tool_id,
        input=_In(),
        idempotency_key="pre-effect-reclaim",
    )
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store, lease_seconds=30)
    from tests.unit.runtime.tools.test_idempotent_invoker import _state_with_enforce_allow

    state = _state_with_enforce_allow()
    operation_identity = compute_invocation_operation_identity(request.tool_id, request.input)
    first = coordinator.before_external_effect(state=state, contract=contract, request=request)
    assert first.claim_context is not None
    authority = PreEffectSuspendedWorkRecoveryAuthority(owner_id="recovery", fence=2)
    assert store.reconcile_abandoned_pre_effect_not_started(
        state.tenant_id,
        request.idempotency_key,
        operation_identity,
        recovery_authority=authority,
    )
    second = coordinator.before_external_effect(state=state, contract=contract, request=request)
    assert second.claim_context is not None
    assert store.get_status(state.tenant_id, request.idempotency_key) == InvocationStatus.STARTED


@pytest.mark.parametrize("store_factory", _local_idempotency_factories())
def test_r3_r2_q07_expired_post_effect_uncertain(store_factory: object) -> None:
    store = store_factory()
    acquired = store.claim(_TENANT_A, "post-effect-key", "owner-a", lease_seconds=1)
    assert acquired.claim is not None
    store.admit_external_effect_may_have_started_with_claim(
        _TENANT_A,
        "post-effect-key",
        acquired.claim,
    )
    time.sleep(1.2)
    retry = store.claim(_TENANT_A, "post-effect-key", "owner-b", lease_seconds=30)
    assert retry.outcome == ClaimOutcome.UNCERTAIN
    assert store.get_status(_TENANT_A, "post-effect-key") == InvocationStatus.UNCERTAIN


@pytest.mark.parametrize("store_factory", _local_idempotency_factories())
def test_r3_r2_q08_stale_complete_rejected(store_factory: object) -> None:
    store = store_factory()
    if isinstance(store, InMemoryIdempotencyStore):
        acquired = store.claim(_TENANT_A, "stale-complete", "owner-a", lease_seconds=30)
        assert acquired.claim is not None
        stale = acquired.claim
        entry = store._store[(_TENANT_A, "stale-complete")]  # noqa: SLF001
        current = acquired.claim.model_copy(update={"fence": 2, "owner_id": "owner-b"})
        entry.claim = current
        result = ToolExecutionResult.ok(_Out())
        with pytest.raises(StaleClaimError):
            store.complete_with_claim(_TENANT_A, "stale-complete", stale, result)
        store.complete_with_claim(_TENANT_A, "stale-complete", current, result)
        assert store.get_status(_TENANT_A, "stale-complete") == InvocationStatus.COMPLETED
    else:
        acquired = store.claim(_TENANT_A, "stale-complete-sqlite", "owner-a", lease_seconds=30)
        assert acquired.claim is not None
        stale = InvocationClaim(
            tenant_id=_TENANT_A,
            key="stale-complete-sqlite",
            owner_id=acquired.claim.owner_id,
            lease_expires_at=acquired.claim.lease_expires_at,
            fence=acquired.claim.fence - 1,
        )
        result = ToolExecutionResult.ok(_Out())
        with pytest.raises(StaleClaimError):
            store.complete_with_claim(_TENANT_A, "stale-complete-sqlite", stale, result)


@pytest.mark.parametrize("store_factory", _local_idempotency_factories())
def test_r3_r2_q09_stale_uncertain_rejected(store_factory: object) -> None:
    store = store_factory()
    acquired = store.claim(_TENANT_A, "stale-uncertain", "owner-a", lease_seconds=30)
    assert acquired.claim is not None
    stale = InvocationClaim(
        tenant_id=_TENANT_A,
        key="stale-uncertain",
        owner_id=acquired.claim.owner_id,
        lease_expires_at=acquired.claim.lease_expires_at,
        fence=acquired.claim.fence - 1,
    )
    with pytest.raises(StaleClaimError):
        store.mark_uncertain_with_claim(_TENANT_A, "stale-uncertain", stale)


@pytest.mark.parametrize("store_factory", _local_idempotency_factories())
def test_r3_r2_q10_completed_replay(store_factory: object) -> None:
    store = store_factory()
    calls = 0

    def _physical() -> None:
        nonlocal calls
        calls += 1

    op = compute_invocation_operation_identity("tool.side", _In(value=1))
    first = store.claim(_TENANT_A, "replay-key", "o1", 30, operation_identity=op)
    assert first.claim is not None
    _physical()
    store.complete_with_claim(
        _TENANT_A,
        "replay-key",
        first.claim,
        ToolExecutionResult.ok(_Out(result=7)),
    )
    second = store.claim(_TENANT_A, "replay-key", "o2", 30, operation_identity=op)
    assert second.outcome == ClaimOutcome.REPLAY_COMPLETED
    assert calls == 1


@pytest.mark.parametrize("store_factory", _local_idempotency_factories())
def test_r3_r2_q11_operation_identity_conflict(store_factory: object) -> None:
    store = store_factory()
    op_a = compute_invocation_operation_identity("tool.a", _In(value=1))
    op_b = compute_invocation_operation_identity("tool.b", _In(value=2))
    first = store.claim(_TENANT_A, "conflict-key", "o1", 30, operation_identity=op_a)
    assert first.claim is not None
    store.complete_with_claim(
        _TENANT_A,
        "conflict-key",
        first.claim,
        ToolExecutionResult.ok(_Out()),
    )
    with pytest.raises(IdempotencyOperationConflictError):
        store.claim(_TENANT_A, "conflict-key", "o2", 30, operation_identity=op_b)


@pytest.mark.parametrize("store_factory", _local_idempotency_factories())
def test_r3_r2_q12_tenant_isolation(store_factory: object) -> None:
    store = store_factory()
    a = store.claim(_TENANT_A, "shared-key", "oa", 30)
    b = store.claim(_TENANT_B, "shared-key", "ob", 30)
    assert a.outcome == ClaimOutcome.ACQUIRED
    assert b.outcome == ClaimOutcome.ACQUIRED


class _CrashOnCompleteStore(InMemoryIdempotencyStore):
    def complete_with_claim(self, tenant_id, key, claim, result, completed_ttl_seconds=None):  # noqa: ANN001
        raise RuntimeError("crash before complete_with_claim")


class _ShortLeaseCrashStore(_CrashOnCompleteStore):
    def claim(self, tenant_id, key, owner_id, lease_seconds, operation_identity=None):  # noqa: ANN001
        del lease_seconds
        return super().claim(tenant_id, key, owner_id, 1, operation_identity=operation_identity)


@pytest.mark.asyncio
async def test_r3_r2_q13_crash_after_effect_before_complete() -> None:
    from intergrax.agents.persistence.declarative_tool_executor import execute_declarative_actions

    store = _ShortLeaseCrashStore()
    calls = 0

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return DeclarativeToolInvokeResult(status="success")

    action = {"tool_id": "email.send", "idempotency_key": "decl-crash", "args": {}}
    with pytest.raises(RuntimeError, match="crash before complete"):
        await execute_declarative_actions(
            actions=[action],
            ledger=None,
            invoker=CallableDeclarativeToolInvoker(_invoke),
            idempotency_store=store,
            tenant_id=_TENANT_A,
        )
    assert calls == 1
    time.sleep(1.2)
    with pytest.raises(InvocationUncertaintyError):
        await execute_declarative_actions(
            actions=[action],
            ledger=None,
            invoker=CallableDeclarativeToolInvoker(_invoke),
            idempotency_store=store,
            tenant_id=_TENANT_A,
        )
    assert calls == 1


def test_r3_r2_q14_no_legacy_record_started_in_side_effect_callers() -> None:
    offenders: list[str] = []
    for rel_root in _SIDE_EFFECT_PRODUCTION_SCAN_ROOTS:
        root = _REPO_ROOT / rel_root
        for path in root.rglob("*.py"):
            if "test_" in path.name:
                continue
            rel = str(path.relative_to(_REPO_ROOT)).replace("\\", "/")
            if rel.endswith("idempotency_store.py") or rel.endswith("redis_idempotency_store.py"):
                continue
            text = path.read_text(encoding="utf-8")
            if ".record_started(" not in text and ".record_completed(" not in text:
                continue
            if "idempotency_store" in text or "IdempotencyStore" in text:
                offenders.append(rel)
    assert offenders == []


@pytest.mark.parametrize("queue_factory", _compensation_queue_store_factories())
def test_r3_r2_q15_compensation_enqueue_dedupe(
    tmp_path: Path,
    queue_factory: CompensationQueueStoreFactory,
) -> None:
    store = queue_factory(tmp_path)
    job = _sample_compensation_job(key_suffix="dedupe")
    store.enqueue(job)
    store.enqueue(job.model_copy(update={"job_id": "other"}))
    pending = store.list_pending(_TENANT_A)
    assert len(pending) == 1


def test_r3_r2_q16_compensation_atomic_claim_inmemory() -> None:
    store = InMemoryCompensationQueueStore()
    store.enqueue(_sample_compensation_job(key_suffix="atomic"))
    winners: list[str] = []
    barrier = threading.Barrier(2)

    def racer(owner: str) -> None:
        barrier.wait()
        claims = store.claim_pending(_TENANT_A, owner, lease_seconds=30, limit=1)
        if claims:
            winners.append(owner)

    t1 = threading.Thread(target=racer, args=("w-a",))
    t2 = threading.Thread(target=racer, args=("w-b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(winners) == 1


def test_r3_r2_q16_compensation_atomic_claim_sqlite(tmp_path: Path) -> None:
    db = tmp_path / "q16.db"
    store_a = SQLiteCompensationQueueStore(db)
    store_b = SQLiteCompensationQueueStore(db)
    store_a.enqueue(_sample_compensation_job(key_suffix="sqlite-atomic"))
    winners: list[str] = []
    barrier = threading.Barrier(2)

    def racer(store: SQLiteCompensationQueueStore, owner: str) -> None:
        barrier.wait()
        claims = store.claim_pending(_TENANT_A, owner, lease_seconds=30, limit=1)
        if claims:
            winners.append(owner)

    t1 = threading.Thread(target=racer, args=(store_a, "a"))
    t2 = threading.Thread(target=racer, args=(store_b, "b"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert len(winners) == 1


@pytest.mark.parametrize("queue_factory", _compensation_queue_store_factories())
def test_r3_r2_q17_compensation_fence_supersession(
    tmp_path: Path,
    queue_factory: CompensationQueueStoreFactory,
) -> None:
    store = queue_factory(tmp_path)
    store.enqueue(_sample_compensation_job(key_suffix="fence"))
    first = store.claim_pending(_TENANT_A, "w-a", lease_seconds=30, limit=1)[0]
    store.fail_claim(first, "transient", retryable=True)
    second = store.claim_pending(_TENANT_A, "w-b", lease_seconds=30, limit=1)[0]
    assert second.fence > first.fence
    with pytest.raises(StaleClaimError):
        store.complete_claim(first)
    with pytest.raises(StaleClaimError):
        store.fail_claim(first, "late", retryable=False)


@pytest.mark.parametrize("queue_factory", _compensation_queue_store_factories())
def test_r3_r2_q18_compensation_tenant_isolation(
    tmp_path: Path,
    queue_factory: CompensationQueueStoreFactory,
) -> None:
    store = queue_factory(tmp_path)
    job_a = _sample_compensation_job(tenant_id=_TENANT_A, key_suffix="ta")
    job_b = _sample_compensation_job(tenant_id=_TENANT_B, key_suffix="tb")
    store.enqueue(job_a)
    store.enqueue(job_b)
    claims_a = store.claim_pending(_TENANT_A, "worker-a", lease_seconds=30, limit=10)
    claims_b = store.claim_pending(_TENANT_B, "worker-b", lease_seconds=30, limit=10)
    assert len(claims_a) == 1
    assert len(claims_b) == 1
    assert claims_a[0].tenant_id == _TENANT_A
    assert claims_b[0].tenant_id == _TENANT_B
    store.complete_claim(claims_a[0])
    loaded_b = store.get_by_idempotency_key(_TENANT_B, job_b.request.idempotency_key)
    assert loaded_b is not None
    assert loaded_b.status != CompensationJobStatus.COMPLETED


def test_r3_r2_q19_durable_identity_continuity() -> None:
    job = _sample_compensation_job(key_suffix="identity")
    work = compensation_side_effect_input_from_job(job)
    assert work.tenant_id == job.tenant_id
    assert work.run_id == job.run_id
    assert work.task_id == job.task_id
    assert work.agent_id == job.agent_id
    assert work.step_index == job.step_index
    assert work.idempotency_key == job.request.idempotency_key
    assert work.original_side_effect_id == job.request.original_side_effect_id


@pytest.mark.asyncio
async def test_r3_r2_q20_compensation_authority_unknown_denies() -> None:
    from intergrax.agents.persistence.compensation_tool_invoke_session import (
        bound_compensation_tool_invoke_session,
    )
    from intergrax.runtime.execution.compensation_side_effect import (
        build_runtime_compensation_side_effect_execution,
    )

    invoked = False

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoked
        invoked = True
        return DeclarativeToolInvokeResult(status="success")

    execution = build_runtime_compensation_side_effect_execution(
        tool_session=bound_compensation_tool_invoke_session(
            RecordingExecutionBoundDeclarativeToolInvoker(_invoke),
        ),
        authority=ParentExecutionAuthority.unknown(),
    )
    result = await execution.execute(compensation_side_effect_input_from_job(_sample_compensation_job()))
    assert result.status == "denied"
    assert invoked is False


@pytest.mark.asyncio
async def test_r3_r2_q21_run_id_mismatch_denied() -> None:
    job = _sample_compensation_job(key_suffix="run-mismatch")
    work = compensation_side_effect_input_from_job(job)
    invoked = False

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoked
        invoked = True
        return DeclarativeToolInvokeResult(status="success")

    execution = _build_admitted_compensation_execution(
        RecordingExecutionBoundDeclarativeToolInvoker(_invoke),
    )
    other_run = RunId(str(mint_run_id()))
    attempt = AttemptId(str(mint_attempt_id()))
    with patch(
        "intergrax.runtime.execution.compensation_side_effect.require_active_execution_identity",
        return_value=(other_run, attempt),
    ):
        result = await execution.execute(work)
    assert invoked is False
    assert result.status == "failed"


@pytest.mark.parametrize("queue_factory", _compensation_queue_store_factories())
@pytest.mark.asyncio
async def test_r3_r2_q22_stable_compensation_idempotency_key_on_retryable_reclaim(
    tmp_path: Path,
    queue_factory: CompensationQueueStoreFactory,
) -> None:
    store = queue_factory(tmp_path)
    job = _sample_compensation_job(key_suffix="stable-key")
    store.enqueue(job)
    first = store.claim_pending(_TENANT_A, "w-a", lease_seconds=30, limit=1)[0]
    key_first = first.job.request.idempotency_key
    store.fail_claim(first, "crash-before-complete", retryable=True)
    second = store.claim_pending(_TENANT_A, "w-b", lease_seconds=30, limit=1)[0]
    assert second.job.request.idempotency_key == key_first


@pytest.mark.parametrize(
    ("queue_factory", "idempotency_factory"),
    _paired_compensation_idempotency_factories(),
)
@pytest.mark.asyncio
async def test_r3_r2_q23_retryable_redelivery_canonical_idempotency_replay(
    tmp_path: Path,
    queue_factory: CompensationQueueStoreFactory,
    idempotency_factory: IdempotencyStorePathFactory,
) -> None:
    idem_store = idempotency_factory(tmp_path)
    queue = queue_factory(tmp_path)
    execution, handler, catalog = _build_canonical_compensation_production_execution(idem_store)
    assert isinstance(catalog, CatalogDeclarativeToolInvoker)
    assert isinstance(catalog.tool_invoker._pre_effect_coordinator, IdempotencyPreEffectCoordinator)

    job = _sample_compensation_job(key_suffix="q23-retryable")
    queue.enqueue(job)
    claim = queue.claim_pending(_TENANT_A, "worker-a", lease_seconds=30, limit=1)[0]
    work = compensation_side_effect_input_from_job(claim.job)
    assert (await _execute_compensation_with_lab_governance(execution, work)).status == "success"
    assert handler.calls == 1
    assert idem_store.get_status(_TENANT_A, work.idempotency_key) == InvocationStatus.COMPLETED
    queue.fail_claim(claim, "simulated crash before queue complete", retryable=True)
    reclaim = queue.claim_pending(_TENANT_A, "worker-b", lease_seconds=30, limit=1)[0]
    work_b = compensation_side_effect_input_from_job(reclaim.job)
    assert work_b.idempotency_key == work.idempotency_key
    assert (await _execute_compensation_with_lab_governance(execution, work_b)).status == "success"
    assert handler.calls == 1


@pytest.mark.parametrize("queue_factory", _compensation_queue_store_factories())
@pytest.mark.asyncio
async def test_r3_r2_q30_expired_running_uncertain_not_reclaimable_queue(
    tmp_path: Path,
    queue_factory: CompensationQueueStoreFactory,
) -> None:
    queue = queue_factory(tmp_path)
    job = _sample_compensation_job(key_suffix="uncertain-queue")
    queue.enqueue(job)
    first = queue.claim_pending(_TENANT_A, "worker-a", lease_seconds=1, limit=1)[0]
    time.sleep(1.2)
    second = queue.claim_pending(_TENANT_A, "worker-b", lease_seconds=30, limit=1)
    loaded = queue.get_by_idempotency_key(_TENANT_A, job.request.idempotency_key)
    assert second == []
    assert loaded is not None
    assert loaded.status == CompensationJobStatus.UNCERTAIN


@pytest.mark.parametrize("queue_factory", _compensation_queue_store_factories())
@pytest.mark.asyncio
async def test_r3_r2_q31_current_owner_completes_claim(
    tmp_path: Path,
    queue_factory: CompensationQueueStoreFactory,
) -> None:
    store = queue_factory(tmp_path)
    job = _sample_compensation_job(key_suffix="complete-owner")
    store.enqueue(job)
    claim = store.claim_pending(_TENANT_A, "worker-ok", lease_seconds=30, limit=1)[0]
    store.complete_claim(claim)
    loaded = store.get_by_idempotency_key(_TENANT_A, job.request.idempotency_key)
    assert loaded is not None
    assert loaded.status == CompensationJobStatus.COMPLETED


@pytest.mark.parametrize(
    ("queue_factory", "idempotency_factory"),
    _paired_compensation_idempotency_factories(),
)
@pytest.mark.asyncio
async def test_r3_r2_q32_crash_window_canonical_production_path_no_duplicate_effect(
    tmp_path: Path,
    queue_factory: CompensationQueueStoreFactory,
    idempotency_factory: IdempotencyStorePathFactory,
) -> None:
    idem_store = idempotency_factory(tmp_path)
    queue = queue_factory(tmp_path)
    execution, handler, _catalog = _build_canonical_compensation_production_execution(idem_store)
    job = _sample_compensation_job(key_suffix="q32-crash")
    queue.enqueue(job)
    claim = queue.claim_pending(_TENANT_A, "worker-a", lease_seconds=1, limit=1)[0]
    work = compensation_side_effect_input_from_job(claim.job)
    assert (await _execute_compensation_with_lab_governance(execution, work)).status == "success"
    assert handler.calls == 1
    assert idem_store.get_status(_TENANT_A, work.idempotency_key) == InvocationStatus.COMPLETED
    time.sleep(1.2)
    reclaim = queue.claim_pending(_TENANT_A, "worker-b", lease_seconds=30, limit=1)
    loaded = queue.get_by_idempotency_key(_TENANT_A, job.request.idempotency_key)
    assert reclaim == []
    assert loaded is not None
    assert loaded.status == CompensationJobStatus.UNCERTAIN
    drained = await drain_pending_compensation_jobs(
        queue,
        tenant_id=_TENANT_A,
        side_effect_execution=execution,
        limit=10,
        owner_id="drain-worker",
        lease_seconds=30,
    )
    assert drained == []
    assert handler.calls == 1


def test_r3_r2_q24_queue_claim_not_execution_authority() -> None:
    worker_source = (
        _REPO_ROOT / "intergrax/agents/persistence/compensation_queue_worker.py"
    ).read_text(encoding="utf-8")
    assert "side_effect_execution.execute" in worker_source
    assert "DeclarativeToolInvoker" not in worker_source
    assert "RuntimeToolInvoker" not in worker_source


def test_r3_r2_q25_redis_atomic_claim() -> None:
    store = _redis_store_or_skip()
    import uuid

    key = f"r3r2:race:{uuid.uuid4()}"
    op = compute_invocation_operation_identity("tool.side", _In(value=1))
    a = store.claim(_TENANT_A, key, "a", 30, operation_identity=op)
    b = store.claim(_TENANT_A, key, "b", 30, operation_identity=op)
    outcomes = {a.outcome, b.outcome}
    assert ClaimOutcome.ACQUIRED in outcomes
    assert ClaimOutcome.BLOCKED_ACTIVE in outcomes


def test_r3_r2_q26_redis_post_effect_uncertain() -> None:
    store = _redis_store_or_skip()
    import uuid

    key = f"r3r2:uncertain:{uuid.uuid4()}"
    acquired = store.claim(_TENANT_A, key, "owner", lease_seconds=1)
    assert acquired.claim is not None
    store.admit_external_effect_may_have_started_with_claim(_TENANT_A, key, acquired.claim)
    time.sleep(1.2)
    retry = store.claim(_TENANT_A, key, "owner-b", 30)
    assert retry.outcome == ClaimOutcome.UNCERTAIN


def test_r3_r2_q27_redis_completed_replay() -> None:
    store = _redis_store_or_skip()
    import uuid

    key = f"r3r2:replay:{uuid.uuid4()}"
    op = compute_invocation_operation_identity("tool.side", _In(value=1))
    first = store.claim(_TENANT_A, key, "o1", 30, operation_identity=op)
    assert first.claim is not None
    store.complete_with_claim(_TENANT_A, key, first.claim, ToolExecutionResult.ok(_Out(result=3)))
    second = store.claim(_TENANT_A, key, "o2", 30, operation_identity=op)
    assert second.outcome == ClaimOutcome.REPLAY_COMPLETED


def test_r3_r2_q28_redis_tenant_separation() -> None:
    store = _redis_store_or_skip()
    import uuid

    key = f"r3r2:tenant:{uuid.uuid4()}"
    assert store.claim(_TENANT_A, key, "a", 30).outcome == ClaimOutcome.ACQUIRED
    assert store.claim(_TENANT_B, key, "b", 30).outcome == ClaimOutcome.ACQUIRED


def test_r3_r2_q29_redis_stale_fence_rejected() -> None:
    store = _redis_store_or_skip()
    import uuid

    key = f"r3r2:stale:{uuid.uuid4()}"
    acquired = store.claim(_TENANT_A, key, "owner-a", 30)
    assert acquired.claim is not None
    stale = InvocationClaim(
        tenant_id=_TENANT_A,
        key=key,
        owner_id=acquired.claim.owner_id,
        lease_expires_at=acquired.claim.lease_expires_at,
        fence=acquired.claim.fence - 1,
    )
    with pytest.raises(StaleClaimError):
        store.complete_with_claim(_TENANT_A, key, stale, ToolExecutionResult.ok(_Out()))
    with pytest.raises(StaleClaimError):
        store.mark_uncertain_with_claim(_TENANT_A, key, stale)
