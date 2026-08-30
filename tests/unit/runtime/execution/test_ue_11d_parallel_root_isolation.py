# © Artur Czarnecki. All rights reserved.

"""UE-11D — parallel canonical root execution isolation proofs."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    mint_run_id,
    mint_task_id,
    peek_active_execution_id,
    peek_active_execution_identity,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.runtime.execution.active_execution_budget import (
    peek_active_execution_budget,
    require_active_execution_budget,
)
from intergrax.runtime.execution.agentic import AgentExecutor
from intergrax.runtime.execution.budget.consumption import consume_llm_call
from intergrax.runtime.execution.budget.ledger import (
    ExecutionBudgetLedger,
    InMemoryExecutionBudgetLedger,
    RunBudgetExecutionBudgetLedgerFactory,
    create_execution_budget_ledger,
    fixed_execution_budget_ledger_factory,
)
from intergrax.runtime.execution.facade import Execution
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.result import ExecutionStatus
from intergrax.runtime.execution.runtime import ExecutionRuntime, RootExecutionOptions
from intergrax.runtime.execution.strategy_router import StrategyExecutionRouter
from intergrax.runtime.governance.active_execution_authority import (
    peek_active_execution_authority,
    require_active_execution_authority,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest

pytestmark = pytest.mark.unit

_AUTHORITY_A = ParentExecutionAuthority.scoped(("capability:read",))
_AUTHORITY_B = ParentExecutionAuthority.scoped(("capability:write",))
_ITERATIONS = 10


@dataclass(frozen=True, slots=True)
class AuthorityFingerprint:
    unrestricted: bool
    permission_scopes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConcurrentExecutionContextObservation:
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    authority: AuthorityFingerprint
    budget_execution_id: ExecutionId


@dataclass(frozen=True, slots=True)
class ConcurrentRootObservation:
    label: str
    before_yield: ConcurrentExecutionContextObservation
    during_overlap: ConcurrentExecutionContextObservation
    after_yield: ConcurrentExecutionContextObservation


@dataclass(frozen=True, slots=True)
class ConcurrentPairResult:
    root_a: ConcurrentRootObservation
    root_b: ConcurrentRootObservation
    ledger_a: InMemoryExecutionBudgetLedger
    ledger_b: InMemoryExecutionBudgetLedger
    max_simultaneously_active: int


@dataclass(frozen=True, slots=True)
class CreatedRootLedger:
    run_id: RunId
    attempt_id: AttemptId
    ledger: InMemoryExecutionBudgetLedger


@dataclass
class RecordingExecutionBudgetLedgerFactory:
    """Test-local factory wrapper delegating to production ``RunBudgetExecutionBudgetLedgerFactory``."""

    _delegate: RunBudgetExecutionBudgetLedgerFactory
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    call_count: int = 0
    created: dict[tuple[RunId, AttemptId], CreatedRootLedger] = field(default_factory=dict)

    def create_ledger(
        self,
        run_budget: RunBudget | None = None,
        *,
        tenant_id: str | None = None,
        run_id: RunId | None = None,
        attempt_id: AttemptId | None = None,
    ) -> ExecutionBudgetLedger:
        ledger = self._delegate.create_ledger(
            run_budget,
            tenant_id=tenant_id,
            run_id=run_id,
            attempt_id=attempt_id,
        )
        if not isinstance(ledger, InMemoryExecutionBudgetLedger):
            raise TypeError("production factory must return InMemoryExecutionBudgetLedger")
        with self._lock:
            self.call_count += 1
            if run_id is not None and attempt_id is not None:
                self.created[(run_id, attempt_id)] = CreatedRootLedger(
                    run_id=run_id,
                    attempt_id=attempt_id,
                    ledger=ledger,
                )
        return ledger


@dataclass(frozen=True, slots=True)
class SharedRuntimeConcurrentPairResult:
    root_a: ConcurrentRootObservation
    root_b: ConcurrentRootObservation
    ledger_a: InMemoryExecutionBudgetLedger
    ledger_b: InMemoryExecutionBudgetLedger
    factory_call_count: int
    backend_invocations: int
    max_simultaneously_active: int


class _ActiveBackendTracker:
    __slots__ = ("_active", "_lock", "max_active")

    def __init__(self) -> None:
        self._active = 0
        self.max_active = 0
        self._lock = asyncio.Lock()

    async def enter(self) -> None:
        async with self._lock:
            self._active += 1
            if self._active > self.max_active:
                self.max_active = self._active

    async def leave(self) -> None:
        async with self._lock:
            self._active -= 1


def _authority_fingerprint(authority: ParentExecutionAuthority) -> AuthorityFingerprint:
    return AuthorityFingerprint(
        unrestricted=authority.unrestricted,
        permission_scopes=authority.permission_scopes,
    )


def _capture_context_observation() -> ConcurrentExecutionContextObservation:
    run_id, attempt_id = require_active_execution_identity()
    execution_id = require_active_execution_id()
    authority = require_active_execution_authority()
    budget_state = require_active_execution_budget()
    assert budget_state.execution_id == execution_id
    return ConcurrentExecutionContextObservation(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        authority=_authority_fingerprint(authority),
        budget_execution_id=budget_state.execution_id,
    )


def _assert_clean_caller_context() -> None:
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None
    assert peek_active_execution_authority() is None
    assert peek_active_execution_budget() is None


def _agentic_request(*, run_id: RunId) -> ExecutionRequest[RuntimeRequest, AgentExecutionResult]:
    return ExecutionRequest(
        input=RuntimeRequest(
            agent_id="ue-11d-observer",
            user_id="ue-11d-user",
            session_id="ue-11d-session",
            message="concurrency proof",
            task_id=mint_task_id(),
            run_id=run_id,
        ),
        output_type=AgentExecutionResult,
        capabilities=frozenset({ExecutionCapability.AGENT}),
    )


class SharedConcurrentContextObserverEngine:
    __slots__ = (
        "_barrier",
        "_label_by_run_id",
        "_llm_calls_by_run_id",
        "_observations",
        "_tracker",
        "invocations",
    )

    def __init__(
        self,
        *,
        barrier: asyncio.Barrier,
        tracker: _ActiveBackendTracker,
        llm_calls_by_run_id: dict[RunId, int],
        label_by_run_id: dict[RunId, str],
    ) -> None:
        self._barrier = barrier
        self._tracker = tracker
        self._llm_calls_by_run_id = llm_calls_by_run_id
        self._label_by_run_id = label_by_run_id
        self._observations: dict[RunId, ConcurrentRootObservation] = {}
        self.invocations = 0

    def observation_for(self, run_id: RunId) -> ConcurrentRootObservation:
        observation = self._observations.get(run_id)
        if observation is None:
            raise KeyError(f"missing observation for run_id={run_id}")
        return observation

    async def run_with_result(self, request: RuntimeRequest) -> AgentExecutionResult:
        self.invocations += 1
        before_yield = _capture_context_observation()
        run_id = before_yield.run_id
        if request.run_id != run_id:
            raise RuntimeError("RuntimeRequest run_id does not match active execution identity")
        llm_calls = self._llm_calls_by_run_id[run_id]
        await self._tracker.enter()
        await self._barrier.wait()
        during_overlap = _capture_context_observation()
        await self._barrier.wait()
        await self._tracker.leave()
        after_yield = _capture_context_observation()
        for _ in range(llm_calls):
            consume_llm_call()
        self._observations[run_id] = ConcurrentRootObservation(
            label=self._label_by_run_id[run_id],
            before_yield=before_yield,
            during_overlap=during_overlap,
            after_yield=after_yield,
        )
        return AgentExecutionResult(
            agent_id="ue-11d-observer",
            run_id=run_id,
            status=AgentExecutionStatus.COMPLETED,
        )


class ConcurrentContextObserverEngine:
    __slots__ = (
        "_barrier",
        "_label",
        "_llm_calls",
        "_tracker",
        "invocations",
        "observation",
    )

    def __init__(
        self,
        label: str,
        *,
        barrier: asyncio.Barrier,
        tracker: _ActiveBackendTracker,
        llm_calls: int,
    ) -> None:
        self._label = label
        self._barrier = barrier
        self._tracker = tracker
        self._llm_calls = llm_calls
        self.invocations = 0
        self.observation: ConcurrentRootObservation | None = None

    async def run_with_result(self, request: RuntimeRequest) -> AgentExecutionResult:
        del request
        self.invocations += 1
        before_yield = _capture_context_observation()
        await self._tracker.enter()
        await self._barrier.wait()
        during_overlap = _capture_context_observation()
        await self._barrier.wait()
        await self._tracker.leave()
        after_yield = _capture_context_observation()
        for _ in range(self._llm_calls):
            consume_llm_call()
        self.observation = ConcurrentRootObservation(
            label=self._label,
            before_yield=before_yield,
            during_overlap=during_overlap,
            after_yield=after_yield,
        )
        return AgentExecutionResult(
            agent_id="ue-11d-observer",
            run_id=before_yield.run_id,
            status=AgentExecutionStatus.COMPLETED,
        )


def _build_shared_root_execution(
    *,
    engine: SharedConcurrentContextObserverEngine,
    recording_factory: RecordingExecutionBudgetLedgerFactory,
) -> Execution[
    ExecutionRequest[RuntimeRequest, AgentExecutionResult],
    object,
]:
    router = StrategyExecutionRouter[
        RuntimeRequest,
        AgentExecutionResult,
        object,
    ](agent_executor=AgentExecutor(engine))
    runtime = ExecutionRuntime[
        ExecutionRequest[RuntimeRequest, AgentExecutionResult],
        object,
    ](
        router,
        ledger_factory=recording_factory,
        run_budget=RunBudget(max_llm_calls=10),
    )
    return Execution(runtime)


def _build_root_execution(
    *,
    engine: ConcurrentContextObserverEngine,
    authority: ParentExecutionAuthority,
    run_id: RunId,
    ledger: InMemoryExecutionBudgetLedger,
) -> tuple[
    Execution[
        ExecutionRequest[RuntimeRequest, AgentExecutionResult],
        object,
    ],
    RootExecutionOptions,
]:
    router = StrategyExecutionRouter[
        RuntimeRequest,
        AgentExecutionResult,
        object,
    ](agent_executor=AgentExecutor(engine))
    runtime = ExecutionRuntime[
        ExecutionRequest[RuntimeRequest, AgentExecutionResult],
        object,
    ](
        router,
        ledger_factory=fixed_execution_budget_ledger_factory(ledger),
        run_budget=RunBudget(max_llm_calls=10),
    )
    options = RootExecutionOptions(authority=authority, run_id=run_id)
    return Execution(runtime), options


async def _run_shared_runtime_concurrent_root_pair(
    *,
    llm_calls_a: int,
    llm_calls_b: int,
) -> SharedRuntimeConcurrentPairResult:
    _assert_clean_caller_context()
    barrier = asyncio.Barrier(2)
    tracker = _ActiveBackendTracker()

    run_id_a = mint_run_id()
    run_id_b = mint_run_id()
    llm_calls_by_run_id = {run_id_a: llm_calls_a, run_id_b: llm_calls_b}
    label_by_run_id = {run_id_a: "a", run_id_b: "b"}

    production_delegate = RunBudgetExecutionBudgetLedgerFactory(
        default_run_budget=RunBudget(max_llm_calls=10),
    )
    recording_factory = RecordingExecutionBudgetLedgerFactory(_delegate=production_delegate)
    engine = SharedConcurrentContextObserverEngine(
        barrier=barrier,
        tracker=tracker,
        llm_calls_by_run_id=llm_calls_by_run_id,
        label_by_run_id=label_by_run_id,
    )
    execution = _build_shared_root_execution(
        engine=engine,
        recording_factory=recording_factory,
    )

    options_a = RootExecutionOptions(authority=_AUTHORITY_A, run_id=run_id_a)
    options_b = RootExecutionOptions(authority=_AUTHORITY_B, run_id=run_id_b)

    result_a, result_b = await asyncio.gather(
        execution.execute(_agentic_request(run_id=run_id_a), options=options_a),
        execution.execute(_agentic_request(run_id=run_id_b), options=options_b),
    )

    assert result_a.status is ExecutionStatus.COMPLETED
    assert result_b.status is ExecutionStatus.COMPLETED
    assert engine.invocations == 2
    assert tracker.max_active >= 2
    _assert_clean_caller_context()

    root_a = engine.observation_for(run_id_a)
    root_b = engine.observation_for(run_id_b)

    created_a = recording_factory.created[(run_id_a, root_a.before_yield.attempt_id)]
    created_b = recording_factory.created[(run_id_b, root_b.before_yield.attempt_id)]
    assert created_a.ledger is not created_b.ledger
    assert recording_factory.call_count == 2

    return SharedRuntimeConcurrentPairResult(
        root_a=root_a,
        root_b=root_b,
        ledger_a=created_a.ledger,
        ledger_b=created_b.ledger,
        factory_call_count=recording_factory.call_count,
        backend_invocations=engine.invocations,
        max_simultaneously_active=tracker.max_active,
    )


async def _run_concurrent_root_pair(
    *,
    llm_calls_a: int,
    llm_calls_b: int,
) -> ConcurrentPairResult:
    _assert_clean_caller_context()
    barrier = asyncio.Barrier(2)
    tracker = _ActiveBackendTracker()

    run_id_a = mint_run_id()
    run_id_b = mint_run_id()
    ledger_a = create_execution_budget_ledger(RunBudget(max_llm_calls=10))
    ledger_b = create_execution_budget_ledger(RunBudget(max_llm_calls=10))

    engine_a = ConcurrentContextObserverEngine(
        "a",
        barrier=barrier,
        tracker=tracker,
        llm_calls=llm_calls_a,
    )
    engine_b = ConcurrentContextObserverEngine(
        "b",
        barrier=barrier,
        tracker=tracker,
        llm_calls=llm_calls_b,
    )

    execution_a, options_a = _build_root_execution(
        engine=engine_a,
        authority=_AUTHORITY_A,
        run_id=run_id_a,
        ledger=ledger_a,
    )
    execution_b, options_b = _build_root_execution(
        engine=engine_b,
        authority=_AUTHORITY_B,
        run_id=run_id_b,
        ledger=ledger_b,
    )

    result_a, result_b = await asyncio.gather(
        execution_a.execute(_agentic_request(run_id=run_id_a), options=options_a),
        execution_b.execute(_agentic_request(run_id=run_id_b), options=options_b),
    )

    assert result_a.status is ExecutionStatus.COMPLETED
    assert result_b.status is ExecutionStatus.COMPLETED
    assert engine_a.invocations == 1
    assert engine_b.invocations == 1
    assert engine_a.observation is not None
    assert engine_b.observation is not None
    assert tracker.max_active >= 2
    _assert_clean_caller_context()

    return ConcurrentPairResult(
        root_a=engine_a.observation,
        root_b=engine_b.observation,
        ledger_a=ledger_a,
        ledger_b=ledger_b,
        max_simultaneously_active=tracker.max_active,
    )


def _assert_root_identity_isolation(pair: ConcurrentPairResult | SharedRuntimeConcurrentPairResult) -> None:
    root_a = pair.root_a
    root_b = pair.root_b
    for observation in (root_a, root_b):
        assert observation.before_yield == observation.during_overlap
        assert observation.before_yield == observation.after_yield

    a = root_a.before_yield
    b = root_b.before_yield
    assert a.execution_id != b.execution_id
    assert a.run_id != b.run_id
    assert a.attempt_id != b.attempt_id

    assert root_a.during_overlap.execution_id == a.execution_id
    assert root_b.during_overlap.execution_id == b.execution_id
    assert root_a.during_overlap.run_id == a.run_id
    assert root_b.during_overlap.run_id == b.run_id


def _assert_root_authority_isolation(pair: ConcurrentPairResult | SharedRuntimeConcurrentPairResult) -> None:
    a = pair.root_a.before_yield
    b = pair.root_b.before_yield
    assert a.authority == _authority_fingerprint(_AUTHORITY_A)
    assert b.authority == _authority_fingerprint(_AUTHORITY_B)
    assert a.authority != b.authority
    assert pair.root_a.before_yield.authority == pair.root_a.after_yield.authority
    assert pair.root_b.before_yield.authority == pair.root_b.after_yield.authority
    assert pair.root_a.during_overlap.authority == a.authority
    assert pair.root_b.during_overlap.authority == b.authority


def _ledger_participant_ids(
    ledger: InMemoryExecutionBudgetLedger,
    *,
    attempt_id: AttemptId,
) -> frozenset[ExecutionId]:
    snapshot = ledger.export_snapshot(attempt_id)
    return frozenset(record.execution_id for record in snapshot.records)


def _assert_root_budget_isolation(pair: ConcurrentPairResult | SharedRuntimeConcurrentPairResult) -> None:
    a = pair.root_a.before_yield
    b = pair.root_b.before_yield

    snapshot_a = pair.ledger_a.export_snapshot(a.attempt_id)
    snapshot_b = pair.ledger_b.export_snapshot(b.attempt_id)
    assert snapshot_a.root_shared_consumed.llm_calls == 1
    assert snapshot_b.root_shared_consumed.llm_calls == 2

    participants_a = _ledger_participant_ids(pair.ledger_a, attempt_id=a.attempt_id)
    participants_b = _ledger_participant_ids(pair.ledger_b, attempt_id=b.attempt_id)
    assert b.execution_id not in participants_a
    assert a.execution_id not in participants_b

    for observation in (pair.root_a, pair.root_b):
        overlap = observation.during_overlap
        assert overlap.budget_execution_id == overlap.execution_id


def _assert_no_context_cross_talk(pair: ConcurrentPairResult | SharedRuntimeConcurrentPairResult) -> None:
    _assert_root_identity_isolation(pair)
    _assert_root_authority_isolation(pair)
    _assert_root_budget_isolation(pair)

    a = pair.root_a.before_yield
    b = pair.root_b.before_yield
    assert a.run_id != b.run_id
    assert a.attempt_id != b.attempt_id
    assert a.execution_id != b.execution_id
    assert a.budget_execution_id != b.budget_execution_id
    assert a.authority != b.authority


@pytest.mark.asyncio
@pytest.mark.parametrize("iteration", range(_ITERATIONS))
async def test_ue_11d_shared_runtime_parallel_root_identity_isolation(iteration: int) -> None:
    del iteration
    pair = await _run_shared_runtime_concurrent_root_pair(llm_calls_a=1, llm_calls_b=1)
    _assert_root_identity_isolation(pair)
    assert pair.max_simultaneously_active >= 2
    assert pair.backend_invocations == 2
    assert pair.factory_call_count == 2
    assert pair.ledger_a is not pair.ledger_b


@pytest.mark.asyncio
@pytest.mark.parametrize("iteration", range(_ITERATIONS))
async def test_ue_11d_shared_runtime_parallel_root_authority_isolation(iteration: int) -> None:
    del iteration
    pair = await _run_shared_runtime_concurrent_root_pair(llm_calls_a=1, llm_calls_b=1)
    _assert_root_authority_isolation(pair)


@pytest.mark.asyncio
@pytest.mark.parametrize("iteration", range(_ITERATIONS))
async def test_ue_11d_shared_runtime_parallel_root_budget_isolation(iteration: int) -> None:
    del iteration
    pair = await _run_shared_runtime_concurrent_root_pair(llm_calls_a=1, llm_calls_b=2)
    _assert_root_budget_isolation(pair)
    assert pair.factory_call_count == 2
    assert pair.ledger_a is not pair.ledger_b


@pytest.mark.asyncio
@pytest.mark.parametrize("iteration", range(_ITERATIONS))
async def test_ue_11d_shared_runtime_no_context_cross_talk(iteration: int) -> None:
    del iteration
    pair = await _run_shared_runtime_concurrent_root_pair(llm_calls_a=1, llm_calls_b=2)
    _assert_no_context_cross_talk(pair)
    _assert_clean_caller_context()


@pytest.mark.asyncio
@pytest.mark.parametrize("iteration", range(_ITERATIONS))
async def test_ue_11d_multi_runtime_parallel_root_identity_isolation(iteration: int) -> None:
    del iteration
    pair = await _run_concurrent_root_pair(llm_calls_a=1, llm_calls_b=1)
    _assert_root_identity_isolation(pair)
    assert pair.max_simultaneously_active >= 2


@pytest.mark.asyncio
@pytest.mark.parametrize("iteration", range(_ITERATIONS))
async def test_ue_11d_multi_runtime_parallel_root_authority_isolation(iteration: int) -> None:
    del iteration
    pair = await _run_concurrent_root_pair(llm_calls_a=1, llm_calls_b=1)
    _assert_root_authority_isolation(pair)


@pytest.mark.asyncio
@pytest.mark.parametrize("iteration", range(_ITERATIONS))
async def test_ue_11d_multi_runtime_parallel_root_budget_isolation(iteration: int) -> None:
    del iteration
    pair = await _run_concurrent_root_pair(llm_calls_a=1, llm_calls_b=2)
    _assert_root_budget_isolation(pair)


@pytest.mark.asyncio
@pytest.mark.parametrize("iteration", range(_ITERATIONS))
async def test_ue_11d_multi_runtime_no_context_cross_talk(iteration: int) -> None:
    del iteration
    pair = await _run_concurrent_root_pair(llm_calls_a=1, llm_calls_b=2)
    _assert_no_context_cross_talk(pair)


@pytest.mark.asyncio
async def test_ue_11d_root_identity_freshness_across_repeated_pairs() -> None:
    seen_execution_ids: set[ExecutionId] = set()
    for _ in range(3):
        pair = await _run_shared_runtime_concurrent_root_pair(llm_calls_a=1, llm_calls_b=1)
        execution_ids = {
            pair.root_a.before_yield.execution_id,
            pair.root_b.before_yield.execution_id,
        }
        assert len(execution_ids) == 2
        assert execution_ids.isdisjoint(seen_execution_ids)
        seen_execution_ids.update(execution_ids)
