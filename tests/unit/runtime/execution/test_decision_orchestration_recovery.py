# © Artur Czarnecki. All rights reserved.

"""DS-NEXUS-02 — Decision semantic checkpoint with orchestration recovery participation.

Decision semantic checkpoint remains Decision-owned while Decision-triggered
ORCHESTRATION uses canonical Execution recovery.
"""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.agents.agent_engine import AgentEngine
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.decision_checkpoint import (
    DecisionCheckpointState,
    decision_checkpoint_state,
)
from intergrax.contracts.decision_finalization import (
    DecisionFinalizeGuardState,
    DecisionFinalizationKey,
    decision_finalization_key,
    initial_decision_finalize_guard,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    DecisionLifecycleState,
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
    peek_active_execution_id,
    peek_active_execution_identity,
    require_active_execution_id,
    require_active_execution_identity,
)
from intergrax.runtime.execution import ExecutionCapability
from intergrax.runtime.execution.active_decision_checkpoint_persistence import (
    ActiveDecisionCheckpointPersistenceBinding,
)
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    require_active_decision_lifecycle_host,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    peek_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.active_execution_resume import (
    peek_active_execution_resume_plan,
)
from intergrax.runtime.execution.active_execution_work_port import (
    ActiveExecutionWorkPortBinding,
)
from intergrax.runtime.execution.budget.consumption import consume_llm_call
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.decision_checkpoint_persistence import (
    DecisionCheckpointPersistence,
    load_decision_checkpoint,
    save_decision_checkpoint,
)
from intergrax.runtime.execution.decision_lifecycle_host import CanonicalDecisionLifecycleHost
from intergrax.runtime.execution.execution_work_port import child_execution_work_port
from intergrax.runtime.execution.orchestration import OrchestrationExecutor
from intergrax.runtime.execution.request import ExecutionRequest as NeutralExecutionRequest
from intergrax.runtime.execution.runtime import (
    ExecutionRuntime,
    RootExecutionContext,
)
from intergrax.runtime.execution.task_adapter import TaskExecutionInput
from intergrax.runtime.governance.active_execution_authority import (
    peek_active_execution_authority,
    require_active_execution_authority,
)
from intergrax.runtime.long_running.checkpoint_builder import (
    apply_runtime_checkpoint_to_task,
    build_runtime_checkpoint,
    resolve_task_runtime_checkpoint,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionCheckpointStatus,
    ExecutionTreeRecorder,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.execution.execution_graph import (
    ExecutionGraph,
    ExecutionNode,
)
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, PlanStep
from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy
from intergrax.runtime.nexus.task_classifier import TaskClassification
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_NODE_A = "n_a"
_NODE_B = "n_b"
_NODE_C = "n_c"
_NODE_D = "n_d"
_AGENT_A = "agent_a"
_AGENT_B = "agent_b"
_AGENT_C = "agent_c"
_AGENT_D = "agent_d"
_CAPABILITY = "cap.shared"
_DECISION_CONTRACT_GLOB = "intergrax/contracts/decision*.py"
_FORBIDDEN_DECISION_TOKENS = (
    "intergrax.runtime.nexus",
    "NexusLoop",
    "GraphExecutor",
    "OrchestrationExecutor",
    "StrategyExecutionRouter(",
)
_FORBIDDEN_DECISION_RECOVERY_TOKENS = (
    "prepare_task_for_checkpoint_resume",
    "build_task_checkpoint_resume_plan",
    "bind_active_execution_resume_plan",
    "reset_active_execution_resume_plan",
    "UnifiedTaskRunner",
)
_FORBIDDEN_MAIN_PROOF_RECOVERY_CALLS = (
    "build_task_checkpoint_resume_plan",
    "prepare_task_for_checkpoint_resume",
    "bind_active_execution_resume_plan",
    "reset_active_execution_resume_plan",
    "_run_graph",
)
_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget(max_llm_calls=50))


@dataclass(frozen=True, slots=True)
class _SemanticOutcomePayload:
    marker: str


@dataclass(frozen=True, slots=True)
class _NodeInvocationCounts:
    agent_a: int
    agent_b: int
    agent_c: int
    agent_d: int


@dataclass(slots=True)
class _InterruptCapture:
    decision_checkpoint: DecisionCheckpointState[_SemanticOutcomePayload] | None = None
    decision_key: DecisionFinalizationKey | None = None
    orchestration_child_execution_id: ExecutionId | None = None
    task_id: TaskId | None = None
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None
    hosting_root_execution_id: ExecutionId | None = None
    graph: ExecutionGraph | None = None
    interrupted_task: Task | None = None
    authority_observed_during_orchestration: bool = False


@dataclass(slots=True)
class _GateState:
    gate_open: bool = False


class _GatedCountingAgentEngine(AgentEngine):
    __slots__ = ("_counts", "_gate_state", "_authority_observed")

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        gate_state: _GateState,
        authority_observed: list[bool],
    ) -> None:
        super().__init__(registry)
        self._counts: dict[str, int] = {
            _AGENT_A: 0,
            _AGENT_B: 0,
            _AGENT_C: 0,
            _AGENT_D: 0,
        }
        self._gate_state = gate_state
        self._authority_observed = authority_observed

    async def run_with_result(self, request):
        agent_id = request.agent_id
        if agent_id in self._counts:
            self._counts[agent_id] += 1
        execution_id = require_active_execution_id()
        consume_llm_call()
        authority = require_active_execution_authority()
        self._authority_observed.append(authority is not None)
        if not self._gate_state.gate_open and agent_id == _AGENT_C:
            run_id, _ = require_active_execution_identity()
            return AgentExecutionResult(
                agent_id=agent_id,
                run_id=run_id,
                status=AgentExecutionStatus.FAILED,
                summary="controlled interruption gate",
                errors=["controlled interruption gate"],
            )
        return await super().run_with_result(request)

    def snapshot_counts(self) -> _NodeInvocationCounts:
        return _NodeInvocationCounts(
            agent_a=self._counts[_AGENT_A],
            agent_b=self._counts[_AGENT_B],
            agent_c=self._counts[_AGENT_C],
            agent_d=self._counts[_AGENT_D],
        )


class DecisionFacingOrchestrationProbe:
    """Decision-aware helper that knows only canonical Execution abstractions."""

    __slots__ = ("_access",)

    def __init__(
        self,
        access: ActiveExecutionWorkPortBinding[
            TaskExecutionInput,
            TaskResult,
            TaskResult,
        ],
    ) -> None:
        self._access = access

    async def request_orchestration_work(
        self,
        *,
        message: str,
    ) -> TaskResult:
        typed_port = self._access.require_active()
        request = NeutralExecutionRequest(
            input=TaskExecutionInput(message=message),
            output_type=TaskResult,
            capabilities=frozenset({ExecutionCapability.ORCHESTRATION}),
        )
        return await typed_port.execute(request)


class _GraphOrchestrationBackend:
    """Composition-root orchestration backend — not Decision-facing."""

    __slots__ = ("_capture", "_executor", "_graph")

    def __init__(
        self,
        executor: GraphExecutor,
        graph: ExecutionGraph,
        capture: _InterruptCapture,
    ) -> None:
        self._executor = executor
        self._graph = graph
        self._capture = capture

    async def handle_task(
        self,
        task: Task,
        *,
        run_id: RunId,
        attempt_id: AttemptId | None = None,
    ) -> TaskResult:
        active_run_id, active_attempt_id = require_active_execution_identity()
        del run_id, attempt_id
        orchestration_child_id = require_active_execution_id()
        self._capture.orchestration_child_execution_id = orchestration_child_id
        self._capture.authority_observed_during_orchestration = (
            require_active_execution_authority() is not None
        )
        apply_runtime_checkpoint_to_task(
            task,
            RuntimeCheckpoint(
                run_id=active_run_id,
                attempt_id=active_attempt_id,
                execution_tree=ExecutionTreeRecorder.start_root(
                    task_id=task.task_id,
                    run_id=active_run_id,
                    attempt_id=active_attempt_id,
                    root_execution_id=orchestration_child_id,
                ).snapshot,
            ),
        )
        budget_token = bind_root_execution_budget(
            execution_id=orchestration_child_id,
            ledger=_UNLIMITED_LEDGER,
        )
        try:
            await self._executor.execute(self._graph, task)
        finally:
            reset_active_execution_budget(budget_token)
        active_run_id, _ = require_active_execution_identity()
        return TaskResult(
            task_id=task.task_id,
            run_id=active_run_id,
            state=TaskState.COMPLETED,
        )


class _RecordingDecisionCheckpointPersistence(
    DecisionCheckpointPersistence[_SemanticOutcomePayload],
):
    __slots__ = ("_store", "load_calls", "save_calls")

    def __init__(self) -> None:
        self.load_calls = 0
        self.save_calls = 0
        self._store: dict[
            DecisionFinalizationKey,
            DecisionCheckpointState[_SemanticOutcomePayload],
        ] = {}

    def load(
        self,
        *,
        key: DecisionFinalizationKey,
    ) -> DecisionCheckpointState[_SemanticOutcomePayload] | None:
        self.load_calls += 1
        return self._store.get(key)

    def save(
        self,
        *,
        checkpoint: DecisionCheckpointState[_SemanticOutcomePayload],
    ) -> None:
        self.save_calls += 1
        self._store[checkpoint.finalization.key] = checkpoint


def _sequential_plan(task: Task) -> NexusPlan:
    return NexusPlan(
        task_id=task.task_id,
        classification=TaskClassification.MULTI_AGENT.value,
        steps=[
            PlanStep(
                step_id=_NODE_A,
                agent_id=_AGENT_A,
                capability=_CAPABILITY,
            ),
            PlanStep(
                step_id=_NODE_B,
                agent_id=_AGENT_B,
                capability=_CAPABILITY,
                depends_on=[_NODE_A],
            ),
            PlanStep(
                step_id=_NODE_C,
                agent_id=_AGENT_C,
                capability=_CAPABILITY,
                depends_on=[_NODE_B],
            ),
            PlanStep(
                step_id=_NODE_D,
                agent_id=_AGENT_D,
                capability=_CAPABILITY,
                depends_on=[_NODE_C],
            ),
        ],
    )


class _DeterministicSequentialPlanner:
    """Composition-root planner returning the DS-NEXUS-02 sequential graph."""

    def plan(self, task: Task, registry: AgentRegistry) -> NexusPlan:
        del registry
        return _sequential_plan(task)


class _DeterministicClassifier:
    """Composition-root classifier for deterministic Nexus orchestration."""

    def classify(self, task: Task) -> Task:
        task.runtime.classification.value = TaskClassification.MULTI_AGENT.value
        task.sync_metadata()
        return task


def _sequential_graph(task_id: TaskId) -> ExecutionGraph:
    return ExecutionGraph(
        graph_id="ds-nexus-02-resume-tree",
        task_id=task_id,
        nodes=[
            ExecutionNode(node_id=_NODE_A, agent_id=_AGENT_A, capability=_CAPABILITY),
            ExecutionNode(
                node_id=_NODE_B,
                agent_id=_AGENT_B,
                capability=_CAPABILITY,
                depends_on=[_NODE_A],
            ),
            ExecutionNode(
                node_id=_NODE_C,
                agent_id=_AGENT_C,
                capability=_CAPABILITY,
                depends_on=[_NODE_B],
            ),
            ExecutionNode(
                node_id=_NODE_D,
                agent_id=_AGENT_D,
                capability=_CAPABILITY,
                depends_on=[_NODE_C],
            ),
        ],
    )


def _register_resume_agents(registry: AgentRegistry) -> None:
    for agent_id in (_AGENT_A, _AGENT_B, _AGENT_C, _AGENT_D):
        registry.register(
            UaepPipelineStubAgent(
                agent_id=agent_id,
                capability=_CAPABILITY,
                prefix=agent_id,
                answer_separator=":",
            ),
        )


def _decision_identity_from_hosting_execution(
    *,
    task_id: TaskId,
    tenant_id: str,
) -> DecisionIdentity:
    run_id, attempt_id = require_active_execution_identity()
    execution_id = require_active_execution_id()
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="ds-nexus-02", subject="orchestration-recovery"),
        tenant_id=tenant_id,
        execution=DecisionExecutionLineage(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        ),
    )


def _execution_id_for_node(
    runtime_checkpoint: RuntimeCheckpoint,
    node_id: str,
) -> ExecutionId | None:
    entry = runtime_checkpoint.execution_tree.entry_by_graph_node_id(node_id)
    if entry is None:
        return None
    return entry.execution_id


class _OrchestrationOnlyRouter:
    """Composition-root router delegate for orchestration-only child work."""

    __slots__ = ("_executor", "_task")

    def __init__(self, task: Task, backend: _GraphOrchestrationBackend) -> None:
        self._task = task
        self._executor = OrchestrationExecutor(backend)

    async def execute(
        self,
        request: NeutralExecutionRequest[TaskExecutionInput, TaskResult],
    ) -> TaskResult:
        del request
        return await self._executor.execute(self._task)


@dataclass(frozen=True, slots=True)
class _RootProbeRequest:
    message: str


@dataclass(frozen=True, slots=True)
class _RootProbeResult:
    orchestration_state: TaskState


class _DecisionHostedOrchestrationDelegate:
    __slots__ = (
        "_capture",
        "_checkpoint_access",
        "_gate_state",
        "_graph",
        "_task",
        "_work_port_access",
    )

    def __init__(
        self,
        *,
        task: Task,
        graph: ExecutionGraph,
        capture: _InterruptCapture,
        work_port_access: ActiveExecutionWorkPortBinding[
            TaskExecutionInput,
            TaskResult,
            TaskResult,
        ],
        checkpoint_access: ActiveDecisionCheckpointPersistenceBinding[
            _RecordingDecisionCheckpointPersistence
        ],
        gate_state: _GateState,
    ) -> None:
        self._task = task
        self._graph = graph
        self._capture = capture
        self._work_port_access = work_port_access
        self._checkpoint_access = checkpoint_access
        self._gate_state = gate_state

    async def execute(self, request: _RootProbeRequest) -> _RootProbeResult:
        identity = _decision_identity_from_hosting_execution(
            task_id=self._task.task_id,
            tenant_id=self._task.tenant_id,
        )
        host = require_active_decision_lifecycle_host()
        lifecycle = host.start(identity)
        lifecycle = host.transition(lifecycle, DecisionLifecycleStage.VERIFICATION)
        finalization = initial_decision_finalize_guard(decision_finalization_key(identity))
        checkpoint = decision_checkpoint_state(
            lifecycle=lifecycle,
            finalization=finalization,
        )
        persistence = self._checkpoint_access.require_active()
        save_decision_checkpoint(persistence, checkpoint=checkpoint)
        self._capture.decision_checkpoint = checkpoint
        self._capture.decision_key = decision_finalization_key(identity)

        self._gate_state.gate_open = False
        probe = DecisionFacingOrchestrationProbe(self._work_port_access)
        result = await probe.request_orchestration_work(message=request.message)
        self._capture.interrupted_task = self._task
        self._capture.graph = self._graph
        return _RootProbeResult(orchestration_state=result.state)


@pytest.mark.asyncio
async def test_decision_orchestration_checkpoint_recovery_participation(
    tmp_path: Path,
) -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    hosting_root_execution_id = mint_execution_id()
    tenant_id = "t1"

    registry = AgentRegistry()
    _register_resume_agents(registry)
    gate_state = _GateState(gate_open=False)
    authority_observed: list[bool] = []
    engine = _GatedCountingAgentEngine(
        registry,
        gate_state=gate_state,
        authority_observed=authority_observed,
    )
    interrupt_executor = GraphExecutor(
        registry,
        engine=engine,
        retry_engine=RetryEngine(registry, policy=RetryPolicy(max_retries=0)),
    )

    task = Task(
        task_id=task_id,
        tenant_id=tenant_id,
        user_id="u1",
        message="ds-nexus-02 interrupt",
        context=TaskContext(capability=_CAPABILITY),
    )
    graph = _sequential_graph(task_id)
    capture = _InterruptCapture(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        hosting_root_execution_id=hosting_root_execution_id,
    )

    orchestration_backend = _GraphOrchestrationBackend(
        interrupt_executor,
        graph,
        capture,
    )
    orchestration_router = _OrchestrationOnlyRouter(task, orchestration_backend)
    work_port = child_execution_work_port(orchestration_router, ledger=_UNLIMITED_LEDGER)
    work_port_binding = ActiveExecutionWorkPortBinding.for_port(work_port)
    decision_store = _RecordingDecisionCheckpointPersistence()
    checkpoint_binding = ActiveDecisionCheckpointPersistenceBinding.for_persistence(
        decision_store,
    )

    delegate = _DecisionHostedOrchestrationDelegate(
        task=task,
        graph=graph,
        capture=capture,
        work_port_access=work_port_binding,
        checkpoint_access=checkpoint_binding,
        gate_state=gate_state,
    )
    runtime = ExecutionRuntime(
        delegate,
        decision_lifecycle_host=CanonicalDecisionLifecycleHost(),
        decision_checkpoint_persistence=decision_store,
        execution_work_port_binding=work_port_binding,
    )
    root_context = RootExecutionContext(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=hosting_root_execution_id,
        authority=ParentExecutionAuthority.unrestricted_root(),
        tenant_id=tenant_id,
    )

    await runtime.execute(_RootProbeRequest(message="orchestrate"), root_context)

    saved_decision = capture.decision_checkpoint
    assert saved_decision is not None
    assert capture.decision_key is not None
    assert capture.orchestration_child_execution_id is not None
    assert capture.authority_observed_during_orchestration

    orchestration_root_before = capture.orchestration_child_execution_id
    assert orchestration_root_before != hosting_root_execution_id

    counts_after_interrupt = engine.snapshot_counts()
    assert counts_after_interrupt == _NodeInvocationCounts(
        agent_a=1,
        agent_b=1,
        agent_c=1,
        agent_d=0,
    )

    runtime_before = resolve_task_runtime_checkpoint(task)
    assert runtime_before is not None
    execution_a_before = _execution_id_for_node(runtime_before, _NODE_A)
    execution_b_before = _execution_id_for_node(runtime_before, _NODE_B)
    execution_c_before = _execution_id_for_node(runtime_before, _NODE_C)
    assert execution_a_before is not None
    assert execution_b_before is not None
    assert execution_c_before is not None

    entry_c = runtime_before.execution_tree.entry_by_graph_node_id(_NODE_C)
    assert entry_c is not None
    assert entry_c.status is ExecutionCheckpointStatus.FAILED

    checkpoint = TaskCheckpoint(
        task_id=task_id,
        tenant_id=tenant_id,
        resume_token="rt_ds_nexus_02",
        task_state=TaskState.WAITING_FOR_HUMAN,
        runtime=build_runtime_checkpoint(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
            graph=graph,
        ),
    )
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ds_nexus_02_resume.db")
    store.save(checkpoint)
    loaded = store.get_by_token(task_id, tenant_id, "rt_ds_nexus_02")
    assert loaded is not None and loaded.runtime is not None

    task_resume = Task(
        task_id=task_id,
        tenant_id=tenant_id,
        user_id="u1",
        message="ds-nexus-02 resume",
        context=TaskContext(capability=_CAPABILITY),
    )

    gate_state.gate_open = True
    resume_executor = GraphExecutor(
        registry,
        engine=engine,
        retry_engine=RetryEngine(registry, policy=RetryPolicy(max_retries=0)),
    )
    loop = NexusLoop(
        registry,
        graph_executor=resume_executor,
        planner=_DeterministicSequentialPlanner(),
        classifier=_DeterministicClassifier(),
        retry_engine=RetryEngine(registry, policy=RetryPolicy(max_retries=0)),
    )
    runner = UnifiedTaskRunner(loop)
    await runner.run_task(task_resume, resume_checkpoint=loaded)

    counts_final = engine.snapshot_counts()
    assert counts_final.agent_a == 1
    assert counts_final.agent_b == 1
    assert counts_final.agent_c == 2
    assert counts_final.agent_d == 1

    runtime_after = resolve_task_runtime_checkpoint(task_resume)
    assert runtime_after is not None

    entry_a_after = runtime_after.execution_tree.entry_by_graph_node_id(_NODE_A)
    entry_b_after = runtime_after.execution_tree.entry_by_graph_node_id(_NODE_B)
    assert entry_a_after is not None
    assert entry_a_after.execution_id == execution_a_before
    assert entry_b_after is not None
    assert entry_b_after.execution_id == execution_b_before

    root_entries = [
        entry
        for entry in runtime_after.execution_tree.entries
        if entry.parent_execution_id is None
    ]
    assert len(root_entries) == 1
    resumed_root = root_entries[0].execution_id
    assert resumed_root != orchestration_root_before

    execution_c_after = _execution_id_for_node(runtime_after, _NODE_C)
    assert execution_c_after is not None
    assert execution_c_after != execution_c_before
    entry_c_after = runtime_after.execution_tree.entry_by_graph_node_id(_NODE_C)
    assert entry_c_after is not None
    assert entry_c_after.parent_execution_id == resumed_root
    assert entry_c_after.resumed_from_execution_id == execution_c_before

    execution_d = _execution_id_for_node(runtime_after, _NODE_D)
    assert execution_d is not None
    entry_d = runtime_after.execution_tree.entry_by_graph_node_id(_NODE_D)
    assert entry_d is not None
    assert entry_d.parent_execution_id == resumed_root
    assert entry_d.resumed_from_execution_id is None
    assert entry_d.status is ExecutionCheckpointStatus.COMPLETED

    assert any(authority_observed)

    restored_decision = load_decision_checkpoint(
        decision_store,
        key=capture.decision_key,
    )
    assert restored_decision is not None
    assert restored_decision == saved_decision
    assert restored_decision.lifecycle.stage == DecisionLifecycleStage.VERIFICATION
    assert (
        restored_decision.lifecycle.identity.decision_id
        == saved_decision.lifecycle.identity.decision_id
    )
    assert (
        restored_decision.lifecycle.identity.version
        == saved_decision.lifecycle.identity.version
    )
    assert (
        restored_decision.lifecycle.transition_index
        == saved_decision.lifecycle.transition_index
    )
    assert (
        restored_decision.lifecycle.identity.execution.execution_id
        == hosting_root_execution_id
    )
    assert (
        restored_decision.lifecycle.identity.execution.execution_id
        != resumed_root
    )

    _assert_checkpoint_separation(saved_decision, loaded.runtime)

    assert peek_active_execution_resume_plan() is None
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None
    assert peek_active_execution_authority() is None
    assert peek_active_execution_budget() is None


@pytest.mark.asyncio
async def test_malformed_physical_checkpoint_fails_without_mutating_decision_checkpoint() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    hosting_root_execution_id = mint_execution_id()
    tenant_id = "t1"

    registry = AgentRegistry()
    _register_resume_agents(registry)
    gate_state = _GateState(gate_open=False)
    authority_observed: list[bool] = []
    engine = _GatedCountingAgentEngine(
        registry,
        gate_state=gate_state,
        authority_observed=authority_observed,
    )
    interrupt_executor = GraphExecutor(
        registry,
        engine=engine,
        retry_engine=RetryEngine(registry, policy=RetryPolicy(max_retries=0)),
    )

    task = Task(
        task_id=task_id,
        tenant_id=tenant_id,
        user_id="u1",
        message="ds-nexus-02 malformed",
        context=TaskContext(capability=_CAPABILITY),
    )
    graph = _sequential_graph(task_id)
    capture = _InterruptCapture()
    orchestration_backend = _GraphOrchestrationBackend(
        interrupt_executor,
        graph,
        capture,
    )
    orchestration_router = _OrchestrationOnlyRouter(task, orchestration_backend)
    work_port = child_execution_work_port(orchestration_router, ledger=_UNLIMITED_LEDGER)
    work_port_binding = ActiveExecutionWorkPortBinding.for_port(work_port)
    decision_store = _RecordingDecisionCheckpointPersistence()
    checkpoint_binding = ActiveDecisionCheckpointPersistenceBinding.for_persistence(
        decision_store,
    )
    delegate = _DecisionHostedOrchestrationDelegate(
        task=task,
        graph=graph,
        capture=capture,
        work_port_access=work_port_binding,
        checkpoint_access=checkpoint_binding,
        gate_state=gate_state,
    )
    runtime = ExecutionRuntime(
        delegate,
        decision_lifecycle_host=CanonicalDecisionLifecycleHost(),
        decision_checkpoint_persistence=decision_store,
        execution_work_port_binding=work_port_binding,
    )
    root_context = RootExecutionContext(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=hosting_root_execution_id,
        authority=ParentExecutionAuthority.unrestricted_root(),
        tenant_id=tenant_id,
    )
    await runtime.execute(_RootProbeRequest(message="orchestrate"), root_context)

    saved_decision = capture.decision_checkpoint
    assert saved_decision is not None
    assert capture.decision_key is not None
    saves_before = decision_store.save_calls

    checkpoint = TaskCheckpoint(
        task_id=task_id,
        tenant_id=tenant_id,
        resume_token="rt_ds_nexus_02_bad",
        task_state=TaskState.WAITING_FOR_HUMAN,
        runtime=build_runtime_checkpoint(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
            graph=graph,
        ),
    )
    assert checkpoint.runtime is not None
    mismatched_attempt = mint_attempt_id()
    corrupt_runtime = checkpoint.runtime.model_copy(
        update={"attempt_id": mismatched_attempt},
    )
    corrupt_checkpoint = checkpoint.model_copy(update={"runtime": corrupt_runtime})

    task_resume = Task(
        task_id=task_id,
        tenant_id=tenant_id,
        user_id="u1",
        message="ds-nexus-02 malformed resume",
        context=TaskContext(capability=_CAPABILITY),
    )
    loop = NexusLoop(registry)
    runner = UnifiedTaskRunner(loop)

    with pytest.raises(ValueError, match="attempt_id mismatch"):
        await runner.run_task(task_resume, resume_checkpoint=corrupt_checkpoint)

    reloaded = load_decision_checkpoint(decision_store, key=capture.decision_key)
    assert reloaded is not None
    assert reloaded == saved_decision
    assert decision_store.save_calls == saves_before


def _assert_checkpoint_separation(
    decision_checkpoint: DecisionCheckpointState[_SemanticOutcomePayload],
    runtime_checkpoint: RuntimeCheckpoint,
) -> None:
    decision_fields = {field.name for field in dataclasses.fields(DecisionCheckpointState)}
    assert decision_fields == {"lifecycle", "finalization", "revision"}
    assert type(decision_checkpoint.lifecycle) is DecisionLifecycleState
    assert type(decision_checkpoint.finalization) is DecisionFinalizeGuardState
    assert type(decision_checkpoint) is DecisionCheckpointState

    forbidden_runtime_annotations = (
        DecisionCheckpointState,
        DecisionLifecycleState,
        DecisionFinalizeGuardState,
    )
    for field_info in RuntimeCheckpoint.model_fields.values():
        annotation = field_info.annotation
        for forbidden in forbidden_runtime_annotations:
            assert annotation is not forbidden

    runtime_checkpoint.validate_canonical()


def test_decision_contracts_have_no_nexus_visibility() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    contract_paths = sorted(repo_root.glob(_DECISION_CONTRACT_GLOB))
    assert contract_paths

    for path in contract_paths:
        source = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_DECISION_TOKENS:
            assert token not in source, f"{path} contains forbidden token: {token}"


def test_decision_facing_probe_has_no_nexus_imports() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "DecisionFacingOrchestrationProbe":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom) and child.module is not None:
                assert "intergrax.runtime.nexus" not in child.module
            if isinstance(child, ast.Import):
                for alias in child.names:
                    assert "intergrax.runtime.nexus" not in alias.name


def test_child_execution_runner_used_by_work_port() -> None:
    port_source = Path(
        "intergrax/runtime/execution/execution_work_port.py",
    ).read_text(encoding="utf-8")
    assert "ChildExecutionRunner" in port_source
    assert "child_execution_work_port" in port_source


def _class_source(tree: ast.Module, class_name: str) -> str:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.get_source_segment(
                Path(__file__).read_text(encoding="utf-8"),
                node,
            ) or ""
    raise AssertionError(f"class not found: {class_name}")


def _function_source(tree: ast.Module, function_name: str) -> str:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return ast.get_source_segment(
                Path(__file__).read_text(encoding="utf-8"),
                node,
            ) or ""
    raise AssertionError(f"function not found: {function_name}")


def test_main_recovery_proof_avoids_manual_resume_helpers() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    proof_source = _function_source(
        tree,
        "test_decision_orchestration_checkpoint_recovery_participation",
    )
    for token in _FORBIDDEN_MAIN_PROOF_RECOVERY_CALLS:
        assert token not in proof_source, (
            f"main DS-NEXUS-02 proof must not call {token!r}"
        )


def test_decision_facing_code_does_not_invoke_recovery() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for class_name in (
        "DecisionFacingOrchestrationProbe",
        "_DecisionHostedOrchestrationDelegate",
    ):
        class_source = _class_source(tree, class_name)
        for token in _FORBIDDEN_DECISION_RECOVERY_TOKENS:
            assert token not in class_source, (
                f"{class_name} must not reference recovery token {token!r}"
            )
