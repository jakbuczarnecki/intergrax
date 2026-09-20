# © Artur Czarnecki. All rights reserved.

"""HITL durable restart harness for OBS-UNIVERSAL-SPINE-E2E (testing only)."""

from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import dataclass
from pathlib import Path

from intergrax.agents.harness_reference_agent import HarnessReferenceAgent
from intergrax.applications._shared.diagnostic_read_wiring import (
    HostDiagnosticReadDependencies,
    build_diagnostic_read_service,
)
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType, HumanRequest
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore
from intergrax.contracts.execution_identity import RunId, mint_run_id
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    backing_execution_continuation_state_store,
)
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.task_contract import (
    HumanApprovalResolution,
    TaskExecutionOptions,
    TaskHumanInput,
    TaskLongRunningOptions,
)
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from intergrax.utils import attribute_access
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager
from testing_support.obs_universal_spine.diagnostic_execution_stack import (
    build_diagnostic_nexus_loop,
    build_obs_spine_unified_task_runner,
)

_HITL_CAPABILITY = "hitl.obs_spine"


class ObsSpineHitlAgent(HarnessReferenceAgent):
    """Requests human approval once, then completes when metadata approves."""

    step_run_count = 0

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id="hitl_obs_spine",
            name="OBS Spine HITL Agent",
            description="single human gate for universal spine qualification",
            capabilities=[_HITL_CAPABILITY],
            max_steps=2,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = attribute_access.optional(task_context, "capability", None)
        if capability in (None, _HITL_CAPABILITY):
            return CapabilityMatchResult(
                matched=True,
                agent_id="hitl_obs_spine",
                matched_capabilities=[_HITL_CAPABILITY],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="ok"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def get_steps(self) -> list[AgentStep]:
        return [AgentStep(step_id="review", step_name="review", step_index=0)]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        ObsSpineHitlAgent.step_run_count += 1
        return StepOutput(step_id=step.step_id, summary="pending review")

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step
        if output is not None and (
            ctx.metadata.get("human_approved")
            or (
                ctx.request
                and (
                    ctx.request.metadata.get("human_approved")
                    or ctx.request.metadata.get("human_decision") == "approve"
                )
            )
        ):
            return AgentDecision(type=AgentDecisionType.COMPLETE, reason="approved")
        if ctx.metadata.get("human_approved") or (
            ctx.request
            and (
                ctx.request.metadata.get("human_approved")
                or ctx.request.metadata.get("human_decision") == "approve"
            )
        ):
            return AgentDecision(type=AgentDecisionType.COMPLETE, reason="approved")
        if ctx.metadata.get("human_rejected") or (
            ctx.request
            and (
                ctx.request.metadata.get("human_rejected")
                or ctx.request.metadata.get("human_decision") == "reject"
            )
        ):
            return AgentDecision(
                type=AgentDecisionType.FAIL,
                reason="rejected by operator",
            )
        return AgentDecision(
            type=AgentDecisionType.REQUEST_HUMAN,
            reason="approval required",
            human_request=HumanRequest(
                request_id="hr_obs_spine",
                prompt="Approve OBS spine qualification?",
                options=["approve", "reject"],
            ),
        )


@dataclass(frozen=True, slots=True)
class HitlRestartRuntimeBundle:
    nexus_loop: NexusLoop
    runner: UnifiedTaskRunner
    checkpoint_store: SQLiteTaskCheckpointStore
    runtime_event_store: SQLiteRuntimeEventStore
    read_deps: HostDiagnosticReadDependencies
    causal_store: InMemoryCausalEvidencePersistence
    continuation_backing: ExecutionContinuationDurableBacking
    continuation_state_store: ExecutionContinuationStateStore


def build_hitl_restart_runtime(
    *,
    checkpoint_db: Path,
    runtime_events_db: Path,
    inject_violation: bool = False,
    continuation_backing: ExecutionContinuationDurableBacking | None = None,
    execution_continuation_state_store: ExecutionContinuationStateStore | None = None,
    document_store: object | None = None,
    problem_persistence: object | None = None,
    occurrence_persistence: object | None = None,
) -> HitlRestartRuntimeBundle:
    backing = continuation_backing or ExecutionContinuationDurableBacking()
    continuation_store = (
        execution_continuation_state_store
        or backing_execution_continuation_state_store(backing)
    )
    runtime_store = SQLiteRuntimeEventStore(db_path=runtime_events_db)
    checkpoint_store = SQLiteTaskCheckpointStore(db_path=checkpoint_db)
    loop, _, read_deps = build_diagnostic_nexus_loop(
        inject_violation=inject_violation,
        runtime_event_store=runtime_store,
        checkpoint_store=checkpoint_store,
        primary_agent=ObsSpineHitlAgent(),
        execution_continuation_state_store=continuation_store,
        document_store=document_store,
        problem_persistence=problem_persistence,
        occurrence_persistence=occurrence_persistence,
    )
    return HitlRestartRuntimeBundle(
        nexus_loop=loop,
        runner=build_obs_spine_unified_task_runner(loop),
        checkpoint_store=checkpoint_store,
        runtime_event_store=runtime_store,
        read_deps=read_deps,
        causal_store=InMemoryCausalEvidencePersistence(),
        continuation_backing=backing,
        continuation_state_store=continuation_store,
    )


def _pause_task(*, tenant_id: str = "tenant-obs-spine-hitl") -> Task:
    return Task(
        tenant_id=tenant_id,
        user_id="operator",
        message="obs spine hitl pause",
        context=TaskContext(capability=_HITL_CAPABILITY),
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True),
        ),
    )


def _resume_task(
    *,
    task_id: str,
    resume_token: str,
    human_approved: bool,
    human_rejected: bool = False,
    tenant_id: str = "tenant-obs-spine-hitl",
) -> Task:
    metadata: dict[str, object] = {"resume_token": resume_token}
    if human_approved:
        metadata["human_approved"] = True
    if human_rejected:
        metadata["human_rejected"] = True
    approver = local_development_approver_evidence(
        actor_id="obs-spine-operator",
        tenant_id=tenant_id,
    )
    verdict = None
    if human_approved:
        verdict = HumanResponseVerdict.APPROVE
    elif human_rejected:
        verdict = HumanResponseVerdict.REJECT
    return Task(
        task_id=task_id,
        tenant_id=tenant_id,
        user_id="operator",
        message="obs spine hitl resume",
        context=TaskContext(capability=_HITL_CAPABILITY),
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(
                enabled=True,
                resume_token=resume_token,
            ),
            human=TaskHumanInput(
                verdict=verdict,
                approver=approver,
            ),
        ),
        metadata=metadata,
    )


def _prepare_hitl_resume_task(
    resume: Task,
    *,
    loaded: TaskCheckpoint,
    run_id: RunId,
    human_approved: bool,
    human_rejected: bool,
) -> None:
    approver = local_development_approver_evidence(
        actor_id="obs-spine-operator",
        tenant_id=resume.tenant_id,
    )
    resume.options.human.approver = approver
    if not human_approved and not human_rejected:
        return
    verdict = HumanResponseVerdict.APPROVE if human_approved else HumanResponseVerdict.REJECT
    pause_record = resume.runtime.governance.pause_record
    pause_id = pause_record.pause_id if pause_record is not None else "hr_obs_spine_pause"
    human_request_id = (
        pause_record.human_request_id if pause_record is not None else "hr_obs_spine"
    )
    execution_id = None
    if loaded.runtime is not None and loaded.runtime.execution_tree.entries:
        execution_id = loaded.runtime.execution_tree.entries[0].execution_id
    resume.runtime.governance.hitl_resolution = HumanApprovalResolution(
        task_id=resume.task_id,
        pause_id=pause_id,
        human_request_id=human_request_id,
        verdict=verdict,
        approver=approver,
        resolved_at=datetime.now(UTC).isoformat(),
        run_id=str(run_id),
        attempt_id=loaded.runtime.attempt_id if loaded.runtime is not None else None,
        execution_id=execution_id,
    )


@dataclass(frozen=True, slots=True)
class HitlIdentitySnapshot:
    tenant_id: str
    task_id: str
    run_id: RunId
    attempt_id: str
    execution_id: str


async def run_hitl_pause_resume_after_runtime_rebuild(
    *,
    checkpoint_db: Path,
    runtime_events_db: Path,
    human_approved: bool,
    human_rejected: bool = False,
) -> tuple[HitlIdentitySnapshot, HitlIdentitySnapshot, TaskState, HitlRestartRuntimeBundle]:
    ObsSpineHitlAgent.step_run_count = 0
    run_id = mint_run_id()
    continuation_backing = ExecutionContinuationDurableBacking()
    runtime_a = build_hitl_restart_runtime(
        checkpoint_db=checkpoint_db,
        runtime_events_db=runtime_events_db,
        continuation_backing=continuation_backing,
    )
    tenant_id = "tenant-obs-spine-hitl"
    paused = await runtime_a.runner.run_task(_pause_task(), run_id=run_id)
    assert paused.state is TaskState.WAITING_FOR_HUMAN
    assert paused.summary.resume_token
    loaded = runtime_a.checkpoint_store.get_latest(paused.task_id, tenant_id)
    assert loaded is not None and loaded.runtime is not None
    before = HitlIdentitySnapshot(
        tenant_id=tenant_id,
        task_id=paused.task_id,
        run_id=run_id,
        attempt_id=str(loaded.runtime.attempt_id),
        execution_id=str(loaded.runtime.execution_tree.entries[0].execution_id),
    )

    del runtime_a

    runtime_b = build_hitl_restart_runtime(
        checkpoint_db=checkpoint_db,
        runtime_events_db=runtime_events_db,
        continuation_backing=continuation_backing,
    )
    resume_task = _resume_task(
        task_id=paused.task_id,
        resume_token=paused.summary.resume_token,
        human_approved=human_approved,
        human_rejected=human_rejected,
    )
    _prepare_hitl_resume_task(
        resume_task,
        loaded=loaded,
        run_id=run_id,
        human_approved=human_approved,
        human_rejected=human_rejected,
    )
    resumed = await runtime_b.runner.run_task(
        resume_task,
        run_id=run_id,
        attempt_id=loaded.runtime.attempt_id,
        resume_checkpoint=loaded,
    )
    latest = runtime_b.checkpoint_store.get_latest(paused.task_id, tenant_id)
    assert latest is not None and latest.runtime is not None
    after = HitlIdentitySnapshot(
        tenant_id=before.tenant_id,
        task_id=resumed.task_id,
        run_id=run_id,
        attempt_id=str(latest.runtime.attempt_id),
        execution_id=str(latest.runtime.execution_tree.entries[0].execution_id),
    )

    reconstructor = ExecutionReconstructor(
        runtime_events=runtime_b.runtime_event_store,
        causal_evidence=runtime_b.causal_store,
    )
    reconstruction = reconstructor.reconstruct_execution(
        tenant_id,
        resumed.task_id,
        run_id,
    )
    assert reconstruction.has_runtime_events
    read_service = build_diagnostic_read_service(runtime_b.read_deps)
    _ = read_service.list_problems(tenant_id=tenant_id)

    all_run_events = runtime_b.runtime_event_store.list_for_run(run_id, tenant_id=tenant_id)
    terminal_events = [
        event
        for event in all_run_events
        if event.event_type in {RuntimeEventType.TASK_COMPLETED, RuntimeEventType.TASK_FAILED}
        and event.task_id == resumed.task_id
    ]
    assert terminal_events, (
        f"expected terminal runtime event for task {resumed.task_id!r} run {run_id!r}, "
        f"got state={resumed.state!r}"
    )
    if human_approved and not human_rejected:
        assert terminal_events[-1].event_type is RuntimeEventType.TASK_COMPLETED
    if human_rejected:
        assert terminal_events[-1].event_type is RuntimeEventType.TASK_FAILED

    return before, after, resumed.state, runtime_b
