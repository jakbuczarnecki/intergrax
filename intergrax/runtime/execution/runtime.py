# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical root execution lifecycle (UE-10R1)."""

from __future__ import annotations

from contextvars import Token
from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.contracts.admitted_root_governance_identity import AdmittedRootGovernanceIdentity
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_capacity_admission import (
    ExecutionCapacityAdmissionPort,
    ExecutionCapacityAdmissionRequest,
    ExecutionCapacityPermit,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)
from intergrax.contracts.execution_failure_evidence import (
    ExecutionFailureEvidenceRecorder,
)
from intergrax.contracts.execution_lineage import ExecutionLineagePersistence
from intergrax.runtime.execution.failure_evidence.active_context import (
    ActiveExecutionEvidenceContext,
    bind_active_execution_evidence_context,
    reset_active_execution_evidence_context,
)
from intergrax.runtime.execution.active_decision_checkpoint_persistence import (
    bind_active_decision_checkpoint_persistence,
    reset_active_decision_checkpoint_persistence,
)
from intergrax.runtime.execution.active_decision_finalization_persistence import (
    bind_active_decision_finalization_persistence,
    reset_active_decision_finalization_persistence,
)
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    bind_active_decision_lifecycle_host,
    reset_active_decision_lifecycle_host,
)
from intergrax.runtime.execution.active_execution_work_port import (
    ActiveExecutionWorkPortBinding,
    bind_active_execution_work_port,
    reset_active_execution_work_port,
)
from intergrax.runtime.execution.decision_checkpoint_persistence import (
    DecisionCheckpointPersistence,
)
from intergrax.runtime.execution.decision_finalization_persistence import (
    DecisionFinalizationPersistence,
)
from intergrax.contracts.execution_deadline.admission import (
    ExecutionCancellationView,
    ExecutionProtectedWorkAdmissionPort,
)
from intergrax.contracts.execution_deadline.persistence_port import (
    ExecutionDeadlinePersistenceError,
)
from intergrax.runtime.execution.deadline_scope import (
    bind_active_execution_deadline_scope,
    reset_active_execution_deadline_scope,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.persistence import RunBudgetPersistence
from intergrax.runtime.execution.deadline_authority import ExecutionDeadlineAuthorityResolver
from intergrax.runtime.execution.protected_work_admission import (
    CanonicalHardProtectedWorkAdmission,
    ComposedProtectedWorkAdmission,
    StaticCancellationView,
)
from intergrax.contracts.execution_continuation_state_store import (
    ExecutionContinuationStateStore,
)
from intergrax.runtime.execution.active_execution_continuation_store import (
    bind_active_execution_continuation_state_store,
    reset_active_execution_continuation_state_store,
)
from intergrax.runtime.execution.boundary import (
    ExecutionAdmissionHook,
    ExecutionBoundary,
    ExecutionDelegate,
    ExecutionIdentityBinding,
)
from intergrax.runtime.execution.budget.ledger import (
    ExecutionBudgetLedgerFactory,
    RunBudgetExecutionBudgetLedgerFactory,
)
from intergrax.runtime.execution.decision_lifecycle_host import DecisionLifecycleHost
from intergrax.runtime.execution.lineage.root_activation import (
    activate_root_execution_lineage,
    build_root_lineage_admission_hook,
    deactivate_root_execution_lineage,
    merge_lineage_root_admission_hooks,
    validate_root_lineage_inputs,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.execution.identity_authority import (
    RootTaskIdentity,
    mint_root_execution_identity,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.resume_planner import (
    execution_identity_from_checkpoint,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")
CheckpointPayloadT = TypeVar("CheckpointPayloadT")
WorkInputT = TypeVar("WorkInputT")
WorkOutputT = TypeVar("WorkOutputT")
WorkResultT = TypeVar("WorkResultT")


@dataclass(frozen=True, slots=True)
class RootExecutionContext:
    """Typed root lifecycle inputs bound before strategy routing."""

    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    authority: ParentExecutionAuthority
    governance_identity: AdmittedRootGovernanceIdentity | None = None
    tenant_id: str | None = None
    workspace_id: str | None = None
    principal_id: str | None = None
    task_id: TaskId | None = None
    segment_predecessor_root_execution_id: ExecutionId | None = None


@dataclass(frozen=True, slots=True)
class RootExecutionOptions:
    """Optional inputs for resolving a root execution context."""

    authority: ParentExecutionAuthority
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None
    execution_id: ExecutionId | None = None
    governance_identity: AdmittedRootGovernanceIdentity | None = None
    tenant_id: str | None = None
    task_id: TaskId | None = None
    segment_predecessor_root_execution_id: ExecutionId | None = None
    resume_checkpoint: TaskCheckpoint | None = None


def resolve_root_task_identity(
    *,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    execution_id: ExecutionId | None = None,
    resume_checkpoint: TaskCheckpoint | None = None,
) -> RootTaskIdentity:
    """Resolve root identity for lifecycle admission (checkpoint interpretation lives here)."""
    if resume_checkpoint is not None and resume_checkpoint.runtime is not None:
        checkpoint_run_id, checkpoint_attempt_id = execution_identity_from_checkpoint(
            resume_checkpoint,
        )
        checkpoint_tree = resume_checkpoint.runtime.execution_tree
        checkpoint_root_execution_id = next(
            entry.execution_id
            for entry in checkpoint_tree.entries
            if entry.parent_execution_id is None
        )
        if run_id is not None and run_id != checkpoint_run_id:
            raise ValueError(
                "explicit run_id conflicts with resume checkpoint identity: "
                f"{run_id!r} != {checkpoint_run_id!r}"
            )
        if attempt_id is not None and attempt_id != checkpoint_attempt_id:
            raise ValueError(
                "explicit attempt_id conflicts with resume checkpoint identity: "
                f"{attempt_id!r} != {checkpoint_attempt_id!r}"
            )
        if execution_id is not None and execution_id != checkpoint_root_execution_id:
            raise ValueError(
                "explicit execution_id conflicts with resume checkpoint identity: "
                f"{execution_id!r} != {checkpoint_root_execution_id!r}"
            )
        return mint_root_execution_identity(
            run_id=checkpoint_run_id,
            attempt_id=checkpoint_attempt_id,
            execution_id=execution_id,
        )
    return mint_root_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )


def resolve_root_execution_context(
    options: RootExecutionOptions,
) -> RootExecutionContext:
    """Resolve typed root context; mints RunId and AttemptId when omitted."""
    identity = resolve_root_task_identity(
        run_id=options.run_id,
        attempt_id=options.attempt_id,
        execution_id=options.execution_id,
        resume_checkpoint=options.resume_checkpoint,
    )
    governance_identity = options.governance_identity
    tenant_id = options.tenant_id
    if governance_identity is not None:
        tenant_id = governance_identity.tenant_id
    workspace_id = (
        governance_identity.workspace_id if governance_identity is not None else None
    )
    principal_id = (
        governance_identity.principal_id if governance_identity is not None else None
    )
    return RootExecutionContext(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
        execution_id=identity.execution_id,
        authority=options.authority,
        governance_identity=governance_identity,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        principal_id=principal_id,
        task_id=options.task_id,
        segment_predecessor_root_execution_id=options.segment_predecessor_root_execution_id,
    )


class ExecutionRuntime(Generic[RequestT, ResultT]):
    """
    Canonical root execution lifecycle owner (UE-10R1).

    Resolves root identity, binds authority and budget, routes through
    :class:`ExecutionBoundary` and :class:`StrategyExecutionRouter`.

    Continuation capability is **disabled** when ``continuation_state_store`` is
    ``None``; inject a :class:`~intergrax.contracts.execution_continuation_state_store.ExecutionContinuationStateStore`
    implementation to enable mandatory four-ID progress enforcement.
    """

    __slots__ = (
        "_delegate",
        "_ledger_factory",
        "_run_budget",
        "_admission_hooks",
        "_decision_lifecycle_host",
        "_decision_checkpoint_persistence",
        "_decision_finalization_persistence",
        "_execution_work_port_binding",
        "_execution_lineage_persistence",
        "_execution_capacity_admission",
        "_failure_evidence_recorder",
        "_continuation_state_store",
        "_deadline_authority_resolver",
        "_run_budget_persistence",
        "_protected_work_admission_contributors",
        "_root_cancellation_view",
    )

    def __init__(
        self,
        delegate: ExecutionDelegate[RequestT, ResultT],
        *,
        ledger_factory: ExecutionBudgetLedgerFactory | None = None,
        run_budget: RunBudget | None = None,
        admission_hooks: tuple[ExecutionAdmissionHook[RequestT], ...] = (),
        execution_lineage_persistence: ExecutionLineagePersistence | None = None,
        decision_lifecycle_host: DecisionLifecycleHost | None = None,
        decision_checkpoint_persistence: (
            DecisionCheckpointPersistence[CheckpointPayloadT] | None
        ) = None,
        decision_finalization_persistence: (
            DecisionFinalizationPersistence[CheckpointPayloadT] | None
        ) = None,
        execution_work_port_binding: (
            ActiveExecutionWorkPortBinding[WorkInputT, WorkOutputT, WorkResultT] | None
        ) = None,
        execution_capacity_admission: ExecutionCapacityAdmissionPort | None = None,
        failure_evidence_recorder: ExecutionFailureEvidenceRecorder | None = None,
        continuation_state_store: ExecutionContinuationStateStore | None = None,
        deadline_authority_resolver: ExecutionDeadlineAuthorityResolver | None = None,
        run_budget_persistence: RunBudgetPersistence | None = None,
        protected_work_admission_contributors: tuple[
            ExecutionProtectedWorkAdmissionPort,
            ...,
        ] = (),
        root_cancellation_view: ExecutionCancellationView | None = None,
    ) -> None:
        self._delegate = delegate
        self._ledger_factory = (
            ledger_factory
            if ledger_factory is not None
            else RunBudgetExecutionBudgetLedgerFactory(default_run_budget=run_budget)
        )
        self._run_budget = run_budget
        self._admission_hooks = admission_hooks
        self._decision_lifecycle_host = decision_lifecycle_host
        self._decision_checkpoint_persistence = decision_checkpoint_persistence
        self._decision_finalization_persistence = decision_finalization_persistence
        self._execution_work_port_binding = execution_work_port_binding
        self._execution_lineage_persistence = execution_lineage_persistence
        self._execution_capacity_admission = execution_capacity_admission
        self._failure_evidence_recorder = failure_evidence_recorder
        self._continuation_state_store = continuation_state_store
        self._deadline_authority_resolver = deadline_authority_resolver
        self._run_budget_persistence = run_budget_persistence
        self._protected_work_admission_contributors = protected_work_admission_contributors
        self._root_cancellation_view = root_cancellation_view
        if run_budget_persistence is not None and deadline_authority_resolver is None:
            raise ExecutionDeadlinePersistenceError(
                "durable run budget persistence requires deadline authority resolver",
            )
        if (
            run_budget_persistence is not None
            and run_budget is not None
            and run_budget.max_wall_time_seconds is not None
            and deadline_authority_resolver is None
        ):
            raise ExecutionDeadlinePersistenceError(
                "durable execution with wall-time budget requires deadline authority resolver",
            )

    async def execute(
        self,
        request: RequestT,
        root_context: RootExecutionContext,
        *,
        held_root_capacity_permit: ExecutionCapacityPermit | None = None,
    ) -> ResultT:
        capacity_permit: ExecutionCapacityPermit | None = held_root_capacity_permit
        acquired_capacity = False
        if capacity_permit is None and self._execution_capacity_admission is not None:
            capacity_permit = await self._execution_capacity_admission.acquire(
                ExecutionCapacityAdmissionRequest(
                    tenant_id=root_context.tenant_id,
                    task_id=root_context.task_id,
                    run_id=root_context.run_id,
                    attempt_id=root_context.attempt_id,
                    execution_id=root_context.execution_id,
                ),
            )
            acquired_capacity = True
        execution_id = root_context.execution_id
        try:
            return await self._execute_with_capacity(
                request,
                root_context,
                execution_id=execution_id,
            )
        finally:
            if capacity_permit is not None and (
                acquired_capacity or held_root_capacity_permit is not None
            ):
                await capacity_permit.release()

    async def _execute_with_capacity(
        self,
        request: RequestT,
        root_context: RootExecutionContext,
        *,
        execution_id: ExecutionId,
    ) -> ResultT:
        ledger = self._ledger_factory.create_ledger(
            self._run_budget,
            tenant_id=root_context.tenant_id,
            run_id=root_context.run_id,
            attempt_id=root_context.attempt_id,
        )
        binding = ExecutionIdentityBinding(
            run_id=root_context.run_id,
            attempt_id=root_context.attempt_id,
            execution_id=execution_id,
            task_id=root_context.task_id,
        )
        admission_hooks = self._admission_hooks
        lineage_token = None
        degradation_token = None
        if self._execution_lineage_persistence is not None:
            lineage_scope = validate_root_lineage_inputs(
                tenant_id=root_context.tenant_id,
                task_id=root_context.task_id,
                run_id=root_context.run_id,
                attempt_id=root_context.attempt_id,
                execution_id=execution_id,
            )
            _, lineage_token, degradation_token = activate_root_execution_lineage(
                persistence=self._execution_lineage_persistence,
                scope=lineage_scope,
                root_execution_id=execution_id,
                predecessor_root_execution_id=root_context.segment_predecessor_root_execution_id,
            )
            lineage_hook = build_root_lineage_admission_hook(
                persistence=self._execution_lineage_persistence,
                scope=lineage_scope,
                segment_root_execution_id=execution_id,
                execution_id=execution_id,
            )
            admission_hooks = merge_lineage_root_admission_hooks(
                lineage_hook,
                self._admission_hooks,
            )
        boundary = ExecutionBoundary[RequestT, ResultT](
            self._delegate,
            admission_hooks=admission_hooks,
            identity=binding,
            authority=root_context.authority,
            continuation_state_store=self._continuation_state_store,
        )
        deadline_resolution = None
        if self._run_budget_persistence is not None:
            if self._deadline_authority_resolver is None:
                raise ExecutionDeadlinePersistenceError(
                    "durable run execution requires deadline authority resolver",
                )
            if root_context.tenant_id is None:
                raise ValueError(
                    "durable run execution requires tenant_id on root execution context",
                )
        if (
            self._deadline_authority_resolver is not None
            and root_context.tenant_id is not None
        ):
            existing_run_materialized = False
            if self._run_budget_persistence is not None:
                existing_run_materialized = (
                    self._run_budget_persistence.load_snapshot(
                        tenant_id=root_context.tenant_id,
                        run_id=root_context.run_id,
                    )
                    is not None
                )
            deadline_resolution = self._deadline_authority_resolver.resolve_for_root(
                tenant_id=root_context.tenant_id,
                run_id=root_context.run_id,
                run_budget=self._run_budget,
                existing_run_materialized=existing_run_materialized,
            )
        if (
            self._run_budget_persistence is not None
            and self._run_budget is not None
            and self._run_budget.max_wall_time_seconds is not None
            and deadline_resolution is None
        ):
            raise ExecutionDeadlinePersistenceError(
                "durable execution could not resolve deadline authority for wall-time budget",
            )
        budget_token = bind_root_execution_budget(
            execution_id=execution_id,
            ledger=ledger,
            run_budget=self._run_budget,
            deadline_projection=(
                deadline_resolution.projection if deadline_resolution is not None else None
            ),
        )
        deadline_scope_tokens: tuple[Token, Token] | None = None
        if deadline_resolution is not None:
            cancellation_view = self._root_cancellation_view
            if cancellation_view is None:
                cancellation_view = StaticCancellationView(cancelled=False)
            monotonic_clock = self._deadline_authority_resolver.monotonic_clock
            canonical_admission = CanonicalHardProtectedWorkAdmission(
                projection=deadline_resolution.projection,
                cancellation_view=cancellation_view,
                monotonic_clock=monotonic_clock,
            )
            admission = ComposedProtectedWorkAdmission(
                canonical=canonical_admission,
                contributors=self._protected_work_admission_contributors,
            )
            deadline_scope_tokens = bind_active_execution_deadline_scope(
                projection=deadline_resolution.projection,
                admission=admission,
                monotonic_clock=monotonic_clock,
            )
        host_token = None
        persistence_token = None
        finalization_token = None
        work_port_token = None
        evidence_token = None
        governance_identity_token = None
        admitted = root_context.governance_identity
        if admitted is not None:
            governance_identity_token = bind_active_execution_governance_identity(
                ActiveExecutionGovernanceIdentity(
                    tenant_id=admitted.tenant_id,
                    workspace_id=admitted.workspace_id,
                    principal_id=admitted.principal_id,
                ),
            )
        if self._failure_evidence_recorder is not None:
            if root_context.tenant_id is None or root_context.task_id is None:
                raise ValueError(
                    "failure evidence recorder requires tenant_id and task_id "
                    "on root execution context",
                )
            evidence_token = bind_active_execution_evidence_context(
                ActiveExecutionEvidenceContext(
                    tenant_id=root_context.tenant_id,
                    task_id=root_context.task_id,
                    run_id=root_context.run_id,
                    attempt_id=root_context.attempt_id,
                    recorder=self._failure_evidence_recorder,
                ),
            )
        continuation_token: Token[ExecutionContinuationStateStore | None] | None = None
        if self._continuation_state_store is not None:
            continuation_token = bind_active_execution_continuation_state_store(
                self._continuation_state_store,
            )
        try:
            if self._decision_lifecycle_host is not None:
                host_token = bind_active_decision_lifecycle_host(
                    self._decision_lifecycle_host,
                )
            if self._decision_checkpoint_persistence is not None:
                persistence_token = bind_active_decision_checkpoint_persistence(
                    self._decision_checkpoint_persistence,
                )
            if self._decision_finalization_persistence is not None:
                finalization_token = bind_active_decision_finalization_persistence(
                    self._decision_finalization_persistence,
                )
            if self._execution_work_port_binding is not None:
                work_port_token = bind_active_execution_work_port(
                    self._execution_work_port_binding,
                )
            return await boundary.execute(request)
        finally:
            if continuation_token is not None:
                reset_active_execution_continuation_state_store(continuation_token)
            if evidence_token is not None:
                reset_active_execution_evidence_context(evidence_token)
            if governance_identity_token is not None:
                reset_active_execution_governance_identity(governance_identity_token)
            if lineage_token is not None and degradation_token is not None:
                deactivate_root_execution_lineage(lineage_token, degradation_token)
            if work_port_token is not None:
                reset_active_execution_work_port(work_port_token)
            if persistence_token is not None:
                reset_active_decision_checkpoint_persistence(persistence_token)
            if finalization_token is not None:
                reset_active_decision_finalization_persistence(finalization_token)
            if host_token is not None:
                reset_active_decision_lifecycle_host(host_token)
            if deadline_scope_tokens is not None:
                reset_active_execution_deadline_scope(*deadline_scope_tokens)
            reset_active_execution_budget(budget_token)
