# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical checkpoint resume validation (NPSC-5E/R2)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, ValidationError

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineagePersistence,
    build_execution_lineage_attempt_scope,
)
from intergrax.contracts.execution_terminal import (
    ExecutionTerminalError,
    ExecutionTerminalOutcome,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.cancellation.resume_admission import CheckpointNotResumableError
from intergrax.runtime.execution.execution_terminal.service import ExecutionTerminalService
from intergrax.runtime.long_running.execution_tree_checkpoint import ExecutionTreeSnapshot
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.runtime_checkpoint import (
    CANONICAL_RUNTIME_CHECKPOINT_SCHEMA_VERSION,
    RuntimeCheckpoint,
)
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_state import TaskState

CANONICAL_TASK_CHECKPOINT_SCHEMA_VERSION = "task_checkpoint.v1"

_RESUMABLE_CHECKPOINT_STATES = frozenset(
    {
        TaskState.WAITING_FOR_HUMAN,
        TaskState.WAITING_FOR_RESOURCES,
        TaskState.NEEDS_MORE_INFORMATION,
    }
)


class CheckpointResumeEligibility(StrEnum):
    ALLOW_RESUME = "ALLOW_RESUME"
    REJECT_STALE = "REJECT_STALE"
    REJECT_IDENTITY = "REJECT_IDENTITY"
    REJECT_TERMINAL = "REJECT_TERMINAL"
    REJECT_CANCELLED = "REJECT_CANCELLED"
    REJECT_LINEAGE = "REJECT_LINEAGE"
    REJECT_SCHEMA = "REJECT_SCHEMA"
    REJECT_MALFORMED = "REJECT_MALFORMED"
    REJECT_STATE = "REJECT_STATE"
    REJECT_GOVERNANCE = "REJECT_GOVERNANCE"
    REJECT_AUTHORITY = "REJECT_AUTHORITY"
    REJECT_TENANT = "REJECT_TENANT"


class CheckpointResumeValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    eligibility: CheckpointResumeEligibility
    reason: str = ""


class CheckpointResumeValidationError(RuntimeError):
    """Fail-closed resume denial from canonical checkpoint validation."""

    def __init__(self, result: CheckpointResumeValidationResult) -> None:
        self.result = result
        super().__init__(result.reason or result.eligibility.value)


def root_execution_id_from_tree(tree: ExecutionTreeSnapshot) -> ExecutionId:
    for entry in tree.entries:
        if entry.parent_execution_id is None:
            return entry.execution_id
    raise ValueError("execution tree missing root entry")


def validate_runtime_checkpoint_schema(
    runtime: RuntimeCheckpoint,
) -> CheckpointResumeValidationResult:
    if runtime.schema_version != CANONICAL_RUNTIME_CHECKPOINT_SCHEMA_VERSION:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_SCHEMA,
            reason=(
                "unsupported runtime checkpoint schema_version: "
                f"{runtime.schema_version!r}"
            ),
        )
    return CheckpointResumeValidationResult(
        eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
    )


def validate_task_checkpoint_schema(
    checkpoint: TaskCheckpoint,
) -> CheckpointResumeValidationResult:
    if checkpoint.schema_version != CANONICAL_TASK_CHECKPOINT_SCHEMA_VERSION:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_SCHEMA,
            reason=(
                "unsupported task checkpoint schema_version: "
                f"{checkpoint.schema_version!r}"
            ),
        )
    return CheckpointResumeValidationResult(
        eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
    )


def validate_checkpoint_identity_binding(
    checkpoint: TaskCheckpoint,
    *,
    target_task_id: TaskId | str,
    target_tenant_id: str,
    target_run_id: RunId | str | None = None,
    target_attempt_id: AttemptId | str | None = None,
    target_root_execution_id: ExecutionId | str | None = None,
) -> CheckpointResumeValidationResult:
    resolved_task_id = validate_task_id(target_task_id)
    if checkpoint.task_id != resolved_task_id:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_IDENTITY,
            reason=(
                f"checkpoint task_id mismatch: {checkpoint.task_id!r} != {resolved_task_id!r}"
            ),
        )
    if checkpoint.tenant_id != target_tenant_id:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_TENANT,
            reason=(
                f"checkpoint tenant mismatch: {checkpoint.tenant_id!r} != {target_tenant_id!r}"
            ),
        )
    runtime = checkpoint.runtime
    if runtime is None:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_MALFORMED,
            reason="checkpoint missing canonical runtime identity",
        )
    schema_result = validate_runtime_checkpoint_schema(runtime)
    if schema_result.eligibility is not CheckpointResumeEligibility.ALLOW_RESUME:
        return schema_result
    try:
        runtime.validate_canonical()
    except (ValueError, ValidationError) as exc:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_MALFORMED,
            reason=str(exc),
        )
    if runtime.execution_tree.task_id != resolved_task_id:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_IDENTITY,
            reason="execution tree task_id mismatch with checkpoint task",
        )
    if target_run_id is not None:
        resolved_run_id = validate_run_id(target_run_id)
        if runtime.run_id != resolved_run_id:
            return CheckpointResumeValidationResult(
                eligibility=CheckpointResumeEligibility.REJECT_IDENTITY,
                reason=(
                    f"checkpoint run_id mismatch: {runtime.run_id!r} != {resolved_run_id!r}"
                ),
            )
        if runtime.execution_tree.run_id != resolved_run_id:
            return CheckpointResumeValidationResult(
                eligibility=CheckpointResumeEligibility.REJECT_IDENTITY,
                reason="execution tree run_id mismatch with checkpoint run_id",
            )
    if target_attempt_id is not None:
        resolved_attempt_id = validate_attempt_id(target_attempt_id)
        if runtime.attempt_id != resolved_attempt_id:
            return CheckpointResumeValidationResult(
                eligibility=CheckpointResumeEligibility.REJECT_IDENTITY,
                reason=(
                    "checkpoint attempt_id mismatch: "
                    f"{runtime.attempt_id!r} != {resolved_attempt_id!r}"
                ),
            )
        if runtime.execution_tree.attempt_id != resolved_attempt_id:
            return CheckpointResumeValidationResult(
                eligibility=CheckpointResumeEligibility.REJECT_IDENTITY,
                reason="execution tree attempt_id mismatch with checkpoint attempt_id",
            )
    if target_root_execution_id is not None:
        from intergrax.contracts.execution_identity import validate_execution_id

        resolved_root = validate_execution_id(target_root_execution_id)
        checkpoint_root = root_execution_id_from_tree(runtime.execution_tree)
        if checkpoint_root != resolved_root:
            return CheckpointResumeValidationResult(
                eligibility=CheckpointResumeEligibility.REJECT_IDENTITY,
                reason=(
                    "checkpoint root execution mismatch: "
                    f"{checkpoint_root!r} != {resolved_root!r}"
                ),
            )
    return CheckpointResumeValidationResult(
        eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
    )


def _checkpoint_logical_revision(checkpoint: TaskCheckpoint) -> int | None:
    return checkpoint.revision


def _checkpoint_is_superseded_by(
    checkpoint: TaskCheckpoint,
    latest_checkpoint: TaskCheckpoint,
) -> bool:
    if checkpoint.checkpoint_id == latest_checkpoint.checkpoint_id:
        return False
    checkpoint_revision = _checkpoint_logical_revision(checkpoint)
    latest_revision = _checkpoint_logical_revision(latest_checkpoint)
    if checkpoint_revision is not None and latest_revision is not None:
        return checkpoint_revision < latest_revision
    if latest_revision is not None and checkpoint_revision is None:
        return True
    if checkpoint_revision is not None and latest_revision is None:
        return False
    return checkpoint.checkpoint_id != latest_checkpoint.checkpoint_id


def validate_checkpoint_not_stale(
    checkpoint: TaskCheckpoint,
    latest_checkpoint: TaskCheckpoint | None,
) -> CheckpointResumeValidationResult:
    if latest_checkpoint is None:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
        )
    if not _checkpoint_is_superseded_by(checkpoint, latest_checkpoint):
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
        )
    return CheckpointResumeValidationResult(
        eligibility=CheckpointResumeEligibility.REJECT_STALE,
        reason=(
            "checkpoint superseded by newer durable checkpoint "
            f"{latest_checkpoint.checkpoint_id!r}"
        ),
    )


def validate_checkpoint_lineage_cross_reference(
    checkpoint: TaskCheckpoint,
    persistence: ExecutionLineagePersistence | None,
    *,
    require_durable_lineage: bool = False,
) -> CheckpointResumeValidationResult:
    runtime = checkpoint.runtime
    if runtime is None:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_MALFORMED,
            reason="checkpoint missing canonical runtime identity",
        )
    if persistence is None:
        if require_durable_lineage:
            return CheckpointResumeValidationResult(
                eligibility=CheckpointResumeEligibility.REJECT_LINEAGE,
                reason="durable lineage required but persistence unavailable",
            )
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
        )
    scope = build_execution_lineage_attempt_scope(
        tenant_id=checkpoint.tenant_id,
        task_id=runtime.execution_tree.task_id,
        run_id=runtime.run_id,
        attempt_id=runtime.attempt_id,
    )
    attempt_state = persistence.read_attempt_lineage_state(scope)
    if attempt_state is None:
        if require_durable_lineage and persistence.is_durable:
            return CheckpointResumeValidationResult(
                eligibility=CheckpointResumeEligibility.REJECT_LINEAGE,
                reason="required durable lineage state missing for checkpoint attempt",
            )
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
        )
    if attempt_state.degraded:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_LINEAGE,
            reason="checkpoint attempt lineage is degraded; resume blocked",
        )
    seal = persistence.read_seal(scope)
    if seal is not None:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_TERMINAL,
            reason=(
                "checkpoint attempt lineage sealed with "
                f"{seal.closure_kind.value!r}"
            ),
        )
    checkpoint_root = root_execution_id_from_tree(runtime.execution_tree)
    if (
        attempt_state.active_segment_root_execution_id is not None
        and attempt_state.active_segment_root_execution_id != checkpoint_root
    ):
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_LINEAGE,
            reason="checkpoint root execution does not match durable lineage segment",
        )
    return CheckpointResumeValidationResult(
        eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
    )


def _authority_does_not_exceed_current(
    checkpoint_authority: ParentExecutionAuthority,
    current_authority: ParentExecutionAuthority,
) -> bool:
    if checkpoint_authority.unrestricted and not current_authority.unrestricted:
        return False
    if current_authority.unrestricted:
        return True
    if checkpoint_authority.unrestricted:
        return True
    checkpoint_scopes = set(checkpoint_authority.permission_scopes)
    current_scopes = set(current_authority.permission_scopes)
    return checkpoint_scopes <= current_scopes


def _parse_checkpoint_historical_authority(
    checkpoint: TaskCheckpoint,
) -> CheckpointResumeValidationResult | ParentExecutionAuthority:
    try:
        snapshot_task = Task.model_validate(checkpoint.task_snapshot)
    except ValidationError as exc:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_MALFORMED,
            reason=str(exc),
        )
    if snapshot_task.execution_authority is None:
        return ParentExecutionAuthority.unknown()
    return snapshot_task.execution_authority


def narrow_resume_execution_authority(
    current_authority: ParentExecutionAuthority,
    historical_authority: ParentExecutionAuthority,
) -> ParentExecutionAuthority:
    """Monotonic narrowing: authoritative current ∩ historical checkpoint bound."""
    if historical_authority.is_unknown:
        return current_authority
    if current_authority.is_unknown:
        return ParentExecutionAuthority.unknown()
    if historical_authority.unrestricted and current_authority.unrestricted:
        return ParentExecutionAuthority.unrestricted_root()
    if historical_authority.unrestricted:
        return current_authority
    if current_authority.unrestricted:
        return historical_authority
    common = tuple(
        sorted(
            set(current_authority.permission_scopes)
            & set(historical_authority.permission_scopes),
        ),
    )
    return ParentExecutionAuthority.scoped(common)


def resolve_resume_execution_authority(
    checkpoint: TaskCheckpoint,
    current_task: Task | None,
) -> ParentExecutionAuthority | None:
    """Narrow authoritative current authority by historical checkpoint constraint."""
    if current_task is None:
        return None
    current_authority = current_task.execution_authority
    if current_authority is None:
        return None
    historical_result = _parse_checkpoint_historical_authority(checkpoint)
    if isinstance(historical_result, CheckpointResumeValidationResult):
        raise CheckpointResumeValidationError(historical_result)
    return narrow_resume_execution_authority(current_authority, historical_result)


def validate_checkpoint_resume_authority(
    checkpoint: TaskCheckpoint,
    current_task: Task | None,
) -> CheckpointResumeValidationResult:
    historical_result = _parse_checkpoint_historical_authority(checkpoint)
    if isinstance(historical_result, CheckpointResumeValidationResult):
        return historical_result
    historical_authority = historical_result
    if current_task is None:
        if not historical_authority.is_unknown:
            return CheckpointResumeValidationResult(
                eligibility=CheckpointResumeEligibility.REJECT_AUTHORITY,
                reason="authoritative current task authority required for resume",
            )
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
        )
    current_authority = current_task.execution_authority
    if current_authority is None:
        if not historical_authority.is_unknown:
            return CheckpointResumeValidationResult(
                eligibility=CheckpointResumeEligibility.REJECT_AUTHORITY,
                reason="authoritative current authority required; checkpoint cannot grant authority",
            )
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
        )
    narrowed = narrow_resume_execution_authority(current_authority, historical_authority)
    if not _authority_does_not_exceed_current(narrowed, historical_authority):
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_AUTHORITY,
            reason="resume authority exceeds historical checkpoint bound",
        )
    if not _authority_does_not_exceed_current(narrowed, current_authority):
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_AUTHORITY,
            reason="resume authority exceeds authoritative current bound",
        )
    return CheckpointResumeValidationResult(
        eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
    )


def validate_checkpoint_authority_expansion(
    checkpoint: TaskCheckpoint,
    proposed_authority: ParentExecutionAuthority,
) -> CheckpointResumeValidationResult:
    try:
        snapshot_task = Task.model_validate(checkpoint.task_snapshot)
    except ValidationError as exc:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_MALFORMED,
            reason=str(exc),
        )
    checkpoint_authority = snapshot_task.execution_authority or ParentExecutionAuthority.unknown()
    if not _authority_does_not_exceed_current(checkpoint_authority, proposed_authority):
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_AUTHORITY,
            reason="checkpoint authority expansion via resume is blocked",
        )
    return CheckpointResumeValidationResult(
        eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
    )


def validate_checkpoint_governance_freshness(
    policy_decision: PolicyDecision | None,
) -> CheckpointResumeValidationResult:
    if policy_decision is None:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
        )
    if policy_decision.action is PolicyAction.DENY:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_GOVERNANCE,
            reason=policy_decision.reason or "current governance denies resume",
        )
    return CheckpointResumeValidationResult(
        eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
    )


def _terminal_outcome_to_eligibility(
    outcome: ExecutionTerminalOutcome,
) -> CheckpointResumeEligibility:
    if outcome is ExecutionTerminalOutcome.CANCELLED:
        return CheckpointResumeEligibility.REJECT_CANCELLED
    return CheckpointResumeEligibility.REJECT_TERMINAL


def validate_checkpoint_terminal_gate(
    checkpoint: TaskCheckpoint,
    execution_terminal: ExecutionTerminalService | None,
) -> CheckpointResumeValidationResult:
    if execution_terminal is None:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
        )
    try:
        record = execution_terminal.get_terminal_record(
            tenant_id=checkpoint.tenant_id,
            task_id=checkpoint.task_id,
        )
    except ExecutionTerminalError:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_TERMINAL,
            reason="execution terminal authority is corrupt or unavailable",
        )
    if record is None:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
        )
    return CheckpointResumeValidationResult(
        eligibility=_terminal_outcome_to_eligibility(record.outcome),
        reason=f"task execution reached terminal outcome {record.outcome.value!r}",
    )


def validate_checkpoint_resumable_state(
    checkpoint: TaskCheckpoint,
) -> CheckpointResumeValidationResult:
    if checkpoint.task_state not in _RESUMABLE_CHECKPOINT_STATES:
        return CheckpointResumeValidationResult(
            eligibility=CheckpointResumeEligibility.REJECT_STATE,
            reason=f"checkpoint state {checkpoint.task_state.value!r} is not resumable",
        )
    return CheckpointResumeValidationResult(
        eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
    )


def _merge_failure(
    current: CheckpointResumeValidationResult,
    next_result: CheckpointResumeValidationResult,
) -> CheckpointResumeValidationResult:
    if current.eligibility is not CheckpointResumeEligibility.ALLOW_RESUME:
        return current
    return next_result


def evaluate_checkpoint_resume_eligibility(
    checkpoint: TaskCheckpoint,
    *,
    target_task_id: TaskId | str,
    target_tenant_id: str,
    target_run_id: RunId | str | None = None,
    target_attempt_id: AttemptId | str | None = None,
    target_root_execution_id: ExecutionId | str | None = None,
    latest_checkpoint: TaskCheckpoint | None = None,
    execution_terminal: ExecutionTerminalService | None = None,
    execution_lineage_persistence: ExecutionLineagePersistence | None = None,
    require_durable_lineage: bool = False,
    current_task: Task | None = None,
    policy_decision: PolicyDecision | None = None,
) -> CheckpointResumeValidationResult:
    result = CheckpointResumeValidationResult(
        eligibility=CheckpointResumeEligibility.ALLOW_RESUME,
    )
    result = _merge_failure(result, validate_task_checkpoint_schema(checkpoint))
    result = _merge_failure(
        result,
        validate_checkpoint_resumable_state(checkpoint),
    )
    result = _merge_failure(
        result,
        validate_checkpoint_identity_binding(
            checkpoint,
            target_task_id=target_task_id,
            target_tenant_id=target_tenant_id,
            target_run_id=target_run_id,
            target_attempt_id=target_attempt_id,
            target_root_execution_id=target_root_execution_id,
        ),
    )
    result = _merge_failure(
        result,
        validate_checkpoint_not_stale(checkpoint, latest_checkpoint),
    )
    result = _merge_failure(
        result,
        validate_checkpoint_terminal_gate(checkpoint, execution_terminal),
    )
    result = _merge_failure(
        result,
        validate_checkpoint_lineage_cross_reference(
            checkpoint,
            execution_lineage_persistence,
            require_durable_lineage=require_durable_lineage,
        ),
    )
    result = _merge_failure(
        result,
        validate_checkpoint_governance_freshness(policy_decision),
    )
    result = _merge_failure(
        result,
        validate_checkpoint_resume_authority(checkpoint, current_task),
    )
    return result


def assert_checkpoint_resume_eligible(
    checkpoint: TaskCheckpoint,
    *,
    target_task_id: TaskId | str,
    target_tenant_id: str,
    target_run_id: RunId | str | None = None,
    target_attempt_id: AttemptId | str | None = None,
    target_root_execution_id: ExecutionId | str | None = None,
    latest_checkpoint: TaskCheckpoint | None = None,
    execution_terminal: ExecutionTerminalService | None = None,
    execution_lineage_persistence: ExecutionLineagePersistence | None = None,
    require_durable_lineage: bool = False,
    current_task: Task | None = None,
    policy_decision: PolicyDecision | None = None,
) -> None:
    result = evaluate_checkpoint_resume_eligibility(
        checkpoint,
        target_task_id=target_task_id,
        target_tenant_id=target_tenant_id,
        target_run_id=target_run_id,
        target_attempt_id=target_attempt_id,
        target_root_execution_id=target_root_execution_id,
        latest_checkpoint=latest_checkpoint,
        execution_terminal=execution_terminal,
        execution_lineage_persistence=execution_lineage_persistence,
        require_durable_lineage=require_durable_lineage,
        current_task=current_task,
        policy_decision=policy_decision,
    )
    if result.eligibility is CheckpointResumeEligibility.ALLOW_RESUME:
        return
    raise CheckpointResumeValidationError(result)


def assert_checkpoint_persistable(task: Task, runtime: RuntimeCheckpoint) -> None:
    if task.state not in _RESUMABLE_CHECKPOINT_STATES:
        raise CheckpointNotResumableError(
            f"checkpoint persist blocked: task state {task.state.value!r} is not resumable",
        )
    schema_result = validate_runtime_checkpoint_schema(runtime)
    if schema_result.eligibility is not CheckpointResumeEligibility.ALLOW_RESUME:
        raise CheckpointResumeValidationError(schema_result)
    runtime.validate_canonical()
