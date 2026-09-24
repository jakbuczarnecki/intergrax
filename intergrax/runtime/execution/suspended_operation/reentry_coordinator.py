# © Artur Czarnecki. All rights reserved.

"""Canonical suspended work re-entry coordinator (UCA-6C-R6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from intergrax.contracts.agent_governance_hitl import AgentGovernanceGrantLifecycleState
from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
    ExecutionContinuationPort,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationClaimOutcome,
    SuspendedOperationMutationOutcome,
)
from intergrax.contracts.execution.suspended_operation.codec import (
    SuspendedOperationCodecRegistry,
    SuspendedOperationKind,
)
from intergrax.contracts.execution.suspended_operation.payload_catalog import (
    ExecutionBoundCatalogToolOperationPayload,
)
from intergrax.contracts.execution.suspended_operation.reentry import (
    ExecutionSuspendedWorkReentryDisposition,
    ExecutionSuspendedWorkReentryRequest,
    ExecutionSuspendedWorkReentryResult,
)
from intergrax.contracts.execution.crash_injection import (
    ExecutionSuspendedWorkReentryCrashCheckpoint,
    ExecutionSuspendedWorkReentryCrashInjectionPort,
)
from intergrax.contracts.execution.execution_terminal_outcome_by_execution_id import (
    ExecutionTerminalOutcomeByExecutionIdDisposition,
    ExecutionTerminalOutcomeByExecutionIdStore,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution.suspended_operation.store import (
    SuspendedExecutionOperationStore,
)
from intergrax.contracts.execution_deadline.clock import UtcClockPort
from intergrax.runtime.execution.deadline_authority.system_clocks import SystemUtcClock
from intergrax.runtime.agent_governance.errors import (
    ToolGovernanceApprovalRequiredError,
    ToolGovernanceDeniedError,
)
from intergrax.runtime.execution.suspended_operation.payload_digest import (
    digest_suspended_operation_envelope,
)
from intergrax.runtime.execution.suspended_operation.agent_governance_reentry_grant import (
    AgentGovernanceReentryGrantError,
    is_agent_governance_invocation_scope,
    prepare_agent_governance_grant_for_reentry,
)
from intergrax.runtime.execution.suspended_operation.pause_required import (
    ExecutionSuspendedWorkPauseRequired,
)
from intergrax.runtime.human.agent_governance_human_approval_grant import (
    AgentGovernanceHumanApprovalGrantCoordinator,
)
from intergrax.runtime.human.declarative_hitl_grant import (
    DeclarativeHitlGrantCoordinator,
)
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
)
from intergrax.runtime.execution.suspended_operation.suspended_work_agent_governance_approval_consumption import (
    SuspendedWorkAgentGovernanceApprovalConsumption,
)
from intergrax.runtime.execution.suspended_operation.crash_injection import (
    NoOpExecutionSuspendedWorkReentryCrashInjection,
)
from intergrax.runtime.nexus.tools.continuation_aware_catalog_tool_host import (
    ContinuationAwareCatalogToolHost,
)
from intergrax.runtime.nexus.tools.nexus_execution_bound_catalog_tool_invoker import (
    NexusExecutionBoundCatalogToolInvoker,
)
from intergrax.runtime.task.execution_continuation_projection import (
    continuation_projection_allows_replacement,
)
from intergrax.runtime.task.task import Task
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry.read import ToolRegistryRead
from intergrax.tools.durable_invocation_wiring_binding_resolver import (
    DurableToolInvocationWiringBindingResolutionError,
    DurableToolInvocationWiringBindingResolver,
)
from intergrax.tools.execution_models import ToolExecutionResult


@dataclass(frozen=True, slots=True)
class ExecutionSuspendedWorkReentryCoordinator:
    """Execution-owned re-entry: claim → reconstruct → ToolRuntime → consume."""

    store: SuspendedExecutionOperationStore
    continuation_port: ExecutionContinuationPort
    tool_registry: ToolRegistryRead
    catalog_host: ContinuationAwareCatalogToolHost
    catalog_invoker: NexusExecutionBoundCatalogToolInvoker
    codec_registry: SuspendedOperationCodecRegistry
    binding_resolver: DurableToolInvocationWiringBindingResolver
    claim_owner_id: str
    task_checkpoint_store: TaskCheckpointPersistence | None = None
    terminal_outcome_store: ExecutionTerminalOutcomeByExecutionIdStore | None = None
    crash_injection: ExecutionSuspendedWorkReentryCrashInjectionPort = field(
        default_factory=NoOpExecutionSuspendedWorkReentryCrashInjection,
    )
    default_lease_seconds: int = 120
    utc_clock: UtcClockPort = field(default_factory=SystemUtcClock)

    def reenter_after_resume(
        self,
        request: ExecutionSuspendedWorkReentryRequest,
        *,
        task: Task | None = None,
    ) -> ExecutionSuspendedWorkReentryResult:
        pending = self.continuation_port.get_pending(
            ExecutionContinuationLookup(continuation_id=request.continuation_id),
        )
        if pending.lifecycle_state is not ExecutionContinuationLifecycleState.RESUMED:
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.NOT_READY,
                reason_detail="continuation_not_resumed",
            )
        if task is not None:
            _reconcile_task_projection_for_resumed_suspended_reentry(task, pending)
            HumanPauseCoordinator.project_continuation(task, pending)
        descriptor = self.store.load_active_for_continuation(request.continuation_id)
        if descriptor is None:
            reconciled = self._reconcile_consumed_without_terminal_outcome(
                request=request,
            )
            if reconciled is not None:
                return reconciled
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.NOT_READY,
                reason_detail="no_active_suspended_operation",
            )
        if descriptor.identity != request.identity:
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.REJECTED,
                reason_detail="identity_mismatch",
            )
        governed = pending.governed_correlation
        if governed is None or descriptor.invocation_scope_id != governed.operation_id:
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.REJECTED,
                reason_detail="invocation_scope_mismatch",
            )
        if (
            digest_suspended_operation_envelope(descriptor.payload)
            != descriptor.payload_digest
        ):
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.REJECTED,
                reason_detail="payload_digest_mismatch",
            )
        authority = request.claim_authority
        if authority.owner_id != self.claim_owner_id:
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.REJECTED,
                reason_detail="claim_owner_mismatch",
            )
        if authority.materialization_revision != descriptor.materialization_revision:
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.REJECTED,
                reason_detail="stale_materialization_revision",
            )
        if authority.pause_generation != descriptor.pause_generation:
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.REJECTED,
                reason_detail="stale_pause_generation",
            )

        now = datetime.now(timezone.utc)
        if (
            descriptor.materialization_state
            is SuspendedOperationMaterializationState.CLAIMED
        ):
            ownership = descriptor.claim_ownership
            if ownership is None:
                return ExecutionSuspendedWorkReentryResult(
                    disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                    reason_detail="claim_missing_descriptor",
                )
            if ownership.owner_id != authority.owner_id:
                return ExecutionSuspendedWorkReentryResult(
                    disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                    reason_detail="stale_claim_owner",
                )
            if authority.fence != ownership.fence:
                return ExecutionSuspendedWorkReentryResult(
                    disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                    reason_detail="stale_claim_fence",
                )
            if ownership.lease_expires_at <= now:
                return ExecutionSuspendedWorkReentryResult(
                    disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                    reason_detail="claim_lease_expired",
                )
            claimed = descriptor
        else:
            if descriptor.claim_ownership is not None:
                return ExecutionSuspendedWorkReentryResult(
                    disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                    reason_detail="claim_ownership_unexpected",
                )
            if authority.fence != 0:
                return ExecutionSuspendedWorkReentryResult(
                    disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                    reason_detail="stale_claim_fence",
                )
            lease_expires = now + timedelta(seconds=self.default_lease_seconds)
            claim = self.store.claim(
                suspended_operation_id=descriptor.suspended_operation_id,
                expected_materialization_revision=authority.materialization_revision,
                owner_id=authority.owner_id,
                lease_expires_at=lease_expires,
            )
            if claim.outcome is not SuspendedOperationClaimOutcome.CLAIMED:
                return ExecutionSuspendedWorkReentryResult(
                    disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                    reason_detail=claim.outcome.value,
                )
            claimed = claim.descriptor
            if claimed is None or claimed.claim_ownership is None:
                return ExecutionSuspendedWorkReentryResult(
                    disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                    reason_detail="claim_missing_descriptor",
                )

        claim_ownership = claimed.claim_ownership
        if claim_ownership is None:
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                reason_detail="claim_missing_descriptor",
            )

        codec = self.codec_registry.resolve(
            SuspendedOperationKind.EXECUTION_BOUND_CATALOG_TOOL,
            claimed.payload.payload_schema_version,
        )
        payload = codec.decode(claimed.payload)
        if type(payload) is not ExecutionBoundCatalogToolOperationPayload:
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                reason_detail="unsupported_payload_type",
            )

        grant: DeclarativeHitlApprovalGrant | None = None
        agent_grant_prepare = None
        if task is not None:
            grant = task.runtime.governance.declarative_hitl_grant
            if grant is None:
                grant = DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)

        try:
            invoke_request = _reconstruct_invoke_request(
                payload,
                binding_resolver=self.binding_resolver,
            )
        except DurableToolInvocationWiringBindingResolutionError as exc:
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                reason_detail=exc.code,
            )
        state = self.catalog_invoker.build_runtime_state(invoke_request)
        if grant is not None:
            state.declarative_hitl_grant = grant

        agent_grant_record = (
            task.runtime.governance.agent_governance_human_approval_grant
            if task is not None
            else None
        )
        agent_grant_applied = (
            agent_grant_record is not None
            and agent_grant_record.lifecycle_state
            is AgentGovernanceGrantLifecycleState.APPLIED
        )
        if (
            task is not None
            and self.task_checkpoint_store is not None
            and (
                is_agent_governance_invocation_scope(claimed.invocation_scope_id)
                or agent_grant_applied
            )
        ):
            contract = self._tool_contract(payload.tool_id)
            pause_generation = claimed.pause_generation
            if agent_grant_applied and not is_agent_governance_invocation_scope(
                claimed.invocation_scope_id,
            ):
                pause_generation = 1
            try:
                agent_grant_prepare = prepare_agent_governance_grant_for_reentry(
                    task=task,
                    checkpoint_store=self.task_checkpoint_store,
                    payload=payload,
                    contract=contract,
                    state=state,
                    claim_ownership=claim_ownership,
                    pause_generation=pause_generation,
                    lease_seconds=self.default_lease_seconds,
                )
            except AgentGovernanceReentryGrantError as exc:
                return ExecutionSuspendedWorkReentryResult(
                    disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                    reason_detail=str(exc),
                )
            state.verified_agent_governance_human_approval = (
                agent_grant_prepare.verified
            )
            state.agent_governance_approval_consumption = (
                SuspendedWorkAgentGovernanceApprovalConsumption(
                    task=task,
                    checkpoint_store=self.task_checkpoint_store,
                    lifecycle_record=agent_grant_prepare.lifecycle_record,
                    claim_ownership=agent_grant_prepare.claim_ownership,
                )
            )

        self.crash_injection.raise_if_scheduled(
            ExecutionSuspendedWorkReentryCrashCheckpoint.AFTER_CLAIM_BEFORE_TOOL_RUNTIME,
        )
        try:
            tool_result = self.catalog_host.invoke(
                state=state,
                request=invoke_request,
                declarative_grant=grant,
                task=task,
                reentry_claimed_descriptor=claimed,
            )
        except ExecutionSuspendedWorkPauseRequired:
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.PAUSED_FOR_NEXT_AUTHORITY,
                reason_detail="authority_sequential_reblock",
            )
        except ToolGovernanceDeniedError:
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                reason_detail="agent_governance_denied",
            )
        except ToolGovernanceApprovalRequiredError:
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                reason_detail="agent_governance_approval_still_required",
            )
        finally:
            state.agent_governance_approval_consumption = None

        if isinstance(tool_result, ToolExecutionResult) and tool_result.success:
            self.crash_injection.raise_if_scheduled(
                ExecutionSuspendedWorkReentryCrashCheckpoint.AFTER_EFFECT_COMMIT_BEFORE_CONSUME,
            )
            consumed = self.store.mark_consumed(
                suspended_operation_id=claimed.suspended_operation_id,
                expected_materialization_revision=claimed.materialization_revision,
                owner_id=claim_ownership.owner_id,
                fence=claim_ownership.fence,
                expected_pause_generation=claimed.pause_generation,
            )
            if consumed.outcome is not SuspendedOperationMutationOutcome.APPLIED:
                return ExecutionSuspendedWorkReentryResult(
                    disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                    reason_detail="mark_consumed_failed",
                    tool_result=tool_result,
                )
            self.crash_injection.raise_if_scheduled(
                ExecutionSuspendedWorkReentryCrashCheckpoint.AFTER_CONSUME_BEFORE_TERMINAL,
            )
            if (
                task is not None
                and self.task_checkpoint_store is not None
                and is_agent_governance_invocation_scope(claimed.invocation_scope_id)
            ):
                AgentGovernanceHumanApprovalGrantCoordinator.terminalize_after_successful_consumption(
                    task,
                    checkpoint_store=self.task_checkpoint_store,
                )
            if self.terminal_outcome_store is not None:
                self.terminal_outcome_store.record_terminal_disposition(
                    request.identity.execution_id,
                    ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED,
                )
            self.crash_injection.raise_if_scheduled(
                ExecutionSuspendedWorkReentryCrashCheckpoint.AFTER_TERMINAL_BEFORE_RETURN,
            )
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.COMPLETED,
                tool_result=tool_result,
            )
        if self.terminal_outcome_store is not None:
            self.terminal_outcome_store.record_terminal_disposition(
                request.identity.execution_id,
                ExecutionTerminalOutcomeByExecutionIdDisposition.FAILED,
            )
        return ExecutionSuspendedWorkReentryResult(
            disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
            tool_result=tool_result,
            reason_detail="tool_invocation_failed",
        )

    def _reconcile_consumed_without_terminal_outcome(
        self,
        *,
        request: ExecutionSuspendedWorkReentryRequest,
    ) -> ExecutionSuspendedWorkReentryResult | None:
        materialized = self.store.load_materialized_for_continuation(
            request.continuation_id,
        )
        if materialized is None:
            return None
        if (
            materialized.materialization_state
            is not SuspendedOperationMaterializationState.CONSUMED
        ):
            return None
        if materialized.identity != request.identity:
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.REJECTED,
                reason_detail="identity_mismatch",
            )
        if self.terminal_outcome_store is None:
            return None
        recorded = self.terminal_outcome_store.get_recorded_disposition(
            request.identity.execution_id,
        )
        if recorded is None:
            self.terminal_outcome_store.record_terminal_disposition(
                request.identity.execution_id,
                ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED,
            )
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.COMPLETED,
                reason_detail="consumed_terminal_reconciled",
            )
        if recorded is ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED:
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.NOT_READY,
                reason_detail="execution_already_terminal",
            )
        return ExecutionSuspendedWorkReentryResult(
            disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
            reason_detail="consumed_terminal_conflict",
        )

    def _tool_contract(self, tool_id: str) -> ToolContract:
        return self.tool_registry.get(tool_id).contract


def _reconcile_task_projection_for_resumed_suspended_reentry(
    task: Task,
    pending: PendingExecutionContinuation,
) -> None:
    """Allow RESUMED re-entry when Task still projects a prior authority generation."""
    if pending.lifecycle_state is not ExecutionContinuationLifecycleState.RESUMED:
        return
    gov = task.runtime.governance
    if gov.projected_continuation_id == pending.continuation_id:
        return
    if gov.projected_continuation_id is None:
        return
    if continuation_projection_allows_replacement(
        gov.projected_continuation_lifecycle_state,
    ):
        return
    gov.projected_continuation_id = None
    gov.projected_continuation_revision = None
    gov.projected_continuation_lifecycle_state = None
    gov.projected_continuation_payload_digest = None


def _reconstruct_invoke_request(
    payload: ExecutionBoundCatalogToolOperationPayload,
    *,
    binding_resolver: DurableToolInvocationWiringBindingResolver,
) -> ExecutionBoundCatalogToolInvokeRequest:
    wiring_resolver = None
    if payload.wiring_resolver_kind == "fixed_sandbox_session":
        if not payload.sandbox_session_id:
            raise DurableToolInvocationWiringBindingResolutionError(
                "sandbox_session_id_missing",
                "fixed_sandbox_session payload requires sandbox_session_id",
            )
        wiring_resolver = binding_resolver.resolve_fixed_sandbox_session_wiring(
            sandbox_session_id=payload.sandbox_session_id,
            tenant_id=payload.tenant_id,
            task_id=payload.task_id,
        )
    elif payload.sandbox_session_id is not None:
        raise DurableToolInvocationWiringBindingResolutionError(
            "unsupported_wiring_resolver_kind",
            f"unsupported wiring_resolver_kind: {payload.wiring_resolver_kind}",
        )
    return ExecutionBoundCatalogToolInvokeRequest(
        tool_id=payload.tool_id,
        input=payload.tool_input,
        tenant_id=payload.tenant_id,
        task_id=payload.task_id,
        run_id=payload.run_id,
        agent_id=payload.agent_id,
        step_id=payload.step_id,
        correlation_request_id=payload.correlation_request_id,
        wiring_resolver=wiring_resolver,
        idempotency_key=payload.idempotency_key,
    )


__all__ = [
    "ExecutionSuspendedWorkReentryCoordinator",
]
