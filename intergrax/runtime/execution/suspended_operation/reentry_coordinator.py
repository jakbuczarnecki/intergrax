# © Artur Czarnecki. All rights reserved.

"""Canonical suspended work re-entry coordinator (UCA-6C-R6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from intergrax.contracts.declarative_hitl import DeclarativeHitlApprovalGrant
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
    ExecutionContinuationPort,
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
from intergrax.contracts.execution.suspended_operation.store import (
    SuspendedExecutionOperationStore,
)
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
    mark_agent_governance_grant_applied_after_governance,
    prepare_agent_governance_grant_for_reentry,
)
from intergrax.runtime.human.agent_governance_human_approval_grant import (
    AgentGovernanceHumanApprovalGrantCoordinator,
)
from intergrax.runtime.human.declarative_hitl_grant import (
    DeclarativeHitlGrantCoordinator,
)
from intergrax.runtime.long_running.persistence_contract import (
    TaskCheckpointPersistence,
)
from intergrax.runtime.nexus.tools.continuation_aware_catalog_tool_host import (
    ContinuationAwareCatalogToolHost,
)
from intergrax.runtime.nexus.tools.nexus_execution_bound_catalog_tool_invoker import (
    NexusExecutionBoundCatalogToolInvoker,
)
from intergrax.runtime.task.task import Task
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
    catalog_host: ContinuationAwareCatalogToolHost
    catalog_invoker: NexusExecutionBoundCatalogToolInvoker
    codec_registry: SuspendedOperationCodecRegistry
    binding_resolver: DurableToolInvocationWiringBindingResolver
    claim_owner_id: str
    task_checkpoint_store: TaskCheckpointPersistence | None = None
    default_lease_seconds: int = 120

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
        descriptor = self.store.load_active_for_continuation(request.continuation_id)
        if descriptor is None:
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

        lease_expires = datetime.now(timezone.utc) + timedelta(
            seconds=self.default_lease_seconds,
        )
        claim = self.store.claim(
            suspended_operation_id=descriptor.suspended_operation_id,
            expected_materialization_revision=descriptor.materialization_revision,
            owner_id=self.claim_owner_id,
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

        if (
            task is not None
            and self.task_checkpoint_store is not None
            and is_agent_governance_invocation_scope(claimed.invocation_scope_id)
        ):
            contract = self.catalog_host._tool_invoker.registry.get(
                payload.tool_id,
            ).contract
            pending = task.runtime.governance.agent_governance_hitl_pending
            pause_generation = (
                pending.requirement.pause_generation if pending is not None else 1
            )
            try:
                agent_grant_prepare = prepare_agent_governance_grant_for_reentry(
                    task=task,
                    checkpoint_store=self.task_checkpoint_store,
                    payload=payload,
                    contract=contract,
                    state=state,
                    claim_ownership=claimed.claim_ownership,
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

        try:
            tool_result = self.catalog_host.invoke(
                state=state,
                request=invoke_request,
                declarative_grant=grant,
                task=task,
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

        if agent_grant_prepare is not None and task is not None:
            assert self.task_checkpoint_store is not None
            lifecycle_record = (
                task.runtime.governance.agent_governance_human_approval_grant
            )
            if lifecycle_record is None:
                return ExecutionSuspendedWorkReentryResult(
                    disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                    reason_detail="agent_governance_grant_missing_after_invoke",
                )
            try:
                mark_agent_governance_grant_applied_after_governance(
                    task=task,
                    checkpoint_store=self.task_checkpoint_store,
                    lifecycle_record=lifecycle_record,
                    claim_ownership=agent_grant_prepare.claim_ownership,
                )
            except AgentGovernanceReentryGrantError as exc:
                return ExecutionSuspendedWorkReentryResult(
                    disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                    reason_detail=str(exc),
                    tool_result=tool_result
                    if isinstance(tool_result, ToolExecutionResult)
                    else None,
                )
        if isinstance(tool_result, ToolExecutionResult) and tool_result.success:
            consumed = self.store.mark_consumed(
                suspended_operation_id=claimed.suspended_operation_id,
                expected_materialization_revision=claimed.materialization_revision,
                owner_id=claimed.claim_ownership.owner_id,
                fence=claimed.claim_ownership.fence,
            )
            if consumed.outcome is not SuspendedOperationMutationOutcome.APPLIED:
                return ExecutionSuspendedWorkReentryResult(
                    disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                    reason_detail="mark_consumed_failed",
                    tool_result=tool_result,
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
            return ExecutionSuspendedWorkReentryResult(
                disposition=ExecutionSuspendedWorkReentryDisposition.COMPLETED,
                tool_result=tool_result,
            )
        return ExecutionSuspendedWorkReentryResult(
            disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
            tool_result=tool_result,
            reason_detail="tool_invocation_failed",
        )


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
