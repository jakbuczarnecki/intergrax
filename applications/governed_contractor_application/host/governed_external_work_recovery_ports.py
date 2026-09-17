# © Artur Czarnecki. All rights reserved.

"""GR-7-A7-R1 — governed external-work recovery repeat and HITL host ports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from external_contractor_adapter.external_effect_outcome_projection import (
    external_work_provider_mutation_attempted,
)
from external_contractor_adapter.external_work_adapter import ExternalWorkAdapter
from external_contractor_adapter.side_effect_actions import ACTION_CREATE_EXTERNAL_WORK
from governed_contractor_application.host.provider_invocation_lifecycle import (
    build_provider_invocation_outcome,
    classify_provider_invocation_status,
    persist_provider_invocation_outcome,
)
from governed_contractor_application.host.stores import ContinuationStateStore
from intergrax.contracts.enterprise_reliability.effect_contract import ExternalEffectContract
from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryEscalationContext,
)
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId
from intergrax.contracts.external_work import ExternalWorkCreateRequest
from intergrax.contracts.governed_continuation import (
    ContinuationReason,
    GovernedContinuationRequest,
)
from intergrax.contracts.governed_continuation_correlation import GovernedContinuationCorrelation
from intergrax.contracts.governed_execution_result import (
    external_work_decision_action_for_provider_operation,
    external_work_invocation_operation_matches_effect_contract,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
)
from intergrax.contracts.provider_invocation_store import ProviderInvocationStore
from intergrax.runtime.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryHitlResult,
    ProviderInvocationRecoveryRepeatResult,
)


@dataclass(frozen=True, slots=True)
class GovernedExternalWorkProviderRecoveryRepeatPort:
    """Canonical repeat: fresh GR-6 authorization + A3 durable lifecycle."""

    adapter: ExternalWorkAdapter
    invocation_store: ProviderInvocationStore
    principal_id: str
    tenant_id: str
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    create_request_for_invocation: Callable[[ProviderInvocation], ExternalWorkCreateRequest]
    clock: Callable[[], datetime]
    host_execution_id: str

    def supports_provider_operation(self, operation: str) -> bool:
        action = external_work_decision_action_for_provider_operation(operation)
        return action == ACTION_CREATE_EXTERNAL_WORK

    def execute_idempotent_repeat(
        self,
        *,
        original_invocation: ProviderInvocation,
        original_outcome: ProviderInvocationOutcome,
        effect_contract: ExternalEffectContract,
    ) -> ProviderInvocationRecoveryRepeatResult:
        _ = original_outcome
        if not self.supports_provider_operation(original_invocation.operation):
            raise RuntimeError("repeat execution not supported for operation")
        if not external_work_invocation_operation_matches_effect_contract(
            contract_operation_key=effect_contract.operation_key,
            invocation_operation=original_invocation.operation,
        ):
            raise RuntimeError("effect contract operation mismatch for repeat")

        idempotency_key = (original_invocation.idempotency_key or "").strip()
        if not idempotency_key:
            raise RuntimeError("idempotency_key required for repeat")

        action = external_work_decision_action_for_provider_operation(
            original_invocation.operation,
        )
        if action != ACTION_CREATE_EXTERNAL_WORK:
            raise RuntimeError("recovery repeat not composed for operation")

        started = self.clock()
        repeat_invocation = original_invocation.model_copy(
            update={
                "invocation_id": f"inv-{uuid4().hex}",
                "started_at": started,
                "request_digest": original_invocation.request_digest,
            },
        )
        base_request = self.create_request_for_invocation(original_invocation)
        repeat_request = base_request.model_copy(
            update={
                "provider_id": original_invocation.provider_id,
                "task_id": original_invocation.task_id,
                "run_id": original_invocation.run_id,
                "idempotency_key": idempotency_key,
                "correlation_id": (
                    original_invocation.correlation_id or base_request.correlation_id
                ),
            },
        )

        adapter_result = self.adapter.create_and_map(
            repeat_request,
            principal_id=self.principal_id,
            tenant_id=self.tenant_id,
            provider_invocation=repeat_invocation,
        )

        stored = self.invocation_store.get_invocation(repeat_invocation.invocation_id)
        if stored is None:
            raise RuntimeError("repeat invocation intent not durable before provider dispatch")

        attempted = external_work_provider_mutation_attempted(
            adapter_result,
            policy_denied=False,
        )
        mutation_count = 1 if attempted else 0
        status = classify_provider_invocation_status(
            adapter_result=adapter_result,
            provider_mutation_attempted=attempted,
        )
        if status is not None:
            outcome = build_provider_invocation_outcome(
                invocation_id=repeat_invocation.invocation_id,
                status=status,
                completed_at=self.clock(),
                adapter_result=adapter_result,
                execution_id=self.host_execution_id,
                action=action,
            )
            persist_provider_invocation_outcome(self.invocation_store, outcome)

        if mutation_count == 1:
            persisted = self.invocation_store.get_invocation(repeat_invocation.invocation_id)
            if persisted is None:
                raise RuntimeError("repeat invocation missing after provider dispatch")
            if persisted.provider_id != original_invocation.provider_id:
                raise RuntimeError("repeat provider_id drift")
            if persisted.operation != original_invocation.operation:
                raise RuntimeError("repeat operation drift")
            if (persisted.idempotency_key or "").strip() != idempotency_key:
                raise RuntimeError("repeat idempotency_key drift")
            if not external_work_invocation_operation_matches_effect_contract(
                contract_operation_key=effect_contract.operation_key,
                invocation_operation=persisted.operation,
            ):
                raise RuntimeError("repeat operation canonical binding drift")

        return ProviderInvocationRecoveryRepeatResult(
            repeat_invocation_id=repeat_invocation.invocation_id,
            idempotency_key=idempotency_key,
            provider_mutation_count=mutation_count,
        )


@dataclass(frozen=True, slots=True)
class GovernedExternalWorkProviderRecoveryHitlPort:
    """Surfaces recovery escalation via existing GovernedContinuationRequest store."""

    continuation_store: ContinuationStateStore
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    source_agent_id: str = "governed_external_work.recovery"

    def surface_hitl(
        self,
        escalation: ProviderInvocationRecoveryEscalationContext,
    ) -> ProviderInvocationRecoveryHitlResult:
        invocation = escalation.invocation
        continuation = GovernedContinuationRequest(
            reason=ContinuationReason.COMPLIANCE,
            task_id=invocation.task_id,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            execution_id=self.execution_id,
            source_agent_id=self.source_agent_id,
            prompt=(
                "Provider invocation recovery requires human review "
                f"({escalation.recovery_reason.value})"
            ),
            operation_id=invocation.operation,
            side_effect_scope_id=invocation.idempotency_key,
            side_effect_scope_digest=invocation.request_digest,
            resource_scope=invocation.external_task_id,
            context={
                "recovery_reason": escalation.recovery_reason.value,
                "dispatch_state": escalation.dispatch_state.value,
                "effect_contract_id": escalation.effect_contract_id,
            },
        )
        correlation = GovernedContinuationCorrelation(
            continuation_request_id=continuation.continuation_request_id,
            reason=ContinuationReason.COMPLIANCE,
            task_id=invocation.task_id,
            run_id=self.run_id,
            attempt_id=self.attempt_id,
            execution_id=self.execution_id,
            operation_id=invocation.operation,
            side_effect_scope_id=invocation.idempotency_key,
            side_effect_scope_digest=invocation.request_digest,
        )
        continuation = continuation.model_copy(
            update={"correlation": correlation.model_dump()},
        )
        self.continuation_store.put_continuation(invocation.task_id, continuation)
        return ProviderInvocationRecoveryHitlResult(
            escalation=escalation,
            governed_continuation_request_id=continuation.continuation_request_id,
        )


__all__ = [
    "GovernedExternalWorkProviderRecoveryHitlPort",
    "GovernedExternalWorkProviderRecoveryRepeatPort",
]
