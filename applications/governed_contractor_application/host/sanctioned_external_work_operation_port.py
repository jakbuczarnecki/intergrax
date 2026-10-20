# © Artur Czarnecki. All rights reserved.

"""Host adapter: typed sanctioned external-work port → ``GovernedExternalWorkOrchestrator``."""

from __future__ import annotations

from governed_contractor_application.host.orchestrator import (
    GovernedExternalWorkOrchestrator,
    META_PROVIDER_INVOCATION_ID,
    OrchestratorStepResult,
)
from intergrax.contracts.sanctioned_external_work_operation import (
    SanctionedExternalWorkDomainSummary,
    SanctionedExternalWorkMutationOperation,
    SanctionedExternalWorkOperationPort,
    SanctionedExternalWorkOperationRequest,
    SanctionedExternalWorkOperationResult,
)


class GovernedSanctionedExternalWorkOperationPort(SanctionedExternalWorkOperationPort):
    """Session-scoped orchestrator facade — prior-create authority is the orchestrator only."""

    def __init__(
        self,
        orchestrator: GovernedExternalWorkOrchestrator,
    ) -> None:
        self._orchestrator = orchestrator

    def execute(
        self,
        request: SanctionedExternalWorkOperationRequest,
    ) -> SanctionedExternalWorkOperationResult:
        if request.operation is SanctionedExternalWorkMutationOperation.CREATE:
            return self._execute_create(request)
        if request.operation is SanctionedExternalWorkMutationOperation.ACCEPT_QUOTE:
            return self._execute_accept(request)
        if request.operation is SanctionedExternalWorkMutationOperation.CANCEL:
            return self._execute_cancel(request)
        return self._deny(
            reason="unsupported_sanctioned_operation",
            operation=request.operation,
        )

    def _execute_create(
        self,
        request: SanctionedExternalWorkOperationRequest,
    ) -> SanctionedExternalWorkOperationResult:
        payload = request.create
        if payload is None:
            return self._deny(
                reason="sanctioned_create_payload_missing",
                operation=SanctionedExternalWorkMutationOperation.CREATE,
            )
        step = self._orchestrator.create(
            task_id=str(payload.task_id),
            run_id=str(payload.run_id),
            principal_id=payload.principal_id,
            tenant_id=payload.tenant_id,
            metadata=payload.adapter_metadata.to_adapter_metadata(),
            execution_id=str(payload.execution_id) if payload.execution_id is not None else None,
        )
        return self._map_step(step, operation=SanctionedExternalWorkMutationOperation.CREATE)

    def _execute_accept(
        self,
        request: SanctionedExternalWorkOperationRequest,
    ) -> SanctionedExternalWorkOperationResult:
        payload = request.accept_quote
        if payload is None:
            return self._deny(
                reason="sanctioned_accept_payload_missing",
                operation=SanctionedExternalWorkMutationOperation.ACCEPT_QUOTE,
            )
        execution_id = str(payload.execution_id)
        create_result = self._resolve_prior_create_result(
            execution_id,
            tenant_id=payload.tenant_id,
            principal_id=payload.principal_id,
        )
        if create_result is None:
            return self._deny(
                reason="prior_create_snapshot_missing",
                operation=SanctionedExternalWorkMutationOperation.ACCEPT_QUOTE,
                execution_id=execution_id,
            )
        step = self._orchestrator.accept(
            execution_id=execution_id,
            create_result=create_result,
            acceptance=payload.acceptance,
            idempotency_key=payload.acceptance_idempotency_key,
            principal_id=payload.principal_id,
            tenant_id=payload.tenant_id,
            metadata=payload.adapter_metadata.to_adapter_metadata(),
        )
        return self._map_step(
            step,
            operation=SanctionedExternalWorkMutationOperation.ACCEPT_QUOTE,
        )

    def _execute_cancel(
        self,
        request: SanctionedExternalWorkOperationRequest,
    ) -> SanctionedExternalWorkOperationResult:
        payload = request.cancel
        if payload is None:
            return self._deny(
                reason="sanctioned_cancel_payload_missing",
                operation=SanctionedExternalWorkMutationOperation.CANCEL,
            )
        execution_id = str(payload.execution_id)
        create_result = self._resolve_prior_create_result(
            execution_id,
            tenant_id=payload.tenant_id,
            principal_id=payload.principal_id,
        )
        if create_result is None:
            return self._deny(
                reason="prior_create_snapshot_missing",
                operation=SanctionedExternalWorkMutationOperation.CANCEL,
                execution_id=execution_id,
            )
        step = self._orchestrator.cancel(
            execution_id=execution_id,
            create_result=create_result,
            principal_id=payload.principal_id,
            tenant_id=payload.tenant_id,
            idempotency_key=payload.idempotency_key,
            metadata=payload.adapter_metadata.to_adapter_metadata(),
        )
        return self._map_step(step, operation=SanctionedExternalWorkMutationOperation.CANCEL)

    def _deny(
        self,
        *,
        reason: str,
        operation: SanctionedExternalWorkMutationOperation,
        execution_id: str | None = None,
    ) -> SanctionedExternalWorkOperationResult:
        return SanctionedExternalWorkOperationResult(
            admitted=False,
            reason=reason,
            execution_id=execution_id,
            domain_summary=SanctionedExternalWorkDomainSummary(
                used=False,
                reason=reason,
                sanctioned_operation=operation.value,
                host_state=None,
                execution_id=execution_id,
                provider_invocation_id=None,
            ),
        )

    def _resolve_prior_create_result(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        principal_id: str,
    ) -> ExternalWorkAdapterResult | None:
        return self._orchestrator.resolve_create_adapter_result_for_follow_on(
            execution_id,
            tenant_id=tenant_id,
            principal_id=principal_id,
        )

    def _map_step(
        self,
        step: OrchestratorStepResult,
        *,
        operation: SanctionedExternalWorkMutationOperation,
    ) -> SanctionedExternalWorkOperationResult:
        adapter = step.adapter_result
        provider_invocation_id: str | None = None
        if adapter is not None and adapter.metadata:
            raw = adapter.metadata.get(META_PROVIDER_INVOCATION_ID)
            if raw is not None and str(raw).strip():
                provider_invocation_id = str(raw).strip()
        if adapter is not None:
            domain_summary = SanctionedExternalWorkDomainSummary.from_adapter_summary(
                adapter.to_domain_summary(),
                sanctioned_operation=operation.value,
                host_state=step.state.value,
                execution_id=step.execution_id,
                provider_invocation_id=provider_invocation_id,
            )
        else:
            domain_summary = SanctionedExternalWorkDomainSummary(
                used=False,
                reason=step.reason,
                sanctioned_operation=operation.value,
                host_state=step.state.value,
                execution_id=step.execution_id,
                provider_invocation_id=provider_invocation_id,
            )
        admitted = adapter is not None and adapter.used
        return SanctionedExternalWorkOperationResult(
            admitted=admitted,
            reason=step.reason,
            execution_id=step.execution_id,
            provider_invocation_id=provider_invocation_id,
            host_state=step.state.value,
            adapter_used=adapter.used if adapter is not None else False,
            adapter_reason=adapter.reason if adapter is not None else None,
            provider_mutation_dispatched=(
                adapter.provider_mutation_dispatched if adapter is not None else False
            ),
            domain_summary=domain_summary,
        )


__all__ = ["GovernedSanctionedExternalWorkOperationPort"]
