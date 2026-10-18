# © Artur Czarnecki. All rights reserved.

"""Host adapter: typed sanctioned external-work port → ``GovernedExternalWorkOrchestrator``."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from external_contractor_adapter.schemas.adapt_result import ExternalWorkAdapterResult
from governed_contractor_application.host.orchestrator import (
    GovernedExternalWorkOrchestrator,
    META_PROVIDER_INVOCATION_ID,
    OrchestratorStepResult,
)
from intergrax.contracts.sanctioned_external_work_operation import (
    SanctionedExternalWorkMutationOperation,
    SanctionedExternalWorkOperationPort,
    SanctionedExternalWorkOperationRequest,
    SanctionedExternalWorkOperationResult,
)


class GovernedSanctionedExternalWorkOperationPort(SanctionedExternalWorkOperationPort):
    """Session-scoped orchestrator facade with authoritative create snapshot cache."""

    def __init__(
        self,
        orchestrator: GovernedExternalWorkOrchestrator,
        *,
        create_result_by_execution_id: MutableMapping[str, ExternalWorkAdapterResult]
        | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._create_results: MutableMapping[str, ExternalWorkAdapterResult] = (
            create_result_by_execution_id if create_result_by_execution_id is not None else {}
        )

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
        return SanctionedExternalWorkOperationResult(
            admitted=False,
            reason="unsupported_sanctioned_operation",
        )

    def _execute_create(
        self,
        request: SanctionedExternalWorkOperationRequest,
    ) -> SanctionedExternalWorkOperationResult:
        payload = request.create
        if payload is None:
            return SanctionedExternalWorkOperationResult(
                admitted=False,
                reason="sanctioned_create_payload_missing",
            )
        step = self._orchestrator.create(
            task_id=str(payload.task_id),
            run_id=str(payload.run_id),
            principal_id=payload.principal_id,
            tenant_id=payload.tenant_id,
            metadata=dict(payload.metadata),
            execution_id=str(payload.execution_id) if payload.execution_id is not None else None,
        )
        return self._map_step(step, operation=SanctionedExternalWorkMutationOperation.CREATE)

    def _execute_accept(
        self,
        request: SanctionedExternalWorkOperationRequest,
    ) -> SanctionedExternalWorkOperationResult:
        payload = request.accept_quote
        if payload is None:
            return SanctionedExternalWorkOperationResult(
                admitted=False,
                reason="sanctioned_accept_payload_missing",
            )
        execution_id = str(payload.execution_id)
        create_result = self._resolve_prior_create_result(execution_id)
        if create_result is None:
            return SanctionedExternalWorkOperationResult(
                admitted=False,
                reason="prior_create_snapshot_missing",
                execution_id=execution_id,
            )
        step = self._orchestrator.accept(
            execution_id=execution_id,
            create_result=create_result,
            acceptance=payload.acceptance,
            idempotency_key=payload.acceptance_idempotency_key,
            principal_id=payload.principal_id,
            tenant_id=payload.tenant_id,
            metadata=dict(payload.metadata),
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
            return SanctionedExternalWorkOperationResult(
                admitted=False,
                reason="sanctioned_cancel_payload_missing",
            )
        execution_id = str(payload.execution_id)
        create_result = self._resolve_prior_create_result(execution_id)
        if create_result is None:
            return SanctionedExternalWorkOperationResult(
                admitted=False,
                reason="prior_create_snapshot_missing",
                execution_id=execution_id,
            )
        step = self._orchestrator.cancel(
            execution_id=execution_id,
            create_result=create_result,
            principal_id=payload.principal_id,
            tenant_id=payload.tenant_id,
            idempotency_key=payload.idempotency_key,
            metadata=dict(payload.metadata),
        )
        return self._map_step(step, operation=SanctionedExternalWorkMutationOperation.CANCEL)

    def _resolve_prior_create_result(
        self,
        execution_id: str,
    ) -> ExternalWorkAdapterResult | None:
        cached = self._create_results.get(execution_id)
        if cached is not None and cached.snapshot is not None:
            return cached
        return None

    def _map_step(
        self,
        step: OrchestratorStepResult,
        *,
        operation: SanctionedExternalWorkMutationOperation,
    ) -> SanctionedExternalWorkOperationResult:
        adapter = step.adapter_result
        if (
            operation is SanctionedExternalWorkMutationOperation.CREATE
            and adapter is not None
            and adapter.snapshot is not None
            and step.execution_id
        ):
            self._create_results[step.execution_id] = adapter
        provider_invocation_id: str | None = None
        if adapter is not None and adapter.metadata:
            raw = adapter.metadata.get(META_PROVIDER_INVOCATION_ID)
            if raw is not None and str(raw).strip():
                provider_invocation_id = str(raw).strip()
        domain_summary: dict[str, Any]
        if adapter is not None:
            domain_summary = dict(adapter.to_domain_summary())
        else:
            domain_summary = {"used": False, "reason": step.reason}
        domain_summary["sanctioned_operation"] = operation.value
        domain_summary["host_state"] = step.state.value
        if step.execution_id:
            domain_summary["execution_id"] = step.execution_id
        if provider_invocation_id:
            domain_summary["provider_invocation_id"] = provider_invocation_id
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
