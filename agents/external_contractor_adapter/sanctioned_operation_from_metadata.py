# © Artur Czarnecki. All rights reserved.

"""Map reflex step metadata → typed sanctioned external-work operation requests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from external_contractor_adapter.external_work_adapter import (
    META_ACCEPTANCE_IDEMPOTENCY_KEY,
    META_IDEMPOTENCY_KEY,
    META_PRINCIPAL_ID,
    META_QUOTE_ACCEPTANCE,
    META_TENANT_ID,
    _optional_meta_str,
    _parse_acceptance,
    _require_meta_str,
)
from intergrax.contracts.execution_identity import (
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.sanctioned_external_work_operation import (
    META_PRIOR_EXECUTION_ID,
    META_SANCTIONED_MUTATION_OPERATION,
    SanctionedExternalWorkAcceptQuoteRequest,
    SanctionedExternalWorkCancelRequest,
    SanctionedExternalWorkCreateRequest,
    SanctionedExternalWorkMutationOperation,
    SanctionedExternalWorkOperationRequest,
)


def resolve_sanctioned_operation_from_metadata(
    metadata: Mapping[str, Any],
    *,
    task_id: str,
    run_id: str | None,
    message: str,
    default_operation: SanctionedExternalWorkMutationOperation = (
        SanctionedExternalWorkMutationOperation.CREATE
    ),
) -> SanctionedExternalWorkOperationRequest:
    raw_op = metadata.get(META_SANCTIONED_MUTATION_OPERATION)
    if raw_op is None or not str(raw_op).strip():
        operation = default_operation
    else:
        operation = SanctionedExternalWorkMutationOperation(str(raw_op).strip())
    principal_id = _require_meta_str(metadata, META_PRINCIPAL_ID, field="principal_id")
    tenant_id = _require_meta_str(metadata, META_TENANT_ID, field="tenant_id")
    validated_task = validate_task_id(task_id)
    validated_run = validate_run_id(run_id or task_id)

    if operation is SanctionedExternalWorkMutationOperation.CREATE:
        acceptance = _parse_acceptance(metadata.get(META_QUOTE_ACCEPTANCE))
        if acceptance is not None:
            raise ValueError(
                "create_operation_with_acceptance_evidence_use_accept_quote_operation",
            )
        execution_raw = metadata.get(META_PRIOR_EXECUTION_ID)
        execution_id = (
            validate_execution_id(str(execution_raw).strip())
            if execution_raw is not None and str(execution_raw).strip()
            else None
        )
        return SanctionedExternalWorkOperationRequest(
            operation=operation,
            create=SanctionedExternalWorkCreateRequest(
                task_id=validated_task,
                run_id=validated_run,
                principal_id=principal_id,
                tenant_id=tenant_id,
                execution_id=execution_id,
                metadata=dict(metadata),
                message=message,
            ),
        )

    prior_execution = metadata.get(META_PRIOR_EXECUTION_ID)
    if prior_execution is None or not str(prior_execution).strip():
        raise ValueError("prior_execution_id_required_for_follow_up_mutation")
    execution_id = validate_execution_id(str(prior_execution).strip())

    if operation is SanctionedExternalWorkMutationOperation.ACCEPT_QUOTE:
        acceptance = _parse_acceptance(metadata.get(META_QUOTE_ACCEPTANCE))
        if acceptance is None:
            raise ValueError("quote_acceptance_evidence_required")
        acceptance_key = _optional_meta_str(metadata, META_ACCEPTANCE_IDEMPOTENCY_KEY)
        if not acceptance_key:
            raise ValueError("acceptance_idempotency_key_required")
        return SanctionedExternalWorkOperationRequest(
            operation=operation,
            accept_quote=SanctionedExternalWorkAcceptQuoteRequest(
                execution_id=execution_id,
                principal_id=principal_id,
                tenant_id=tenant_id,
                acceptance=acceptance,
                acceptance_idempotency_key=acceptance_key,
                metadata=dict(metadata),
            ),
        )

    if operation is SanctionedExternalWorkMutationOperation.CANCEL:
        idem = _optional_meta_str(metadata, META_IDEMPOTENCY_KEY)
        if not idem:
            raise ValueError("cancel_idempotency_key_required")
        return SanctionedExternalWorkOperationRequest(
            operation=operation,
            cancel=SanctionedExternalWorkCancelRequest(
                execution_id=execution_id,
                principal_id=principal_id,
                tenant_id=tenant_id,
                idempotency_key=idem,
                metadata=dict(metadata),
            ),
        )

    raise ValueError(f"unsupported_sanctioned_operation:{operation}")


__all__ = ["resolve_sanctioned_operation_from_metadata"]
