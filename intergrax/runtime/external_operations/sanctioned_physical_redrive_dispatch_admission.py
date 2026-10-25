# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Authoritative dispatch admission for sanctioned Class B physical redrive."""

from __future__ import annotations

from pydantic import BaseModel

from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationIntentState,
    ExternalOperationPhysicalState,
    ExternalOperationState,
)
from intergrax.contracts.external_operation_redrive_effect_binding import (
    ExternalOperationRedriveEffectBindingMismatchError,
    assert_supplied_contract_matches_bound_identity,
)
from intergrax.contracts.external_operation_redrive_trusted_effect_contracts import (
    assert_sanctioned_class_b_catalog_tool_provenance,
    resolve_sanctioned_trusted_redrive_effect_binding_for_catalog_tool,
)
from intergrax.runtime.external_operations.certified_owner_loss_composition import (
    bound_worker_execution_epoch_is_revoked,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    ExternalOperationStateStore,
)
from intergrax.runtime.external_operations.redrive_external_operation_authority import (
    _bound_identity_from_state,
    _registry_contract_matching_binding,
)
from intergrax.runtime.tools.operation_identity import compute_invocation_operation_identity
from intergrax.tools.registry.runtime import RegisteredTool


class SanctionedPhysicalRedriveDispatchRejectedError(RuntimeError):
    """Recovery dispatch does not match durable binding or redrive admission."""


def assert_sanctioned_physical_redrive_dispatch_admitted(
    store: ExternalOperationStateStore,
    *,
    tenant_id: str,
    durable_operation_id: str,
    catalog_tool_id: str,
    registered_tool: RegisteredTool,
    physical_idempotency_key: str | None,
    tool_input: BaseModel,
    physical_attempt_sequence: int,
) -> ExternalOperationState:
    """
    Fail closed unless the requested tool dispatch matches the durable Class B binding.

    Recovery authority for effect A must never authorize physical effect B.
    """
    if physical_attempt_sequence < 2:
        raise ValueError("physical_attempt_sequence must be >= 2 for sanctioned redrive")
    normalized_op = durable_operation_id.strip()
    if not normalized_op:
        raise ValueError("durable_operation_id is required for sanctioned redrive")
    normalized_tenant = tenant_id.strip()
    if not normalized_tenant:
        raise SanctionedPhysicalRedriveDispatchRejectedError(
            "tenant_id is required for sanctioned physical redrive",
        )
    record = store.load(normalized_tenant, normalized_op)
    if record is None:
        raise SanctionedPhysicalRedriveDispatchRejectedError(
            "durable external operation not found for redrive",
        )
    if record.tenant_id != normalized_tenant:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "tenant_id does not match durable operation record",
        )
    if record.operation_id != normalized_op:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "durable_operation_id does not match durable operation record",
        )
    if not record.consumed_redrive_admission_ids:
        raise SanctionedPhysicalRedriveDispatchRejectedError(
            "redrive authority not transferred",
        )
    if record.physical_state is not ExternalOperationPhysicalState.NOT_STARTED:
        raise SanctionedPhysicalRedriveDispatchRejectedError(
            f"redrive requires physical NOT_STARTED, got {record.physical_state.value}",
        )
    if record.intent_state is not ExternalOperationIntentState.ACTIVE:
        raise SanctionedPhysicalRedriveDispatchRejectedError(
            f"redrive requires active intent, got {record.intent_state.value}",
        )
    if bound_worker_execution_epoch_is_revoked(current=record):
        raise SanctionedPhysicalRedriveDispatchRejectedError(
            "revoked worker execution epoch cannot dispatch sanctioned physical redrive",
        )
    if record.bound_redrive_effect_contract_digest is None:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "durable operation lacks Class B redrive effect binding",
        )

    normalized_catalog = catalog_tool_id.strip()
    if registered_tool.contract.tool_id != normalized_catalog:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "registered catalog tool does not match redrive request tool_id",
        )
    assert_sanctioned_class_b_catalog_tool_provenance(
        catalog_tool_id=normalized_catalog,
        registered_tool=registered_tool,
    )
    bound_contract_id, bound_provider_id, bound_binding_tool_id = (
        resolve_sanctioned_trusted_redrive_effect_binding_for_catalog_tool(
            normalized_catalog,
        )
    )
    if record.bound_redrive_effect_contract_id != bound_contract_id:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "catalog tool redrive contract does not match durable operation binding",
        )
    if record.bound_redrive_provider_id != bound_provider_id:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "catalog tool provider does not match durable operation binding",
        )
    if record.bound_redrive_tool_id != bound_binding_tool_id:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "catalog tool execution identity does not match durable operation binding",
        )

    effect_fingerprint = compute_invocation_operation_identity(
        normalized_catalog,
        tool_input,
    ).operation_fingerprint
    bound = _bound_identity_from_state(record)
    registry_contract = _registry_contract_matching_binding(bound)
    assert_supplied_contract_matches_bound_identity(
        bound=bound,
        supplied_contract=registry_contract,
        provider_id=bound_provider_id,
        tool_id=bound_binding_tool_id,
        physical_idempotency_key=physical_idempotency_key,
        physical_effect_fingerprint=effect_fingerprint,
    )
    return record


__all__ = [
    "SanctionedPhysicalRedriveDispatchRejectedError",
    "assert_sanctioned_physical_redrive_dispatch_admitted",
]
