# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Authoritative durable redrive effect binding on the tool external-operation path."""

from __future__ import annotations

from intergrax.contracts.external_operation_redrive_effect_binding import (
    ExternalOperationRedriveEffectBindingMismatchError,
)
from intergrax.contracts.external_operation_redrive_trusted_effect_contracts import (
    assert_sanctioned_class_b_catalog_tool_provenance,
    catalog_tool_has_sanctioned_redrive_effect_binding,
    require_complete_physical_effect_identity_values,
    resolve_sanctioned_trusted_redrive_effect_binding_for_catalog_tool,
    resolve_trusted_redrive_effect_contract,
    trusted_redrive_contract_requires_complete_physical_identity,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    ExternalOperationStateStore,
)
from intergrax.runtime.external_operations.redrive_external_operation_authority import (
    bind_authoritative_redrive_effect_to_operation,
)


def bind_sanctioned_redrive_effect_before_tool_physical_execution(
    store: ExternalOperationStateStore,
    *,
    tenant_id: str,
    operation_id: str,
    expected_revision: int,
    catalog_tool_id: str,
    catalog_tool_tags: tuple[str, ...] | None = None,
    physical_idempotency_key: str | None = None,
    physical_effect_fingerprint: str | None = None,
) -> int:
    """Record immutable trusted effect identity; returns updated store revision."""
    if not catalog_tool_has_sanctioned_redrive_effect_binding(catalog_tool_id):
        return expected_revision
    assert_sanctioned_class_b_catalog_tool_provenance(
        catalog_tool_id=catalog_tool_id,
        catalog_tool_tags=catalog_tool_tags,
    )
    contract_id, provider_id, binding_tool_id = (
        resolve_sanctioned_trusted_redrive_effect_binding_for_catalog_tool(
            catalog_tool_id,
        )
    )
    trusted = resolve_trusted_redrive_effect_contract(contract_id)
    if trusted_redrive_contract_requires_complete_physical_identity(trusted.contract_id):
        require_complete_physical_effect_identity_values(
            physical_idempotency_key=physical_idempotency_key,
            physical_effect_fingerprint=physical_effect_fingerprint,
        )
    updated = bind_authoritative_redrive_effect_to_operation(
        store,
        tenant_id=tenant_id,
        operation_id=operation_id,
        expected_revision=expected_revision,
        trusted_effect_contract_id=contract_id,
        provider_id=provider_id,
        tool_id=binding_tool_id,
        physical_idempotency_key=physical_idempotency_key,
        physical_effect_fingerprint=physical_effect_fingerprint,
    )
    return updated.revision


__all__ = ["bind_sanctioned_redrive_effect_before_tool_physical_execution"]
