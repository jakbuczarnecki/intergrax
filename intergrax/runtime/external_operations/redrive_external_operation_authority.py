# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""TL8 sanctioned redrive authority transfer after reconciliation admission."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.effect_contract import ExternalEffectContract
from intergrax.contracts.external_operation_cancellation import (
    ExternalOperationPhysicalState,
)
from intergrax.contracts.external_operation_redrive_effect_binding import (
    BoundExternalOperationRedriveEffectIdentity,
    ExternalOperationRedriveEffectBindingImmutableError,
    ExternalOperationRedriveEffectBindingMismatchError,
    ExternalOperationRedriveEffectBindingError,
    assert_supplied_contract_matches_bound_identity,
    bound_identity_from_trusted_contract,
    compute_trusted_effect_contract_binding_digest,
)
from intergrax.contracts.external_operation_redrive_authority import (
    ExternalOperationRedriveEffectClass,
    ExternalOperationRedriveEffectClassRejectedError,
    assert_redrive_effect_class_claim_matches_authoritative,
    resolve_authoritative_redrive_effect_class,
)
from intergrax.contracts.external_operation_redrive_trusted_effect_contracts import (
    assert_trusted_redrive_binding_matches_execution_context,
    resolve_trusted_redrive_effect_contract,
)
from intergrax.runtime.external_operations.external_operation_state_store import (
    ExternalOperationStateStore,
    ExternalOperationStateUpdate,
    StaleExternalOperationStateError,
)
from intergrax.runtime.external_operations.recovery_external_operation_gate import (
    ExternalOperationReconciliationAdmissionError,
    admit_unknown_external_operation_for_reconciliation,
)
from intergrax.contracts.external_operation_cancellation import ExternalOperationState


def _redrive_admission_id(
    *,
    tenant_id: str,
    operation_id: str,
    store_revision: int,
) -> str:
    return f"redrive-admit:{tenant_id}:{operation_id}:rev{store_revision}"


def _bound_identity_from_state(
    state: ExternalOperationState,
) -> BoundExternalOperationRedriveEffectIdentity:
    if not state.bound_redrive_effect_contract_digest:
        raise ExternalOperationRedriveEffectBindingError(
            f"operation {state.operation_id!r} lacks durable redrive effect binding",
        )
    return BoundExternalOperationRedriveEffectIdentity(
        contract_id=state.bound_redrive_effect_contract_id or "",
        operation_key=state.bound_redrive_effect_operation_key or "",
        contract_digest=state.bound_redrive_effect_contract_digest,
        provider_id=state.bound_redrive_provider_id,
        tool_id=state.bound_redrive_tool_id,
        physical_idempotency_key=state.bound_physical_idempotency_key,
        physical_effect_fingerprint=state.bound_physical_effect_fingerprint,
        binding_revision=state.bound_redrive_effect_binding_revision or 1,
    )


def _registry_contract_matching_binding(
    bound: BoundExternalOperationRedriveEffectIdentity,
) -> ExternalEffectContract:
    trusted = resolve_trusted_redrive_effect_contract(bound.contract_id)
    registry_identity = bound_identity_from_trusted_contract(
        trusted,
        provider_id=bound.provider_id,
        tool_id=bound.tool_id,
        physical_idempotency_key=bound.physical_idempotency_key,
        physical_effect_fingerprint=bound.physical_effect_fingerprint,
    )
    if registry_identity.contract_digest != bound.contract_digest:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "durable binding digest is not in the trusted registry",
        )
    if registry_identity.operation_key != bound.operation_key:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "durable binding operation_key is not in the trusted registry",
        )
    return trusted


def bind_authoritative_redrive_effect_to_operation(
    store: ExternalOperationStateStore,
    *,
    tenant_id: str,
    operation_id: str,
    expected_revision: int,
    trusted_effect_contract_id: str,
    provider_id: str | None = None,
    tool_id: str | None = None,
    physical_idempotency_key: str | None = None,
    physical_effect_fingerprint: str | None = None,
) -> ExternalOperationState:
    """Record immutable authoritative effect identity before physical execution."""
    trusted = resolve_trusted_redrive_effect_contract(trusted_effect_contract_id)
    assert_trusted_redrive_binding_matches_execution_context(
        trusted,
        provider_id=provider_id,
        tool_id=tool_id,
    )
    identity = bound_identity_from_trusted_contract(
        trusted,
        provider_id=provider_id,
        tool_id=tool_id,
        physical_idempotency_key=physical_idempotency_key,
        physical_effect_fingerprint=physical_effect_fingerprint,
    )
    current = store.load(tenant_id, operation_id)
    if current is None:
        raise ExternalOperationRedriveEffectBindingError(
            f"external operation {operation_id!r} not found",
        )
    if current.tenant_id != tenant_id:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "tenant_id does not match durable operation record",
        )
    if current.operation_id != operation_id:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "operation_id does not match durable operation record",
        )
    if current.bound_redrive_effect_contract_digest is not None:
        existing = _bound_identity_from_state(current)
        if (
            existing.contract_digest != identity.contract_digest
            or existing.provider_id != identity.provider_id
            or existing.tool_id != identity.tool_id
            or existing.physical_idempotency_key != identity.physical_idempotency_key
            or existing.physical_effect_fingerprint != identity.physical_effect_fingerprint
        ):
            raise ExternalOperationRedriveEffectBindingImmutableError(
                "redrive effect binding is immutable once recorded",
            )
        return current
    if current.physical_state is not ExternalOperationPhysicalState.NOT_STARTED:
        raise ExternalOperationRedriveEffectBindingImmutableError(
            "redrive effect binding must be recorded before physical execution",
        )
    return store.compare_and_set(
        tenant_id,
        operation_id,
        expected_revision=expected_revision,
        update=ExternalOperationStateUpdate(
            bound_redrive_effect_contract_id=identity.contract_id,
            bound_redrive_effect_operation_key=identity.operation_key,
            bound_redrive_effect_contract_digest=identity.contract_digest,
            bound_redrive_provider_id=identity.provider_id,
            bound_redrive_tool_id=identity.tool_id,
            bound_physical_idempotency_key=identity.physical_idempotency_key,
            bound_physical_effect_fingerprint=identity.physical_effect_fingerprint,
            bound_redrive_effect_binding_revision=identity.binding_revision,
        ),
    )


def transfer_redrive_authority_after_reconciliation(
    store: ExternalOperationStateStore,
    *,
    tenant_id: str,
    operation_id: str,
    expected_revision: int,
    trusted_effect_contract: ExternalEffectContract,
    worker_instance_id: str,
    worker_ownership_epoch: int,
    caller_claimed_effect_class: ExternalOperationRedriveEffectClass | None = None,
    provider_id: str | None = None,
    tool_id: str | None = None,
    physical_idempotency_key: str | None = None,
    physical_effect_fingerprint: str | None = None,
) -> ExternalOperationState:
    """Atomic durable authority transfer for eligible Class B effects (qual + controlled)."""
    current = store.load(tenant_id, operation_id)
    if current is None:
        raise ExternalOperationRedriveEffectBindingError(
            f"external operation {operation_id!r} not found",
        )
    if current.tenant_id != tenant_id:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "tenant_id does not match durable operation record",
        )
    bound = _bound_identity_from_state(current)
    assert_supplied_contract_matches_bound_identity(
        bound=bound,
        supplied_contract=trusted_effect_contract,
        provider_id=provider_id,
        tool_id=tool_id,
        physical_idempotency_key=physical_idempotency_key,
        physical_effect_fingerprint=physical_effect_fingerprint,
    )
    registry_contract = _registry_contract_matching_binding(bound)
    if (
        compute_trusted_effect_contract_binding_digest(trusted_effect_contract)
        != compute_trusted_effect_contract_binding_digest(registry_contract)
    ):
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "supplied contract does not match trusted registry entry",
        )
    authoritative = resolve_authoritative_redrive_effect_class(registry_contract)
    if caller_claimed_effect_class is not None:
        assert_redrive_effect_class_claim_matches_authoritative(
            caller_claimed_effect_class=caller_claimed_effect_class,
            authoritative_effect_class=authoritative,
        )
    if authoritative is not ExternalOperationRedriveEffectClass.CLASS_B_IDEMPOTENT_OR_RECONCILED:
        raise ExternalOperationRedriveEffectClassRejectedError(
            f"trusted contract {registry_contract.contract_id!r} is not eligible "
            f"for automated redrive ({authoritative.value})",
        )
    admit_unknown_external_operation_for_reconciliation(
        store,
        tenant_id=tenant_id,
        operation_id=operation_id,
    )
    admission_id = _redrive_admission_id(
        tenant_id=tenant_id,
        operation_id=operation_id,
        store_revision=expected_revision,
    )
    try:
        return store.transfer_redrive_worker_execution_authority(
            tenant_id,
            operation_id,
            expected_revision=expected_revision,
            worker_instance_id=worker_instance_id,
            worker_ownership_epoch=worker_ownership_epoch,
            redrive_admission_id=admission_id,
        )
    except StaleExternalOperationStateError:
        raise


__all__ = [
    "ExternalOperationReconciliationAdmissionError",
    "bind_authoritative_redrive_effect_to_operation",
    "transfer_redrive_authority_after_reconciliation",
]
