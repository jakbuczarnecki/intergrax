# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable redrive effect identity bound to external operations (G-A02 / EXT-L4)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectContract,
    validate_external_effect_contract,
)
from intergrax.contracts.external_operation_redrive_authority import (
    ExternalOperationRedriveAuthorityError,
)

_HTTP_IDEMPOTENT_POST_QUAL_CONTRACT_ID = "tool_life.http_idempotent_post.qual.v1"


def _bound_contract_requires_complete_physical_identity(contract_id: str) -> bool:
    return contract_id.strip() == _HTTP_IDEMPOTENT_POST_QUAL_CONTRACT_ID


class ExternalOperationRedriveEffectBindingError(ExternalOperationRedriveAuthorityError):
    """Durable effect binding is missing or inconsistent."""


class ExternalOperationRedriveEffectBindingMismatchError(
    ExternalOperationRedriveEffectBindingError,
):
    """Supplied contract does not match the operation's durable authoritative binding."""


class ExternalOperationRedriveEffectBindingImmutableError(
    ExternalOperationRedriveEffectBindingError,
):
    """Effect binding cannot be changed after it is recorded."""


@dataclass(frozen=True, slots=True)
class BoundExternalOperationRedriveEffectIdentity:
    contract_id: str
    operation_key: str
    contract_digest: str
    provider_id: str | None = None
    tool_id: str | None = None
    physical_idempotency_key: str | None = None
    physical_effect_fingerprint: str | None = None
    binding_revision: int = 1


def compute_trusted_effect_contract_binding_digest(
    contract: ExternalEffectContract,
) -> str:
    validate_external_effect_contract(contract)
    payload = contract.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def bound_identity_from_trusted_contract(
    contract: ExternalEffectContract,
    *,
    provider_id: str | None = None,
    tool_id: str | None = None,
    physical_idempotency_key: str | None = None,
    physical_effect_fingerprint: str | None = None,
) -> BoundExternalOperationRedriveEffectIdentity:
    digest = compute_trusted_effect_contract_binding_digest(contract)
    return BoundExternalOperationRedriveEffectIdentity(
        contract_id=contract.contract_id,
        operation_key=contract.operation_key,
        contract_digest=digest,
        provider_id=provider_id,
        tool_id=tool_id,
        physical_idempotency_key=physical_idempotency_key,
        physical_effect_fingerprint=physical_effect_fingerprint,
    )


def assert_supplied_contract_matches_bound_identity(
    *,
    bound: BoundExternalOperationRedriveEffectIdentity,
    supplied_contract: ExternalEffectContract,
    provider_id: str | None = None,
    tool_id: str | None = None,
    physical_idempotency_key: str | None = None,
    physical_effect_fingerprint: str | None = None,
) -> None:
    supplied_digest = compute_trusted_effect_contract_binding_digest(supplied_contract)
    if supplied_contract.contract_id != bound.contract_id:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "contract_id does not match durable operation binding",
        )
    if supplied_contract.operation_key != bound.operation_key:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "operation_key does not match durable operation binding",
        )
    if supplied_digest != bound.contract_digest:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "contract digest does not match durable operation binding",
        )
    if bound.provider_id is not None and provider_id != bound.provider_id:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "provider_id does not match durable operation binding",
        )
    if bound.tool_id is not None and tool_id != bound.tool_id:
        raise ExternalOperationRedriveEffectBindingMismatchError(
            "tool_id does not match durable operation binding",
        )
    requires_identity = _bound_contract_requires_complete_physical_identity(
        bound.contract_id,
    )
    if requires_identity:
        if not bound.physical_idempotency_key or not bound.physical_effect_fingerprint:
            raise ExternalOperationRedriveEffectBindingMismatchError(
                "durable Class B binding lacks complete physical effect identity",
            )
        if physical_idempotency_key != bound.physical_idempotency_key:
            raise ExternalOperationRedriveEffectBindingMismatchError(
                "physical_idempotency_key does not match durable operation binding",
            )
        if physical_effect_fingerprint != bound.physical_effect_fingerprint:
            raise ExternalOperationRedriveEffectBindingMismatchError(
                "physical_effect_fingerprint does not match durable operation binding",
            )
    else:
        if bound.physical_idempotency_key is not None:
            if physical_idempotency_key != bound.physical_idempotency_key:
                raise ExternalOperationRedriveEffectBindingMismatchError(
                    "physical_idempotency_key does not match durable operation binding",
                )
        if bound.physical_effect_fingerprint is not None:
            if physical_effect_fingerprint != bound.physical_effect_fingerprint:
                raise ExternalOperationRedriveEffectBindingMismatchError(
                    "physical_effect_fingerprint does not match durable operation binding",
                )


__all__ = [
    "BoundExternalOperationRedriveEffectIdentity",
    "ExternalOperationRedriveEffectBindingError",
    "ExternalOperationRedriveEffectBindingImmutableError",
    "ExternalOperationRedriveEffectBindingMismatchError",
    "assert_supplied_contract_matches_bound_identity",
    "bound_identity_from_trusted_contract",
    "compute_trusted_effect_contract_binding_digest",
]
