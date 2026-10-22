# © Artur Czarnecki. All rights reserved.

"""Stable provider invocation identity for governed external-work CREATE (PROD-Q IMPL-01)."""

from __future__ import annotations

from intergrax.contracts.canonical_payload_hash import stable_payload_hash

_STABLE_CREATE_KIND = "governed_external_work_create.v1"


def derive_governed_external_work_create_provider_invocation_id(
    *,
    tenant_id: str,
    principal_id: str,
    execution_id: str,
    provider_id: str,
    idempotency_key: str,
) -> str:
    """
    Canonical CREATE ``invocation_id`` — tenant- and principal-scoped.

    One stable identity per authorized physical CREATE effect (execution + idempotency).
    Retries and process restarts must rebuild the same id without random minting.
    """
    tenant = tenant_id.strip()
    principal = principal_id.strip()
    execution = execution_id.strip()
    provider = provider_id.strip()
    idem = idempotency_key.strip()
    if not tenant:
        raise ValueError("tenant_id_required")
    if not principal:
        raise ValueError("principal_id_required")
    if not execution:
        raise ValueError("execution_id_required")
    if not provider:
        raise ValueError("provider_id_required")
    if not idem:
        raise ValueError("idempotency_key_required")
    digest = stable_payload_hash(
        {
            "kind": _STABLE_CREATE_KIND,
            "tenant_id": tenant,
            "principal_id": principal,
            "execution_id": execution,
            "provider_id": provider,
            "idempotency_key": idem,
        },
    )
    return f"gca-pinv:{digest}"


__all__ = ["derive_governed_external_work_create_provider_invocation_id"]
