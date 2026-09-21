# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Platform provider invocation durability helpers (GR-7-A3 / GR-10-R13)."""

from __future__ import annotations

from datetime import datetime

from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.contracts.provider_invocation_store import (
    ProviderInvocationConflictError,
    ProviderInvocationOutcomeConflictError,
    ProviderInvocationPersistenceError,
    ProviderInvocationStore,
    ProviderInvocationStoreError,
    provider_invocation_outcomes_equivalent,
    provider_invocations_equivalent,
)
from intergrax.runtime.attestation.canonical_json import stable_payload_hash


def persist_provider_invocation_intent(
    store: ProviderInvocationStore,
    invocation: ProviderInvocation,
) -> None:
    """Write-before-effect: raises on conflict or infrastructure failure."""
    try:
        existing = store.get_invocation(invocation.invocation_id)
        if existing is not None:
            if provider_invocations_equivalent(existing, invocation):
                return
            raise ProviderInvocationConflictError(
                f"provider invocation conflict:{invocation.invocation_id}",
            )
        store.put_invocation(invocation)
    except ProviderInvocationStoreError:
        raise
    except Exception as exc:  # noqa: BLE001 — port boundary
        raise ProviderInvocationPersistenceError(str(exc)) from exc


def persist_provider_invocation_outcome(
    store: ProviderInvocationStore,
    outcome: ProviderInvocationOutcome,
) -> None:
    """Post-dispatch outcome durability — never retries provider."""
    try:
        existing = store.get_outcome(outcome.invocation_id)
        if existing is not None:
            if provider_invocation_outcomes_equivalent(existing, outcome):
                return
            raise ProviderInvocationOutcomeConflictError(
                f"provider invocation outcome conflict:{outcome.invocation_id}",
            )
        store.put_outcome(outcome)
    except ProviderInvocationStoreError:
        raise
    except Exception as exc:  # noqa: BLE001 — port boundary
        raise ProviderInvocationPersistenceError(str(exc)) from exc


def build_classified_provider_invocation_outcome(
    *,
    invocation_id: str,
    status: ProviderInvocationStatus,
    completed_at: datetime,
    operation_id: str,
    operation: str,
    detail: str = "",
) -> ProviderInvocationOutcome:
    return ProviderInvocationOutcome(
        invocation_id=invocation_id,
        status=status,
        completed_at=completed_at,
        response_digest=stable_payload_hash(
            {
                "operation_id": operation_id,
                "operation": operation,
                "invocation_status": status.value,
                "detail": detail,
            },
        ),
        external_status=None,
        error_code=None if status is ProviderInvocationStatus.SUCCEEDED else "orchestration_slot_effect",
    )


__all__ = [
    "build_classified_provider_invocation_outcome",
    "persist_provider_invocation_intent",
    "persist_provider_invocation_outcome",
]
