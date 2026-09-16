# © Artur Czarnecki. All rights reserved.

"""Host-owned provider invocation durability (GR-7-A3)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar

from external_contractor_adapter.external_effect_outcome_projection import (
    ExternalWorkSideEffectObservation,
    project_external_work_side_effect_to_effect_outcome,
)
from external_contractor_adapter.schemas.adapt_result import ExternalWorkAdapterResult
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
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

T = TypeVar("T")

_EFFECT_TO_STATUS: dict[ExternalEffectOutcome, ProviderInvocationStatus] = {
    ExternalEffectOutcome.SUCCESS: ProviderInvocationStatus.SUCCEEDED,
    ExternalEffectOutcome.FAILURE: ProviderInvocationStatus.FAILED,
    ExternalEffectOutcome.UNKNOWN: ProviderInvocationStatus.UNKNOWN,
}


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


def classify_provider_invocation_status(
    *,
    adapter_result: ExternalWorkAdapterResult,
    provider_mutation_attempted: bool,
) -> ProviderInvocationStatus | None:
    if not provider_mutation_attempted:
        return None
    observation = ExternalWorkSideEffectObservation(
        provider_mutation_attempted=True,
        adapter_result=adapter_result,
        reason=adapter_result.reason or "",
    )
    effect = project_external_work_side_effect_to_effect_outcome(observation)
    if effect is None:
        return ProviderInvocationStatus.FAILED
    return _EFFECT_TO_STATUS[effect]


def build_provider_invocation_outcome(
    *,
    invocation_id: str,
    status: ProviderInvocationStatus,
    completed_at: datetime,
    adapter_result: ExternalWorkAdapterResult,
    execution_id: str,
    action: str,
) -> ProviderInvocationOutcome:
    return ProviderInvocationOutcome(
        invocation_id=invocation_id,
        status=status,
        completed_at=completed_at,
        response_digest=stable_payload_hash(
            {
                "execution_id": execution_id,
                "action": action,
                "status": (
                    adapter_result.status.value
                    if adapter_result.status is not None
                    else "unknown"
                ),
                "invocation_status": status.value,
            }
        ),
        external_status=(
            adapter_result.status.value if adapter_result.status is not None else None
        ),
        error_code=(
            adapter_result.error_code.value
            if adapter_result.error_code is not None
            else None
        ),
    )


@dataclass(frozen=True, slots=True)
class GovernedProviderInvocationDispatchGate:
    """Production dispatch gate: intent durability then provider execute."""

    store: ProviderInvocationStore

    def dispatch_after_intent_persisted(
        self,
        invocation: ProviderInvocation,
        execute: Callable[[], T],
    ) -> T:
        persist_provider_invocation_intent(self.store, invocation)
        return execute()


__all__ = [
    "GovernedProviderInvocationDispatchGate",
    "build_provider_invocation_outcome",
    "classify_provider_invocation_status",
    "persist_provider_invocation_intent",
    "persist_provider_invocation_outcome",
]
