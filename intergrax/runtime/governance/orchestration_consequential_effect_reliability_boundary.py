# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ProviderInvocation-backed orchestration slot reliability (GR-7 composition, GR-10-R13)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TypeVar

from intergrax.contracts.execution_identity import require_active_execution_identity
from intergrax.contracts.orchestration_consequential_effect_reliability import (
    OrchestrationConsequentialEffectReliabilityPort,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.contracts.provider_invocation_store import ProviderInvocationStore
from intergrax.runtime.attestation.canonical_json import stable_payload_hash
from intergrax.runtime.enterprise_reliability.provider_invocation_lifecycle import (
    build_classified_provider_invocation_outcome,
    persist_provider_invocation_intent,
    persist_provider_invocation_outcome,
)
from intergrax.runtime.enterprise_reliability.provider_invocation_reliability_early_lifecycle import (
    emit_dispatch_attempted,
    emit_intent_persisted,
    emit_outcome_persisted,
)

T = TypeVar("T")

_ORCHESTRATION_SLOT_PROVIDER_ID = "platform.orchestration.topology_slot"


class OrchestrationConsequentialEffectUncertaintyError(RuntimeError):
    """Physical effect may have occurred; outcome is UNKNOWN — no blind retry implied."""


class OrchestrationConsequentialEffectDefinitiveFailureError(RuntimeError):
    """Typed adapter/slot contract proves the physical effect did not occur."""


def orchestration_slot_invocation_id(
    *,
    tenant_id: str,
    provider_id: str,
    slot_id: str,
    idempotency_key: str,
) -> str:
    """Canonical invocation identity — tenant- and slot-scoped (no cross-tenant collision)."""
    digest = stable_payload_hash(
        {
            "tenant_id": tenant_id,
            "provider_id": provider_id,
            "slot_id": slot_id,
            "idempotency_key": idempotency_key,
        },
    )
    return f"orch-slot:{digest}"


@dataclass(frozen=True, slots=True)
class ProviderInvocationOrchestrationConsequentialEffectReliabilityBoundary(
    OrchestrationConsequentialEffectReliabilityPort,
):
    """Persist invocation intent, run admitted effect once, persist classified outcome."""

    store: ProviderInvocationStore
    clock: Callable[[], datetime]
    tenant_id: str = "platform"
    provider_id: str = _ORCHESTRATION_SLOT_PROVIDER_ID

    def _mint_invocation(
        self,
        *,
        slot_id: str,
        operation_id: str,
        idempotency_key: str,
    ) -> ProviderInvocation:
        run_id, _attempt_id = require_active_execution_identity()
        started_at = self.clock()
        request_digest = stable_payload_hash(
            {
                "tenant_id": self.tenant_id,
                "slot_id": slot_id,
                "operation_id": operation_id,
                "idempotency_key": idempotency_key,
            },
        )
        invocation_id = orchestration_slot_invocation_id(
            tenant_id=self.tenant_id,
            provider_id=self.provider_id,
            slot_id=slot_id,
            idempotency_key=idempotency_key,
        )
        return ProviderInvocation(
            invocation_id=invocation_id,
            provider_id=self.provider_id,
            operation=f"topology_slot:{slot_id}",
            task_id=operation_id,
            run_id=str(run_id),
            correlation_id=operation_id,
            idempotency_key=idempotency_key,
            request_digest=request_digest,
            started_at=started_at,
        )

    async def execute_admitted_effect(
        self,
        *,
        slot_id: str,
        operation_id: str,
        idempotency_key: str,
        execute: Callable[[], Awaitable[T]],
    ) -> T:
        invocation = self._mint_invocation(
            slot_id=slot_id,
            operation_id=operation_id,
            idempotency_key=idempotency_key,
        )
        prior_invocation = self.store.get_invocation(invocation.invocation_id)
        prior_outcome = self.store.get_outcome(invocation.invocation_id)
        if prior_outcome is not None:
            self._raise_for_persisted_outcome(prior_outcome)
        if prior_invocation is not None:
            raise OrchestrationConsequentialEffectUncertaintyError(
                "durable invocation intent without persisted outcome — effect may have occurred",
            )

        recorded_at = self.clock()
        persist_provider_invocation_intent(self.store, invocation)
        emit_intent_persisted(
            invocation=invocation,
            tenant_id=self.tenant_id,
            effect_contract_id=None,
            execution_id=None,
            attempt_id=None,
            recorded_at=recorded_at,
            observer=None,
        )
        emit_dispatch_attempted(
            invocation=invocation,
            tenant_id=self.tenant_id,
            effect_contract_id=None,
            execution_id=None,
            attempt_id=None,
            recorded_at=recorded_at,
            provider_mutation_attempted=True,
            observer=None,
        )

        status = ProviderInvocationStatus.SUCCEEDED
        detail = ""
        try:
            result = await execute()
        except OrchestrationConsequentialEffectDefinitiveFailureError as exc:
            status = ProviderInvocationStatus.FAILED
            detail = str(exc)
            self._persist_outcome(invocation, status, detail, operation_id)
            raise
        except TimeoutError as exc:
            status = ProviderInvocationStatus.UNKNOWN
            detail = str(exc)
            self._persist_outcome(invocation, status, detail, operation_id)
            raise OrchestrationConsequentialEffectUncertaintyError(detail) from exc
        except Exception as exc:
            status = ProviderInvocationStatus.UNKNOWN
            detail = str(exc)
            self._persist_outcome(invocation, status, detail, operation_id)
            raise OrchestrationConsequentialEffectUncertaintyError(detail) from exc

        self._persist_outcome(invocation, status, detail, operation_id)
        return result

    @staticmethod
    def _raise_for_persisted_outcome(outcome: ProviderInvocationOutcome) -> None:
        if outcome.status is ProviderInvocationStatus.SUCCEEDED:
            raise OrchestrationConsequentialEffectUncertaintyError(
                "orchestration slot effect already recorded as succeeded",
            )
        if outcome.status is ProviderInvocationStatus.FAILED:
            raise OrchestrationConsequentialEffectDefinitiveFailureError(
                "orchestration slot effect already recorded as not executed",
            )
        raise OrchestrationConsequentialEffectUncertaintyError(
            "orchestration slot effect outcome remains unknown",
        )

    def _persist_outcome(
        self,
        invocation: ProviderInvocation,
        status: ProviderInvocationStatus,
        detail: str,
        operation_id: str,
    ) -> None:
        outcome = build_classified_provider_invocation_outcome(
            invocation_id=invocation.invocation_id,
            status=status,
            completed_at=self.clock(),
            operation_id=operation_id,
            operation=invocation.operation,
            detail=detail,
        )
        persist_provider_invocation_outcome(self.store, outcome)
        emit_outcome_persisted(
            invocation=invocation,
            outcome=outcome,
            tenant_id=self.tenant_id,
            effect_contract_id=None,
            execution_id=None,
            recorded_at=self.clock(),
            observer=None,
        )


__all__ = [
    "OrchestrationConsequentialEffectDefinitiveFailureError",
    "OrchestrationConsequentialEffectUncertaintyError",
    "ProviderInvocationOrchestrationConsequentialEffectReliabilityBoundary",
    "orchestration_slot_invocation_id",
]
