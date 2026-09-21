# © Artur Czarnecki. All rights reserved.

"""Test doubles for orchestration consequential effect reliability (GR-10-R13)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from intergrax.contracts.orchestration_consequential_effect_reliability import (
    OrchestrationConsequentialEffectReliabilityPort,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
)
from intergrax.contracts.provider_invocation_store import (
    ProviderInvocationConflictError,
    ProviderInvocationOutcomeConflictError,
    provider_invocation_outcomes_equivalent,
    provider_invocations_equivalent,
)

T = TypeVar("T")


@dataclass
class PassthroughOrchestrationConsequentialEffectReliability(
    OrchestrationConsequentialEffectReliabilityPort,
):
    """Reliability boundary double — executes admitted effect without durability."""

    calls: int = 0
    last_idempotency_key: str | None = None

    async def execute_admitted_effect(
        self,
        *,
        slot_id: str,
        operation_id: str,
        idempotency_key: str,
        execute: Callable[[], Awaitable[T]],
    ) -> T:
        self.calls += 1
        self.last_idempotency_key = idempotency_key
        return await execute()


@dataclass
class RecordingOrchestrationConsequentialEffectReliability(
    OrchestrationConsequentialEffectReliabilityPort,
):
    """Records admitted operations for architecture / e2e proofs."""

    admitted: list[tuple[str, str, str]] = field(default_factory=list)

    async def execute_admitted_effect(
        self,
        *,
        slot_id: str,
        operation_id: str,
        idempotency_key: str,
        execute: Callable[[], Awaitable[T]],
    ) -> T:
        self.admitted.append((slot_id, operation_id, idempotency_key))
        return await execute()


class DurableTestProviderInvocationStore:
    """In-memory store marked durable for production composition proofs."""

    def __init__(self) -> None:
        self._invocations: dict[str, ProviderInvocation] = {}
        self._outcomes: dict[str, ProviderInvocationOutcome] = {}

    @property
    def is_durable(self) -> bool:
        return True

    def put_invocation(self, invocation: ProviderInvocation) -> None:
        existing = self._invocations.get(invocation.invocation_id)
        if existing is not None:
            if not provider_invocations_equivalent(existing, invocation):
                raise ProviderInvocationConflictError(
                    f"provider invocation conflict:{invocation.invocation_id}",
                )
            return
        self._invocations[invocation.invocation_id] = invocation

    def get_invocation(self, invocation_id: str) -> ProviderInvocation | None:
        return self._invocations.get(invocation_id)

    def put_outcome(self, outcome: ProviderInvocationOutcome) -> None:
        existing = self._outcomes.get(outcome.invocation_id)
        if existing is not None:
            if not provider_invocation_outcomes_equivalent(existing, outcome):
                raise ProviderInvocationOutcomeConflictError(
                    f"provider invocation outcome conflict:{outcome.invocation_id}",
                )
            return
        self._outcomes[outcome.invocation_id] = outcome

    def get_outcome(self, invocation_id: str) -> ProviderInvocationOutcome | None:
        return self._outcomes.get(invocation_id)


__all__ = [
    "DurableTestProviderInvocationStore",
    "PassthroughOrchestrationConsequentialEffectReliability",
    "RecordingOrchestrationConsequentialEffectReliability",
]
