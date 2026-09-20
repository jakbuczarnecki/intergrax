# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Post-admission reliability for orchestration consequential physical effects (GR-10-R13)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Protocol, TypeVar

T = TypeVar("T")


class OrchestrationConsequentialEffectReliabilityOutcome(StrEnum):
    """Typed reliability classification after Governance admission — not permission."""

    SUCCEEDED = "SUCCEEDED"
    DEFINITIVE_FAILURE = "DEFINITIVE_FAILURE"
    UNKNOWN = "UNKNOWN"


class OrchestrationConsequentialEffectReliabilityPort(Protocol):
    """Canonical boundary between Governance ALLOW and provider/slot physical effect."""

    async def execute_admitted_effect(
        self,
        *,
        slot_id: str,
        operation_id: str,
        idempotency_key: str,
        execute: Callable[[], Awaitable[T]],
    ) -> T:
        """Run one admitted consequential effect with durable intent and classified outcome."""


__all__ = [
    "OrchestrationConsequentialEffectReliabilityOutcome",
    "OrchestrationConsequentialEffectReliabilityPort",
]
