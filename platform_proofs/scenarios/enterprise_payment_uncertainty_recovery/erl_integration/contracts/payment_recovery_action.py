"""Scenario payment recovery execution — business actions behind platform recovery decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class PaymentRecoveryBusinessAction(StrEnum):
    """Enterprise workflow steps — not platform lifecycle verbs."""

    RESUME_FULFILLMENT = "resume_fulfillment"
    RELEASE_RESERVATION = "release_reservation"
    OPERATIONAL_FOLLOW_UP = "operational_follow_up"


class PaymentRecoveryExecutionStatus(StrEnum):
    """Scenario-local execution tracking — never exported to ``intergrax/``."""

    SUCCESS = "success"
    WAITING = "waiting"
    ESCALATED = "escalated"


@dataclass(frozen=True, slots=True)
class PaymentRecoveryActionResult:
    """Outcome of a scenario recovery action — complements platform ``RecoveryDecision``."""

    action: PaymentRecoveryBusinessAction
    status: PaymentRecoveryExecutionStatus
    detail: str = ""


class PaymentRecoveryActionPort(Protocol):
    """Execute payment recovery business steps — replaceable lab or production adapter."""

    def execute(
        self,
        *,
        correlation_id: str,
        tenant_id: str,
        action: PaymentRecoveryBusinessAction,
        resolution_rationale: str,
    ) -> PaymentRecoveryActionResult:
        """Perform the requested business recovery step without ERL lifecycle mutation."""
