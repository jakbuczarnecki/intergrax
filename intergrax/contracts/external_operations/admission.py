# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Operation admission boundary — decides; never executes (R1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.external_operations.intent import ExternalOperationIntent


class OperationAdmissionVerdict(StrEnum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"


class OperationAdmissionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: OperationAdmissionVerdict
    reason: str = Field(min_length=1, max_length=1024)
    approval_id: str | None = Field(default=None, min_length=1, max_length=128)
    decision_id: str | None = Field(default=None, min_length=1, max_length=128)
    risk_score: float | None = Field(default=None, ge=0.0, le=1.0)


@dataclass(frozen=True, slots=True)
class ExternalOperationAdmissionContext:
    """Governance and diagnostic hooks for admission evaluation."""

    tenant_id: str
    provider_id: str | None = None
    production_target: bool = False
    decision_id: str | None = None
    human_approval_granted: bool = False
    approval_id: str | None = None


@runtime_checkable
class ExternalOperationAdmission(Protocol):
    """Admission port — no provider I/O."""

    def evaluate(
        self,
        intent: ExternalOperationIntent,
        context: ExternalOperationAdmissionContext,
    ) -> OperationAdmissionDecision:
        """Return ALLOW, DENY, or REQUIRES_APPROVAL with operator-readable reason."""
        ...
