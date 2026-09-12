# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Preventive action audit trail (PREVENTIVE R7)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class PreventiveActionOutcome(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    DENIED = "DENIED"
    REJECTED = "REJECTED"
    PENDING = "PENDING"


@dataclass(frozen=True, slots=True)
class PreventiveActionAuditRecord:
    """Answers: why did the platform perform (or refuse) this preventive change?"""

    proposal_id: str
    tenant_id: str
    risk_signal_refs: tuple[str, ...]
    recommendation_refs: tuple[str, ...]
    approval_refs: tuple[str, ...]
    external_operation_id: str | None
    execution_id: str | None
    outcome: PreventiveActionOutcome
    recorded_at: datetime
    action_type: str = ""
    admission_reason: str = ""

    def __post_init__(self) -> None:
        if not self.proposal_id.startswith("pract_prop_"):
            raise ValueError("proposal_id must be pract_prop_*")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.risk_signal_refs:
            raise ValueError("risk_signal_refs required")


__all__ = ["PreventiveActionAuditRecord", "PreventiveActionOutcome"]
