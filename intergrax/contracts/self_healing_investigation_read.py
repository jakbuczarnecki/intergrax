# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Operator investigation attachment for self-healing (SELF-HEALING R1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class RelatedSelfHealingHistoryEntryView:
    """Readonly self-healing trail on DiagnosticInvestigationView — not execution authority."""

    decision_id: str
    strategy_id: str
    tenant_id: str
    evidence_refs: tuple[str, ...]
    approval_refs: tuple[str, ...]
    external_operation_id: str | None
    execution_id: str | None
    outcome: str
    recorded_at: datetime
    action_type: str = ""
    admission_reason: str = ""
    justification: str = ""


__all__ = ["RelatedSelfHealingHistoryEntryView"]
