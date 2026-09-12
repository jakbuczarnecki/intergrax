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


@dataclass(frozen=True, slots=True)
class RelatedSelfHealingWorkflowHistoryEntryView:
    """Workflow lifecycle trail on DiagnosticInvestigationView — read-only."""

    workflow_id: str
    strategy_id: str
    plan_id: str
    tenant_id: str
    from_state: str
    to_state: str
    actor: str
    reason: str
    recorded_at: datetime


__all__ = [
    "RelatedSelfHealingHistoryEntryView",
    "RelatedSelfHealingWorkflowHistoryEntryView",
]
