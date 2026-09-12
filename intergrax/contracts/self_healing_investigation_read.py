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
class HealingExecutionTimelineStageView:
    """One stage on the operator healing execution timeline — read-only."""

    phase: str
    label: str
    recorded_at: datetime | None
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HealingExecutionTimelineView:
    """R3 healing execution lifecycle projection — not execution authority."""

    workflow_id: str
    strategy_id: str
    plan_id: str
    stages: tuple[HealingExecutionTimelineStageView, ...]


@dataclass(frozen=True, slots=True)
class AdaptiveHealingInsightView:
    """R4 adaptive strategy ranking projection — advisory only, not execution authority."""

    tenant_id: str
    status: str
    overall_confidence: float
    recommended_strategy_order: tuple[str, ...]
    strategy_ranking_summary: tuple[str, ...]
    confidence_explanation: str
    adaptive_insights: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()


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
    "AdaptiveHealingInsightView",
    "HealingExecutionTimelineStageView",
    "HealingExecutionTimelineView",
    "RelatedSelfHealingHistoryEntryView",
    "RelatedSelfHealingWorkflowHistoryEntryView",
]
