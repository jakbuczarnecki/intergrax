# © Artur Czarnecki. All rights reserved.

"""Scenario-owned planner decision observability payloads (APP-2A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.runtime.nexus.tracing.trace_models import DEFAULT_REDACTED_TEXT, DiagnosticPayload


@dataclass(frozen=True, slots=True)
class IncidentPlannerDecisionDiagV1(DiagnosticPayload):
    """One material autonomous evidence-gathering planner decision."""

    round_index: int
    investigation_phase: str
    objective: str
    selected_tool_ids: tuple[str, ...]
    evidence_basis_tool_call_ids: tuple[str, ...]
    evidence_basis_evidence_ids: tuple[str, ...]
    incident_line_id: str
    incident_station_id: str

    @classmethod
    def schema_id(cls) -> str:
        return "incident.planner_decision.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "investigation_phase": self.investigation_phase,
            "objective": self.objective,
            "selected_tool_ids": list(self.selected_tool_ids),
            "evidence_basis_tool_call_ids": list(self.evidence_basis_tool_call_ids),
            "evidence_basis_evidence_ids": list(self.evidence_basis_evidence_ids),
            "incident_line_id": self.incident_line_id,
            "incident_station_id": self.incident_station_id,
        }

    def redact(self) -> IncidentPlannerDecisionDiagV1:
        return IncidentPlannerDecisionDiagV1(
            round_index=self.round_index,
            investigation_phase=self.investigation_phase,
            objective=DEFAULT_REDACTED_TEXT,
            selected_tool_ids=self.selected_tool_ids,
            evidence_basis_tool_call_ids=self.evidence_basis_tool_call_ids,
            evidence_basis_evidence_ids=self.evidence_basis_evidence_ids,
            incident_line_id=self.incident_line_id,
            incident_station_id=DEFAULT_REDACTED_TEXT,
        )


@dataclass(frozen=True, slots=True)
class IncidentPlannerStopDiagV1(DiagnosticPayload):
    """Structured stop semantics for autonomous evidence gathering."""

    investigation_phase: str
    stop_reason: str
    loop_iterations: int
    tool_invocations: int
    selected_tool_order: tuple[str, ...]

    @classmethod
    def schema_id(cls) -> str:
        return "incident.planner_stop.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "investigation_phase": self.investigation_phase,
            "stop_reason": self.stop_reason,
            "loop_iterations": self.loop_iterations,
            "tool_invocations": self.tool_invocations,
            "selected_tool_order": list(self.selected_tool_order),
        }

    def redact(self) -> IncidentPlannerStopDiagV1:
        return self


@dataclass(frozen=True, slots=True)
class IncidentScopeRejectionDiagV1(DiagnosticPayload):
    """Planner-selected tool args rejected before provider execution."""

    tool_id: str
    rejection_code: str
    investigation_phase: str

    @classmethod
    def schema_id(cls) -> str:
        return "incident.scope_rejection.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "rejection_code": self.rejection_code,
            "investigation_phase": self.investigation_phase,
        }

    def redact(self) -> IncidentScopeRejectionDiagV1:
        return self


@dataclass(frozen=True, slots=True)
class IncidentReasoningUpdateDiagV1(DiagnosticPayload):
    investigation_phase: str
    preferred_hypothesis_id: str
    uncertainty_class: str
    hypothesis_count: int
    information_gap_count: int

    @classmethod
    def schema_id(cls) -> str:
        return "incident.reasoning_update.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "investigation_phase": self.investigation_phase,
            "preferred_hypothesis_id": self.preferred_hypothesis_id,
            "uncertainty_class": self.uncertainty_class,
            "hypothesis_count": self.hypothesis_count,
            "information_gap_count": self.information_gap_count,
        }

    def redact(self) -> IncidentReasoningUpdateDiagV1:
        return IncidentReasoningUpdateDiagV1(
            investigation_phase=self.investigation_phase,
            preferred_hypothesis_id=self.preferred_hypothesis_id,
            uncertainty_class=DEFAULT_REDACTED_TEXT,
            hypothesis_count=self.hypothesis_count,
            information_gap_count=self.information_gap_count,
        )


@dataclass(frozen=True, slots=True)
class IncidentEvidenceGapDiagV1(DiagnosticPayload):
    investigation_phase: str
    information_gap: str

    @classmethod
    def schema_id(cls) -> str:
        return "incident.evidence_gap.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "investigation_phase": self.investigation_phase,
            "information_gap": self.information_gap,
        }

    def redact(self) -> IncidentEvidenceGapDiagV1:
        return IncidentEvidenceGapDiagV1(
            investigation_phase=self.investigation_phase,
            information_gap=DEFAULT_REDACTED_TEXT,
        )


@dataclass(frozen=True, slots=True)
class IncidentClaimProposedDiagV1(DiagnosticPayload):
    claim_id: str
    statement: str
    supporting_evidence_count: int

    @classmethod
    def schema_id(cls) -> str:
        return "incident.claim_proposed.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "supporting_evidence_count": self.supporting_evidence_count,
        }

    def redact(self) -> IncidentClaimProposedDiagV1:
        return IncidentClaimProposedDiagV1(
            claim_id=self.claim_id,
            statement=DEFAULT_REDACTED_TEXT,
            supporting_evidence_count=self.supporting_evidence_count,
        )


@dataclass(frozen=True, slots=True)
class IncidentClaimRevisedDiagV1(DiagnosticPayload):
    claim_id: str
    statement: str
    critic_feedback: tuple[str, ...]

    @classmethod
    def schema_id(cls) -> str:
        return "incident.claim_revised.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "critic_feedback": list(self.critic_feedback),
        }

    def redact(self) -> IncidentClaimRevisedDiagV1:
        return IncidentClaimRevisedDiagV1(
            claim_id=self.claim_id,
            statement=DEFAULT_REDACTED_TEXT,
            critic_feedback=self.critic_feedback,
        )


@dataclass(frozen=True, slots=True)
class IncidentCompletionIntentDiagV1(DiagnosticPayload):
    completion_intent: str
    unresolved_reason: str | None

    @classmethod
    def schema_id(cls) -> str:
        return "incident.completion_intent.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "completion_intent": self.completion_intent,
            "unresolved_reason": self.unresolved_reason,
        }

    def redact(self) -> IncidentCompletionIntentDiagV1:
        return IncidentCompletionIntentDiagV1(
            completion_intent=self.completion_intent,
            unresolved_reason=DEFAULT_REDACTED_TEXT if self.unresolved_reason else None,
        )
