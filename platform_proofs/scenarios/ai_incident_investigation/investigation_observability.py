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
