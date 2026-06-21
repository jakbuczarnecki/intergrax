# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Trace timeline descriptive facets (HEP Band 2ae · EVID-TRACE-02)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class TracePolicyOutcome(str, Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    HITL_REQUIRED = "hitl_required"


class TraceBudgetStatus(str, Enum):
    OK = "ok"
    WARN = "warn"
    EXCEEDED = "exceeded"


class TraceHitlStatus(str, Enum):
    REQUESTED = "requested"
    WAITING = "waiting"
    APPROVED = "approved"
    REJECTED = "rejected"


class TraceEvidenceOrigin(str, Enum):
    MOCK = "mock"
    TRACE = "trace"
    REPORT = "report"
    RUNTIME_EVENT = "runtime_event"


class TraceScenarioPhase(str, Enum):
    STARTED = "started"
    EVIDENCE = "evidence"
    PASSED = "passed"
    FAILED = "failed"


class TracePolicyFacet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: TracePolicyOutcome
    tool_or_action: str = ""
    reason: str = ""


class TraceBudgetFacet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: TraceBudgetStatus
    metric: str = ""
    value: str = ""
    limit: str = ""


class TraceHitlFacet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: TraceHitlStatus
    subject: str = ""
    reason: str = ""


class TraceEvidenceFacet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin: TraceEvidenceOrigin
    ref: str = ""
    description: str = ""


class TraceScenarioLifecycleFacet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: TraceScenarioPhase
    scenario_id: str


class TraceTimelineEventFacets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: TracePolicyFacet | None = None
    budget: TraceBudgetFacet | None = None
    hitl: TraceHitlFacet | None = None
    evidence: TraceEvidenceFacet | None = None
    scenario_lifecycle: TraceScenarioLifecycleFacet | None = None


def validate_trace_timeline_event_facets(event: Any) -> None:
    """Raise ``ValueError`` when ``event.facets`` violates EVID-TRACE-02 rules."""
    facets: TraceTimelineEventFacets | None = getattr(event, "facets", None)
    if facets is None:
        return

    from intergrax.runtime.evidence.trace_timeline_contracts import TraceTimelineEventKind

    if facets.evidence is not None:
        if event.kind is not TraceTimelineEventKind.EVIDENCE_EMITTED:
            raise ValueError(
                "evidence facet is only allowed on EVIDENCE_EMITTED events"
            )
        if not facets.evidence.ref.strip():
            raise ValueError("evidence facet ref must not be empty")

    if facets.scenario_lifecycle is not None:
        lifecycle = facets.scenario_lifecycle
        if not lifecycle.scenario_id.strip():
            raise ValueError("scenario_lifecycle scenario_id must not be empty")

        if event.scenario_id is not None and event.scenario_id != lifecycle.scenario_id:
            raise ValueError(
                "scenario_lifecycle scenario_id must match event.scenario_id"
            )

        expected_phase_by_kind = {
            TraceTimelineEventKind.SCENARIO_STARTED: TraceScenarioPhase.STARTED,
            TraceTimelineEventKind.EVIDENCE_EMITTED: TraceScenarioPhase.EVIDENCE,
            TraceTimelineEventKind.SCENARIO_PASSED: TraceScenarioPhase.PASSED,
            TraceTimelineEventKind.SCENARIO_FAILED: TraceScenarioPhase.FAILED,
        }
        expected_phase = expected_phase_by_kind.get(event.kind)
        if expected_phase is None or lifecycle.phase is not expected_phase:
            raise ValueError(
                f"scenario_lifecycle phase {lifecycle.phase.value} "
                f"does not match event kind {event.kind.value}"
            )
