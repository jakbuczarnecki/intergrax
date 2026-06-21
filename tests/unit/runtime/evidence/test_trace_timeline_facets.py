# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.evidence.trace_timeline_contracts import (
    TraceTimeline,
    TraceTimelineEventKind,
    TraceTimelineKind,
    TraceTimelineSourceKind,
    TraceTimelineSourceRef,
    create_trace_timeline_event,
    validate_trace_timeline,
)
from intergrax.runtime.evidence.trace_timeline_facets import (
    TraceBudgetFacet,
    TraceBudgetStatus,
    TraceEvidenceFacet,
    TraceEvidenceOrigin,
    TraceHitlFacet,
    TraceHitlStatus,
    TracePolicyFacet,
    TracePolicyOutcome,
    TraceScenarioLifecycleFacet,
    TraceScenarioPhase,
    TraceTimelineEventFacets,
    validate_trace_timeline_event_facets,
)

pytestmark = pytest.mark.unit

_EVIDENCE_ROOT = Path(__file__).resolve().parents[3] / "intergrax" / "runtime" / "evidence"


def _evidence_event(
    *,
    sequence: int,
    scenario_id: str,
    facets: TraceTimelineEventFacets | None = None,
) -> object:
    return create_trace_timeline_event(
        kind=TraceTimelineEventKind.EVIDENCE_EMITTED,
        sequence=sequence,
        title="Evidence emitted",
        scenario_id=scenario_id,
        source_refs=[
            TraceTimelineSourceRef(
                kind=TraceTimelineSourceKind.EVIDENCE_REF,
                ref=f"mock:{scenario_id}",
            )
        ],
        facets=facets,
    )


def test_policy_denied_facet_passes_on_evidence_emitted_for_tool_denied_by_policy() -> None:
    event = _evidence_event(
        sequence=3,
        scenario_id="tool_denied_by_policy",
        facets=TraceTimelineEventFacets(
            policy=TracePolicyFacet(
                outcome=TracePolicyOutcome.DENIED,
                tool_or_action="dangerous_tool",
                reason="policy denied tool execution",
            )
        ),
    )
    validate_trace_timeline_event_facets(event)


def test_budget_exceeded_facet_passes_on_evidence_emitted_for_budget_exceeded_handled() -> None:
    event = _evidence_event(
        sequence=3,
        scenario_id="budget_exceeded_handled",
        facets=TraceTimelineEventFacets(
            budget=TraceBudgetFacet(
                status=TraceBudgetStatus.EXCEEDED,
                metric="tokens",
                value="1200",
                limit="1000",
            )
        ),
    )
    validate_trace_timeline_event_facets(event)


def test_hitl_scenario_allows_policy_hitl_required_and_hitl_requested() -> None:
    event = _evidence_event(
        sequence=3,
        scenario_id="high_risk_tool_hitl",
        facets=TraceTimelineEventFacets(
            policy=TracePolicyFacet(
                outcome=TracePolicyOutcome.HITL_REQUIRED,
                tool_or_action="high_risk_tool",
            ),
            hitl=TraceHitlFacet(status=TraceHitlStatus.REQUESTED),
        ),
    )
    validate_trace_timeline_event_facets(event)


def test_evidence_mock_facet_passes_only_on_evidence_emitted() -> None:
    event = _evidence_event(
        sequence=3,
        scenario_id="basic_run_completed",
        facets=TraceTimelineEventFacets(
            evidence=TraceEvidenceFacet(
                origin=TraceEvidenceOrigin.MOCK,
                ref="mock:basic_run_completed",
            )
        ),
    )
    validate_trace_timeline_event_facets(event)


def test_evidence_facet_on_scenario_passed_raises_value_error() -> None:
    event = create_trace_timeline_event(
        kind=TraceTimelineEventKind.SCENARIO_PASSED,
        sequence=4,
        title="Scenario passed",
        scenario_id="basic_run_completed",
        facets=TraceTimelineEventFacets(
            evidence=TraceEvidenceFacet(
                origin=TraceEvidenceOrigin.MOCK,
                ref="mock:basic_run_completed",
            )
        ),
    )
    with pytest.raises(ValueError, match="evidence facet is only allowed on EVIDENCE_EMITTED"):
        validate_trace_timeline_event_facets(event)


def test_evidence_facet_with_empty_ref_raises_value_error() -> None:
    event = _evidence_event(
        sequence=3,
        scenario_id="basic_run_completed",
        facets=TraceTimelineEventFacets(
            evidence=TraceEvidenceFacet(
                origin=TraceEvidenceOrigin.MOCK,
                ref="   ",
            )
        ),
    )
    with pytest.raises(ValueError, match="evidence facet ref must not be empty"):
        validate_trace_timeline_event_facets(event)


def test_scenario_lifecycle_started_matches_scenario_started() -> None:
    event = create_trace_timeline_event(
        kind=TraceTimelineEventKind.SCENARIO_STARTED,
        sequence=2,
        title="Scenario started",
        scenario_id="basic_run_completed",
        facets=TraceTimelineEventFacets(
            scenario_lifecycle=TraceScenarioLifecycleFacet(
                phase=TraceScenarioPhase.STARTED,
                scenario_id="basic_run_completed",
            )
        ),
    )
    validate_trace_timeline_event_facets(event)


def test_scenario_lifecycle_passed_matches_scenario_passed() -> None:
    event = create_trace_timeline_event(
        kind=TraceTimelineEventKind.SCENARIO_PASSED,
        sequence=4,
        title="Scenario passed",
        scenario_id="basic_run_completed",
        facets=TraceTimelineEventFacets(
            scenario_lifecycle=TraceScenarioLifecycleFacet(
                phase=TraceScenarioPhase.PASSED,
                scenario_id="basic_run_completed",
            )
        ),
    )
    validate_trace_timeline_event_facets(event)


def test_scenario_lifecycle_failed_matches_scenario_failed() -> None:
    event = create_trace_timeline_event(
        kind=TraceTimelineEventKind.SCENARIO_FAILED,
        sequence=4,
        title="Scenario failed",
        scenario_id="basic_run_completed",
        facets=TraceTimelineEventFacets(
            scenario_lifecycle=TraceScenarioLifecycleFacet(
                phase=TraceScenarioPhase.FAILED,
                scenario_id="basic_run_completed",
            )
        ),
    )
    validate_trace_timeline_event_facets(event)


def test_scenario_lifecycle_scenario_id_must_match_event_scenario_id() -> None:
    event = create_trace_timeline_event(
        kind=TraceTimelineEventKind.SCENARIO_STARTED,
        sequence=2,
        title="Scenario started",
        scenario_id="basic_run_completed",
        facets=TraceTimelineEventFacets(
            scenario_lifecycle=TraceScenarioLifecycleFacet(
                phase=TraceScenarioPhase.STARTED,
                scenario_id="other_scenario",
            )
        ),
    )
    with pytest.raises(ValueError, match="scenario_lifecycle scenario_id must match event.scenario_id"):
        validate_trace_timeline_event_facets(event)


def test_scenario_lifecycle_passed_on_evidence_emitted_raises_value_error() -> None:
    event = _evidence_event(
        sequence=3,
        scenario_id="basic_run_completed",
        facets=TraceTimelineEventFacets(
            scenario_lifecycle=TraceScenarioLifecycleFacet(
                phase=TraceScenarioPhase.PASSED,
                scenario_id="basic_run_completed",
            )
        ),
    )
    with pytest.raises(ValueError, match="scenario_lifecycle phase passed does not match event kind evidence_emitted"):
        validate_trace_timeline_event_facets(event)


def test_validate_trace_timeline_rejects_invalid_facets() -> None:
    timeline = TraceTimeline(
        timeline_id="timeline-facets-invalid",
        kind=TraceTimelineKind.CORE_CERTIFICATION,
        title="Invalid facets timeline",
        events=[
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.CERTIFICATION_STARTED,
                sequence=1,
                title="Certification started",
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.SCENARIO_PASSED,
                sequence=2,
                title="Scenario passed",
                scenario_id="basic_run_completed",
                facets=TraceTimelineEventFacets(
                    evidence=TraceEvidenceFacet(
                        origin=TraceEvidenceOrigin.MOCK,
                        ref="mock:basic_run_completed",
                    )
                ),
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.CERTIFICATION_COMPLETED,
                sequence=3,
                title="Certification completed",
            ),
        ],
    )
    with pytest.raises(ValueError, match="evidence facet is only allowed on EVIDENCE_EMITTED"):
        validate_trace_timeline(timeline)


def test_evidence_modules_have_no_applications_or_agents_imports() -> None:
    forbidden = ("applications.", "agents.", "from applications", "from agents")
    for path in _EVIDENCE_ROOT.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{path.name} contains forbidden import token: {token}"
