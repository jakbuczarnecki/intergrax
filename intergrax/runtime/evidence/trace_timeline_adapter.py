# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Certification report → trace timeline adapter (HEP Band 2ae · EVID-TRACE-03)."""

from __future__ import annotations

from pathlib import Path

from intergrax.runtime.evidence.certification_report import CoreCertificationReport
from intergrax.runtime.evidence.core_certification_spec import CORE_CERTIFICATION_EVIDENCE_KIND
from intergrax.runtime.evidence.scenario_contracts import (
    CoreEvidenceRef,
    CoreScenarioResult,
    CoreScenarioStatus,
    EvidenceRefKind,
    get_core_scenario_contract,
)
from intergrax.runtime.evidence.trace_timeline_contracts import (
    TraceTimeline,
    TraceTimelineEventKind,
    TraceTimelineKind,
    TraceTimelineSeverity,
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
)

_CERTIFICATION_REPORT_SCENARIO_ID = "certification_report_emitted"


def load_core_certification_report(report_path: Path) -> CoreCertificationReport:
    """Load a HEP-1 ``report.json`` artifact."""
    resolved = report_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"certification report not found: {resolved}")
    return CoreCertificationReport.model_validate_json(resolved.read_text(encoding="utf-8"))


def _scenario_title(scenario_id: str) -> str:
    contract = get_core_scenario_contract(scenario_id)
    return contract.title if contract is not None else scenario_id


def _evidence_origin(ref: CoreEvidenceRef) -> TraceEvidenceOrigin:
    if ref.kind is EvidenceRefKind.CERTIFICATION_REPORT:
        return TraceEvidenceOrigin.REPORT
    return TraceEvidenceOrigin.MOCK


def _primary_evidence_ref(evidence_refs: list[CoreEvidenceRef], scenario_id: str) -> str:
    if evidence_refs:
        return evidence_refs[0].ref
    return f"mock:{scenario_id}"


def _first_evidence_ref(
    evidence_refs: list[CoreEvidenceRef],
    kind: EvidenceRefKind,
) -> CoreEvidenceRef | None:
    for ref in evidence_refs:
        if ref.kind is kind:
            return ref
    return None


def _build_evidence_facets(
    scenario_id: str,
    evidence_refs: list[CoreEvidenceRef],
) -> TraceTimelineEventFacets:
    primary_ref = _primary_evidence_ref(evidence_refs, scenario_id)
    primary = evidence_refs[0] if evidence_refs else None
    origin = _evidence_origin(primary) if primary is not None else TraceEvidenceOrigin.MOCK
    evidence_description = primary.description if primary is not None else ""
    if origin is TraceEvidenceOrigin.MOCK and evidence_description:
        evidence_description = f"{evidence_description} ({CORE_CERTIFICATION_EVIDENCE_KIND})"
    elif origin is TraceEvidenceOrigin.MOCK:
        evidence_description = CORE_CERTIFICATION_EVIDENCE_KIND

    facets = TraceTimelineEventFacets(
        evidence=TraceEvidenceFacet(
            origin=origin,
            ref=primary_ref,
            description=evidence_description,
        ),
        scenario_lifecycle=TraceScenarioLifecycleFacet(
            phase=TraceScenarioPhase.EVIDENCE,
            scenario_id=scenario_id,
        ),
    )

    policy_ref = _first_evidence_ref(evidence_refs, EvidenceRefKind.POLICY_DECISION)
    if policy_ref is not None:
        if scenario_id == "tool_denied_by_policy":
            outcome = TracePolicyOutcome.DENIED
        elif scenario_id == "high_risk_tool_hitl":
            outcome = TracePolicyOutcome.HITL_REQUIRED
        else:
            outcome = TracePolicyOutcome.ALLOWED
        facets.policy = TracePolicyFacet(
            outcome=outcome,
            reason=policy_ref.description or policy_ref.ref,
        )
        if outcome is TracePolicyOutcome.HITL_REQUIRED:
            facets.hitl = TraceHitlFacet(status=TraceHitlStatus.REQUESTED)

    budget_ref = _first_evidence_ref(evidence_refs, EvidenceRefKind.BUDGET_TICK)
    if budget_ref is not None:
        facets.budget = TraceBudgetFacet(
            status=TraceBudgetStatus.EXCEEDED,
            metric=budget_ref.kind.value,
            value=budget_ref.ref,
        )

    return facets


def _append_scenario_events(
    events: list,
    *,
    result: CoreScenarioResult,
    sequence: int,
) -> int:
    scenario_id = result.scenario_id
    title = _scenario_title(scenario_id)

    events.append(
        create_trace_timeline_event(
            kind=TraceTimelineEventKind.SCENARIO_STARTED,
            sequence=sequence,
            title=f"Scenario started: {title}",
            message=result.message or title,
            scenario_id=scenario_id,
            facets=TraceTimelineEventFacets(
                scenario_lifecycle=TraceScenarioLifecycleFacet(
                    phase=TraceScenarioPhase.STARTED,
                    scenario_id=scenario_id,
                )
            ),
        )
    )
    sequence += 1

    if result.evidence_refs:
        events.append(
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.EVIDENCE_EMITTED,
                sequence=sequence,
                title=f"Evidence emitted: {title}",
                scenario_id=scenario_id,
                source_refs=[
                    TraceTimelineSourceRef(
                        kind=TraceTimelineSourceKind.EVIDENCE_REF,
                        ref=ref.ref,
                        description=ref.description,
                    )
                    for ref in result.evidence_refs
                ],
                facets=_build_evidence_facets(scenario_id, result.evidence_refs),
            )
        )
        sequence += 1

    if result.status is CoreScenarioStatus.PASSED:
        events.append(
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.SCENARIO_PASSED,
                sequence=sequence,
                title=f"Scenario passed: {title}",
                message=result.message,
                scenario_id=scenario_id,
                facets=TraceTimelineEventFacets(
                    scenario_lifecycle=TraceScenarioLifecycleFacet(
                        phase=TraceScenarioPhase.PASSED,
                        scenario_id=scenario_id,
                    )
                ),
            )
        )
        return sequence + 1

    failed_kind = TraceTimelineEventKind.SCENARIO_FAILED
    events.append(
        create_trace_timeline_event(
            kind=failed_kind,
            sequence=sequence,
            title=f"Scenario failed: {title}",
            message=result.message or "scenario failed",
            scenario_id=scenario_id,
            severity=TraceTimelineSeverity.ERROR,
            facets=TraceTimelineEventFacets(
                scenario_lifecycle=TraceScenarioLifecycleFacet(
                    phase=TraceScenarioPhase.FAILED,
                    scenario_id=scenario_id,
                )
            ),
        )
    )
    return sequence + 1


def build_timeline_from_certification_report(
    report: CoreCertificationReport,
    *,
    source_report_path: str | None = None,
) -> TraceTimeline:
    """Build a canonical trace timeline from a HEP-1 certification report."""
    output_dir = Path(report.output_dir)
    json_path = str(output_dir / "report.json")
    md_path = str(output_dir / "report.md")
    resolved_source = source_report_path or json_path

    events = [
        create_trace_timeline_event(
            kind=TraceTimelineEventKind.CERTIFICATION_STARTED,
            sequence=1,
            title="Certification started",
            message=(
                f"Core certification {report.certification_level.value} "
                f"({CORE_CERTIFICATION_EVIDENCE_KIND} evidence from report); "
                f"run {report.certification_run_id}"
            ),
            source_refs=[
                TraceTimelineSourceRef(
                    kind=TraceTimelineSourceKind.CERTIFICATION_REPORT,
                    ref=report.certification_run_id,
                    description=(
                        "HEP-1 report-derived timeline; not live runtime trace"
                    ),
                )
            ],
        )
    ]
    sequence = 2

    for result in report.scenario_results:
        if result.scenario_id == _CERTIFICATION_REPORT_SCENARIO_ID:
            continue
        sequence = _append_scenario_events(events, result=result, sequence=sequence)

    from intergrax.runtime.evidence.trace_timeline_contracts import (
        TraceTimelineArtifactKind,
        TraceTimelineArtifactRef,
    )

    events.append(
        create_trace_timeline_event(
            kind=TraceTimelineEventKind.REPORT_WRITTEN,
            sequence=sequence,
            title="Certification report written",
            message="JSON and Markdown certification report artifacts written",
            artifact_refs=[
                TraceTimelineArtifactRef(
                    kind=TraceTimelineArtifactKind.REPORT_JSON,
                    path=json_path,
                    description="core certification report.json",
                ),
                TraceTimelineArtifactRef(
                    kind=TraceTimelineArtifactKind.REPORT_MARKDOWN,
                    path=md_path,
                    description="core certification report.md",
                ),
            ],
        )
    )
    sequence += 1

    completion_message = (
        f"Certification completed: {report.scenarios_passed}/{report.scenarios_total} passed"
    )
    if not report.passed:
        completion_message = (
            f"Certification completed with failures: "
            f"{report.scenarios_failed} failed, "
            f"{report.scenarios_passed}/{report.scenarios_total} passed"
        )

    events.append(
        create_trace_timeline_event(
            kind=TraceTimelineEventKind.CERTIFICATION_COMPLETED,
            sequence=sequence,
            title="Certification completed",
            message=completion_message,
            source_refs=[
                TraceTimelineSourceRef(
                    kind=TraceTimelineSourceKind.CERTIFICATION_REPORT,
                    ref=resolved_source,
                    description="source certification report artifact",
                )
            ],
        )
    )

    timeline = TraceTimeline(
        timeline_id=report.certification_run_id,
        kind=TraceTimelineKind.CORE_CERTIFICATION,
        title=(
            f"Core certification {report.certification_level.value} timeline "
            f"(report-derived {CORE_CERTIFICATION_EVIDENCE_KIND})"
        ),
        events=events,
        source_report_path=resolved_source,
        generated_at=report.generated_at,
    )
    validate_trace_timeline(timeline)
    return timeline
