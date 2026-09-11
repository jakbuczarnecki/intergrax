# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J-O2-P2 completion alignment producer reachability qualification probe."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.decision_system.completion_eligibility import CompletionEligibilityStatus
from intergrax.runtime.diagnostics.completion_alignment_diag import CompletionAlignmentDiagV1
from platform_proofs.scenarios.ai_incident_investigation.application.evidence_completion_gate import (
    evaluate_ai_incident_completion_eligibility,
)
from platform_proofs.scenarios.ai_incident_investigation.application.evidence_requirement_semantics import (
    RESOLVED_INCIDENT_EVIDENCE_REQUIREMENT_PROVIDER,
)
from platform_proofs.scenarios.ai_incident_investigation.application.runtime_composition import (
    trace_reader_from_composition,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    ScenarioExecutionResult,
    execute_resolved_skeleton,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.runtime_bundle import (
    FixtureRuntimeBundle,
    build_fixture_runtime_bundle,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_coverage_analysis import (
    AlignmentCoverageMetrics,
    _aggregate_coverage_metrics,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_coverage_evidence import (
    extract_coverage_evidence,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    COMPLETION_ALIGNMENT_TRACE_SCHEMA,
    TraceReadbackStatus,
)
from testing_support.decision_e2e.local_qualification_session.trace_readback import (
    read_typed_alignment_events,
)


@dataclass(frozen=True, slots=True)
class CompletionAlignmentProducerReachabilityResult:
    execution_result: ScenarioExecutionResult
    completion_eligibility_status: CompletionEligibilityStatus
    trace_events: tuple[dict[str, object], ...]
    alignment_events_count: int
    typed_alignment_events: tuple[CompletionAlignmentDiagV1, ...]
    trace_readback_status: TraceReadbackStatus
    coverage_alignment: AlignmentCoverageMetrics


def _trace_events_for_execution(
    fixture_bundle: FixtureRuntimeBundle,
    result: ScenarioExecutionResult,
) -> tuple[dict[str, object], ...]:
    provenance = result.execution_provenance
    if provenance is None:
        return ()
    reader = trace_reader_from_composition(fixture_bundle.bundle.runtime_composition)
    if reader is None:
        return ()
    persisted = reader.read_run(
        str(provenance.platform_run_id),
        provenance.execution_tenant_id,
    )
    return tuple(dict(item) for item in persisted.events if isinstance(item, dict))


async def run_completion_alignment_producer_reachability_probe(
    *,
    fixture_bundle: FixtureRuntimeBundle | None = None,
) -> CompletionAlignmentProducerReachabilityResult:
    resolved_fixture = fixture_bundle or build_fixture_runtime_bundle()
    execution_result = await execute_resolved_skeleton(resolved_fixture.bundle)
    eligibility = evaluate_ai_incident_completion_eligibility(
        evidence_nodes=execution_result.evidence_nodes,
        provider=RESOLVED_INCIDENT_EVIDENCE_REQUIREMENT_PROVIDER,
    )
    trace_events = _trace_events_for_execution(resolved_fixture, execution_result)
    readback = read_typed_alignment_events(trace_events, trace_available=True)
    run_id = execution_result.execution_provenance
    coverage_run = {
        "run_id": str(run_id.platform_run_id) if run_id is not None else "",
        "trace_available": True,
        "trace_events": list(trace_events),
    }
    coverage_evidence = (extract_coverage_evidence(coverage_run),)
    coverage_alignment, _, _, _ = _aggregate_coverage_metrics(coverage_evidence)
    return CompletionAlignmentProducerReachabilityResult(
        execution_result=execution_result,
        completion_eligibility_status=eligibility.status,
        trace_events=trace_events,
        alignment_events_count=len(readback.events),
        typed_alignment_events=readback.events,
        trace_readback_status=readback.status,
        coverage_alignment=coverage_alignment,
    )


def assert_canonical_match_alignment_event(
    event: CompletionAlignmentDiagV1,
) -> None:
    from intergrax.runtime.diagnostics.completion_alignment_diag import (
        AlignmentDirection,
        AlignmentStatus,
    )

    assert event.schema_id() == COMPLETION_ALIGNMENT_TRACE_SCHEMA
    assert event.alignment_status is AlignmentStatus.MATCH
    assert event.alignment_direction is AlignmentDirection.NONE


def assert_reverse_mismatch_alignment_event(
    event: CompletionAlignmentDiagV1,
) -> None:
    from intergrax.runtime.diagnostics.completion_alignment_diag import (
        AlignmentDirection,
        AlignmentStatus,
    )

    assert event.schema_id() == COMPLETION_ALIGNMENT_TRACE_SCHEMA
    assert event.alignment_status is AlignmentStatus.MISMATCH
    assert event.alignment_direction is AlignmentDirection.REVERSE
    assert event.correctable is True


__all__ = [
    "CompletionAlignmentProducerReachabilityResult",
    "assert_canonical_match_alignment_event",
    "assert_reverse_mismatch_alignment_event",
    "run_completion_alignment_producer_reachability_probe",
]
