# © Artur Czarnecki. All rights reserved.

"""DIAG R7 — operator investigation read model qualification (R7-A1–A7)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.diagnostic_investigation import DiagnosticRootCauseStatus
from intergrax.contracts.execution_identity import mint_execution_id
from intergrax.contracts.multi_agent_failure_localization import FailureBoundaryCertainty
from intergrax.runtime.diagnostics.decision_context_read_models import DecisionContextReadStatus
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.diagnostic_lineage_projection import (
    project_execution_lineage_view,
)
from intergrax.runtime.diagnostics.diagnostic_operator_investigation_projection import (
    project_investigation_view,
)
from intergrax.runtime.diagnostics.diagnostic_operator_investigation_read_models import (
    DiagnosticImpactNodeHealth,
)
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticOccurrenceReadStatus,
    DiagnosticProblemDetail,
    DiagnosticProblemOccurrenceView,
    grouping_provenance_from_problem_provenance,
)
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingMethod,
    problem_grouping_subject_ref_for_execution,
)
from testing_support.runtime.decision_execution_lineage_r4_harness import (
    build_decision_execution_lineage_r4_harness,
)
from intergrax.contracts.diagnostic_extension_evidence import DiagnosticExtensionEvidence
from intergrax.runtime.diagnostics.diagnostic_extension_registry import (
    DiagnosticExtensionRegistry,
)
from testing_support.runtime.diagnostic_extension_spi_r5_harness import (
    _DEFAULT_NAMESPACE,
    build_diagnostic_extension_spi_r5_harness,
)
from testing_support.runtime.execution_failure_evidence_r2_closure_harness import (
    build_execution_failure_evidence_r2_closure_harness,
)
from testing_support.runtime.multi_agent_failure_localization_r1_harness import (
    build_multi_agent_failure_localization_scenario,
    open_lineage_chain,
    open_lineage_fan_out,
    record_execution_failed,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FORBIDDEN_SYMBOLS = (
    "ApplicationDiagnosticEngine",
    "ApplicationProblemStore",
    "OperatorDiagnosticEngine",
    "HeuristicRootCause",
)
_OBSERVED_AT = datetime(2026, 9, 11, 14, 0, tzinfo=UTC)
_STRATEGY = DeterministicProblemGroupingStrategy()


def _problem_detail_for_scenario(scenario) -> DiagnosticProblemDetail:
    problem = sample_problem(tenant_id=scenario.tenant_id)
    grouping = grouping_provenance_from_problem_provenance(problem.provenance)
    return DiagnosticProblemDetail(
        problem_id=problem.problem_id,
        tenant_id=problem.tenant_id,
        status=problem.status,
        first_seen_at=problem.first_seen_at,
        last_seen_at=problem.last_seen_at,
        occurrence_count=1,
        record_version=problem.record_version,
        grouping_provenance=grouping,
        occurrence_aggregate_health=problem.occurrence_aggregate_health,
        occurrences=(),
        returned_occurrence_count=0,
        total_occurrence_count=1,
        is_occurrences_truncated=False,
    )


def _investigation_from_localization_scenario(scenario):
    reconstruction = scenario.reconstructor.reconstruct_execution(
        scenario.tenant_id,
        scenario.task_id,
        scenario.run_id,
    )
    lifecycle = scenario.lifecycle_analyzer.analyze(reconstruction)
    assessment = scenario.assessment_builder.assess(reconstruction, lifecycle)
    subject_ref = problem_grouping_subject_ref_for_execution(
        tenant_id=scenario.tenant_id,
        task_id=scenario.task_id,
        run_id=scenario.run_id,
    )
    occurrence = DiagnosticProblemOccurrenceView(
        subject_ref=subject_ref,
        observed_at=_OBSERVED_AT,
        strategy_id=_STRATEGY.strategy_id,
        strategy_version=_STRATEGY.strategy_version,
        method=ProblemGroupingMethod.DETERMINISTIC,
        read_status=DiagnosticOccurrenceReadStatus.AVAILABLE,
        assessment=assessment,
        execution_lineage=project_execution_lineage_view(reconstruction),
    )
    detail = _problem_detail_for_scenario(scenario)
    return project_investigation_view(
        problem_detail=detail,
        occurrence=occurrence,
        reconstruction=reconstruction,
    )


async def _run_failure_and_investigate(harness):
    class _FailingLeaf:
        async def execute(self, request: object) -> None:
            raise RuntimeError("r7-failure")

    class _Root:
        async def execute(self, request: object) -> None:
            await harness.child_runner.execute(
                request=object(),
                delegate=_FailingLeaf(),
            )

    harness.bind_root_delegate(_Root())
    context = harness.resolve_root_context()
    try:
        await harness.runtime.execute(object(), context)
    except RuntimeError:
        pass
    harness.seed_terminal_lifecycle_events(context, failed=True)
    harness.run_terminal_diagnostics(context)
    problem_id = harness.read_service.list_problems(tenant_id=harness.tenant_id).problems[
        0
    ].problem_id
    return harness.read_service.get_investigation(
        tenant_id=harness.tenant_id,
        problem_id=problem_id,
    )


@pytest.mark.asyncio
async def test_r7_a1_simple_failure_investigation() -> None:
    harness = build_execution_failure_evidence_r2_closure_harness()
    result = await _run_failure_and_investigate(harness)
    assert result.investigation is not None
    view = result.investigation
    assert view.failure_boundary is not None
    assert any(
        item.confidence.value == "proven"
        for item in view.evidence_summary
    )
    assert view.failure_investigation.root_cause_status is DiagnosticRootCauseStatus.UNKNOWN


def test_r7_a2_nested_failure_boundary_and_impact() -> None:
    e1, e2, e3, e4 = [mint_execution_id() for _ in range(4)]
    scenario = build_multi_agent_failure_localization_scenario()
    open_lineage_chain(scenario, (e1, e2, e3, e4))
    record_execution_failed(scenario, e4)
    view = _investigation_from_localization_scenario(scenario)
    boundary_ids = {
        str(b.execution_id) for b in view.failure_investigation.failure_boundaries
    }
    assert boundary_ids == {str(e4)}
    assert str(view.impact_graph.root_execution_id) == str(e1)
    assert str(e1) in {str(x) for x in view.affected_execution_ids}


def test_r7_a3_multi_agent_sibling_isolation() -> None:
    e1, e2, e3, e4 = [mint_execution_id() for _ in range(4)]
    scenario = build_multi_agent_failure_localization_scenario()
    open_lineage_fan_out(
        scenario,
        e1,
        ((e2, e1), (e3, e1), (e4, e1)),
    )
    record_execution_failed(scenario, e3)
    view = _investigation_from_localization_scenario(scenario)
    failed_nodes = {
        str(n.execution_id)
        for n in view.impact_graph.nodes
        if n.health is DiagnosticImpactNodeHealth.FAILED
    }
    assert failed_nodes == {str(e3)}
    healthy = {
        str(n.execution_id)
        for n in view.impact_graph.nodes
        if n.health is DiagnosticImpactNodeHealth.HEALTHY
    }
    assert {str(e2), str(e4)} <= healthy


@pytest.mark.asyncio
async def test_r7_a4_decision_context_without_causal_inference() -> None:
    harness = build_decision_execution_lineage_r4_harness()
    decision_id = mint_decision_id()

    class _FailingLeaf:
        async def execute(self, request: object) -> None:
            raise RuntimeError("r7-a4")

    class _Root:
        async def execute(self, request: object) -> None:
            await harness.execution.child_runner.execute(
                request=object(),
                delegate=_FailingLeaf(),
            )

    harness.execution.bind_root_delegate(_Root())
    context = harness.execution.resolve_root_context()
    identity = DecisionIdentity(
        decision_id=decision_id,
        version=initial_decision_version(),
        scope=DecisionScope(namespace="qualification", subject="r7-a4"),
        tenant_id=harness.tenant_id,
        execution=DecisionExecutionLineage(
            task_id=harness.execution.task_id,
            run_id=context.run_id,
            attempt_id=context.attempt_id,
            execution_id=context.execution_id,
        ),
    )
    harness.append_correlation(identity)
    try:
        await harness.execution.runtime.execute(object(), context)
    except RuntimeError:
        pass
    harness.execution.seed_terminal_lifecycle_events(context, failed=True)
    harness.execution.run_terminal_diagnostics(context)
    problem_id = harness.read_service.list_problems(tenant_id=harness.tenant_id).problems[
        0
    ].problem_id
    result = harness.read_service.get_investigation(
        tenant_id=harness.tenant_id,
        problem_id=problem_id,
    )
    view = result.investigation
    assert view is not None
    assert view.decision_context is not None
    assert view.decision_context.read_status is DecisionContextReadStatus.AVAILABLE
    assert not any(item.is_causal_claim for item in view.evidence_summary)
    assert str(decision_id) in view.failure_investigation.related_decision_ids


@pytest.mark.asyncio
async def test_r7_a5_missing_evidence_explicit_uncertainty() -> None:
    harness = build_execution_failure_evidence_r2_closure_harness()
    result = await _run_failure_and_investigate(harness)
    view = result.investigation
    assert view is not None
    assert view.failure_investigation.explicit_unknowns
    assert any("Timeline ordering" in item for item in view.investigation_limitations)


class _R7EvidenceContributor:
    contributor_id = "r7-contributor"
    evidence_namespace = _DEFAULT_NAMESPACE
    priority = 0

    def collect(self, context) -> tuple[DiagnosticExtensionEvidence, ...]:
        scope = context.to_evidence_scope()
        return (
            DiagnosticExtensionEvidence.mint(
                scope=scope,
                evidence_namespace=self.evidence_namespace,
                kind=f"{self.evidence_namespace}.probe",
                summary="r7 extension probe",
            ),
        )


@pytest.mark.asyncio
async def test_r7_a6_plugin_enrichment_problem_authority_unchanged() -> None:
    registry = DiagnosticExtensionRegistry(
        evidence_contributors=(_R7EvidenceContributor(),),
    )
    harness = build_diagnostic_extension_spi_r5_harness(registry=registry)
    result = await _run_failure_and_investigate(harness.execution)
    view = result.investigation
    assert view is not None
    assert view.extension_enrichment is not None
    assert view.problem.problem_id is not None
    problems = harness.read_service.list_problems(tenant_id=harness.tenant_id)
    assert len(problems.problems) == 1


@pytest.mark.asyncio
async def test_r7_a7_tenant_isolation() -> None:
    harness = build_execution_failure_evidence_r2_closure_harness()
    await _run_failure_and_investigate(harness)
    problem_id = harness.read_service.list_problems(tenant_id=harness.tenant_id).problems[
        0
    ].problem_id
    foreign = harness.read_service.get_investigation(
        tenant_id="foreign-tenant-r7",
        problem_id=problem_id,
    )
    assert foreign.investigation is None


@pytest.mark.unit
@pytest.mark.gate
def test_r7_quality_gates_no_second_engine_and_no_heuristic_root_cause() -> None:
    intergrax_root = _REPO_ROOT / "intergrax"
    hits: list[str] = []
    for path in intergrax_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for symbol in _FORBIDDEN_SYMBOLS:
            if symbol in text:
                hits.append(f"{path.relative_to(_REPO_ROOT)}:{symbol}")
    assert hits == []

    scenario = build_multi_agent_failure_localization_scenario()
    e1, e2, e3 = mint_execution_id(), mint_execution_id(), mint_execution_id()
    open_lineage_chain(scenario, (e1, e2, e3))
    record_execution_failed(scenario, e3)
    view = _investigation_from_localization_scenario(scenario)
    assert view.failure_investigation.root_cause_status is DiagnosticRootCauseStatus.UNKNOWN
    assert not hasattr(view, "root_cause_execution_id")
    assert view.assistant_payload.root_cause_status is DiagnosticRootCauseStatus.UNKNOWN
    for boundary in view.failure_investigation.failure_boundaries:
        assert boundary.certainty is FailureBoundaryCertainty.PROVEN
