# © Artur Czarnecki. All rights reserved.

"""DiagnosticInvestigationView must surface related predictive risk signals."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.contracts.predictive_investigation_read import RelatedPredictiveRiskSignalView
from intergrax.contracts.predictive_risk import PredictiveRiskScope, PredictiveRiskSeverity
from intergrax.runtime.diagnostics.deterministic_problem_grouping import (
    DeterministicProblemGroupingStrategy,
)
from intergrax.runtime.diagnostics.diagnostic_operator_investigation_projection import (
    project_investigation_view,
)
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticOccurrenceReadStatus,
    DiagnosticProblemDetail,
    DiagnosticProblemOccurrenceView,
    grouping_provenance_from_problem_provenance,
)
from intergrax.runtime.diagnostics.persistence_conformance import sample_problem, sample_subject_refs
from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingMethod
from intergrax.runtime.prediction import (
    FailurePatternAnalyzer,
    LatencyTrendAnalyzer,
    PredictionEngine,
    PredictiveAnalyzerRegistry,
)
from intergrax.runtime.prediction.predictive_investigation_projection import (
    project_related_risk_signals,
)
from tests.unit.runtime.prediction.conftest import crm_agent_showcase_context

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_STRATEGY = DeterministicProblemGroupingStrategy()
_OBSERVED = datetime(2026, 9, 11, 14, 0, tzinfo=UTC)


def test_predictive_read_model_enrichment() -> None:
    engine = PredictionEngine(
        registry=PredictiveAnalyzerRegistry(
            (LatencyTrendAnalyzer(), FailurePatternAnalyzer()),
        ),
    )
    prediction = engine.analyze(crm_agent_showcase_context())
    related = project_related_risk_signals(prediction.signals)
    assert related

    problem = sample_problem(tenant_id="tenant-demo")
    grouping = grouping_provenance_from_problem_provenance(problem.provenance)
    detail = DiagnosticProblemDetail(
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
    occurrence = DiagnosticProblemOccurrenceView(
        subject_ref=sample_subject_refs(problem)[0],
        observed_at=_OBSERVED,
        strategy_id=_STRATEGY.strategy_id,
        strategy_version=_STRATEGY.strategy_version,
        method=ProblemGroupingMethod.DETERMINISTIC,
        read_status=DiagnosticOccurrenceReadStatus.AVAILABLE,
        assessment=None,
        execution_lineage=None,
    )
    investigation = project_investigation_view(
        problem_detail=detail,
        occurrence=occurrence,
        reconstruction=None,
        related_risk_signals=related,
    )

    assert investigation.related_risk_signals == related
    assert investigation.related_risk_signals[0].scope is PredictiveRiskScope.COMPONENT
    assert investigation.related_risk_signals[0].severity in PredictiveRiskSeverity
    assert investigation.related_risk_signals[0].analyzer_id in (
        "latency_trend",
        "failure_pattern",
    )
