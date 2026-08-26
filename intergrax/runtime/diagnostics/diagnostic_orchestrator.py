# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""
Canonical cross-run diagnostic orchestration (DIAG-7).

Synchronous application-service composition over the existing DIAG-2→5 spine.
Does not schedule work, subscribe to events, or introduce new diagnostic truth.
"""

from __future__ import annotations

from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessmentBuilder
from intergrax.runtime.diagnostics.diagnostic_orchestration_models import (
    DiagnosticExecutionAnalysis,
    DiagnosticExecutionScope,
    DiagnosticOrchestrationRequest,
    DiagnosticOrchestrationResult,
    validate_orchestration_request,
)
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.diagnostics.lifecycle_analysis import LifecycleAnomalyAnalyzer
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingAssessmentInput,
    ProblemGroupingEngine,
)
from intergrax.runtime.diagnostics.problem_grouping_features import (
    ProblemGroupingFeatureSourceFacts,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemLifecycleEngine


class DiagnosticOrchestrator:
    """
    Canonical platform entry point for explicit multi-execution diagnostic processing.

    Composes existing DIAG-2→5 components only. Does not read canonical persistence
    directly, infer root cause, or expose raw runtime payload in public results.
    """

    def __init__(
        self,
        execution_reconstructor: ExecutionReconstructor,
        lifecycle_analyzer: LifecycleAnomalyAnalyzer,
        assessment_builder: DiagnosticAssessmentBuilder,
        grouping_engine: ProblemGroupingEngine,
        problem_lifecycle_engine: ProblemLifecycleEngine,
    ) -> None:
        self._execution_reconstructor = execution_reconstructor
        self._lifecycle_analyzer = lifecycle_analyzer
        self._assessment_builder = assessment_builder
        self._grouping_engine = grouping_engine
        self._problem_lifecycle_engine = problem_lifecycle_engine

    def run(self, request: DiagnosticOrchestrationRequest) -> DiagnosticOrchestrationResult:
        tenant_id = validate_orchestration_request(request)

        assessment_inputs: list[ProblemGroupingAssessmentInput] = []
        execution_results: list[DiagnosticExecutionAnalysis] = []

        for scope in request.executions:
            analysis = self._analyze_execution_scope(tenant_id, scope)
            assessment_inputs.append(analysis.assessment_input)
            execution_results.append(analysis.execution_analysis)

        grouping_result = self._grouping_engine.group(
            tuple(assessment_inputs),
            strategy_id=request.grouping_strategy_id,
        )
        lifecycle_result = self._problem_lifecycle_engine.reconcile(
            grouping_result,
            observed_at=request.observed_at,
        )

        return DiagnosticOrchestrationResult(
            tenant_id=tenant_id,
            execution_results=tuple(execution_results),
            grouping_result=grouping_result,
            lifecycle_result=lifecycle_result,
        )

    def _analyze_execution_scope(
        self,
        tenant_id: str,
        scope: DiagnosticExecutionScope,
    ) -> _ScopedExecutionAnalysis:
        reconstruction = self._execution_reconstructor.reconstruct_execution(
            tenant_id,
            scope.task_id,
            scope.run_id,
        )
        lifecycle = self._lifecycle_analyzer.analyze(reconstruction)
        assessment = self._assessment_builder.assess(reconstruction, lifecycle)
        source_facts = ProblemGroupingFeatureSourceFacts(
            tenant_id=tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            reconstruction=reconstruction,
            problem_signals=scope.problem_signals,
        )
        return _ScopedExecutionAnalysis(
            assessment_input=ProblemGroupingAssessmentInput(
                assessment=assessment,
                feature_source_facts=source_facts,
            ),
            execution_analysis=DiagnosticExecutionAnalysis(
                tenant_id=tenant_id,
                task_id=scope.task_id,
                run_id=scope.run_id,
                assessment=assessment,
                runtime_history_completeness=reconstruction.runtime_history_completeness,
                has_runtime_events=reconstruction.has_runtime_events,
                has_transport_evidence=reconstruction.has_transport_evidence,
            ),
        )


class _ScopedExecutionAnalysis:
    """Internal pairing of grouping input and bounded execution analysis."""

    __slots__ = ("assessment_input", "execution_analysis")

    def __init__(
        self,
        *,
        assessment_input: ProblemGroupingAssessmentInput,
        execution_analysis: DiagnosticExecutionAnalysis,
    ) -> None:
        self.assessment_input = assessment_input
        self.execution_analysis = execution_analysis
