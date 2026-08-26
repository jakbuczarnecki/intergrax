# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed request/result contracts for cross-run diagnostic orchestration (DIAG-7)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.execution_identity import RunId, TaskId, validate_run_id, validate_task_id
from intergrax.runtime.diagnostics.diagnostic_assessment import DiagnosticAssessment
from intergrax.runtime.diagnostics.execution_reconstruction import RuntimeHistoryCompleteness
from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingResult,
    ProblemGroupingStrategyId,
)
from intergrax.runtime.diagnostics.problem_lifecycle import ProblemLifecycleResult
from intergrax.runtime.observability.problem_signal import PlatformProblemSignal

from intergrax.runtime.diagnostics.signal_diagnostic_assessment import SignalDiagnosticAssessment

MAX_DIAGNOSTIC_ORCHESTRATION_EXECUTIONS = 100
MAX_DIAGNOSTIC_ORCHESTRATION_SIGNAL_SUBJECTS = 100


class DiagnosticOrchestrationIntegrityError(Exception):
    """Raised when orchestration input or scope validation fails before processing."""


@dataclass(frozen=True, slots=True)
class DiagnosticExecutionScope:
    """One tenant/task/run execution input for cross-run diagnostic orchestration."""

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    problem_signals: tuple[PlatformProblemSignal, ...] = ()


@dataclass(frozen=True, slots=True)
class DiagnosticSignalSubjectScope:
    """One tenant/application/instance non-execution diagnostic input."""

    tenant_id: str
    application_id: str
    instance_id: str
    problem_signals: tuple[PlatformProblemSignal, ...] = ()


@dataclass(frozen=True, slots=True)
class DiagnosticOrchestrationRequest:
    """
    Explicit multi-subject diagnostic processing request.

    One invocation processes exactly one tenant across execution scopes and/or
    non-execution signal subjects. At least one subject input is required.

    Cost is O(N) over subject scopes plus grouping strategy cost.
    """

    tenant_id: str
    grouping_strategy_id: ProblemGroupingStrategyId
    observed_at: datetime
    executions: tuple[DiagnosticExecutionScope, ...] = ()
    signal_subjects: tuple[DiagnosticSignalSubjectScope, ...] = ()


@dataclass(frozen=True, slots=True)
class DiagnosticExecutionAnalysis:
    """
    Bounded per-execution diagnostic output for one orchestration scope.

    Does not expose ``ExecutionReconstruction``, runtime events, causal evidence,
    or raw payload-bearing canonical objects.
    """

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    assessment: DiagnosticAssessment
    runtime_history_completeness: RuntimeHistoryCompleteness
    has_runtime_events: bool
    has_transport_evidence: bool


@dataclass(frozen=True, slots=True)
class DiagnosticSignalSubjectAnalysis:
    """Bounded per-application-instance diagnostic output for one signal subject scope."""

    tenant_id: str
    application_id: str
    instance_id: str
    assessment: SignalDiagnosticAssessment


@dataclass(frozen=True, slots=True)
class DiagnosticOrchestrationResult:
    """Auditable outcome of one synchronous cross-run diagnostic orchestration."""

    tenant_id: str
    execution_results: tuple[DiagnosticExecutionAnalysis, ...]
    signal_subject_results: tuple[DiagnosticSignalSubjectAnalysis, ...]
    grouping_result: ProblemGroupingResult
    lifecycle_result: ProblemLifecycleResult


def _require_tenant_id(tenant_id: str) -> str:
    if type(tenant_id) is not str:
        raise TypeError("tenant_id must be str")
    normalized = tenant_id.strip()
    if not normalized:
        raise ValueError("tenant_id must be non-empty and not whitespace-only")
    if tenant_id != normalized:
        raise ValueError("tenant_id must not contain leading or trailing whitespace")
    return normalized


def _validate_observed_at(observed_at: datetime) -> None:
    if type(observed_at) is not datetime:
        raise TypeError("observed_at must be datetime")
    if observed_at.tzinfo is None or observed_at.tzinfo.utcoffset(observed_at) is None:
        raise ValueError("observed_at must be timezone-aware")


def _validate_application_id(application_id: str) -> str:
    if type(application_id) is not str:
        raise TypeError("application_id must be str")
    normalized = application_id.strip()
    if not normalized:
        raise ValueError("application_id must be non-empty and not whitespace-only")
    if application_id != normalized:
        raise ValueError("application_id must not contain leading or trailing whitespace")
    return normalized


def _validate_instance_id(instance_id: str) -> str:
    if type(instance_id) is not str:
        raise TypeError("instance_id must be str")
    normalized = instance_id.strip()
    if not normalized:
        raise ValueError("instance_id must be non-empty and not whitespace-only")
    if instance_id != normalized:
        raise ValueError("instance_id must not contain leading or trailing whitespace")
    return normalized


def validate_orchestration_request(request: DiagnosticOrchestrationRequest) -> str:
    """
    Validate orchestration request before any reconstruction reads.

    Returns the normalized tenant id for the invocation.
    """
    tenant_id = _require_tenant_id(request.tenant_id)
    _validate_observed_at(request.observed_at)
    execution_count = len(request.executions)
    signal_subject_count = len(request.signal_subjects)
    total_subjects = execution_count + signal_subject_count
    if total_subjects < 1:
        raise DiagnosticOrchestrationIntegrityError(
            "orchestration requires at least one execution or signal subject scope",
        )
    if execution_count > MAX_DIAGNOSTIC_ORCHESTRATION_EXECUTIONS:
        raise DiagnosticOrchestrationIntegrityError(
            f"orchestration exceeds max execution scopes "
            f"({MAX_DIAGNOSTIC_ORCHESTRATION_EXECUTIONS})",
        )
    if signal_subject_count > MAX_DIAGNOSTIC_ORCHESTRATION_SIGNAL_SUBJECTS:
        raise DiagnosticOrchestrationIntegrityError(
            f"orchestration exceeds max signal subject scopes "
            f"({MAX_DIAGNOSTIC_ORCHESTRATION_SIGNAL_SUBJECTS})",
        )

    seen_executions: set[tuple[TaskId, RunId]] = set()
    for scope in request.executions:
        scope_tenant_id = _require_tenant_id(scope.tenant_id)
        if scope_tenant_id != tenant_id:
            raise DiagnosticOrchestrationIntegrityError(
                "all execution scopes must match request tenant_id",
            )
        task_id = validate_task_id(scope.task_id)
        run_id = validate_run_id(scope.run_id)
        key = (task_id, run_id)
        if key in seen_executions:
            raise DiagnosticOrchestrationIntegrityError(
                "duplicate execution scope in orchestration request",
            )
        seen_executions.add(key)

    seen_signal_subjects: set[tuple[str, str]] = set()
    for scope in request.signal_subjects:
        scope_tenant_id = _require_tenant_id(scope.tenant_id)
        if scope_tenant_id != tenant_id:
            raise DiagnosticOrchestrationIntegrityError(
                "all signal subject scopes must match request tenant_id",
            )
        application_id = _validate_application_id(scope.application_id)
        instance_id = _validate_instance_id(scope.instance_id)
        key = (application_id, instance_id)
        if key in seen_signal_subjects:
            raise DiagnosticOrchestrationIntegrityError(
                "duplicate signal subject scope in orchestration request",
            )
        seen_signal_subjects.add(key)

    return tenant_id
