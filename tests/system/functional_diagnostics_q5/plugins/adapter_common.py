# © Artur Czarnecki. All rights reserved.

"""Shared adapter utilities for Q5 domain qualification plugins."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from intergrax.core.qualification.functional_diagnostic_expectation import (
    QualificationCaseComparison,
    QualificationCaseExpectation,
    QualificationFunctionalOutcome,
)
from intergrax.core.qualification.functional_qualification_attempts import (
    QualificationAttemptRecord,
    QualificationPreconditionStatus,
)
from intergrax.core.qualification.functional_qualification_case import (
    QualificationNormalizedCaseResult,
    QualificationRepeatabilityGroup,
)
from intergrax.core.qualification.functional_qualification_fidelity import (
    QualificationGateResult,
    core_gate_pass,
    count_gate_failures,
)
from intergrax.core.qualification.functional_qualification_identity import (
    FunctionalQualificationPluginId,
    QualificationGateId as CoreGateId,
)
from intergrax.core.qualification.functional_qualification_metrics import (
    QualificationPluginMetrics,
    compute_qualification_metrics,
    metrics_pass,
)
from intergrax.core.qualification.functional_qualification_plugin import QualificationPluginDescriptor
from intergrax.core.qualification.functional_qualification_result import (
    QualificationPluginResult,
    QualificationReportSection,
)
from intergrax.core.qualification.functional_qualification_verdict import (
    QualificationVerdict,
    aggregate_plugin_verdict,
    verdict_from_domain_string,
)
from intergrax.runtime.diagnostics.functional_diagnostic_analyzer import FunctionalDiagnosticAnalyzer
from intergrax.runtime.diagnostics.functional_operator_projection import FunctionalOperatorOutcomeStatus


@dataclass(frozen=True, slots=True)
class DomainRunRecord:
    case_id: str
    task_id: str
    run_id: str
    comparison: QualificationCaseComparison
    functional_outcome: QualificationFunctionalOutcome
    diag_first_failed_check: str | None
    operator_outcome: str | None
    repeat_group: str | None
    identity_fidelity_match: bool
    authoritative_attempt_index: int | None = None
    attempt_history: tuple[QualificationAttemptRecord, ...] = ()
    prerequisite_exhausted: bool = False
    blocked_reason: str | None = None


@dataclass(frozen=True, slots=True)
class DomainQualificationReport:
    verdict: str
    records: tuple[DomainRunRecord, ...]
    blocked_reason: str | None
    repeatability_pass: bool
    fidelity_pass: bool
    decision_independence_pass: bool
    domain_extra_gates: tuple[QualificationGateResult, ...] = ()


@dataclass(frozen=True, slots=True)
class DomainAdapterConfig:
    plugin_id: FunctionalQualificationPluginId
    domain: str
    display_name: str
    artifact_ref: str
    domain_artifact_ref: str
    tenant_id: str | None
    healthy_case_ids: frozenset[str]
    inconclusive_case_ids: frozenset[str]
    expectation_for_record: Callable[[DomainRunRecord], QualificationCaseExpectation]
    repeatability_group_ids: frozenset[str]
    signature_for_record: Callable[[DomainRunRecord], tuple[str, ...]]
    extra_report_sections: tuple[QualificationReportSection, ...] = ()


def build_plugin_descriptor(config: DomainAdapterConfig) -> QualificationPluginDescriptor:
    return QualificationPluginDescriptor(
        plugin_id=config.plugin_id,
        domain=config.domain,
        version="1.0.0",
        display_name=config.display_name,
        contract_version="functional_qualification_v1",
        qualification_level="enterprise",
        required_capabilities=(),
    )


def build_plugin_result(
    config: DomainAdapterConfig,
    report: DomainQualificationReport,
) -> QualificationPluginResult:
    domain_verdict = verdict_from_domain_string(report.verdict)
    cases = _normalize_cases(config, report.records)
    repeatability_groups = _repeatability_groups(config, report.records)
    metrics = compute_qualification_metrics(cases, repeatability_groups=repeatability_groups)
    if not report.repeatability_pass:
        metrics = QualificationPluginMetrics(
            total_cases=metrics.total_cases,
            matched_cases=metrics.matched_cases,
            mismatched_cases=metrics.mismatched_cases,
            false_positives=metrics.false_positives,
            false_negatives=metrics.false_negatives,
            inconclusive_correct_cases=metrics.inconclusive_correct_cases,
            stage_matched_cases=metrics.stage_matched_cases,
            stage_accuracy_percent=metrics.stage_accuracy_percent,
            inconclusive_accuracy_percent=metrics.inconclusive_accuracy_percent,
            repeatability_pass=False,
            full_case_match_rate=metrics.full_case_match_rate,
        )
    gates = _build_gates(report, metrics)
    verdict = aggregate_plugin_verdict(
        domain_verdict=domain_verdict,
        gate_failures=count_gate_failures(gates),
    )
    analyzer = FunctionalDiagnosticAnalyzer
    return QualificationPluginResult(
        plugin_id=config.plugin_id,
        verdict=verdict,
        metrics=metrics,
        gate_results=gates,
        case_results=cases,
        artifact_ref=config.artifact_ref,
        blocked_reason=report.blocked_reason,
        report_sections=config.extra_report_sections,
        analyzer_class=analyzer.__name__,
        analyzer_module=analyzer.__module__,
        domain_artifact_ref=config.domain_artifact_ref,
    )


def _normalize_cases(
    config: DomainAdapterConfig,
    records: tuple[DomainRunRecord, ...],
) -> tuple[QualificationNormalizedCaseResult, ...]:
    normalized: list[QualificationNormalizedCaseResult] = []
    for record in records:
        expectation = config.expectation_for_record(record)
        expected_first = (
            str(expectation.expected_first_proven_failed_check)
            if expectation.expected_first_proven_failed_check is not None
            else None
        )
        expects_failure = (
            expectation.expected_functional_outcome is QualificationFunctionalOutcome.FAILED
        )
        expects_inconclusive = (
            expectation.expected_operator_outcome is FunctionalOperatorOutcomeStatus.INCONCLUSIVE
        )
        normalized.append(
            QualificationNormalizedCaseResult(
                case_id=record.case_id,
                task_id=record.task_id,
                run_id=record.run_id,
                attempt_id=None,
                tenant_id=config.tenant_id,
                comparison=record.comparison,
                functional_outcome=record.functional_outcome,
                diag_first_failed_check=record.diag_first_failed_check,
                operator_outcome=record.operator_outcome,
                repeat_group=record.repeat_group,
                identity_fidelity_match=record.identity_fidelity_match,
                is_healthy_case=record.case_id in config.healthy_case_ids,
                expects_functional_failure=expects_failure,
                expects_inconclusive=expects_inconclusive,
                expected_first_failed_check=expected_first,
                authoritative_attempt_index=record.authoritative_attempt_index,
                attempt_history=record.attempt_history,
                attempt_count=len(record.attempt_history) if record.attempt_history else 1,
                prerequisite_miss_count=sum(
                    1
                    for item in record.attempt_history
                    if item.precondition_status is QualificationPreconditionStatus.NOT_SATISFIED
                ),
                prerequisite_exhausted=record.prerequisite_exhausted,
                blocked_reason=record.blocked_reason,
            ),
        )
    return tuple(normalized)


def _repeatability_groups(
    config: DomainAdapterConfig,
    records: tuple[DomainRunRecord, ...],
) -> tuple[QualificationRepeatabilityGroup, ...]:
    groups: dict[str, list[tuple[str, ...]]] = {}
    for record in records:
        if record.repeat_group is None or record.repeat_group not in config.repeatability_group_ids:
            continue
        groups.setdefault(record.repeat_group, []).append(config.signature_for_record(record))
    return tuple(
        QualificationRepeatabilityGroup(group_id=group_id, signatures=tuple(signatures))
        for group_id, signatures in sorted(groups.items())
    )


def _build_gates(
    report: DomainQualificationReport,
    metrics: QualificationPluginMetrics,
) -> tuple[QualificationGateResult, ...]:
    gates: list[QualificationGateResult] = [
        core_gate_pass(
            CoreGateId.COMPARISON_PASS.value,
            passed=metrics_pass(metrics),
            summary=f"matched={metrics.matched_cases}/{metrics.total_cases}",
        ),
        core_gate_pass(
            CoreGateId.PLUGIN_EXECUTION_COMPLETED.value,
            passed=report.verdict != "BLOCKED",
            summary=report.blocked_reason,
        ),
        core_gate_pass(
            CoreGateId.EVIDENCE_SCOPE_INTEGRITY.value,
            passed=report.fidelity_pass,
        ),
        core_gate_pass(
            CoreGateId.ORACLE_INDEPENDENCE.value,
            passed=report.decision_independence_pass,
        ),
    ]
    gates.extend(report.domain_extra_gates)
    return tuple(gates)


def comparison_signature(record: DomainRunRecord) -> tuple[str, ...]:
    failed_checks = tuple(
        f"{item.check_id}:{item.expected_status.value}->{item.actual_status.value}"
        for item in record.comparison.check_mismatches
    )
    return (
        record.comparison.result.value,
        record.diag_first_failed_check or "",
        record.operator_outcome or "",
        ",".join(failed_checks),
    )
