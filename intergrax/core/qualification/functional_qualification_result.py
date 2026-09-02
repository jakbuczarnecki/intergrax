# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed result models for functional qualification (DIAG-FUNCTIONAL-Q5)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.core.qualification.functional_qualification_case import QualificationNormalizedCaseResult
from intergrax.core.qualification.functional_qualification_fidelity import QualificationGateResult
from intergrax.core.qualification.functional_qualification_identity import FunctionalQualificationPluginId
from intergrax.core.qualification.functional_qualification_metrics import QualificationPluginMetrics
from intergrax.core.qualification.functional_qualification_verdict import QualificationVerdict
from intergrax.knowledge.contracts.validation import JsonObject, JsonValue


@dataclass(frozen=True, slots=True)
class QualificationReportSection:
    title: str
    summary: str
    rows: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class QualificationPluginResult:
    plugin_id: FunctionalQualificationPluginId
    verdict: QualificationVerdict
    metrics: QualificationPluginMetrics
    gate_results: tuple[QualificationGateResult, ...]
    case_results: tuple[QualificationNormalizedCaseResult, ...]
    artifact_ref: str | None
    blocked_reason: str | None
    report_sections: tuple[QualificationReportSection, ...]
    analyzer_class: str
    analyzer_module: str
    domain_artifact_ref: str | None = None


@dataclass(frozen=True, slots=True)
class QualificationRunReport:
    schema_version: str
    run_id: str
    verdict: QualificationVerdict
    plan_plugin_ids: tuple[FunctionalQualificationPluginId, ...]
    plugin_results: tuple[QualificationPluginResult, ...]
    aggregate_metrics: QualificationPluginMetrics
    analyzer_identity: tuple[str, str]
    domain_specific_analyzer_count: int
    extension_change_surface: JsonObject
    cross_domain_isolation_pass: bool
    collision_safety_pass: bool


def plugin_result_to_json(plugin: QualificationPluginResult) -> JsonObject:
    return {
        "plugin_id": plugin.plugin_id.value,
        "verdict": plugin.verdict.value,
        "blocked_reason": plugin.blocked_reason,
        "artifact_ref": plugin.artifact_ref,
        "domain_artifact_ref": plugin.domain_artifact_ref,
        "analyzer_class": plugin.analyzer_class,
        "analyzer_module": plugin.analyzer_module,
        "metrics": _metrics_to_json(plugin.metrics),
        "gate_results": [_gate_to_json(gate) for gate in plugin.gate_results],
        "case_results": [_case_to_json(case) for case in plugin.case_results],
        "report_sections": [_section_to_json(section) for section in plugin.report_sections],
    }


def _metrics_to_json(metrics: QualificationPluginMetrics) -> JsonObject:
    return {
        "total_cases": metrics.total_cases,
        "matched_cases": metrics.matched_cases,
        "mismatched_cases": metrics.mismatched_cases,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "inconclusive_correct_cases": metrics.inconclusive_correct_cases,
        "stage_matched_cases": metrics.stage_matched_cases,
        "stage_accuracy_percent": metrics.stage_accuracy_percent,
        "inconclusive_accuracy_percent": metrics.inconclusive_accuracy_percent,
        "repeatability_pass": metrics.repeatability_pass,
        "full_case_match_rate": metrics.full_case_match_rate,
        "total_attempts": metrics.total_attempts,
        "prerequisite_misses": metrics.prerequisite_misses,
        "cases_requiring_retry": metrics.cases_requiring_retry,
        "prerequisite_exhaustions": metrics.prerequisite_exhaustions,
        "prerequisite_success_rate": metrics.prerequisite_success_rate,
    }


def _gate_to_json(gate: QualificationGateResult) -> JsonObject:
    return {
        "gate_id": gate.gate_id,
        "status": gate.status.value,
        "summary": gate.summary,
    }


def _case_to_json(case: QualificationNormalizedCaseResult) -> JsonObject:
    return {
        "case_id": case.case_id,
        "task_id": case.task_id,
        "run_id": case.run_id,
        "attempt_id": case.attempt_id,
        "tenant_id": case.tenant_id,
        "comparison_result": case.comparison.result.value,
        "functional_outcome": case.functional_outcome.value,
        "diag_first_failed_check": case.diag_first_failed_check,
        "operator_outcome": case.operator_outcome,
        "repeat_group": case.repeat_group,
        "identity_fidelity_match": case.identity_fidelity_match,
        "authoritative_attempt_index": case.authoritative_attempt_index,
        "attempt_count": case.attempt_count,
        "prerequisite_exhausted": case.prerequisite_exhausted,
        "blocked_reason": case.blocked_reason,
        "attempt_history": [
            {
                "attempt_index": attempt.attempt_index,
                "precondition_status": attempt.precondition_status.value,
                "task_id": attempt.task_id,
                "run_id": attempt.run_id,
                "summary": attempt.summary,
            }
            for attempt in case.attempt_history
        ],
    }


def _section_to_json(section: QualificationReportSection) -> JsonObject:
  rows: list[JsonValue] = [{"key": key, "value": value} for key, value in section.rows]
  return {
      "title": section.title,
      "summary": section.summary,
      "rows": rows,
  }


__all__ = [
    "QualificationPluginResult",
    "QualificationReportSection",
    "QualificationRunReport",
    "plugin_result_to_json",
]
