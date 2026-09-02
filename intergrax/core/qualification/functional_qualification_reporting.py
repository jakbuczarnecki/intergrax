# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Machine and human report serialization for functional qualification (DIAG-FUNCTIONAL-Q5)."""

from __future__ import annotations

import json
from pathlib import Path

from intergrax.core.qualification.functional_qualification_result import (
    QualificationRunReport,
    plugin_result_to_json,
)
from intergrax.knowledge.contracts.validation import JsonObject


def qualification_run_report_to_json(report: QualificationRunReport) -> JsonObject:
    analyzer_class, analyzer_module = report.analyzer_identity
    return {
        "schema_version": report.schema_version,
        "run_id": report.run_id,
        "verdict": report.verdict.value,
        "plan_plugin_ids": [plugin_id.value for plugin_id in report.plan_plugin_ids],
        "plugin_results": [plugin_result_to_json(item) for item in report.plugin_results],
        "aggregate_metrics": {
            "total_cases": report.aggregate_metrics.total_cases,
            "matched_cases": report.aggregate_metrics.matched_cases,
            "mismatched_cases": report.aggregate_metrics.mismatched_cases,
            "false_positives": report.aggregate_metrics.false_positives,
            "false_negatives": report.aggregate_metrics.false_negatives,
            "inconclusive_correct_cases": report.aggregate_metrics.inconclusive_correct_cases,
            "stage_matched_cases": report.aggregate_metrics.stage_matched_cases,
            "stage_accuracy_percent": report.aggregate_metrics.stage_accuracy_percent,
            "inconclusive_accuracy_percent": report.aggregate_metrics.inconclusive_accuracy_percent,
            "repeatability_pass": report.aggregate_metrics.repeatability_pass,
            "full_case_match_rate": report.aggregate_metrics.full_case_match_rate,
            "total_attempts": report.aggregate_metrics.total_attempts,
            "prerequisite_misses": report.aggregate_metrics.prerequisite_misses,
            "cases_requiring_retry": report.aggregate_metrics.cases_requiring_retry,
            "prerequisite_exhaustions": report.aggregate_metrics.prerequisite_exhaustions,
            "prerequisite_success_rate": report.aggregate_metrics.prerequisite_success_rate,
        },
        "analyzer_identity": {
            "class": analyzer_class,
            "module": analyzer_module,
            "domain_specific_analyzer_count": report.domain_specific_analyzer_count,
        },
        "extension_change_surface": report.extension_change_surface,
        "cross_domain_isolation_pass": report.cross_domain_isolation_pass,
        "collision_safety_pass": report.collision_safety_pass,
    }


def write_qualification_run_report(path: Path, report: QualificationRunReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = qualification_run_report_to_json(report)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


__all__ = [
    "qualification_run_report_to_json",
    "write_qualification_run_report",
]
