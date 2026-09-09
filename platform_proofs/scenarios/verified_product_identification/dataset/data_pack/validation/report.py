"""Validation report serialization helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.contracts import (
    FullDataPackValidationReport,
)


def write_validation_report_json(path: Path, report: FullDataPackValidationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def write_validation_report_markdown(path: Path, report: FullDataPackValidationReport) -> None:
    summary = report.summary
    lines = [
        "# Full Data Pack Validation Report",
        "",
        f"- verdict: {report.verdict.value}",
        f"- artifact_root: {summary.artifact_root}",
        f"- validation_timestamp_utc: {summary.validation_timestamp_utc}",
        f"- expected_record_count: {summary.expected_record_count}",
        f"- observed_relational_count: {summary.observed_relational_count}",
        f"- observed_embedding_count: {summary.observed_embedding_count}",
        f"- relational_shard_count: {summary.relational_shard_count}",
        f"- embedding_shard_count: {summary.embedding_shard_count}",
        f"- ready_shard_count: {summary.ready_shard_count}",
        f"- first_invalid_shard_ordinal: {summary.first_invalid_shard_ordinal}",
        f"- non_finite_vector_count: {summary.non_finite_vector_count}",
        f"- zero_vector_count: {summary.zero_vector_count}",
        f"- semantic_hash_mismatch_count: {summary.semantic_hash_mismatch_count}",
        f"- finalized_artifact_valid: {'YES' if summary.finalized_artifact_valid else 'NO'}",
        "",
        "## Coverage",
        "",
        f"- passed: {report.coverage_summary.passed}",
        f"- first_gap_ordinal: {report.coverage_summary.first_gap_ordinal}",
        f"- first_overlap_ordinal: {report.coverage_summary.first_overlap_ordinal}",
        "",
        "## Duplicates",
        "",
        f"- passed: {report.duplicate_summary.passed}",
        f"- duplicate_global_row_index_count: {report.duplicate_summary.duplicate_global_row_index_count}",
        f"- duplicate_source_ref_count: {report.duplicate_summary.duplicate_source_ref_count}",
        f"- duplicate_logical_point_id_count: {report.duplicate_summary.duplicate_logical_point_id_count}",
        "",
        "## Failed Checks",
        "",
    ]
    for check in report.all_checks:
        if check.status.value != "PASS":
            lines.append(f"- {check.name}: {check.detail}")
    if len(lines) == lines.index("## Failed Checks") + 3:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
