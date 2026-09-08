"""JSON and Markdown reporting for vector DB round-trip qualification."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.qualification.vector_db_roundtrip.contracts import (
    VectorDbRoundTripQualificationReport,
)


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    msg = f"unsupported type for vector DB round-trip qualification JSON: {type(value)!r}"
    raise TypeError(msg)


def vector_db_roundtrip_report_to_json(report: VectorDbRoundTripQualificationReport) -> str:
    return json.dumps(asdict(report), indent=2, default=_json_default, sort_keys=True)


def write_vector_db_roundtrip_report_json(
    path: Path,
    report: VectorDbRoundTripQualificationReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(vector_db_roundtrip_report_to_json(report), encoding="utf-8")


def render_vector_db_roundtrip_markdown_report(
    report: VectorDbRoundTripQualificationReport,
) -> str:
    lines: list[str] = [
        "# Vector DB Round-Trip Qualification Report",
        "",
        f"Task: {report.task_id}",
        f"Status: {report.status.value}",
        f"Git SHA: {report.git_sha}",
        "",
        "## Pilot artifact",
        "",
        f"- Root: `{report.pilot_root}`",
        f"- Relational checksum: `{report.pilot_relational_checksum}`",
        f"- Embedding checksum: `{report.pilot_embedding_checksum}`",
        f"- Relational count: {report.artifact_integrity.relational_count}",
        f"- Embedding count: {report.artifact_integrity.embedding_count}",
        f"- Source-ref parity: {'PASS' if report.artifact_integrity.source_ref_parity else 'FAIL'}",
        f"- Finite vectors: {'PASS' if report.artifact_integrity.finite_vectors else 'FAIL'}",
        f"- Non-zero vectors: {'PASS' if report.artifact_integrity.non_zero_vectors else 'FAIL'}",
        "",
        "## Model",
        "",
        f"- Provider: {report.provider}",
        f"- Model: {report.model}",
        f"- Revision: {report.revision}",
        f"- Dimension: {report.dimension}",
        f"- Model load count: {report.model_load_count}",
        "",
        "## Policy",
        "",
        f"- Document policy: {report.document_policy}",
        f"- Document budget: {report.document_token_budget}",
        f"- Effective provider ceiling: {report.effective_provider_ceiling}",
        f"- Query policy changed: {'YES' if report.query_policy_changed else 'NO'}",
        "",
        "## Qdrant",
        "",
        f"- Collection: `{report.qdrant.collection_name}`",
        f"- Metric: {report.qdrant.metric}",
        f"- Dimension: {report.qdrant.dimension}",
        f"- Point count: {report.qdrant.point_count}",
        (
            "- Temporary isolated collection: "
            f"{'YES' if report.qdrant.temporary_isolated_collection else 'NO'}"
        ),
        f"- Cleanup: {'PASS' if report.qdrant.cleanup_passed else 'FAIL'}",
        "",
        "## Embedding calls",
        "",
        f"- Document embedding calls: {report.document_embedding_calls}",
        f"- Query embedding calls: {report.query_embedding_calls}",
        "",
        "## Self-vector probes",
        "",
        f"- Count: {len(report.self_probes)}",
        f"- Passed: {sum(1 for probe in report.self_probes if probe.passed)}",
        (
            "- Minimum self cosine: "
            f"{min((probe.self_cosine_score for probe in report.self_probes), default=0.0):.6f}"
        ),
        "",
        "## Real query parity",
        "",
        f"- Query count: {len(report.query_evidence)}",
        f"- Tie-aware top-1 parity: {report.metrics.tie_aware_top1_parity_rate * 100:.2f}%",
        f"- Mean Recall@5: {report.metrics.mean_recall_at_5:.6f}",
        f"- Mean Recall@10: {report.metrics.mean_recall_at_10:.6f}",
        f"- Mean absolute score delta: {report.metrics.mean_absolute_score_delta:.8f}",
        f"- Max absolute score delta: {report.metrics.max_absolute_score_delta:.8f}",
        f"- Unknown logical IDs: {report.metrics.unknown_logical_point_id_count}",
        f"- Source-ref mismatches: {report.metrics.source_ref_mismatch_count}",
        "",
        "## Transport and ranking",
        "",
        (
            "- Embedding transport correctness: "
            f"{'PASS' if report.metrics.embedding_transport_correctness else 'FAIL'}"
        ),
        (
            "- Vector index ranking parity: "
            f"{'PASS' if report.metrics.vector_index_ranking_parity else 'FAIL'}"
        ),
        "",
        "## Final verdict",
        "",
        f"Safe to proceed with full 3,770,377 embedding build: "
        f"{'YES' if report.status.value == 'PASS' else 'NO'}",
    ]
    if report.known_gaps:
        lines.extend(["", "## Known gaps", ""])
        lines.extend(f"- {gap}" for gap in report.known_gaps)
    return "\n".join(lines) + "\n"
