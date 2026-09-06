"""Typed proof evidence report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)


@dataclass(frozen=True, slots=True)
class ValidationSectionResult:
    status: str
    detail: str


@dataclass(frozen=True, slots=True)
class RetrievalMetricRow:
    query_id: str
    channel: str
    recall_at_1: float | None
    recall_at_5: float | None
    recall_at_10: float | None
    mrr_at_10: float | None
    ndcg_at_10: float | None
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class DataPackProofReport:
    status: DataPackStatus
    data_pack_identity: str
    record_count: int
    relational_validation: ValidationSectionResult
    embedding_validation: ValidationSectionResult
    cross_ref_validation: ValidationSectionResult
    checksum_validation: ValidationSectionResult
    semantic_text_hash_validation: ValidationSectionResult
    relational_load: ValidationSectionResult
    vector_load: ValidationSectionResult
    zero_embedding_calls: ValidationSectionResult
    retrieval_metrics: tuple[RetrievalMetricRow, ...]
    mapping_validation: ValidationSectionResult
    negative_match_validation: ValidationSectionResult
    provider_configuration: str
    idempotent_reload: ValidationSectionResult
    warnings: tuple[str, ...]
    known_gaps: tuple[str, ...]


def _section_to_dict(section: ValidationSectionResult) -> dict[str, str]:
    return {"status": section.status, "detail": section.detail}


def proof_report_to_json_dict(report: DataPackProofReport) -> dict[str, object]:
    return {
        "status": report.status.value,
        "data_pack_identity": report.data_pack_identity,
        "record_count": report.record_count,
        "relational_validation": _section_to_dict(report.relational_validation),
        "embedding_validation": _section_to_dict(report.embedding_validation),
        "cross_ref_validation": _section_to_dict(report.cross_ref_validation),
        "checksum_validation": _section_to_dict(report.checksum_validation),
        "semantic_text_hash_validation": _section_to_dict(report.semantic_text_hash_validation),
        "relational_load": _section_to_dict(report.relational_load),
        "vector_load": _section_to_dict(report.vector_load),
        "zero_embedding_calls": _section_to_dict(report.zero_embedding_calls),
        "retrieval_metrics": [
            {
                "query_id": row.query_id,
                "channel": row.channel,
                "recall_at_1": row.recall_at_1,
                "recall_at_5": row.recall_at_5,
                "recall_at_10": row.recall_at_10,
                "mrr_at_10": row.mrr_at_10,
                "ndcg_at_10": row.ndcg_at_10,
                "passed": row.passed,
                "detail": row.detail,
            }
            for row in report.retrieval_metrics
        ],
        "mapping_validation": _section_to_dict(report.mapping_validation),
        "negative_match_validation": _section_to_dict(report.negative_match_validation),
        "provider_configuration": report.provider_configuration,
        "idempotent_reload": _section_to_dict(report.idempotent_reload),
        "warnings": list(report.warnings),
        "known_gaps": list(report.known_gaps),
    }


def write_proof_report(path: Path, report: DataPackProofReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(proof_report_to_json_dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)
