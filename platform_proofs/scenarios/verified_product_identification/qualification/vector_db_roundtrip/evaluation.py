"""Provider-neutral vector round-trip evaluation logic."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding import (
    EmbeddingDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    VPI_CANONICAL_EMBEDDING_DIMENSION,
    VPI_CANONICAL_EMBEDDING_MODEL,
    VPI_CANONICAL_EMBEDDING_PROVIDER,
    VPI_CANONICAL_EMBEDDING_REVISION,
    source_ref_key,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
    RelationalDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.qualification.vector_db_roundtrip.contracts import (
    MAX_SCORE_DELTA,
    MIN_RECALL_AT_10,
    MIN_RECALL_AT_5,
    PILOT_RECORD_COUNT,
    PilotArtifactIntegrity,
    QualificationRunStatus,
    QUERY_CASE_COUNT,
    RankedVectorHit,
    SELF_PROBE_MIN_SCORE,
    TIE_EPSILON,
    VectorDbRoundTripMetrics,
    VectorRoundTripQueryEvidence,
    VectorSelfProbeEvidence,
)


class VectorValidationError(ValueError):
    """Raised when stored vectors fail structural validation."""


def validate_embedding_vector(vector: Sequence[float], *, expected_dimension: int) -> None:
    if len(vector) != expected_dimension:
        msg = (
            f"vector length {len(vector)} != expected dimension {expected_dimension}"
        )
        raise VectorValidationError(msg)
    array = np.asarray(vector, dtype=np.float64)
    if not np.isfinite(array).all():
        raise VectorValidationError("vector contains non-finite values")
    norm = float(np.linalg.norm(array))
    if math.isclose(norm, 0.0):
        raise VectorValidationError("vector has zero L2 norm")


def validate_embedding_matrix(matrix: NDArray[np.float64]) -> None:
    if matrix.ndim != 2:
        msg = f"embedding matrix must be 2D, got ndim={matrix.ndim}"
        raise VectorValidationError(msg)
    if matrix.shape[0] == 0:
        raise VectorValidationError("embedding matrix must have at least one row")
    if not np.isfinite(matrix).all():
        raise VectorValidationError("embedding matrix contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0):
        raise VectorValidationError("embedding matrix contains zero-norm vectors")


def evaluate_pilot_artifact_integrity(
    relational_records: Sequence[RelationalDataPackRecord],
    embedding_records: Sequence[EmbeddingDataPackRecord],
) -> PilotArtifactIntegrity:
    failure_reasons: list[str] = []
    relational_count = len(relational_records)
    embedding_count = len(embedding_records)
    if relational_count != PILOT_RECORD_COUNT:
        failure_reasons.append(
            f"relational_count={relational_count} expected={PILOT_RECORD_COUNT}"
        )
    if embedding_count != PILOT_RECORD_COUNT:
        failure_reasons.append(
            f"embedding_count={embedding_count} expected={PILOT_RECORD_COUNT}"
        )

    relational_refs = {source_ref_key(record.source_ref) for record in relational_records}
    embedding_refs = {source_ref_key(record.source_ref) for record in embedding_records}
    source_ref_parity = relational_refs == embedding_refs
    if not source_ref_parity:
        failure_reasons.append("relational and embedding source_ref sets differ")

    logical_point_ids = [record.logical_point_id for record in embedding_records]
    logical_point_ids_unique = len(set(logical_point_ids)) == len(logical_point_ids)
    if not logical_point_ids_unique:
        failure_reasons.append("logical_point_id values are not unique")

    finite_vectors = True
    non_zero_vectors = True
    provider_match = True
    model_match = True
    revision_match = True
    dimension_ok = True

    for record in embedding_records:
        if record.embedding_dimension != VPI_CANONICAL_EMBEDDING_DIMENSION:
            dimension_ok = False
            failure_reasons.append(
                f"unexpected embedding_dimension={record.embedding_dimension}"
            )
            break
        try:
            validate_embedding_vector(
                record.dense_embedding,
                expected_dimension=VPI_CANONICAL_EMBEDDING_DIMENSION,
            )
        except VectorValidationError as exc:
            finite_vectors = False
            non_zero_vectors = False
            failure_reasons.append(str(exc))
            break
        if record.embedding_provider != VPI_CANONICAL_EMBEDDING_PROVIDER:
            provider_match = False
        if record.embedding_model != VPI_CANONICAL_EMBEDDING_MODEL:
            model_match = False
        if record.embedding_model_revision != VPI_CANONICAL_EMBEDDING_REVISION:
            revision_match = False
        if not record.logical_point_id.strip():
            failure_reasons.append("empty logical_point_id encountered")
            break
        if not record.semantic_text_hash.strip():
            failure_reasons.append("missing semantic_text_hash")
            break

    if not provider_match:
        failure_reasons.append("embedding provider mismatch")
    if not model_match:
        failure_reasons.append("embedding model mismatch")
    if not revision_match:
        failure_reasons.append("embedding revision mismatch")

    passed = (
        relational_count == PILOT_RECORD_COUNT
        and embedding_count == PILOT_RECORD_COUNT
        and source_ref_parity
        and dimension_ok
        and finite_vectors
        and non_zero_vectors
        and provider_match
        and model_match
        and revision_match
        and logical_point_ids_unique
        and not failure_reasons
    )
    return PilotArtifactIntegrity(
        relational_count=relational_count,
        embedding_count=embedding_count,
        source_ref_parity=source_ref_parity,
        embedding_dimension=VPI_CANONICAL_EMBEDDING_DIMENSION,
        finite_vectors=finite_vectors,
        non_zero_vectors=non_zero_vectors,
        provider_match=provider_match,
        model_match=model_match,
        revision_match=revision_match,
        logical_point_ids_unique=logical_point_ids_unique,
        passed=passed,
        failure_reasons=tuple(failure_reasons),
    )


def build_embedding_matrix(
    embedding_records: Sequence[EmbeddingDataPackRecord],
) -> NDArray[np.float64]:
    matrix = np.asarray(
        [list(record.dense_embedding) for record in embedding_records],
        dtype=np.float64,
    )
    validate_embedding_matrix(matrix)
    return matrix


def compute_cosine_scores(
    corpus_embeddings: NDArray[np.float64],
    query_vector: NDArray[np.float64],
) -> NDArray[np.float64]:
    validate_embedding_matrix(corpus_embeddings)
    if query_vector.ndim != 1:
        msg = f"query vector must be 1D, got ndim={query_vector.ndim}"
        raise VectorValidationError(msg)
    if query_vector.shape[0] != corpus_embeddings.shape[1]:
        msg = (
            f"query dimension {query_vector.shape[0]} != corpus dimension "
            f"{corpus_embeddings.shape[1]}"
        )
        raise VectorValidationError(msg)
    if not np.isfinite(query_vector).all():
        raise VectorValidationError("query vector contains non-finite values")
    query_norm = float(np.linalg.norm(query_vector))
    if math.isclose(query_norm, 0.0):
        raise VectorValidationError("query vector has zero L2 norm")
    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1)
    if np.any(corpus_norms == 0.0):
        raise VectorValidationError("corpus contains zero-norm vectors")
    return corpus_embeddings @ query_vector / (corpus_norms * query_norm)


def rank_by_cosine_deterministic(
    corpus_embeddings: NDArray[np.float64],
    logical_point_ids: Sequence[str],
    query_vector: NDArray[np.float64],
    *,
    top_k: int,
) -> tuple[RankedVectorHit, ...]:
    if len(logical_point_ids) != corpus_embeddings.shape[0]:
        msg = "logical_point_ids length must match corpus row count"
        raise ValueError(msg)
    if top_k <= 0:
        msg = "top_k must be > 0"
        raise ValueError(msg)
    scores = compute_cosine_scores(corpus_embeddings, query_vector)
    limit = min(top_k, scores.shape[0])
    order = sorted(
        range(scores.shape[0]),
        key=lambda index: (-float(scores[index]), logical_point_ids[index]),
    )[:limit]
    return tuple(
        RankedVectorHit(
            logical_point_id=logical_point_ids[index],
            rank=rank,
            cosine_score=float(scores[index]),
        )
        for rank, index in enumerate(order, start=1)
    )


def tie_aware_top1_ids(
    corpus_embeddings: NDArray[np.float64],
    logical_point_ids: Sequence[str],
    query_vector: NDArray[np.float64],
) -> tuple[str, ...]:
    scores = compute_cosine_scores(corpus_embeddings, query_vector)
    best_score = float(np.max(scores))
    threshold = best_score - TIE_EPSILON
    tied = [
        logical_point_ids[index]
        for index, score in enumerate(scores)
        if float(score) >= threshold
    ]
    return tuple(sorted(tied))


def exact_neighbor_recall_at_k(
    baseline_ids: Sequence[str],
    candidate_ids: Sequence[str],
    *,
    k: int,
) -> float:
    if k <= 0:
        msg = "k must be > 0"
        raise ValueError(msg)
    baseline_top = tuple(baseline_ids[:k])
    if not baseline_top:
        return 0.0
    candidate_set = set(candidate_ids[:k])
    matches = sum(1 for logical_id in baseline_top if logical_id in candidate_set)
    return matches / len(baseline_top)


def _score_delta_for_matched_hits(
    baseline_hits: Sequence[RankedVectorHit],
    candidate_scores: Mapping[str, float],
) -> tuple[float, float, int]:
    deltas: list[float] = []
    for hit in baseline_hits:
        candidate_score = candidate_scores.get(hit.logical_point_id)
        if candidate_score is None:
            continue
        deltas.append(abs(hit.cosine_score - candidate_score))
    if not deltas:
        return 0.0, 0.0, 0
    return float(sum(deltas) / len(deltas)), float(max(deltas)), len(deltas)


def evaluate_query_round_trip(
    *,
    query_id: str,
    query_text: str,
    query_vector: NDArray[np.float64],
    corpus_embeddings: NDArray[np.float64],
    logical_point_ids: Sequence[str],
    qdrant_hits: Sequence[RankedVectorHit],
    source_ref_by_logical_id: Mapping[str, SourceRecordRef],
    qdrant_source_refs: Mapping[str, SourceRecordRef],
) -> VectorRoundTripQueryEvidence:
    baseline_top10 = rank_by_cosine_deterministic(
        corpus_embeddings,
        logical_point_ids,
        query_vector,
        top_k=10,
    )
    baseline_top5 = baseline_top10[:5]
    baseline_top1_ids = tie_aware_top1_ids(
        corpus_embeddings,
        logical_point_ids,
        query_vector,
    )
    qdrant_top1_id = qdrant_hits[0].logical_point_id if qdrant_hits else ""
    top1_parity = qdrant_top1_id in baseline_top1_ids if qdrant_top1_id else False
    qdrant_ids = tuple(hit.logical_point_id for hit in qdrant_hits)
    recall_at_5 = exact_neighbor_recall_at_k(
        tuple(hit.logical_point_id for hit in baseline_top5),
        qdrant_ids,
        k=5,
    )
    recall_at_10 = exact_neighbor_recall_at_k(
        tuple(hit.logical_point_id for hit in baseline_top10),
        qdrant_ids,
        k=10,
    )
    candidate_scores = {
        hit.logical_point_id: hit.cosine_score for hit in qdrant_hits
    }
    mean_delta, max_delta, _ = _score_delta_for_matched_hits(
        baseline_top10,
        candidate_scores,
    )
    unknown_logical_ids = 0
    source_ref_mismatches = 0
    for hit in qdrant_hits:
        if hit.logical_point_id not in source_ref_by_logical_id:
            unknown_logical_ids += 1
            continue
        expected = source_ref_by_logical_id[hit.logical_point_id]
        returned = qdrant_source_refs.get(hit.logical_point_id)
        if returned is None or source_ref_key(expected) != source_ref_key(returned):
            source_ref_mismatches += 1
    return VectorRoundTripQueryEvidence(
        query_id=query_id,
        query_text=query_text,
        baseline_top1_ids=baseline_top1_ids,
        qdrant_top1_id=qdrant_top1_id,
        top1_parity=top1_parity,
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        mean_absolute_score_delta=mean_delta,
        max_absolute_score_delta=max_delta,
        unknown_logical_point_ids=unknown_logical_ids,
        source_ref_mismatches=source_ref_mismatches,
    )


def aggregate_query_metrics(
    query_evidence: Sequence[VectorRoundTripQueryEvidence],
) -> VectorDbRoundTripMetrics:
    if not query_evidence:
        return VectorDbRoundTripMetrics(
            tie_aware_top1_parity_rate=0.0,
            mean_recall_at_5=0.0,
            mean_recall_at_10=0.0,
            mean_absolute_score_delta=0.0,
            max_absolute_score_delta=0.0,
            unknown_logical_point_id_count=0,
            source_ref_mismatch_count=0,
            embedding_transport_correctness=False,
            vector_index_ranking_parity=False,
        )
    top1_matches = sum(1 for item in query_evidence if item.top1_parity)
    mean_recall_at_5 = sum(item.recall_at_5 for item in query_evidence) / len(query_evidence)
    mean_recall_at_10 = sum(item.recall_at_10 for item in query_evidence) / len(query_evidence)
    mean_abs_delta = (
        sum(item.mean_absolute_score_delta for item in query_evidence) / len(query_evidence)
    )
    max_abs_delta = max(item.max_absolute_score_delta for item in query_evidence)
    unknown_ids = sum(item.unknown_logical_point_ids for item in query_evidence)
    source_ref_mismatches = sum(item.source_ref_mismatches for item in query_evidence)
    transport_ok = unknown_ids == 0 and source_ref_mismatches == 0 and max_abs_delta <= MAX_SCORE_DELTA
    ranking_ok = (
        top1_matches == len(query_evidence)
        and mean_recall_at_5 >= MIN_RECALL_AT_5
        and mean_recall_at_10 >= MIN_RECALL_AT_10
        and unknown_ids == 0
        and source_ref_mismatches == 0
    )
    return VectorDbRoundTripMetrics(
        tie_aware_top1_parity_rate=top1_matches / len(query_evidence),
        mean_recall_at_5=mean_recall_at_5,
        mean_recall_at_10=mean_recall_at_10,
        mean_absolute_score_delta=mean_abs_delta,
        max_absolute_score_delta=max_abs_delta,
        unknown_logical_point_id_count=unknown_ids,
        source_ref_mismatch_count=source_ref_mismatches,
        embedding_transport_correctness=transport_ok,
        vector_index_ranking_parity=ranking_ok,
    )


def evaluate_self_probe(
    *,
    pilot_index: int,
    expected_logical_point_id: str,
    expected_source_ref: SourceRecordRef,
    returned_logical_point_id: str,
    returned_source_ref: SourceRecordRef,
    self_cosine_score: float,
) -> VectorSelfProbeEvidence:
    top1_identity_match = returned_logical_point_id == expected_logical_point_id
    metadata_identity_match = (
        source_ref_key(expected_source_ref) == source_ref_key(returned_source_ref)
    )
    passed = (
        top1_identity_match
        and metadata_identity_match
        and self_cosine_score >= SELF_PROBE_MIN_SCORE
    )
    return VectorSelfProbeEvidence(
        pilot_index=pilot_index,
        logical_point_id=expected_logical_point_id,
        offer_id=expected_source_ref.offer_id.value,
        catalog_id=expected_source_ref.catalog_id,
        source_revision=expected_source_ref.source_revision,
        returned_logical_point_id=returned_logical_point_id,
        self_cosine_score=self_cosine_score,
        top1_identity_match=top1_identity_match,
        metadata_identity_match=metadata_identity_match,
        passed=passed,
    )


def evaluate_hard_gate(
    *,
    artifact_integrity: PilotArtifactIntegrity,
    self_probes: Sequence[VectorSelfProbeEvidence],
    query_evidence: Sequence[VectorRoundTripQueryEvidence],
    metrics: VectorDbRoundTripMetrics,
    document_embedding_calls: int,
    query_embedding_calls: int,
    qdrant_point_count: int,
    qdrant_dimension: int,
    qdrant_metric: str,
    dense_search_available: bool,
) -> QualificationRunStatus:
    if not artifact_integrity.passed:
        return QualificationRunStatus.ARTIFACT_INTEGRITY_FAIL
    if document_embedding_calls != 0:
        return QualificationRunStatus.FAIL
    if query_embedding_calls != QUERY_CASE_COUNT:
        return QualificationRunStatus.FAIL
    if qdrant_point_count != PILOT_RECORD_COUNT:
        return QualificationRunStatus.FAIL
    if qdrant_dimension != VPI_CANONICAL_EMBEDDING_DIMENSION:
        return QualificationRunStatus.FAIL
    if qdrant_metric != "cosine":
        return QualificationRunStatus.FAIL
    if not dense_search_available:
        return QualificationRunStatus.FAIL
    if len(self_probes) != 8 or not all(probe.passed for probe in self_probes):
        return QualificationRunStatus.FAIL
    if not metrics.embedding_transport_correctness:
        return QualificationRunStatus.FAIL
    if not metrics.vector_index_ranking_parity:
        return QualificationRunStatus.VECTOR_INDEX_RETRIEVAL_REGRESSION
    if len(query_evidence) != QUERY_CASE_COUNT:
        return QualificationRunStatus.FAIL
    return QualificationRunStatus.PASS
