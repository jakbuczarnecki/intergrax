"""Immutable contracts for vector DB round-trip retrieval qualification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

QUALIFICATION_TASK_ID = "VPI-IMPLEMENTATION-5C4E5A"
PILOT_RECORD_COUNT = 1000
QUERY_CASE_COUNT = 32
SELF_PROBE_INDICES: tuple[int, ...] = (0, 1, 7, 31, 127, 255, 511, 999)
TIE_EPSILON = 1e-6
MAX_SCORE_DELTA = 1e-4
SELF_PROBE_MIN_SCORE = 0.9999
MIN_RECALL_AT_5 = 0.99
MIN_RECALL_AT_10 = 0.99


class QualificationRunStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    ARTIFACT_INTEGRITY_FAIL = "ARTIFACT_INTEGRITY_FAIL"
    VECTOR_INDEX_RETRIEVAL_REGRESSION = "VECTOR_INDEX_RETRIEVAL_REGRESSION"
    RESOURCE_PRECONDITION_FAIL = "RESOURCE_PRECONDITION_FAIL"
    CORRECTION_REQUIRED = "CORRECTION_REQUIRED"


@dataclass(frozen=True, slots=True)
class PilotArtifactIntegrity:
    relational_count: int
    embedding_count: int
    source_ref_parity: bool
    embedding_dimension: int
    finite_vectors: bool
    non_zero_vectors: bool
    provider_match: bool
    model_match: bool
    revision_match: bool
    logical_point_ids_unique: bool
    passed: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RankedVectorHit:
    logical_point_id: str
    rank: int
    cosine_score: float


@dataclass(frozen=True, slots=True)
class VectorSelfProbeEvidence:
    pilot_index: int
    logical_point_id: str
    offer_id: str
    catalog_id: str
    source_revision: str | None
    returned_logical_point_id: str
    self_cosine_score: float
    top1_identity_match: bool
    metadata_identity_match: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class VectorRoundTripQueryEvidence:
    query_id: str
    query_text: str
    baseline_top1_ids: tuple[str, ...]
    qdrant_top1_id: str
    top1_parity: bool
    recall_at_5: float
    recall_at_10: float
    mean_absolute_score_delta: float
    max_absolute_score_delta: float
    unknown_logical_point_ids: int
    source_ref_mismatches: int


@dataclass(frozen=True, slots=True)
class VectorDbRoundTripMetrics:
    tie_aware_top1_parity_rate: float
    mean_recall_at_5: float
    mean_recall_at_10: float
    mean_absolute_score_delta: float
    max_absolute_score_delta: float
    unknown_logical_point_id_count: int
    source_ref_mismatch_count: int
    embedding_transport_correctness: bool
    vector_index_ranking_parity: bool


@dataclass(frozen=True, slots=True)
class QdrantIndexSnapshot:
    collection_name: str
    metric: str
    dimension: int
    point_count: int
    dense_search_available: bool
    temporary_isolated_collection: bool
    cleanup_passed: bool


@dataclass(frozen=True, slots=True)
class VectorDbRoundTripQualificationReport:
    task_id: str
    status: QualificationRunStatus
    git_sha: str
    pilot_root: str
    pilot_relational_checksum: str
    pilot_embedding_checksum: str
    provider: str
    model: str
    revision: str
    dimension: int
    document_policy: str
    document_token_budget: int
    effective_provider_ceiling: int
    query_policy_changed: bool
    qdrant: QdrantIndexSnapshot
    document_embedding_calls: int
    query_embedding_calls: int
    model_load_count: int
    artifact_integrity: PilotArtifactIntegrity
    self_probes: tuple[VectorSelfProbeEvidence, ...]
    query_evidence: tuple[VectorRoundTripQueryEvidence, ...]
    metrics: VectorDbRoundTripMetrics
    known_gaps: tuple[str, ...]
