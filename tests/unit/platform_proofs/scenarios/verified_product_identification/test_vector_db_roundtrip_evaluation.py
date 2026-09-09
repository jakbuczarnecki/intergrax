"""CPU-only tests for vector DB round-trip qualification logic."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.qualification.vector_db_roundtrip.contracts import (
    PILOT_RECORD_COUNT,
    QUERY_CASE_COUNT,
    PilotArtifactIntegrity,
    QualificationRunStatus,
    RankedVectorHit,
    VectorDbRoundTripMetrics,
    VectorRoundTripQueryEvidence,
    VectorSelfProbeEvidence,
)
from platform_proofs.scenarios.verified_product_identification.qualification.vector_db_roundtrip.evaluation import (
    VectorValidationError,
    aggregate_query_metrics,
    compute_cosine_scores,
    evaluate_hard_gate,
    evaluate_query_round_trip,
    exact_neighbor_recall_at_k,
    rank_by_cosine_deterministic,
    tie_aware_top1_ids,
    validate_embedding_matrix,
    validate_embedding_vector,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_EVALUATION_PATH = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/qualification/vector_db_roundtrip/evaluation.py"
)


def _source_ref(offer_id: str) -> SourceRecordRef:
    return SourceRecordRef(
        offer_id=ProductOfferId(offer_id),
        catalog_id="wdc-v2-selected",
        source_revision=None,
    )


def test_exact_cosine_ranking_is_deterministic() -> None:
    corpus = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float64)
    query = np.asarray([1.0, 0.0], dtype=np.float64)
    ranked = rank_by_cosine_deterministic(
        corpus,
        ("a", "b", "c"),
        query,
        top_k=2,
    )
    assert [hit.logical_point_id for hit in ranked] == ["a", "b"]


def test_zero_norm_vector_rejection() -> None:
    with pytest.raises(VectorValidationError, match="zero L2 norm"):
        validate_embedding_vector((0.0, 0.0), expected_dimension=2)


def test_non_finite_vector_rejection() -> None:
    with pytest.raises(VectorValidationError, match="non-finite"):
        validate_embedding_vector((1.0, float("nan")), expected_dimension=2)
    matrix = np.asarray([[1.0, float("inf")]], dtype=np.float64)
    with pytest.raises(VectorValidationError, match="non-finite"):
        validate_embedding_matrix(matrix)


def test_deterministic_tie_handling() -> None:
    corpus = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    query = np.asarray([1.0, 0.0], dtype=np.float64)
    top1_ids = tie_aware_top1_ids(corpus, ("b-id", "a-id", "c-id"), query)
    assert top1_ids == ("a-id", "b-id")


def test_top1_parity_with_tie_set() -> None:
    corpus = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    query = np.asarray([1.0, 0.0], dtype=np.float64)
    baseline_top1 = tie_aware_top1_ids(corpus, ("a", "b", "c"), query)
    evidence = evaluate_query_round_trip(
        query_id="q-0001",
        query_text="probe",
        query_vector=query,
        corpus_embeddings=corpus,
        logical_point_ids=("a", "b", "c"),
        qdrant_hits=(
            RankedVectorHit(logical_point_id="b", rank=1, cosine_score=1.0),
            RankedVectorHit(logical_point_id="a", rank=2, cosine_score=1.0),
        ),
        source_ref_by_logical_id={
            "a": _source_ref("a"),
            "b": _source_ref("b"),
            "c": _source_ref("c"),
        },
        qdrant_source_refs={
            "a": _source_ref("a"),
            "b": _source_ref("b"),
        },
    )
    assert evidence.top1_parity is True
    assert baseline_top1 == ("a", "b")


def test_recall_at_5_and_10() -> None:
    baseline = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
    candidate = ("a", "x", "b", "y", "c", "d", "e", "f", "g", "h")
    assert exact_neighbor_recall_at_k(baseline, candidate, k=5) == 0.6
    assert exact_neighbor_recall_at_k(baseline, candidate, k=10) == 0.8


def test_score_delta_calculation() -> None:
    corpus = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float64)
    query = np.asarray([1.0, 0.0], dtype=np.float64)
    baseline = rank_by_cosine_deterministic(corpus, ("a", "b", "c"), query, top_k=3)
    evidence = evaluate_query_round_trip(
        query_id="q-0002",
        query_text="probe",
        query_vector=query,
        corpus_embeddings=corpus,
        logical_point_ids=("a", "b", "c"),
        qdrant_hits=tuple(
            RankedVectorHit(
                logical_point_id=hit.logical_point_id,
                rank=hit.rank,
                cosine_score=hit.cosine_score - 1e-5,
            )
            for hit in baseline
        ),
        source_ref_by_logical_id={
            "a": _source_ref("a"),
            "b": _source_ref("b"),
            "c": _source_ref("c"),
        },
        qdrant_source_refs={
            "a": _source_ref("a"),
            "b": _source_ref("b"),
            "c": _source_ref("c"),
        },
    )
    assert evidence.max_absolute_score_delta == pytest.approx(1e-5)
    assert evidence.mean_absolute_score_delta == pytest.approx(1e-5)


def test_source_identity_mismatch_detection() -> None:
    corpus = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    query = np.asarray([1.0, 0.0], dtype=np.float64)
    evidence = evaluate_query_round_trip(
        query_id="q-0003",
        query_text="probe",
        query_vector=query,
        corpus_embeddings=corpus,
        logical_point_ids=("a", "b"),
        qdrant_hits=(RankedVectorHit(logical_point_id="a", rank=1, cosine_score=1.0),),
        source_ref_by_logical_id={"a": _source_ref("a"), "b": _source_ref("b")},
        qdrant_source_refs={"a": _source_ref("wrong")},
    )
    assert evidence.source_ref_mismatches == 1


def _passing_metrics() -> VectorDbRoundTripMetrics:
    return VectorDbRoundTripMetrics(
        tie_aware_top1_parity_rate=1.0,
        mean_recall_at_5=1.0,
        mean_recall_at_10=1.0,
        mean_absolute_score_delta=0.0,
        max_absolute_score_delta=0.0,
        unknown_logical_point_id_count=0,
        source_ref_mismatch_count=0,
        embedding_transport_correctness=True,
        vector_index_ranking_parity=True,
    )


def _artifact_pass() -> PilotArtifactIntegrity:
    return PilotArtifactIntegrity(
        relational_count=PILOT_RECORD_COUNT,
        embedding_count=PILOT_RECORD_COUNT,
        source_ref_parity=True,
        embedding_dimension=1024,
        finite_vectors=True,
        non_zero_vectors=True,
        provider_match=True,
        model_match=True,
        revision_match=True,
        logical_point_ids_unique=True,
        passed=True,
        failure_reasons=(),
    )


def _self_probe_pass() -> VectorSelfProbeEvidence:
    return VectorSelfProbeEvidence(
        pilot_index=0,
        logical_point_id="lp-0",
        offer_id="offer-0",
        catalog_id="wdc-v2-selected",
        source_revision=None,
        returned_logical_point_id="lp-0",
        self_cosine_score=1.0,
        top1_identity_match=True,
        metadata_identity_match=True,
        passed=True,
    )


def _query_evidence_pass() -> VectorRoundTripQueryEvidence:
    return VectorRoundTripQueryEvidence(
        query_id="q-0001",
        query_text="probe",
        baseline_top1_ids=("a",),
        qdrant_top1_id="a",
        top1_parity=True,
        recall_at_5=1.0,
        recall_at_10=1.0,
        mean_absolute_score_delta=0.0,
        max_absolute_score_delta=0.0,
        unknown_logical_point_ids=0,
        source_ref_mismatches=0,
    )


def test_hard_gate_pass() -> None:
    status = evaluate_hard_gate(
        artifact_integrity=_artifact_pass(),
        self_probes=tuple(_self_probe_pass() for _ in range(8)),
        query_evidence=tuple(_query_evidence_pass() for _ in range(QUERY_CASE_COUNT)),
        metrics=_passing_metrics(),
        document_embedding_calls=0,
        query_embedding_calls=QUERY_CASE_COUNT,
        qdrant_point_count=PILOT_RECORD_COUNT,
        qdrant_dimension=1024,
        qdrant_metric="cosine",
        dense_search_available=True,
    )
    assert status is QualificationRunStatus.PASS


def test_hard_gate_failure_on_self_probe_mismatch() -> None:
    failed_probe = VectorSelfProbeEvidence(
        pilot_index=0,
        logical_point_id="lp-0",
        offer_id="offer-0",
        catalog_id="wdc-v2-selected",
        source_revision=None,
        returned_logical_point_id="lp-1",
        self_cosine_score=0.5,
        top1_identity_match=False,
        metadata_identity_match=False,
        passed=False,
    )
    status = evaluate_hard_gate(
        artifact_integrity=_artifact_pass(),
        self_probes=(failed_probe,) + tuple(_self_probe_pass() for _ in range(7)),
        query_evidence=tuple(_query_evidence_pass() for _ in range(QUERY_CASE_COUNT)),
        metrics=_passing_metrics(),
        document_embedding_calls=0,
        query_embedding_calls=QUERY_CASE_COUNT,
        qdrant_point_count=PILOT_RECORD_COUNT,
        qdrant_dimension=1024,
        qdrant_metric="cosine",
        dense_search_available=True,
    )
    assert status is QualificationRunStatus.FAIL


def test_hard_gate_failure_on_ranking_parity_regression() -> None:
    metrics = VectorDbRoundTripMetrics(
        tie_aware_top1_parity_rate=0.5,
        mean_recall_at_5=0.5,
        mean_recall_at_10=0.5,
        mean_absolute_score_delta=0.0,
        max_absolute_score_delta=0.0,
        unknown_logical_point_id_count=0,
        source_ref_mismatch_count=0,
        embedding_transport_correctness=True,
        vector_index_ranking_parity=False,
    )
    status = evaluate_hard_gate(
        artifact_integrity=_artifact_pass(),
        self_probes=tuple(_self_probe_pass() for _ in range(8)),
        query_evidence=tuple(_query_evidence_pass() for _ in range(QUERY_CASE_COUNT)),
        metrics=metrics,
        document_embedding_calls=0,
        query_embedding_calls=QUERY_CASE_COUNT,
        qdrant_point_count=PILOT_RECORD_COUNT,
        qdrant_dimension=1024,
        qdrant_metric="cosine",
        dense_search_available=True,
    )
    assert status is QualificationRunStatus.VECTOR_INDEX_RETRIEVAL_REGRESSION


def test_provider_neutral_core_has_no_qdrant_imports() -> None:
    source = _EVALUATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)
    assert not any("qdrant" in module for module in imported_modules)


def test_evaluation_module_has_no_forbidden_patterns() -> None:
    source = _EVALUATION_PATH.read_text(encoding="utf-8")
    forbidden_fragments = (
        "dict[str, Any]",
        "dict[str, object]",
        "getattr(",
        "setattr(",
        "hasattr(",
        "inspect.",
    )
    for fragment in forbidden_fragments:
        assert fragment not in source
    tree = ast.parse(source)
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    assert {"getattr", "setattr", "hasattr", "inspect"}.isdisjoint(names)


def test_compute_cosine_scores_matches_manual_value() -> None:
    corpus = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    query = np.asarray([1.0, 0.0], dtype=np.float64)
    scores = compute_cosine_scores(corpus, query)
    assert scores.tolist() == [1.0, 0.0]
