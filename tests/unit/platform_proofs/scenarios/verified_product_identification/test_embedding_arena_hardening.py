"""Hardening tests for VPI embedding arena lifecycle and correctness."""

from __future__ import annotations

import ast
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    WdcIdentifierEntry,
    WdcKeyValuePair,
    WdcSourceOffer,
)
from platform_proofs.scenarios.verified_product_identification.arena.composition.candidates import (
    BASELINE_CANDIDATE_ID,
)
from platform_proofs.scenarios.verified_product_identification.arena.composition.input_policies import (
    StaticEmbeddingInputTransformation,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.candidates import (
    EmbeddingArenaCandidate,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.classification import (
    EmbeddingArenaCandidateEligibility,
    EmbeddingLicenseClassification,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.errors import (
    EmbeddingArenaBenchmarkGroundTruthError,
    EmbeddingArenaTokenizerUnavailableError,
    EmbeddingArenaTruncationProfileError,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.input_policy import (
    EmbeddingInputPolicyRef,
    EmbeddingInputRole,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.query_benchmark import (
    ArenaSourceRef,
    EmbeddingArenaQueryCase,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    RetrievalQualityMetrics,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.stage_evaluation_scope import (
    compute_stage_content_fingerprint,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.finalist_selection import (
    StageBCandidateEvidence,
    select_stage_c_finalist_ids,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.ground_truth import (
    resolve_relevant_indices_or_fail,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.retrieval_evaluation import (
    evaluate_retrieval_quality_for_scope,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.stage_scope import (
    build_stage_evaluation_scope,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.candidate_execution_session import (
    EmbeddingArenaCandidateExecutionSession,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.embedding_port import (
    transform_texts_for_role,
)
from platform_proofs.scenarios.verified_product_identification.arena.integration.truncation_probe import (
    profile_truncation_for_texts,
)
from platform_proofs.scenarios.verified_product_identification.arena.sampling.arena_sample import (
    ArenaSampleRecord,
    derive_strata_tags,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    EmbeddingProbeResult,
    ValidationStatus,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.classification import (
    QueryDifficultyClass,
)

pytestmark = pytest.mark.unit


def _sample_offer(
    *,
    offer_id: str,
    title: str | None = "Example Product Title",
    brand: str | None = "BrandX",
    description: str | None = "Short description",
    identifiers: tuple[WdcIdentifierEntry, ...] = (),
    cluster_id: int | None = None,
) -> WdcSourceOffer:
    return WdcSourceOffer(
        offer_id=offer_id,
        cluster_id=cluster_id,
        category="electronics",
        identifiers=identifiers,
        title=title,
        description=description,
        brand=brand,
        price="9.99",
        key_value_pairs=(
            WdcKeyValuePair(source_key="color", source_value="black", raw_value="black"),
        ),
        spec_table_content="Voltage: 12V",
    )


def _sample_record(offer_id: str, *, cluster_id: int | None = None) -> ArenaSampleRecord:
    offer = _sample_offer(
        offer_id=offer_id,
        identifiers=(WdcIdentifierEntry(source_key="gtin", source_value=f"ID-{offer_id}"),),
        cluster_id=cluster_id,
    )
    return ArenaSampleRecord(
        offer_id=offer_id,
        global_row_index=int(offer_id.split("-")[-1]),
        semantic_text=f"{offer.brand} {offer.title} {offer.description}",
        source_offer=offer,
        strata_tags=derive_strata_tags(offer),
    )


@dataclass
class _CountingEmbeddingPort:
    construction_count: int = 0
    embed_batch_count: int = 0
    close_count: int = 0
    dimension: int = 3

    def probe(self) -> EmbeddingProbeResult:
        return EmbeddingProbeResult(
            status=ValidationStatus.PASS,
            provider="fake",
            model="fake",
            resolved_dimension=self.dimension,
            probe_vector_count=1,
            detail="fake probe",
        )

    def embed_batch(self, texts: tuple[str, ...] | list[str]) -> tuple[tuple[float, ...], ...]:
        self.embed_batch_count += 1
        rows: list[tuple[float, ...]] = []
        for index, text in enumerate(texts):
            vector = [0.0] * self.dimension
            vector[index % self.dimension] = 1.0 + float(len(text) % 3) * 0.01
            rows.append(tuple(vector))
        return tuple(rows)

    def close(self) -> None:
        self.close_count += 1


def _counting_port_factory(port: _CountingEmbeddingPort):
    def factory(*_args, **_kwargs) -> _CountingEmbeddingPort:
        port.construction_count += 1
        return port

    return factory


def _fake_candidate(
    *,
    query_policy_id: str = "query-policy",
    document_policy_id: str = "document-policy",
) -> EmbeddingArenaCandidate:
    query_ref = EmbeddingInputPolicyRef(
        policy_id=query_policy_id,
        policy_version="v1",
        query_instruction_summary="query",
        document_instruction_summary="document",
    )
    document_ref = EmbeddingInputPolicyRef(
        policy_id=document_policy_id,
        policy_version="v1",
        query_instruction_summary="query",
        document_instruction_summary="document",
    )
    return EmbeddingArenaCandidate(
        candidate_id="fake",
        provider="hf",
        model="fake/model",
        expected_dimension=3,
        license_classification=EmbeddingLicenseClassification.ELIGIBLE_COMMERCIAL,
        license_identifier="MIT",
        license_reference="https://example.com",
        license_reason="test",
        query_instruction_policy=query_ref,
        document_instruction_policy=document_ref,
        semantic_input_policy_id="fake-v1",
        max_sequence_length=None,
        trust_remote_code_required=False,
        normalization_expected=True,
        eligibility_status=EmbeddingArenaCandidateEligibility.ELIGIBLE,
        is_baseline=False,
        fixed_provider_batch_size=8,
    )


def test_stage_b_scope_ground_truth_stays_within_corpus() -> None:
    records = tuple(_sample_record(f"offer-{index}") for index in range(600))
    stage_b_scope = build_stage_evaluation_scope(
        stage_name="stage_b",
        records=records[:500],
    )
    for case in stage_b_scope.query_cases:
        for source_ref in case.relevant_source_refs:
            assert source_ref.offer_id in stage_b_scope.offer_index


def test_stage_b_and_c_scopes_are_deterministic() -> None:
    records = tuple(_sample_record(f"offer-{index}") for index in range(1000))
    stage_b_first = build_stage_evaluation_scope(stage_name="stage_b", records=records[:500])
    stage_b_second = build_stage_evaluation_scope(stage_name="stage_b", records=records[:500])
    stage_c_first = build_stage_evaluation_scope(stage_name="stage_c", records=records[:1000])
    stage_c_second = build_stage_evaluation_scope(stage_name="stage_c", records=records[:1000])
    assert stage_b_first.query_cases == stage_b_second.query_cases
    assert stage_c_first.query_cases == stage_c_second.query_cases
    assert stage_b_first.content_fingerprint == stage_b_second.content_fingerprint
    assert stage_c_first.content_fingerprint != stage_b_first.content_fingerprint


def test_invalid_relevant_source_ref_fails_closed() -> None:
    case = EmbeddingArenaQueryCase(
        case_id="q-0001",
        query_text="BrandX Example Product Title",
        difficulty=QueryDifficultyClass.TITLE_BRAND,
        relevant_source_refs=(ArenaSourceRef(offer_id="missing-offer", global_row_index=99),),
        provenance="test",
        benchmark_only_cluster_evidence=None,
        hard_negative_offer_ids=(),
        is_long_input_query=False,
    )
    offer_index = {"offer-0": 0}
    with pytest.raises(EmbeddingArenaBenchmarkGroundTruthError):
        resolve_relevant_indices_or_fail(case, offer_index)


def test_duplicate_relevant_source_ref_fails_closed() -> None:
    case = EmbeddingArenaQueryCase(
        case_id="q-0002",
        query_text="BrandX Example Product Title",
        difficulty=QueryDifficultyClass.TITLE_BRAND,
        relevant_source_refs=(
            ArenaSourceRef(offer_id="offer-0", global_row_index=0),
            ArenaSourceRef(offer_id="offer-0", global_row_index=0),
        ),
        provenance="test",
        benchmark_only_cluster_evidence=None,
        hard_negative_offer_ids=(),
        is_long_input_query=False,
    )
    with pytest.raises(EmbeddingArenaBenchmarkGroundTruthError):
        resolve_relevant_indices_or_fail(case, {"offer-0": 0})


def test_invalid_benchmark_is_not_treated_as_no_hit() -> None:
    from platform_proofs.scenarios.verified_product_identification.arena.contracts.stage_evaluation_scope import (
        EmbeddingArenaStageEvaluationScope,
    )

    case = EmbeddingArenaQueryCase(
        case_id="q-0003",
        query_text="BrandX Example Product Title",
        difficulty=QueryDifficultyClass.TITLE_BRAND,
        relevant_source_refs=(ArenaSourceRef(offer_id="outside", global_row_index=1),),
        provenance="test",
        benchmark_only_cluster_evidence=None,
        hard_negative_offer_ids=(),
        is_long_input_query=False,
    )
    records = tuple(_sample_record("offer-0") for _ in range(1))
    scope = build_stage_evaluation_scope(stage_name="stage_b", records=records)
    corpus = np.eye(1, dtype=np.float64)
    query_embeddings = np.eye(1, dtype=np.float64)
    scope_with_bad_case = scope
    with pytest.raises(EmbeddingArenaBenchmarkGroundTruthError):
        evaluate_retrieval_quality_for_scope(
            scope=EmbeddingArenaStageEvaluationScope(
                stage_name=scope.stage_name,
                records=scope.records,
                query_cases=(case,),
                offer_index=scope.offer_index,
                corpus_size=scope.corpus_size,
                benchmark_version=scope.benchmark_version,
                sample_version=scope.sample_version,
                content_fingerprint=scope.content_fingerprint,
            ),
            corpus_embeddings=corpus,
            query_embeddings=query_embeddings,
            expected_dimension=1,
        )


def test_session_uses_single_provider_for_documents_and_queries() -> None:
    port = _CountingEmbeddingPort()
    candidate = _fake_candidate(
        query_policy_id="bge-m3",
        document_policy_id="bge-m3",
    )
    with EmbeddingArenaCandidateExecutionSession(
        candidate,
        provider_batch_size=8,
        device="cpu",
        port_factory=_counting_port_factory(port),
    ) as session:
        session.warmup(("warmup text",))
        session.embed_documents(("doc-a", "doc-b"), expected_dimension=3)
        session.embed_queries(("query-a", "query-b", "query-c"), expected_dimension=3)
        session.measure_query_latency(("latency-query",), expected_dimension=3, repetitions=2)
    assert port.construction_count == 1
    assert port.embed_batch_count >= 3
    assert port.close_count == 1


def test_session_close_runs_after_failure() -> None:
    port = _CountingEmbeddingPort()

    class _FailingPort(_CountingEmbeddingPort):
        def embed_batch(self, texts: tuple[str, ...] | list[str]) -> tuple[tuple[float, ...], ...]:
            raise RuntimeError("embed failed")

    failing_port = _FailingPort()

    def factory(*_args, **_kwargs) -> _FailingPort:
        failing_port.construction_count += 1
        return failing_port

    candidate = _fake_candidate(
        query_policy_id="bge-m3",
        document_policy_id="bge-m3",
    )
    with pytest.raises(RuntimeError, match="embed failed"):
        with EmbeddingArenaCandidateExecutionSession(
            candidate,
            provider_batch_size=8,
            device="cpu",
            port_factory=factory,
        ) as session:
            session.embed_documents(("doc",), expected_dimension=3)
    assert failing_port.close_count == 1


def test_query_and_document_policies_resolve_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_transformation = StaticEmbeddingInputTransformation(
        policy_ref=EmbeddingInputPolicyRef(
            policy_id="query-policy",
            policy_version="v1",
            query_instruction_summary="query",
            document_instruction_summary="document",
        ),
        query_prefix="QUERY::",
        document_prefix="DOCUMENT::",
    )
    document_transformation = StaticEmbeddingInputTransformation(
        policy_ref=EmbeddingInputPolicyRef(
            policy_id="document-policy",
            policy_version="v1",
            query_instruction_summary="query",
            document_instruction_summary="document",
        ),
        query_prefix="WRONG-QUERY::",
        document_prefix="DOC::",
    )

    def _resolve(policy_id: str) -> StaticEmbeddingInputTransformation:
        if policy_id == "query-policy":
            return query_transformation
        if policy_id == "document-policy":
            return document_transformation
        raise ValueError(policy_id)

    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.arena.integration.embedding_port.resolve_input_transformation",
        _resolve,
    )
    candidate = _fake_candidate(
        query_policy_id="query-policy",
        document_policy_id="document-policy",
    )
    assert transform_texts_for_role(
        candidate,
        role=EmbeddingInputRole.QUERY,
        canonical_texts=("text",),
    ) == ("QUERY::text",)
    assert transform_texts_for_role(
        candidate,
        role=EmbeddingInputRole.DOCUMENT,
        canonical_texts=("text",),
    ) == ("DOC::text",)


def _stage_b_evidence(
    candidate_id: str,
    *,
    recall_at_10: float,
    throughput: float,
    runtime_ok: bool = True,
) -> StageBCandidateEvidence:
    return StageBCandidateEvidence(
        candidate_id=candidate_id,
        is_baseline=candidate_id == BASELINE_CANDIDATE_ID,
        license_eligible=True,
        runtime_ok=runtime_ok,
        throughput_records_per_second=throughput,
        quality_metrics=RetrievalQualityMetrics(
            recall_at_1=recall_at_10,
            recall_at_5=recall_at_10,
            recall_at_10=recall_at_10,
            mrr_at_10=recall_at_10,
            ndcg_at_10=recall_at_10,
            query_count=10,
        ),
    )


def test_finalist_selection_is_order_independent() -> None:
    evidence = (
        _stage_b_evidence("challenger-a", recall_at_10=0.91, throughput=12.0),
        _stage_b_evidence(BASELINE_CANDIDATE_ID, recall_at_10=0.90, throughput=10.4),
        _stage_b_evidence("challenger-b", recall_at_10=0.89, throughput=15.0),
        _stage_b_evidence("challenger-c", recall_at_10=0.88, throughput=8.0),
    )
    expected = select_stage_c_finalist_ids(
        evidence,
        baseline_candidate_id=BASELINE_CANDIDATE_ID,
        baseline_throughput=10.4,
    )
    for permutation in itertools.permutations(evidence):
        actual = select_stage_c_finalist_ids(
            permutation,
            baseline_candidate_id=BASELINE_CANDIDATE_ID,
            baseline_throughput=10.4,
        )
        assert actual == expected


def test_rejected_candidates_cannot_enter_finalists() -> None:
    evidence = (
        _stage_b_evidence(BASELINE_CANDIDATE_ID, recall_at_10=0.90, throughput=10.4),
        _stage_b_evidence("failed-runtime", recall_at_10=0.99, throughput=20.0, runtime_ok=False),
    )
    finalists = select_stage_c_finalist_ids(
        evidence,
        baseline_candidate_id=BASELINE_CANDIDATE_ID,
        baseline_throughput=10.4,
    )
    assert "failed-runtime" not in finalists
    assert BASELINE_CANDIDATE_ID in finalists


def test_stage_c_finalist_hard_limit_with_throughput_boosted_challengers() -> None:
    baseline_throughput = 10.0
    evidence = (
        _stage_b_evidence(BASELINE_CANDIDATE_ID, recall_at_10=0.90, throughput=baseline_throughput),
        _stage_b_evidence("challenger-a", recall_at_10=0.95, throughput=baseline_throughput * 1.5),
        _stage_b_evidence("challenger-b", recall_at_10=0.94, throughput=baseline_throughput * 1.6),
        _stage_b_evidence("challenger-c", recall_at_10=0.93, throughput=baseline_throughput * 1.7),
        _stage_b_evidence("challenger-d", recall_at_10=0.92, throughput=baseline_throughput * 1.8),
    )
    finalists = select_stage_c_finalist_ids(
        evidence,
        baseline_candidate_id=BASELINE_CANDIDATE_ID,
        baseline_throughput=baseline_throughput,
        max_finalists=3,
    )
    assert len(finalists) == 3


def test_stage_c_finalist_baseline_counts_toward_limit() -> None:
    baseline_throughput = 10.0
    evidence = (
        _stage_b_evidence(BASELINE_CANDIDATE_ID, recall_at_10=0.90, throughput=baseline_throughput),
        _stage_b_evidence("challenger-a", recall_at_10=0.95, throughput=baseline_throughput * 1.3),
        _stage_b_evidence("challenger-b", recall_at_10=0.94, throughput=baseline_throughput * 1.4),
        _stage_b_evidence("challenger-c", recall_at_10=0.93, throughput=baseline_throughput * 1.5),
        _stage_b_evidence("challenger-d", recall_at_10=0.92, throughput=baseline_throughput * 1.6),
    )
    finalists = select_stage_c_finalist_ids(
        evidence,
        baseline_candidate_id=BASELINE_CANDIDATE_ID,
        baseline_throughput=baseline_throughput,
        max_finalists=3,
    )
    assert len(finalists) == 3
    assert BASELINE_CANDIDATE_ID in finalists
    assert sum(1 for candidate_id in finalists if candidate_id != BASELINE_CANDIDATE_ID) == 2


def test_stage_c_finalist_selection_permutation_determinism() -> None:
    baseline_throughput = 10.0
    evidence = (
        _stage_b_evidence(BASELINE_CANDIDATE_ID, recall_at_10=0.90, throughput=baseline_throughput),
        _stage_b_evidence("challenger-a", recall_at_10=0.95, throughput=baseline_throughput * 2.0),
        _stage_b_evidence("challenger-b", recall_at_10=0.94, throughput=baseline_throughput * 3.0),
        _stage_b_evidence("challenger-c", recall_at_10=0.93, throughput=baseline_throughput * 4.0),
        _stage_b_evidence("challenger-d", recall_at_10=0.92, throughput=baseline_throughput * 5.0),
    )
    expected = select_stage_c_finalist_ids(
        evidence,
        baseline_candidate_id=BASELINE_CANDIDATE_ID,
        baseline_throughput=baseline_throughput,
        max_finalists=3,
    )
    for permutation in itertools.permutations(evidence):
        actual = select_stage_c_finalist_ids(
            permutation,
            baseline_candidate_id=BASELINE_CANDIDATE_ID,
            baseline_throughput=baseline_throughput,
            max_finalists=3,
        )
        assert actual == expected


def test_stage_c_finalist_throughput_does_not_break_limit() -> None:
    baseline_throughput = 10.0
    evidence = (
        _stage_b_evidence(BASELINE_CANDIDATE_ID, recall_at_10=0.90, throughput=baseline_throughput),
        _stage_b_evidence("challenger-a", recall_at_10=0.80, throughput=baseline_throughput * 2.0),
        _stage_b_evidence("challenger-b", recall_at_10=0.79, throughput=baseline_throughput * 3.0),
        _stage_b_evidence("challenger-c", recall_at_10=0.78, throughput=baseline_throughput * 4.0),
        _stage_b_evidence("challenger-d", recall_at_10=0.77, throughput=baseline_throughput * 5.0),
    )
    finalists = select_stage_c_finalist_ids(
        evidence,
        baseline_candidate_id=BASELINE_CANDIDATE_ID,
        baseline_throughput=baseline_throughput,
        max_finalists=3,
    )
    assert len(finalists) <= 3


@pytest.mark.parametrize("max_finalists", [1, 2])
def test_stage_c_finalist_small_limits(max_finalists: int) -> None:
    baseline_throughput = 10.0
    evidence = (
        _stage_b_evidence(BASELINE_CANDIDATE_ID, recall_at_10=0.90, throughput=baseline_throughput),
        _stage_b_evidence("challenger-a", recall_at_10=0.95, throughput=baseline_throughput * 2.0),
        _stage_b_evidence("challenger-b", recall_at_10=0.94, throughput=baseline_throughput * 3.0),
    )
    finalists = select_stage_c_finalist_ids(
        evidence,
        baseline_candidate_id=BASELINE_CANDIDATE_ID,
        baseline_throughput=baseline_throughput,
        max_finalists=max_finalists,
    )
    assert len(finalists) == max_finalists


def test_stage_c_finalist_invalid_max_finalists_raises() -> None:
    evidence = (
        _stage_b_evidence(BASELINE_CANDIDATE_ID, recall_at_10=0.90, throughput=10.0),
    )
    with pytest.raises(ValueError, match="max_finalists must be > 0"):
        select_stage_c_finalist_ids(
            evidence,
            baseline_candidate_id=BASELINE_CANDIDATE_ID,
            baseline_throughput=10.0,
            max_finalists=0,
        )


def test_stage_c_finalist_quality_ranks_before_throughput() -> None:
    baseline_throughput = 10.0
    evidence = (
        _stage_b_evidence(BASELINE_CANDIDATE_ID, recall_at_10=0.90, throughput=baseline_throughput),
        _stage_b_evidence("quality-leader", recall_at_10=0.95, throughput=baseline_throughput * 1.1),
        _stage_b_evidence("throughput-leader", recall_at_10=0.80, throughput=baseline_throughput * 5.0),
    )
    finalists = select_stage_c_finalist_ids(
        evidence,
        baseline_candidate_id=BASELINE_CANDIDATE_ID,
        baseline_throughput=baseline_throughput,
        max_finalists=2,
    )
    assert finalists == (BASELINE_CANDIDATE_ID, "quality-leader")


def test_stage_c_finalist_alphabetical_tie_break() -> None:
    evidence = (
        _stage_b_evidence("beta", recall_at_10=0.90, throughput=10.0),
        _stage_b_evidence("alpha", recall_at_10=0.90, throughput=10.0),
    )
    finalists = select_stage_c_finalist_ids(
        evidence,
        baseline_candidate_id=BASELINE_CANDIDATE_ID,
        baseline_throughput=10.0,
        max_finalists=1,
    )
    assert finalists == ("alpha",)


def test_truncation_tokenizer_unavailable_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_import_error(*_args, **_kwargs) -> None:
        raise ImportError("no transformers")

    monkeypatch.setitem(__import__("sys").modules, "transformers", None)
    with pytest.raises(EmbeddingArenaTokenizerUnavailableError):
        profile_truncation_for_texts(
            model_name="fake/model",
            texts=("hello",),
            max_supported_tokens=512,
        )


def test_truncation_profile_encoding_failure_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs) -> _BrokenTokenizer:
            return cls()

        def encode(self, _text: str, *, add_special_tokens: bool) -> list[int]:
            raise ValueError("broken tokenizer")

    class _FakeTransformers:
        AutoTokenizer = _BrokenTokenizer

    monkeypatch.setitem(__import__("sys").modules, "transformers", _FakeTransformers())
    with pytest.raises(EmbeddingArenaTruncationProfileError):
        profile_truncation_for_texts(
            model_name="fake/model",
            texts=("hello",),
            max_supported_tokens=512,
        )


def test_truncation_unexpected_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ExplodingTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs) -> _ExplodingTokenizer:
            raise RuntimeError("unexpected tokenizer failure")

    class _FakeTransformers:
        AutoTokenizer = _ExplodingTokenizer

    monkeypatch.setitem(__import__("sys").modules, "transformers", _FakeTransformers())
    with pytest.raises(RuntimeError, match="unexpected tokenizer failure"):
        profile_truncation_for_texts(
            model_name="fake/model",
            texts=("hello",),
            max_supported_tokens=512,
        )


def test_stage_b_candidates_share_same_query_cases() -> None:
    records = tuple(_sample_record(f"offer-{index}") for index in range(500))
    scope_a = build_stage_evaluation_scope(stage_name="stage_b", records=records)
    scope_b = build_stage_evaluation_scope(stage_name="stage_b", records=records)
    assert scope_a.query_cases == scope_b.query_cases


def test_arena_core_has_no_vendor_imports() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    arena_root = (
        repo_root
        / "platform_proofs/scenarios/verified_product_identification/arena"
    )
    forbidden_roots = {
        "torch",
        "sentence_transformers",
        "transformers",
        "qdrant_client",
        "psycopg",
        "asyncpg",
    }
    forbidden_modules = {
        "intergrax.rag.embedding.providers.hf_embedding_provider",
    }
    violations: list[str] = []
    for path in sorted(arena_root.rglob("*.py")):
        if "integration" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    root = module.split(".", 1)[0]
                    if root in forbidden_roots or module in forbidden_modules:
                        violations.append(f"{path.name}: {module}")
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                module = node.module
                root = module.split(".", 1)[0]
                if root in forbidden_roots or module in forbidden_modules:
                    violations.append(f"{path.name}: {module}")
    assert violations == []


def test_content_fingerprint_is_stable() -> None:
    records = tuple(_sample_record(f"offer-{index}") for index in range(3))
    assert compute_stage_content_fingerprint(records) == compute_stage_content_fingerprint(records)
