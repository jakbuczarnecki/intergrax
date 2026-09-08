"""VPI-IMPLEMENTATION-5C4E5A — vector DB round-trip retrieval qualification runner."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from platform_proofs.scenarios.verified_product_identification.application.catalog.derive_search_representation import (
    derive_search_representation,
    flatten_lexical_text,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    load_vpi_embedding_configuration,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    load_vpi_embedding_provider_execution_configuration,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.query_builder import (
    build_query_benchmark_cases,
)
from platform_proofs.scenarios.verified_product_identification.arena.sampling.arena_sample import (
    ArenaSampleRecord,
    derive_strata_tags,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_execution_profiles import (
    PRODUCTION_LOCAL_GPU_PROFILE_ID,
    apply_data_pack_build_execution_profile,
    resolve_data_pack_build_execution_profile,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding_input_policy import (
    VPI_BGE_M3_DOCUMENT_EFFECTIVE_TOKEN_CEILING,
    VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
    VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    VPI_CANONICAL_EMBEDDING_DIMENSION,
    VPI_CANONICAL_EMBEDDING_MODEL,
    VPI_CANONICAL_EMBEDDING_PROVIDER,
    VPI_CANONICAL_EMBEDDING_REVISION,
    source_ref_key,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.embedding_codec import (
    read_embedding_parquet,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.relational_codec import (
    read_relational_parquet,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.artifact.record import (
    EmbeddingArtifactRecord,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.bootstrap import (
    ensure_embedding_provider_integrations_registered,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.qualification.vector_db_roundtrip.contracts import (
    PILOT_RECORD_COUNT,
    QUALIFICATION_TASK_ID,
    QUERY_CASE_COUNT,
    QdrantIndexSnapshot,
    QualificationRunStatus,
    SELF_PROBE_INDICES,
    VectorDbRoundTripQualificationReport,
)
from platform_proofs.scenarios.verified_product_identification.qualification.vector_db_roundtrip.evaluation import (
    aggregate_query_metrics,
    build_embedding_matrix,
    evaluate_hard_gate,
    evaluate_pilot_artifact_integrity,
    evaluate_query_round_trip,
    evaluate_self_probe,
)
from platform_proofs.scenarios.verified_product_identification.qualification.vector_db_roundtrip.qdrant_runtime import (
    QdrantMetricConfigurationError,
    QdrantResourcePreconditionError,
    delete_qualification_collection,
    describe_qdrant_index,
    ensure_qdrant_available,
    make_qualification_collection_name,
    open_qdrant_qualification_runtime,
    query_qdrant_ranked_hits,
)
from platform_proofs.scenarios.verified_product_identification.qualification.vector_db_roundtrip.reporting import (
    render_vector_db_roundtrip_markdown_report,
    write_vector_db_roundtrip_report_json,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.ports import (
    SearchIndexIngestBatch,
    SearchIndexIngestRecord,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    BOOTSTRAP_IMPLEMENTATION_VERSION,
    BootstrapState,
    VpiBootstrapManifest,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.orchestration.search_from_artifact import (
    search_ingest_record_from_artifact,
)

_PILOT_ROOT = _REPO_ROOT / ".tmp" / "session" / "vpi-5c4e5" / "data-pack-pilot"
_SESSION_ROOT = _REPO_ROOT / ".tmp" / "session" / "vpi-5c4e5a"
_RELATIONAL_PATH = _PILOT_ROOT / "relational" / "part-000001.parquet"
_EMBEDDING_PATH = _PILOT_ROOT / "embeddings" / "part-000001.parquet"
_BATCH_SIZE = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _build_arena_records(
    relational_records: tuple,
) -> tuple[ArenaSampleRecord, ...]:
    records: list[ArenaSampleRecord] = []
    for relational_record in relational_records:
        source_offer = parse_wdc_source_offer_json(relational_record.record_json)
        records.append(
            ArenaSampleRecord(
                offer_id=relational_record.source_ref.offer_id.value,
                global_row_index=relational_record.global_row_index,
                semantic_text=relational_record.semantic_text,
                source_offer=source_offer,
                strata_tags=derive_strata_tags(source_offer),
            )
        )
    return tuple(records)


def _select_query_cases(
    arena_records: tuple[ArenaSampleRecord, ...],
) -> tuple:
    all_cases = build_query_benchmark_cases(arena_records)
    ordered = sorted(all_cases, key=lambda case: case.case_id)
    if len(ordered) < QUERY_CASE_COUNT:
        msg = (
            f"arena query benchmark produced {len(ordered)} cases; "
            f"expected at least {QUERY_CASE_COUNT}"
        )
        raise RuntimeError(msg)
    return tuple(ordered[:QUERY_CASE_COUNT])


def _build_search_ingest_records(
    relational_records: tuple,
    embedding_records: tuple,
    *,
    dataset_checksum: str,
) -> tuple[SearchIndexIngestRecord, ...]:
    embedding_by_ref = {
        source_ref_key(record.source_ref): record for record in embedding_records
    }
    search_records: list[SearchIndexIngestRecord] = []
    for relational_record in relational_records:
        embedding_record = embedding_by_ref.get(source_ref_key(relational_record.source_ref))
        if embedding_record is None:
            msg = (
                "missing embedding row for "
                f"{relational_record.source_ref.offer_id.value}"
            )
            raise RuntimeError(msg)
        source_offer = parse_wdc_source_offer_json(relational_record.record_json)
        representation = derive_search_representation(
            source_offer,
            source_ref=relational_record.source_ref,
            derivation_version=relational_record.derivation_version,
        )
        artifact_record = EmbeddingArtifactRecord(
            global_row_index=relational_record.global_row_index,
            logical_point_id=embedding_record.logical_point_id,
            catalog_id=relational_record.source_ref.catalog_id,
            offer_id=relational_record.source_ref.offer_id.value,
            source_revision=relational_record.source_ref.source_revision,
            derivation_version=relational_record.derivation_version,
            semantic_text=relational_record.semantic_text,
            lexical_text=flatten_lexical_text(representation.lexical),
            embedding_provider=embedding_record.embedding_provider,
            embedding_model=embedding_record.embedding_model,
            embedding_dimension=embedding_record.embedding_dimension,
            dense_embedding=embedding_record.dense_embedding,
        )
        search_records.append(
            search_ingest_record_from_artifact(
                artifact_record,
                dataset_checksum=dataset_checksum,
            )
        )
    return tuple(search_records)


def _build_manifest(
    *,
    dataset_checksum: str,
    catalog_id: str,
    derivation_version: str,
) -> VpiBootstrapManifest:
    return VpiBootstrapManifest(
        state=BootstrapState.INITIALIZING,
        dataset_path=str(_RELATIONAL_PATH),
        dataset_checksum=dataset_checksum,
        dataset_record_count=PILOT_RECORD_COUNT,
        search_representation_derivation_version=derivation_version,
        embedding_configuration_version="v1",
        embedding_provider=VPI_CANONICAL_EMBEDDING_PROVIDER,
        embedding_model=VPI_CANONICAL_EMBEDDING_MODEL,
        embedding_dimension=VPI_CANONICAL_EMBEDDING_DIMENSION,
        catalog_schema_version="v1",
        search_index_schema_version="v1",
        bootstrap_implementation_version=BOOTSTRAP_IMPLEMENTATION_VERSION,
        catalog_id=catalog_id,
        source_revision=None,
        checkpoint_batch_ordinal=0,
        checkpoint_rows_processed=PILOT_RECORD_COUNT,
        target_max_records=PILOT_RECORD_COUNT,
        catalog_source_offer_count=PILOT_RECORD_COUNT,
        catalog_identifier_count=PILOT_RECORD_COUNT,
        catalog_structured_attribute_count=PILOT_RECORD_COUNT,
        search_point_count=PILOT_RECORD_COUNT,
    )


def _count_hf_model_loads(embedding_adapter: IntergraxEmbeddingBootstrapAdapter) -> int:
    from intergrax.rag.embedding.providers.hf_embedding_provider import HFEmbeddingProvider

    provider = embedding_adapter.embedding_provider()
    if isinstance(provider, HFEmbeddingProvider):
        return 1 if provider._model is not None else 0  # noqa: SLF001
    return 1


def _write_session_outputs(
    report: VectorDbRoundTripQualificationReport,
    run_log: str,
) -> None:
    _SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    write_vector_db_roundtrip_report_json(
        _SESSION_ROOT / "vector-db-roundtrip-qualification-report.json",
        report,
    )
    (_SESSION_ROOT / "VECTOR_DB_ROUNDTRIP_QUALIFICATION_REPORT.md").write_text(
        render_vector_db_roundtrip_markdown_report(report),
        encoding="utf-8",
    )
    (_SESSION_ROOT / "run.log").write_text(run_log, encoding="utf-8")


def run_qualification() -> VectorDbRoundTripQualificationReport:
    git_sha = _git_sha()
    relational_checksum = _sha256_file(_RELATIONAL_PATH)
    embedding_checksum = _sha256_file(_EMBEDDING_PATH)
    relational_records = read_relational_parquet(_RELATIONAL_PATH)
    embedding_records = read_embedding_parquet(
        _EMBEDDING_PATH,
        expected_dimension=VPI_CANONICAL_EMBEDDING_DIMENSION,
    )
    artifact_integrity = evaluate_pilot_artifact_integrity(
        relational_records,
        embedding_records,
    )
    if not artifact_integrity.passed:
        report = VectorDbRoundTripQualificationReport(
            task_id=QUALIFICATION_TASK_ID,
            status=QualificationRunStatus.ARTIFACT_INTEGRITY_FAIL,
            git_sha=git_sha,
            pilot_root=str(_PILOT_ROOT),
            pilot_relational_checksum=relational_checksum,
            pilot_embedding_checksum=embedding_checksum,
            provider=VPI_CANONICAL_EMBEDDING_PROVIDER,
            model=VPI_CANONICAL_EMBEDDING_MODEL,
            revision=VPI_CANONICAL_EMBEDDING_REVISION,
            dimension=VPI_CANONICAL_EMBEDDING_DIMENSION,
            document_policy=VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
            document_token_budget=VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET,
            effective_provider_ceiling=VPI_BGE_M3_DOCUMENT_EFFECTIVE_TOKEN_CEILING,
            query_policy_changed=False,
            qdrant=QdrantIndexSnapshot(
                collection_name="",
                metric="cosine",
                dimension=0,
                point_count=0,
                dense_search_available=False,
                temporary_isolated_collection=False,
                cleanup_passed=False,
            ),
            document_embedding_calls=0,
            query_embedding_calls=0,
            model_load_count=0,
            artifact_integrity=artifact_integrity,
            self_probes=(),
            query_evidence=(),
            metrics=aggregate_query_metrics(()),
            known_gaps=artifact_integrity.failure_reasons,
        )
        _write_session_outputs(report, "ARTIFACT_INTEGRITY_FAIL\n")
        return report

    logical_point_ids = tuple(record.logical_point_id for record in embedding_records)
    source_ref_by_logical_id = {
        record.logical_point_id: record.source_ref for record in embedding_records
    }
    corpus_matrix = build_embedding_matrix(embedding_records)
    dataset_checksum = hashlib.sha256(
        f"{relational_checksum}:{embedding_checksum}".encode("utf-8")
    ).hexdigest()
    first_relational = relational_records[0]
    manifest = _build_manifest(
        dataset_checksum=dataset_checksum,
        catalog_id=first_relational.source_ref.catalog_id,
        derivation_version=first_relational.derivation_version,
    )
    search_records = _build_search_ingest_records(
        relational_records,
        embedding_records,
        dataset_checksum=dataset_checksum,
    )
    if len(search_records) != PILOT_RECORD_COUNT:
        msg = f"expected {PILOT_RECORD_COUNT} search records, got {len(search_records)}"
        raise RuntimeError(msg)

    try:
        ensure_qdrant_available()
    except (QdrantResourcePreconditionError, QdrantMetricConfigurationError) as exc:
        report = VectorDbRoundTripQualificationReport(
            task_id=QUALIFICATION_TASK_ID,
            status=QualificationRunStatus.RESOURCE_PRECONDITION_FAIL,
            git_sha=git_sha,
            pilot_root=str(_PILOT_ROOT),
            pilot_relational_checksum=relational_checksum,
            pilot_embedding_checksum=embedding_checksum,
            provider=VPI_CANONICAL_EMBEDDING_PROVIDER,
            model=VPI_CANONICAL_EMBEDDING_MODEL,
            revision=VPI_CANONICAL_EMBEDDING_REVISION,
            dimension=VPI_CANONICAL_EMBEDDING_DIMENSION,
            document_policy=VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
            document_token_budget=VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET,
            effective_provider_ceiling=VPI_BGE_M3_DOCUMENT_EFFECTIVE_TOKEN_CEILING,
            query_policy_changed=False,
            qdrant=QdrantIndexSnapshot(
                collection_name="",
                metric="cosine",
                dimension=0,
                point_count=0,
                dense_search_available=False,
                temporary_isolated_collection=False,
                cleanup_passed=False,
            ),
            document_embedding_calls=0,
            query_embedding_calls=0,
            model_load_count=0,
            artifact_integrity=artifact_integrity,
            self_probes=(),
            query_evidence=(),
            metrics=aggregate_query_metrics(()),
            known_gaps=(str(exc),),
        )
        _write_session_outputs(report, f"RESOURCE_PRECONDITION_FAIL: {exc}\n")
        return report

    collection_name = make_qualification_collection_name()
    runtime = open_qdrant_qualification_runtime(collection_name)
    cleanup_passed = False
    document_embedding_calls = 0
    query_embedding_calls = 0
    model_load_count = 0
    self_probes: list = []
    query_evidence: list = []
    qdrant_snapshot = describe_qdrant_index(runtime)
    report: VectorDbRoundTripQualificationReport | None = None
    try:
        runtime.adapter.prepare(manifest)
        runtime.adapter.ingest_batch(
            SearchIndexIngestBatch(batch_ordinal=0, records=search_records)
        )
        qdrant_snapshot = describe_qdrant_index(runtime)

        for pilot_index in SELF_PROBE_INDICES:
            embedding_record = embedding_records[pilot_index]
            query_vector = np.asarray(embedding_record.dense_embedding, dtype=np.float64)
            hits, returned_refs = query_qdrant_ranked_hits(runtime, query_vector, top_k=5)
            top_hit = hits[0]
            returned_ref = returned_refs[top_hit.logical_point_id]
            self_probes.append(
                evaluate_self_probe(
                    pilot_index=pilot_index,
                    expected_logical_point_id=embedding_record.logical_point_id,
                    expected_source_ref=embedding_record.source_ref,
                    returned_logical_point_id=top_hit.logical_point_id,
                    returned_source_ref=returned_ref,
                    self_cosine_score=top_hit.cosine_score,
                )
            )

        ensure_embedding_provider_integrations_registered()
        profile = resolve_data_pack_build_execution_profile(PRODUCTION_LOCAL_GPU_PROFILE_ID)
        apply_data_pack_build_execution_profile(profile)
        embedding_configuration = load_vpi_embedding_configuration()
        execution_configuration = load_vpi_embedding_provider_execution_configuration()
        embedding_adapter = IntergraxEmbeddingBootstrapAdapter(
            embedding_configuration,
            execution_configuration=execution_configuration,
        )
        arena_records = _build_arena_records(relational_records)
        query_cases = _select_query_cases(arena_records)
        query_vectors: list[np.ndarray] = []
        for case in query_cases:
            vectors = embedding_adapter.embed_batch((case.query_text,))
            query_embedding_calls += 1
            query_vectors.append(np.asarray(vectors[0], dtype=np.float64))
        model_load_count = _count_hf_model_loads(embedding_adapter)
        embedding_adapter.close()

        for case, query_vector in zip(query_cases, query_vectors, strict=True):
            qdrant_hits, qdrant_source_refs = query_qdrant_ranked_hits(
                runtime,
                query_vector,
                top_k=10,
            )
            query_evidence.append(
                evaluate_query_round_trip(
                    query_id=case.case_id,
                    query_text=case.query_text,
                    query_vector=query_vector,
                    corpus_embeddings=corpus_matrix,
                    logical_point_ids=logical_point_ids,
                    qdrant_hits=qdrant_hits,
                    source_ref_by_logical_id=source_ref_by_logical_id,
                    qdrant_source_refs=qdrant_source_refs,
                )
            )

        metrics = aggregate_query_metrics(query_evidence)
        status = evaluate_hard_gate(
            artifact_integrity=artifact_integrity,
            self_probes=self_probes,
            query_evidence=query_evidence,
            metrics=metrics,
            document_embedding_calls=document_embedding_calls,
            query_embedding_calls=query_embedding_calls,
            qdrant_point_count=qdrant_snapshot.point_count,
            qdrant_dimension=qdrant_snapshot.dimension,
            qdrant_metric=qdrant_snapshot.metric,
            dense_search_available=qdrant_snapshot.dense_search_available,
        )
        known_gaps: tuple[str, ...] = ()
        if status is QualificationRunStatus.VECTOR_INDEX_RETRIEVAL_REGRESSION:
            known_gaps = (
                "ANN/index ranking parity failed while transport checks may still pass",
            )
        report = VectorDbRoundTripQualificationReport(
            task_id=QUALIFICATION_TASK_ID,
            status=status,
            git_sha=git_sha,
            pilot_root=str(_PILOT_ROOT),
            pilot_relational_checksum=relational_checksum,
            pilot_embedding_checksum=embedding_checksum,
            provider=VPI_CANONICAL_EMBEDDING_PROVIDER,
            model=VPI_CANONICAL_EMBEDDING_MODEL,
            revision=VPI_CANONICAL_EMBEDDING_REVISION,
            dimension=VPI_CANONICAL_EMBEDDING_DIMENSION,
            document_policy=VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
            document_token_budget=VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET,
            effective_provider_ceiling=VPI_BGE_M3_DOCUMENT_EFFECTIVE_TOKEN_CEILING,
            query_policy_changed=False,
            qdrant=qdrant_snapshot,
            document_embedding_calls=document_embedding_calls,
            query_embedding_calls=query_embedding_calls,
            model_load_count=model_load_count,
            artifact_integrity=artifact_integrity,
            self_probes=tuple(self_probes),
            query_evidence=tuple(query_evidence),
            metrics=metrics,
            known_gaps=known_gaps,
        )
    finally:
        runtime.adapter.close()
        cleanup_passed = delete_qualification_collection(
            runtime.collection_name,
            config=runtime.config,
        )
        if report is not None:
            report = replace(
                report,
                qdrant=replace(report.qdrant, cleanup_passed=cleanup_passed),
            )
            _write_session_outputs(report, f"STATUS={report.status.value}\n")
    if report is None:
        msg = "qualification did not produce a report"
        raise RuntimeError(msg)
    return report


def main() -> int:
    report = run_qualification()
    print(f"STATUS: {report.status.value}")
    return 0 if report.status is QualificationRunStatus.PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
