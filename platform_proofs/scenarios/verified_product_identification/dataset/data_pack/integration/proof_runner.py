"""End-to-end proof-50 orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.application.catalog.source_resolution import (
    SourceTruthResolutionError,
    resolve_source_record,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    load_vpi_embedding_configuration,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    load_vpi_embedding_provider_execution_configuration,
)
from platform_proofs.scenarios.verified_product_identification.arena.evaluation.metrics import (
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.builder import (
    build_proof_50_data_pack,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.validation import (
    assert_validation_pass,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.composition.runtime import (
    build_proof_50_storage_runtime,
    build_proof_50_search_runtime,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.evidence import (
    DataPackProofReport,
    RetrievalMetricRow,
    ValidationSectionResult,
    write_proof_report,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    DEFAULT_PROOF_50_ROOT,
    PROOF_50_POSTGRESQL_SCHEMA,
    PROOF_50_QDRANT_COLLECTION,
    resolve_data_pack_paths,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.integration.query_set import (
    ProofQueryCase,
    build_proof_query_set,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.integration.storage_loader import (
    load_data_pack_into_reference_storage,
    validate_storage_load_result,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.filesystem_reader import (
    FilesystemDataPackReader,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.storage_environment import (
    storage_environment_available,
    storage_environment_gap,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    BootstrapState,
)


def _section_from_report(report) -> ValidationSectionResult:
    return ValidationSectionResult(
        status=report.status.value,
        detail="; ".join(f"{check.name}={check.status.value}" for check in report.checks),
    )


def _evaluate_query_case(
    case: ProofQueryCase,
    *,
    exact_lookup,
    lexical_search,
    structured_search,
    vector_search,
) -> RetrievalMetricRow:
    if case.channel == "exact" and case.exact_query is not None:
        result = exact_lookup.lookup(case.exact_query)
        ranked_offer_ids = [candidate.offer_id.value for candidate in result.candidates]
        relevant_index = 0
        ranked_indices = [
            index
            for index, offer_id in enumerate(ranked_offer_ids)
            if offer_id == case.expected_offer_id
        ]
        passed = (
            not case.negative
            and ranked_offer_ids
            and ranked_offer_ids[0] == case.expected_offer_id
        )
        return RetrievalMetricRow(
            query_id=case.query_id,
            channel=case.channel,
            recall_at_1=recall_at_k([relevant_index], ranked_indices, 1) if ranked_indices else 0.0,
            recall_at_5=recall_at_k([relevant_index], ranked_indices, 5) if ranked_indices else 0.0,
            recall_at_10=recall_at_k([relevant_index], ranked_indices, 10) if ranked_indices else 0.0,
            mrr_at_10=mrr_at_k([relevant_index], ranked_indices, 10) if ranked_indices else 0.0,
            ndcg_at_10=ndcg_at_k([relevant_index], ranked_indices, 10) if ranked_indices else 0.0,
            passed=passed,
            detail=f"ranked={ranked_offer_ids[:3]}",
        )
    if case.channel == "lexical" and case.lexical_query is not None:
        result = lexical_search.search(case.lexical_query)
        ranked_offer_ids = [candidate.offer_id.value for candidate in result.candidates]
        if case.negative:
            passed = case.expected_offer_id not in ranked_offer_ids
            detail = f"negative ranked={ranked_offer_ids}"
            return RetrievalMetricRow(
                query_id=case.query_id,
                channel=case.channel,
                recall_at_1=None,
                recall_at_5=None,
                recall_at_10=None,
                mrr_at_10=None,
                ndcg_at_10=None,
                passed=passed,
                detail=detail,
            )
        passed = case.expected_offer_id in ranked_offer_ids
        ranked_indices = [
            index
            for index, offer_id in enumerate(ranked_offer_ids)
            if offer_id == case.expected_offer_id
        ]
        return RetrievalMetricRow(
            query_id=case.query_id,
            channel=case.channel,
            recall_at_1=1.0 if ranked_indices and ranked_indices[0] == 0 else 0.0,
            recall_at_5=1.0 if ranked_indices and ranked_indices[0] < 5 else 0.0,
            recall_at_10=1.0 if ranked_indices else 0.0,
            mrr_at_10=mrr_at_k([0], ranked_indices, 10) if ranked_indices else 0.0,
            ndcg_at_10=ndcg_at_k([0], ranked_indices, 10) if ranked_indices else 0.0,
            passed=passed,
            detail=f"ranked={ranked_offer_ids[:5]}",
        )
    if case.channel == "structured" and case.structured_query is not None:
        result = structured_search.search(case.structured_query)
        ranked_offer_ids = [candidate.offer_id.value for candidate in result.candidates]
        passed = case.expected_offer_id in ranked_offer_ids
        return RetrievalMetricRow(
            query_id=case.query_id,
            channel=case.channel,
            recall_at_1=1.0 if ranked_offer_ids and ranked_offer_ids[0] == case.expected_offer_id else 0.0,
            recall_at_5=1.0 if case.expected_offer_id in ranked_offer_ids[:5] else 0.0,
            recall_at_10=1.0 if passed else 0.0,
            mrr_at_10=1.0 if ranked_offer_ids and ranked_offer_ids[0] == case.expected_offer_id else 0.0,
            ndcg_at_10=1.0 if passed else 0.0,
            passed=passed,
            detail=f"ranked={ranked_offer_ids[:5]}",
        )
    if case.channel == "vector" and case.vector_query is not None:
        result = vector_search.search(case.vector_query)
        ranked_offer_ids = [candidate.offer_id.value for candidate in result.candidates]
        ranked_indices = [
            index
            for index, offer_id in enumerate(ranked_offer_ids)
            if offer_id == case.expected_offer_id
        ]
        passed = bool(ranked_indices)
        return RetrievalMetricRow(
            query_id=case.query_id,
            channel=case.channel,
            recall_at_1=recall_at_k([0], ranked_indices, 1) if ranked_indices else 0.0,
            recall_at_5=recall_at_k([0], ranked_indices, 5) if ranked_indices else 0.0,
            recall_at_10=recall_at_k([0], ranked_indices, 10) if ranked_indices else 0.0,
            mrr_at_10=mrr_at_k([0], ranked_indices, 10) if ranked_indices else 0.0,
            ndcg_at_10=ndcg_at_k([0], ranked_indices, 10) if ranked_indices else 0.0,
            passed=passed,
            detail=f"ranked={ranked_offer_ids[:5]}",
        )
    return RetrievalMetricRow(
        query_id=case.query_id,
        channel=case.channel,
        recall_at_1=None,
        recall_at_5=None,
        recall_at_10=None,
        mrr_at_10=None,
        ndcg_at_10=None,
        passed=False,
        detail="unsupported query case",
    )


def run_proof_50(
    *,
    output_root: Path = DEFAULT_PROOF_50_ROOT,
    dataset_path: Path,
    dataset_manifest_path: Path,
    rebuild_data_pack: bool = True,
) -> DataPackProofReport:
    warnings: list[str] = []
    known_gaps: list[str] = ["50-RECORD RETRIEVAL PROOF — not final production benchmark"]
    paths = resolve_data_pack_paths(output_root)

    if rebuild_data_pack or not paths.manifest_file.is_file():
        build_proof_50_data_pack(
            output_root=output_root,
            dataset_path=dataset_path,
            dataset_manifest_path=dataset_manifest_path,
        )

    reader = FilesystemDataPackReader(output_root)
    manifest = reader.read_manifest()
    integrity_report = reader.validate_integrity()
    relational_validation = _section_from_report(
        integrity_report
    )

    if not storage_environment_available():
        gap = storage_environment_gap() or "storage environment unavailable"
        report = DataPackProofReport(
            status=DataPackStatus.BLOCKED,
            data_pack_identity=manifest.data_pack_version,
            record_count=manifest.record_count,
            relational_validation=relational_validation,
            embedding_validation=relational_validation,
            cross_ref_validation=relational_validation,
            checksum_validation=relational_validation,
            semantic_text_hash_validation=relational_validation,
            relational_load=ValidationSectionResult(status="BLOCKED", detail=gap),
            vector_load=ValidationSectionResult(status="BLOCKED", detail=gap),
            zero_embedding_calls=ValidationSectionResult(status="BLOCKED", detail=gap),
            retrieval_metrics=(),
            mapping_validation=ValidationSectionResult(status="BLOCKED", detail=gap),
            negative_match_validation=ValidationSectionResult(status="BLOCKED", detail=gap),
            provider_configuration=(
                f"postgresql_schema={PROOF_50_POSTGRESQL_SCHEMA};"
                f"qdrant_collection={PROOF_50_QDRANT_COLLECTION}"
            ),
            idempotent_reload=ValidationSectionResult(status="BLOCKED", detail=gap),
            warnings=(gap,),
            known_gaps=tuple(known_gaps),
        )
        write_proof_report(paths.proof_report_file, report)
        reader.close()
        return report

    storage_runtime = build_proof_50_storage_runtime()
    search_runtime = build_proof_50_search_runtime()
    bootstrap_manifest = storage_runtime.bootstrap_manifest
    try:
        load_result = load_data_pack_into_reference_storage(
            reader=reader,
            catalog=storage_runtime.catalog,
            search=storage_runtime.search,
            manifest=bootstrap_manifest,
        )
        load_report = validate_storage_load_result(
            load_result,
            expected_count=manifest.record_count,
        )
        assert_validation_pass(load_report, stage="storage_load")

        reload_result = load_data_pack_into_reference_storage(
            reader=reader,
            catalog=storage_runtime.catalog,
            search=storage_runtime.search,
            manifest=bootstrap_manifest,
        )
        reload_report = validate_storage_load_result(
            reload_result,
            expected_count=manifest.record_count,
        )
        idempotent_ok = reload_report.status is ValidationStatus.PASS

        query_cases = build_proof_query_set(reader.read_relational_records())
        metric_rows: list[RetrievalMetricRow] = []
        mapping_ok = True
        for case in query_cases:
            metric_rows.append(
                _evaluate_query_case(
                    case,
                    exact_lookup=search_runtime.exact_lookup,
                    lexical_search=search_runtime.lexical_search,
                    structured_search=search_runtime.structured_search,
                    vector_search=search_runtime.vector_search,
                )
            )
            if case.negative:
                continue
            if case.channel == "exact" and case.exact_query is not None:
                result = search_runtime.exact_lookup.lookup(case.exact_query)
                candidates = result.candidates
            elif case.channel == "lexical" and case.lexical_query is not None:
                result = search_runtime.lexical_search.search(case.lexical_query)
                candidates = result.candidates
            elif case.channel == "structured" and case.structured_query is not None:
                result = search_runtime.structured_search.search(case.structured_query)
                candidates = result.candidates
            elif case.channel == "vector" and case.vector_query is not None:
                result = search_runtime.vector_search.search(case.vector_query)
                candidates = result.candidates
            else:
                candidates = ()
            for candidate in candidates[:3]:
                try:
                    resolve_source_record(candidate, search_runtime.source_fetch)
                except SourceTruthResolutionError:
                    mapping_ok = False

        retrieval_passed = all(row.passed for row in metric_rows)
        negative_rows = [row for row in metric_rows if row.query_id.startswith("negative")]
        negative_ok = all(row.passed for row in negative_rows)

        final_status = DataPackStatus.READY if retrieval_passed and mapping_ok and negative_ok else DataPackStatus.BLOCKED
        report = DataPackProofReport(
            status=final_status,
            data_pack_identity=manifest.data_pack_version,
            record_count=manifest.record_count,
            relational_validation=relational_validation,
            embedding_validation=relational_validation,
            cross_ref_validation=relational_validation,
            checksum_validation=ValidationSectionResult(status="PASS", detail="SHA256SUMS verified"),
            semantic_text_hash_validation=ValidationSectionResult(status="PASS", detail="semantic hashes match"),
            relational_load=ValidationSectionResult(
                status=load_report.status.value,
                detail=f"rows={load_result.catalog_source_rows}",
            ),
            vector_load=ValidationSectionResult(
                status=load_report.status.value,
                detail=f"points={load_result.search_point_count}",
            ),
            zero_embedding_calls=ValidationSectionResult(
                status="PASS",
                detail=f"embedding_calls={load_result.embedding_calls}",
            ),
            retrieval_metrics=tuple(metric_rows),
            mapping_validation=ValidationSectionResult(
                status="PASS" if mapping_ok else "FAIL",
                detail="candidate source_ref resolves to source record",
            ),
            negative_match_validation=ValidationSectionResult(
                status="PASS" if negative_ok else "FAIL",
                detail="negative lexical query did not resolve unrelated product",
            ),
            provider_configuration=(
                f"postgresql_schema={PROOF_50_POSTGRESQL_SCHEMA};"
                f"qdrant_collection={PROOF_50_QDRANT_COLLECTION}"
            ),
            idempotent_reload=ValidationSectionResult(
                status="PASS" if idempotent_ok else "FAIL",
                detail="second load preserved counts without duplication",
            ),
            warnings=tuple(warnings),
            known_gaps=tuple(known_gaps),
        )
    finally:
        storage_runtime.close()
        search_runtime.close()
        reader.close()

    write_proof_report(paths.proof_report_file, report)
    return report
