"""VPI-IMPLEMENTATION-5C4E4 — bounded CUDA batch throughput qualification runner."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    load_vpi_embedding_configuration,
    validate_resolved_provider_dimension,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    assert_execution_device_available,
    load_vpi_embedding_provider_execution_configuration,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_execution_profiles import (
    PRODUCTION_LOCAL_GPU_PROFILE_ID,
    apply_data_pack_build_execution_profile,
    resolve_data_pack_build_execution_profile,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding_input_policy import (
    VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
    VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    VPI_CANONICAL_EMBEDDING_DIMENSION,
    VPI_CANONICAL_EMBEDDING_MODEL,
    VPI_CANONICAL_EMBEDDING_PROVIDER,
    VPI_CANONICAL_EMBEDDING_REVISION,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.bootstrap import (
    ensure_embedding_provider_integrations_registered,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.canonical_document_embedding_input_policy import (
    resolve_canonical_document_embedding_input_policy,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.model_identity import (
    resolve_embedding_model_identity,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.tokenizer_codec import (
    resolve_embedding_tokenizer_codec,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.contracts import (
    CANONICAL_TOKEN_BUDGET,
    MANDATORY_BATCH_VARIANTS,
    OPTIONAL_BATCH_VARIANT,
    QUALIFICATION_TASK_ID,
    SAMPLE_RECORD_COUNT,
    BoundedCudaBatchThroughputReport,
    OptionalBatch32Decision,
    QualificationRunStatus,
    TokenProfile,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.metrics import (
    collect_token_budget_violations,
    compute_token_profile,
    project_embedding_time,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.reporting import (
    write_bounded_cuda_markdown_report,
    write_bounded_cuda_report_json,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.runtime import (
    count_hf_model_loads,
    measure_batch_variant,
    require_hf_embedding_provider,
    run_cuda_preflight,
    run_warmup,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.sample import (
    load_bounded_qualification_documents,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.selection import (
    select_production_batch_candidate,
    should_run_optional_batch_32,
)

_SESSION_ROOT = _REPO_ROOT / ".tmp" / "session" / "vpi-5c4e4"
_DATASET_PATH = (
    _REPO_ROOT
    / "platform_proofs"
    / "scenarios"
    / "verified_product_identification"
    / "dataset"
    / "processed"
    / "selected_offers.parquet"
)


def _apply_execution_profile() -> None:
    profile = resolve_data_pack_build_execution_profile(PRODUCTION_LOCAL_GPU_PROFILE_ID)
    apply_data_pack_build_execution_profile(profile)


def _count_tokens(
    texts: tuple[str, ...],
    tokenizer_codec,
) -> tuple[int, ...]:
    return tuple(len(tokenizer_codec.encode_text_tokens(text)) for text in texts)


def _enrich_measurement_tokens(
    measurement,
    *,
    token_profile,
):
    return replace(
        measurement,
        p50_tokens=token_profile.p50_tokens,
        p95_tokens=token_profile.p95_tokens,
    )


def _detect_persistent_vram_growth(measurements) -> bool:
    if len(measurements) < 2:
        return False
    first_after = measurements[0].gpu_free_memory_bytes_after
    last_after = measurements[-1].gpu_free_memory_bytes_after
    if first_after <= 0:
        return False
    return last_after < int(first_after * 0.90)


def run_bounded_cuda_batch_throughput_qualification() -> BoundedCudaBatchThroughputReport:
    preflight = run_cuda_preflight()
    known_gaps: list[str] = []
    if not preflight.cuda_available:
        return _resource_fail_report(preflight, known_gaps, "CUDA unavailable")
    if preflight.resource_precondition_fail_reason is not None:
        return _resource_fail_report(
            preflight,
            known_gaps,
            preflight.resource_precondition_fail_reason,
        )

    _apply_execution_profile()
    ensure_embedding_provider_integrations_registered()
    embedding_configuration = load_vpi_embedding_configuration()
    execution_configuration = load_vpi_embedding_provider_execution_configuration()
    assert_execution_device_available(execution_configuration)

    embedding_port = IntergraxEmbeddingBootstrapAdapter(
        embedding_configuration,
        execution_configuration=execution_configuration,
    )
    try:
        probe = embedding_port.probe()
        validate_resolved_provider_dimension(
            configuration=embedding_configuration,
            resolved_dimension=probe.resolved_dimension,
        )
        provider = require_hf_embedding_provider(embedding_port)
        model_load_count = count_hf_model_loads(provider)
        if model_load_count != 1:
            known_gaps.append(
                f"expected model load count 1, observed {model_load_count}"
            )

        policy = resolve_canonical_document_embedding_input_policy(embedding_port)
        if policy.policy_version != VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION:
            msg = (
                "unexpected document embedding input policy version: "
                f"{policy.policy_version}"
            )
            raise RuntimeError(msg)

        documents = load_bounded_qualification_documents(
            dataset_path=_DATASET_PATH,
            record_count=SAMPLE_RECORD_COUNT,
            policy=policy,
            artifact_dir=_SESSION_ROOT / "artifact-root",
        )
        if documents.record_count != SAMPLE_RECORD_COUNT:
            msg = (
                f"expected {SAMPLE_RECORD_COUNT} records, got {documents.record_count}"
            )
            raise RuntimeError(msg)

        tokenizer_codec = resolve_embedding_tokenizer_codec(embedding_port)
        token_counts = _count_tokens(documents.bounded_document_texts, tokenizer_codec)
        token_budget_violations = collect_token_budget_violations(
            token_counts,
            token_budget=CANONICAL_TOKEN_BUDGET,
        )
        if token_budget_violations:
            known_gaps.append(
                "bounded document texts re-encode above token budget after decode round-trip: "
                + "; ".join(token_budget_violations)
            )
        token_profile = compute_token_profile(token_counts)

        run_warmup(
            embedding_port,
            documents.bounded_document_texts,
            provider=provider,
        )

        measurements: list = []
        stop_escalation = False
        previous_smaller = None
        for batch_size in MANDATORY_BATCH_VARIANTS:
            if stop_escalation:
                break
            measurement = measure_batch_variant(
                embedding_port,
                provider=provider,
                document_texts=documents.bounded_document_texts,
                token_counts=token_counts,
                batch_size=batch_size,
                previous_smaller=previous_smaller,
            )
            measurement = _enrich_measurement_tokens(
                measurement,
                token_profile=token_profile,
            )
            measurements.append(measurement)
            if measurement.cuda_oom:
                stop_escalation = True
            previous_smaller = measurement

        batch_by_size = {item.batch_size: item for item in measurements}
        batch_8 = batch_by_size.get(8)
        batch_16 = batch_by_size.get(16)
        optional_batch_32 = OptionalBatch32Decision(
            executed=False,
            reason="batch 16 measurement unavailable",
        )
        if batch_16 is not None:
            optional_batch_32 = should_run_optional_batch_32(
                batch_8=batch_8,
                batch_16=batch_16,
            )
            if optional_batch_32.executed and not stop_escalation:
                measurement = measure_batch_variant(
                    embedding_port,
                    provider=provider,
                    document_texts=documents.bounded_document_texts,
                    token_counts=token_counts,
                    batch_size=OPTIONAL_BATCH_VARIANT,
                    previous_smaller=batch_16,
                )
                measurement = _enrich_measurement_tokens(
                    measurement,
                    token_profile=token_profile,
                )
                measurements.append(measurement)
                if measurement.cuda_oom or not measurement.safe:
                    stop_escalation = True

        measurement_tuple = tuple(measurements)
        production_selection = select_production_batch_candidate(measurement_tuple)
        projections = tuple(
            project_embedding_time(
                batch_size=item.batch_size,
                records_per_second=item.records_per_second,
                safe=item.safe,
            )
            for item in measurement_tuple
        )
        winner_projection = None
        if production_selection.batch_size is not None:
            winner_projection = next(
                (
                    projection
                    for projection in projections
                    if projection.batch_size == production_selection.batch_size
                ),
                None,
            )

        identity = resolve_embedding_model_identity(
            VPI_CANONICAL_EMBEDDING_PROVIDER,
            VPI_CANONICAL_EMBEDDING_MODEL,
        )
        if identity.revision != VPI_CANONICAL_EMBEDDING_REVISION:
            known_gaps.append(
                "resolved HF revision "
                f"{identity.revision} differs from canonical "
                f"{VPI_CANONICAL_EMBEDDING_REVISION}"
            )

        oom_observed = any(item.cuda_oom for item in measurement_tuple)
        persistent_vram_growth = _detect_persistent_vram_growth(measurement_tuple)
        status = QualificationRunStatus.PASS
        if production_selection.batch_size is None:
            status = QualificationRunStatus.FAIL

        return BoundedCudaBatchThroughputReport(
            task_id=QUALIFICATION_TASK_ID,
            status=status,
            python_executable=str(Path(sys.executable)),
            preflight=preflight,
            provider=VPI_CANONICAL_EMBEDDING_PROVIDER,
            model=VPI_CANONICAL_EMBEDDING_MODEL,
            revision=identity.revision,
            dimension=VPI_CANONICAL_EMBEDDING_DIMENSION,
            model_load_count=model_load_count,
            policy_version=policy.policy_version,
            token_budget=VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET,
            dataset_path=documents.dataset_path,
            record_count=documents.record_count,
            selection_method=documents.selection,
            token_profile=token_profile,
            batch_measurements=measurement_tuple,
            optional_batch_32=optional_batch_32,
            production_batch_selection=production_selection,
            projections=projections,
            winner_projection=winner_projection,
            projection_only=True,
            oom_observed=oom_observed,
            system_instability=False,
            persistent_vram_growth=persistent_vram_growth,
            known_gaps=tuple(known_gaps),
        )
    finally:
        embedding_port.close()


def _resource_fail_report(preflight, known_gaps: list[str], reason: str) -> BoundedCudaBatchThroughputReport:
    known_gaps.append(reason)
    return BoundedCudaBatchThroughputReport(
        task_id=QUALIFICATION_TASK_ID,
        status=QualificationRunStatus.RESOURCE_PRECONDITION_FAIL,
        python_executable=str(Path(sys.executable)),
        preflight=preflight,
        provider=VPI_CANONICAL_EMBEDDING_PROVIDER,
        model=VPI_CANONICAL_EMBEDDING_MODEL,
        revision=VPI_CANONICAL_EMBEDDING_REVISION,
        dimension=VPI_CANONICAL_EMBEDDING_DIMENSION,
        model_load_count=0,
        policy_version=VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
        token_budget=VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET,
        dataset_path=str(_DATASET_PATH),
        record_count=0,
        selection_method="not_executed",
        token_profile=TokenProfile(
            total_tokens=0,
            average_tokens_per_record=0.0,
            p50_tokens=0.0,
            p95_tokens=0.0,
            max_tokens=0,
        ),
        batch_measurements=(),
        optional_batch_32=OptionalBatch32Decision(
            executed=False,
            reason=reason,
        ),
        production_batch_selection=select_production_batch_candidate(()),
        projections=(),
        winner_projection=None,
        projection_only=True,
        oom_observed=False,
        system_instability=False,
        persistent_vram_growth=False,
        known_gaps=tuple(known_gaps),
    )


def main() -> int:
    _SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    report = run_bounded_cuda_batch_throughput_qualification()
    write_bounded_cuda_report_json(
        _SESSION_ROOT / "bounded-cuda-batch-throughput-report.json",
        report,
    )
    write_bounded_cuda_markdown_report(
        _SESSION_ROOT / "BOUNDED_CUDA_BATCH_THROUGHPUT_REPORT.md",
        report,
    )
    print(json.dumps({"status": report.status.value}, indent=2))
    if report.status is QualificationRunStatus.RESOURCE_PRECONDITION_FAIL:
        return 2
    if report.status is QualificationRunStatus.FAIL:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
