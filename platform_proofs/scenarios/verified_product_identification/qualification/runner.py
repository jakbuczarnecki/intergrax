"""VPI embedding performance qualification runner — scenario-owned orchestration."""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    EMBEDDING_CONFIGURATION_VERSION,
    load_vpi_embedding_configuration,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    VpiEmbeddingDeviceUnavailableError,
    assert_execution_device_available,
    load_vpi_embedding_provider_execution_configuration,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.composition.materialization_runtime import (
    build_vpi_embedding_materialization_runtime,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.config import (
    load_vpi_embedding_materialization_config,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EmbeddingArtifactState,
)
from platform_proofs.scenarios.verified_product_identification.qualification.batch_selection import (
    select_best_provider_batch_size,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bottleneck import (
    analyze_bottleneck,
)
from platform_proofs.scenarios.verified_product_identification.qualification.contracts.classification import (
    VpiEmbeddingQualificationStatus,
)
from platform_proofs.scenarios.verified_product_identification.qualification.contracts.results import (
    QUALIFICATION_VERSION,
    ArtifactIntegritySnapshot,
    EmbeddingIdentitySnapshot,
    ExecutionConfigurationSnapshot,
    FullBuildEstimate,
    MaterializationQualificationSnapshot,
    RestartQualificationSnapshot,
    StorageQualificationSnapshot,
    VpiEmbeddingQualificationReport,
    WarmupTimingSnapshot,
)
from platform_proofs.scenarios.verified_product_identification.qualification.duration_estimate import (
    estimate_full_build_duration,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.hardware_probe import (
    probe_hardware_runtime_capability,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.microbenchmark import (
    build_embedding_execution_port,
    measure_warmup_timing,
    resolve_provider_execution_proof,
    run_provider_batch_candidate,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.semantic_text_sampler import (
    sample_semantic_texts,
)
from platform_proofs.scenarios.verified_product_identification.qualification.integration.storage_environment import (
    storage_environment_gap,
    storage_environment_available,
)
from platform_proofs.scenarios.verified_product_identification.qualification.text_length_profile import (
    profile_text_lengths,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.config import (
    BootstrapRunMode,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    BootstrapState,
)

FULL_DATASET_RECORD_COUNT = 3_770_377
DEFAULT_MICROBENCHMARK_RECORDS = 192
DEFAULT_RECORD_TARGET = 1000
DEFAULT_PROVIDER_BATCH_CANDIDATES = (16, 32, 64)


def _materialization_snapshot(report) -> MaterializationQualificationSnapshot:
    manifest = report.manifest
    return MaterializationQualificationSnapshot(
        state=report.final_state.value,
        rows=manifest.checkpoint_rows_materialized if manifest is not None else report.rows_materialized,
        shards=manifest.shard_count if manifest is not None else report.shards_committed,
        derive_seconds=report.elapsed_derive_seconds,
        embedding_seconds=report.elapsed_embedding_seconds,
        artifact_write_seconds=report.elapsed_artifact_write_seconds,
        total_seconds=report.elapsed_total_seconds,
        embedding_calls=report.embedding_calls,
        materialization_records_per_second=report.effective_records_per_second,
        embedding_records_per_second=report.embedding_records_per_second,
    )


def _classify_status(
    *,
    materialization_ok: bool,
    restart_ok: bool,
    storage_ok: bool,
    cuda_available: bool,
    explicit_cuda_requested: bool,
) -> VpiEmbeddingQualificationStatus:
    if not materialization_ok:
        return VpiEmbeddingQualificationStatus.FAILED_CORRECTNESS
    if not restart_ok:
        return VpiEmbeddingQualificationStatus.FAILED_CORRECTNESS
    if not storage_ok:
        return VpiEmbeddingQualificationStatus.BLOCKED_STORAGE_ENVIRONMENT
    if explicit_cuda_requested and not cuda_available:
        return VpiEmbeddingQualificationStatus.BLOCKED_GPU
    if materialization_ok and storage_ok and not cuda_available:
        return VpiEmbeddingQualificationStatus.PARTIAL_PASS_GPU
    return VpiEmbeddingQualificationStatus.PASS


def run_vpi_embedding_qualification(
    *,
    record_target: int = DEFAULT_RECORD_TARGET,
    microbenchmark_records: int = DEFAULT_MICROBENCHMARK_RECORDS,
    artifact_dir: Path,
    provider_batch_candidates: tuple[int, ...] = DEFAULT_PROVIDER_BATCH_CANDIDATES,
    run_target_extension: bool = False,
) -> VpiEmbeddingQualificationReport:
    warnings: list[str] = []
    resources_touched: list[str] = [str(artifact_dir)]

    embedding_configuration = load_vpi_embedding_configuration()
    execution_configuration = load_vpi_embedding_provider_execution_configuration()
    materialization_config = load_vpi_embedding_materialization_config(
        max_records_override=record_target,
        artifact_dir_override=artifact_dir,
    )

    gpu_blocked = False
    try:
        assert_execution_device_available(execution_configuration)
    except VpiEmbeddingDeviceUnavailableError as exc:
        hardware = probe_hardware_runtime_capability(
            configured_device=execution_configuration.device,
        )
        return VpiEmbeddingQualificationReport(
            qualification_version=QUALIFICATION_VERSION,
            status=VpiEmbeddingQualificationStatus.BLOCKED_GPU,
            dataset_path=str(materialization_config.dataset_path),
            record_target=record_target,
            microbenchmark_record_count=microbenchmark_records,
            embedding_identity=EmbeddingIdentitySnapshot(
                provider=embedding_configuration.provider,
                model=embedding_configuration.model or "",
                dimension=embedding_configuration.expected_dimension,
                embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
                search_representation_derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
            ),
            hardware=hardware,
            execution_configuration=ExecutionConfigurationSnapshot(
                device=execution_configuration.device,
                outer_materialization_batch_size=materialization_config.embedding_batch_size,
                inner_provider_batch_size=execution_configuration.provider_batch_size,
                max_length=None,
                precision="provider_default",
            ),
            text_length_profile=profile_text_lengths(("blocked",)),
            warmup_timing=WarmupTimingSnapshot(0.0, 0.0, 0.0),
            microbenchmark_results=(),
            selected_provider_batch_size=None,
            selection_rationale=str(exc),
            materialization=None,
            materialization_restart=None,
            target_extension_executed=False,
            target_extension_detail=None,
            artifact_integrity=None,
            postgresql_result=StorageQualificationSnapshot(
                attempted=False,
                state=None,
                catalog_source_rows=None,
                search_point_count=None,
                elapsed_seconds=None,
                detail=str(exc),
            ),
            qdrant_result=StorageQualificationSnapshot(
                attempted=False,
                state=None,
                catalog_source_rows=None,
                search_point_count=None,
                elapsed_seconds=None,
                detail=str(exc),
            ),
            storage_bootstrap=StorageQualificationSnapshot(
                attempted=False,
                state=None,
                catalog_source_rows=None,
                search_point_count=None,
                elapsed_seconds=None,
                detail=str(exc),
            ),
            storage_restart=StorageQualificationSnapshot(
                attempted=False,
                state=None,
                catalog_source_rows=None,
                search_point_count=None,
                elapsed_seconds=None,
                detail=str(exc),
            ),
            zero_storage_embedding_proof="not_executed",
            full_build_estimate=None,
            bottleneck=None,
            warnings=(str(exc),),
            resources_touched=(str(artifact_dir),),
        )

    semantic_texts = sample_semantic_texts(
        materialization_config,
        record_count=microbenchmark_records,
    )
    text_length_profile = profile_text_lengths(semantic_texts)

    microbenchmark_results: list = []
    for candidate_batch_size in provider_batch_candidates:
        microbenchmark_results.append(
            run_provider_batch_candidate(
                embedding_configuration,
                semantic_texts,
                provider_batch_size=candidate_batch_size,
                device=execution_configuration.device,
                expected_dimension=embedding_configuration.expected_dimension,
            )
        )

    selected_batch_size, selection_rationale = select_best_provider_batch_size(
        tuple(microbenchmark_results),
        expected_dimension=embedding_configuration.expected_dimension,
    )
    if selected_batch_size is None:
        selected_batch_size = execution_configuration.provider_batch_size or 32
        selection_rationale = (
            "no passing microbenchmark candidate; "
            f"falling back to provider batch {selected_batch_size}"
        )
        warnings.append(selection_rationale)

    os.environ["VPI_EMBEDDING_PROVIDER_BATCH_SIZE"] = str(selected_batch_size)
    execution_configuration = load_vpi_embedding_provider_execution_configuration()

    execution_proof = resolve_provider_execution_proof(
        embedding_configuration,
        provider_batch_size=selected_batch_size,
        device=execution_configuration.device,
    )
    resolved_device = (
        execution_proof.snapshot.resolved_device
        if execution_proof.snapshot is not None
        else None
    )
    device_proof = (
        execution_proof.snapshot.evidence_source
        if execution_proof.snapshot is not None
        else execution_proof.reason or "unavailable"
    )
    reported_max_length = (
        execution_proof.snapshot.max_length
        if execution_proof.snapshot is not None
        else None
    )
    hardware = probe_hardware_runtime_capability(
        configured_device=execution_configuration.device,
        resolved_provider_device=resolved_device,
        provider_device_proof=device_proof,
    )
    if not hardware.cuda_available:
        gpu_blocked = True
        warnings.append(
            "CUDA unavailable in current torch build; GPU qualification blocked; "
            "using CPU steady-state throughput for estimates"
        )

    warmup_embedding = build_embedding_execution_port(
        embedding_configuration,
        provider_batch_size=selected_batch_size,
        device=execution_configuration.device,
    )
    try:
        warmup_timing = measure_warmup_timing(warmup_embedding, semantic_texts)
    finally:
        warmup_embedding.close()

    materialization_config = load_vpi_embedding_materialization_config(
        max_records_override=record_target,
        artifact_dir_override=artifact_dir,
    )
    orchestrator = build_vpi_embedding_materialization_runtime(
        materialization_config,
        artifact_dir=artifact_dir,
    )
    try:
        materialization_report = orchestrator.run()
    finally:
        orchestrator.dependencies.artifact_writer.close()
        orchestrator.dependencies.embedding.close()

    materialization_ok = (
        materialization_report.final_state is EmbeddingArtifactState.READY
        and materialization_report.manifest is not None
        and materialization_report.manifest.checkpoint_rows_materialized >= record_target
        and materialization_report.validation is not None
        and materialization_report.validation.status is ValidationStatus.PASS
    )
    artifact_integrity = None
    if materialization_report.validation is not None:
        artifact_integrity = ArtifactIntegritySnapshot(
            status=materialization_report.validation.status.value,
            detail=materialization_report.validation.checks[-1].detail
            if materialization_report.validation.checks
            else "",
        )

    restart_orchestrator = build_vpi_embedding_materialization_runtime(
        materialization_config,
        artifact_dir=artifact_dir,
    )
    try:
        restart_started = time.perf_counter()
        restart_report = restart_orchestrator.run()
        restart_elapsed = time.perf_counter() - restart_started
    finally:
        restart_orchestrator.dependencies.artifact_writer.close()
        restart_orchestrator.dependencies.embedding.close()

    restart_ok = (
        restart_report.final_state is EmbeddingArtifactState.READY
        and restart_report.embedding_calls == 0
    )
    materialization_restart = RestartQualificationSnapshot(
        state=restart_report.final_state.value,
        embedding_calls=restart_report.embedding_calls,
        elapsed_seconds=restart_elapsed,
    )

    target_extension_executed = False
    target_extension_detail: str | None = None
    if run_target_extension and materialization_ok:
        extension_target = record_target + 100
        extension_config = load_vpi_embedding_materialization_config(
            max_records_override=extension_target,
            artifact_dir_override=artifact_dir,
        )
        extension_orchestrator = build_vpi_embedding_materialization_runtime(
            extension_config,
            artifact_dir=artifact_dir,
        )
        try:
            extension_report = extension_orchestrator.run()
            target_extension_executed = True
            target_extension_detail = (
                f"extended target to {extension_target}; "
                f"embedding_calls={extension_report.embedding_calls}"
            )
        finally:
            extension_orchestrator.dependencies.artifact_writer.close()
            extension_orchestrator.dependencies.embedding.close()

    postgresql_result = StorageQualificationSnapshot(
        attempted=False,
        state=None,
        catalog_source_rows=None,
        search_point_count=None,
        elapsed_seconds=None,
        detail=None,
    )
    qdrant_result = StorageQualificationSnapshot(
        attempted=False,
        state=None,
        catalog_source_rows=None,
        search_point_count=None,
        elapsed_seconds=None,
        detail=None,
    )
    storage_bootstrap = StorageQualificationSnapshot(
        attempted=False,
        state=None,
        catalog_source_rows=None,
        search_point_count=None,
        elapsed_seconds=None,
        detail=None,
    )
    storage_restart = StorageQualificationSnapshot(
        attempted=False,
        state=None,
        catalog_source_rows=None,
        search_point_count=None,
        elapsed_seconds=None,
        detail=None,
    )
    zero_storage_embedding_proof = "not_executed"

    storage_ok = False
    if storage_environment_available():
        from platform_proofs.scenarios.verified_product_identification.composition.bootstrap_runtime import (
            build_vpi_bootstrap_runtime,
        )
        from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.config import (
            load_vpi_bootstrap_config,
        )

        bootstrap_config = load_vpi_bootstrap_config(
            mode=BootstrapRunMode.VERIFY,
            max_records_override=record_target,
        )
        resources_touched.extend(
            [
                f"postgresql_schema={bootstrap_config.postgresql_schema}",
                f"qdrant_collection={bootstrap_config.qdrant_collection_name}",
            ]
        )
        bootstrap_orchestrator = build_vpi_bootstrap_runtime(bootstrap_config)
        try:
            storage_started = time.perf_counter()
            bootstrap_report = bootstrap_orchestrator.run()
            storage_elapsed = time.perf_counter() - storage_started
        finally:
            bootstrap_orchestrator.dependencies.catalog.close()
            bootstrap_orchestrator.dependencies.search.close()
            bootstrap_orchestrator.dependencies.embedding_artifact.close()

        manifest = bootstrap_report.manifest
        storage_bootstrap = StorageQualificationSnapshot(
            attempted=True,
            state=bootstrap_report.final_state.value,
            catalog_source_rows=manifest.catalog_source_offer_count if manifest else None,
            search_point_count=manifest.search_point_count if manifest else None,
            elapsed_seconds=storage_elapsed,
            detail=bootstrap_report.failure_detail,
        )
        postgresql_result = replace(
            storage_bootstrap,
            detail="catalog bootstrap via PostgreSQLCatalogBootstrapAdapter",
        )
        qdrant_result = replace(
            storage_bootstrap,
            detail="search bootstrap via PlatformSearchIndexBootstrapAdapter",
        )
        storage_ok = bootstrap_report.final_state is BootstrapState.READY

        restart_bootstrap = build_vpi_bootstrap_runtime(bootstrap_config)
        try:
            restart_storage_started = time.perf_counter()
            restart_bootstrap_report = restart_bootstrap.run()
            restart_storage_elapsed = time.perf_counter() - restart_storage_started
        finally:
            restart_bootstrap.dependencies.catalog.close()
            restart_bootstrap.dependencies.search.close()
            restart_bootstrap.dependencies.embedding_artifact.close()
        storage_restart = StorageQualificationSnapshot(
            attempted=True,
            state=restart_bootstrap_report.final_state.value,
            catalog_source_rows=(
                restart_bootstrap_report.manifest.catalog_source_offer_count
                if restart_bootstrap_report.manifest
                else None
            ),
            search_point_count=(
                restart_bootstrap_report.manifest.search_point_count
                if restart_bootstrap_report.manifest
                else None
            ),
            elapsed_seconds=restart_storage_elapsed,
            detail=restart_bootstrap_report.failure_detail,
        )
        zero_storage_embedding_proof = (
            "storage bootstrap orchestrator has no EmbeddingExecutionPort; "
            "artifact-only ingest path"
        )
    else:
        gap = storage_environment_gap()
        detail = gap or "storage environment unavailable"
        warnings.append(detail)
        blocked_snapshot = StorageQualificationSnapshot(
            attempted=False,
            state=None,
            catalog_source_rows=None,
            search_point_count=None,
            elapsed_seconds=None,
            detail=detail,
        )
        postgresql_result = blocked_snapshot
        qdrant_result = blocked_snapshot
        storage_bootstrap = blocked_snapshot
        storage_restart = blocked_snapshot

    explicit_cuda = (
        execution_configuration.device is not None
        and execution_configuration.device.strip().casefold() == "cuda"
    )
    status = _classify_status(
        materialization_ok=materialization_ok,
        restart_ok=restart_ok,
        storage_ok=storage_ok,
        cuda_available=hardware.cuda_available,
        explicit_cuda_requested=explicit_cuda,
    )
    if gpu_blocked and status is VpiEmbeddingQualificationStatus.PASS:
        status = VpiEmbeddingQualificationStatus.PARTIAL_PASS_GPU

    full_build_estimate: FullBuildEstimate | None = None
    bottleneck = None
    if materialization_ok and materialization_report.rows_materialized > 0:
        steady_rps = materialization_report.embedding_records_per_second
        if steady_rps > 0:
            per_record_derive = (
                materialization_report.elapsed_derive_seconds
                / materialization_report.rows_materialized
            )
            per_record_write = (
                materialization_report.elapsed_artifact_write_seconds
                / materialization_report.rows_materialized
            )
            throughput_source = (
                "1k_materialization_gpu"
                if hardware.cuda_available
                else "1k_materialization_cpu"
            )
            full_build_estimate = estimate_full_build_duration(
                record_count=FULL_DATASET_RECORD_COUNT,
                steady_records_per_second=steady_rps,
                derive_seconds_per_record=per_record_derive,
                artifact_write_seconds_per_record=per_record_write,
                throughput_source=throughput_source,
            )
        bottleneck = analyze_bottleneck(
            derive_seconds=materialization_report.elapsed_derive_seconds,
            embedding_seconds=materialization_report.elapsed_embedding_seconds,
            artifact_write_seconds=materialization_report.elapsed_artifact_write_seconds,
        )

    model = embedding_configuration.model
    if model is None:
        msg = "embedding model is required"
        raise RuntimeError(msg)

    return VpiEmbeddingQualificationReport(
        qualification_version=QUALIFICATION_VERSION,
        status=status,
        dataset_path=str(materialization_config.dataset_path),
        record_target=record_target,
        microbenchmark_record_count=microbenchmark_records,
        embedding_identity=EmbeddingIdentitySnapshot(
            provider=embedding_configuration.provider,
            model=model,
            dimension=embedding_configuration.expected_dimension,
            embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
            search_representation_derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        ),
        hardware=hardware,
        execution_configuration=ExecutionConfigurationSnapshot(
            device=execution_configuration.device,
            outer_materialization_batch_size=materialization_config.embedding_batch_size,
            inner_provider_batch_size=selected_batch_size,
            max_length=reported_max_length,
            precision="provider_default",
        ),
        text_length_profile=text_length_profile,
        warmup_timing=warmup_timing,
        microbenchmark_results=tuple(microbenchmark_results),
        selected_provider_batch_size=selected_batch_size,
        selection_rationale=selection_rationale,
        materialization=_materialization_snapshot(materialization_report)
        if materialization_report
        else None,
        materialization_restart=materialization_restart,
        target_extension_executed=target_extension_executed,
        target_extension_detail=target_extension_detail,
        artifact_integrity=artifact_integrity,
        postgresql_result=postgresql_result,
        qdrant_result=qdrant_result,
        storage_bootstrap=storage_bootstrap,
        storage_restart=storage_restart,
        zero_storage_embedding_proof=zero_storage_embedding_proof,
        full_build_estimate=full_build_estimate,
        bottleneck=bottleneck,
        warnings=tuple(warnings),
        resources_touched=tuple(resources_touched),
    )
