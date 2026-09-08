"""Embedding diagnostic execution and controlled experiment orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from intergrax.rag.embedding.registry.execution_config import EmbeddingProviderExecutionConfig

from platform_proofs.scenarios.verified_product_identification.application.catalog.derive_search_representation import (
    build_source_record_ref,
    derive_search_representation,
    derive_search_representation_with_policy,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    load_vpi_embedding_configuration,
    validate_resolved_provider_dimension,
)
from platform_proofs.scenarios.verified_product_identification.application.config.embedding_execution_configuration import (
    VpiEmbeddingProviderExecutionConfiguration,
    assert_execution_device_available,
    load_vpi_embedding_provider_execution_configuration,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.data_package.identity import (
    CANONICAL_CATALOG_ID,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.contracts import (
    BatchLatencyMeasurement,
    DiagnosticExperimentKind,
    EmbeddingDiagnosticReport,
    EmbeddingExperimentResult,
    RecordRepresentationMeasurement,
    TokenDistributionReport,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.metrics import (
    batch_size_for_experiment,
    build_embedding_performance_metrics,
    build_token_distribution_report,
    classify_embedding_bottleneck,
    experiment_kind_for_batch_size,
    resolve_requested_experiments,
    validate_record_limit,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.ports import (
    GpuTelemetryPort,
    TokenCounterPort,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.telemetry import (
    create_gpu_telemetry,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.sample_selection import (
    SelectedDatasetRow,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_execution_profiles import (
    PRODUCTION_LOCAL_GPU_PROFILE_ID,
    apply_data_pack_build_execution_profile,
    resolve_data_pack_build_execution_profile,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.ports import (
    DataPackEmbeddingPort,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.integration.selected_dataset_reader import (
    SelectedDatasetShardReaderPort,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.orchestration.embedding_batches import (
    iter_embedding_slices,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.bootstrap import (
    ensure_embedding_provider_integrations_registered,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.model_identity import (
    resolve_embedding_model_identity,
)


@dataclass(slots=True)
class _BatchProfiler:
    batch_latencies: list[BatchLatencyMeasurement]

    def record_batch(
        self,
        *,
        batch_index: int,
        batch_size: int,
        record_count: int,
        input_tokens: int,
        batch_latency_seconds: float,
        inference_latency_seconds: float,
    ) -> BatchLatencyMeasurement:
        measurement = BatchLatencyMeasurement(
            batch_index=batch_index,
            batch_size=batch_size,
            record_count=record_count,
            input_tokens=input_tokens,
            tokens_processed=input_tokens,
            batch_latency_seconds=batch_latency_seconds,
            inference_latency_seconds=inference_latency_seconds,
        )
        self.batch_latencies.append(measurement)
        return measurement


class ProviderTokenCounter:
    """Adapter that isolates provider tokenizer failures behind a narrow port."""

    def __init__(self, count_tokens: Callable[[str], int]) -> None:
        self._count_tokens = count_tokens

    def count_tokens(self, text: str) -> int:
        return self._count_tokens(text)


def resolve_token_counter(embedding_port: DataPackEmbeddingPort) -> TokenCounterPort | None:
    """Best-effort tokenizer resolution for provider-backed embedding ports."""
    if not isinstance(embedding_port, IntergraxEmbeddingBootstrapAdapter):
        return None
    provider = getattr(embedding_port, "_provider", None)
    if provider is None:
        return None

    def count_tokens(text: str) -> int:
        ensure_model = getattr(provider, "_ensure_model", None)
        if callable(ensure_model):
            ensure_model()
        model = getattr(provider, "_model", None)
        if model is None:
            msg = "embedding provider model is unavailable for token counting"
            raise RuntimeError(msg)
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            msg = "embedding provider tokenizer is unavailable for token counting"
            raise RuntimeError(msg)
        return len(tokenizer.encode(text, add_special_tokens=True))

    return ProviderTokenCounter(count_tokens=count_tokens)


def load_diagnostic_dataset_sample(
    dataset_path: Path,
    *,
    record_limit: int,
) -> tuple[SelectedDatasetRow, ...]:
    validated_limit = validate_record_limit(record_limit)
    reader = SelectedDatasetShardReaderPort(dataset_path)
    return tuple(reader.read_range(0, validated_limit))


def derive_semantic_texts(
    rows: Sequence[SelectedDatasetRow],
    *,
    catalog_id: str = CANONICAL_CATALOG_ID,
    source_revision: str | None = None,
) -> tuple[str, ...]:
    semantic_texts: list[str] = []
    for row in rows:
        source_offer = parse_wdc_source_offer_json(row.record_json)
        source_ref = build_source_record_ref(
            source_offer,
            catalog_id=catalog_id,
            source_revision=source_revision,
        )
        representation = derive_search_representation(source_offer, source_ref=source_ref)
        semantic_texts.append(representation.semantic.semantic_text)
    return tuple(semantic_texts)


def derive_bounded_semantic_texts(
    rows: Sequence[SelectedDatasetRow],
    *,
    catalog_id: str = CANONICAL_CATALOG_ID,
    source_revision: str | None = None,
    representation_policy_profile: str,
) -> tuple[str, ...]:
    semantic_texts: list[str] = []
    for row in rows:
        source_offer = parse_wdc_source_offer_json(row.record_json)
        source_ref = build_source_record_ref(
            source_offer,
            catalog_id=catalog_id,
            source_revision=source_revision,
        )
        representation = derive_search_representation_with_policy(
            source_offer,
            source_ref=source_ref,
            representation_policy_profile=representation_policy_profile,
        )
        semantic_texts.append(representation.semantic.semantic_text)
    return tuple(semantic_texts)


def derive_record_identifiers(
    rows: Sequence[SelectedDatasetRow],
    *,
    catalog_id: str = CANONICAL_CATALOG_ID,
    source_revision: str | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    record_ids: list[str] = []
    source_refs: list[str] = []
    for row in rows:
        source_offer = parse_wdc_source_offer_json(row.record_json)
        source_ref = build_source_record_ref(
            source_offer,
            catalog_id=catalog_id,
            source_revision=source_revision,
        )
        record_ids.append(row.offer_id)
        source_refs.append(
            f"{source_ref.catalog_id}:{source_ref.offer_id.value}"
            + (
                f":{source_ref.source_revision}"
                if source_ref.source_revision is not None
                else ""
            )
        )
    return tuple(record_ids), tuple(source_refs)


def measure_record_representations(
    rows: Sequence[SelectedDatasetRow],
    semantic_texts: Sequence[str],
    token_counter: TokenCounterPort,
) -> tuple[RecordRepresentationMeasurement, ...]:
    if len(rows) != len(semantic_texts):
        msg = "rows and semantic_texts length mismatch"
        raise ValueError(msg)
    measurements: list[RecordRepresentationMeasurement] = []
    for row, semantic_text in zip(rows, semantic_texts, strict=True):
        try:
            token_count = token_counter.count_tokens(semantic_text)
        except Exception as exc:
            msg = (
                f"token counting failed for row {row.global_row_index} "
                f"(offer_id={row.offer_id})"
            )
            raise RuntimeError(msg) from exc
        measurements.append(
            RecordRepresentationMeasurement(
                global_row_index=row.global_row_index,
                semantic_text_length=len(semantic_text),
                character_count=len(semantic_text),
                token_count=token_count,
            )
        )
    return tuple(measurements)


def create_diagnostic_embedding_port(batch_size: int) -> IntergraxEmbeddingBootstrapAdapter:
    ensure_embedding_provider_integrations_registered()
    embedding_configuration = load_vpi_embedding_configuration()
    base_execution = load_vpi_embedding_provider_execution_configuration()
    execution_configuration = VpiEmbeddingProviderExecutionConfiguration(
        execution=EmbeddingProviderExecutionConfig(
            device=base_execution.device,
            batch_size=batch_size,
        ),
    )
    assert_execution_device_available(execution_configuration)
    adapter = IntergraxEmbeddingBootstrapAdapter(
        embedding_configuration,
        execution_configuration=execution_configuration,
    )
    probe = adapter.probe()
    validate_resolved_provider_dimension(
        configuration=embedding_configuration,
        resolved_dimension=probe.resolved_dimension,
    )
    return adapter


def run_embedding_experiment(
    *,
    semantic_texts: Sequence[str],
    token_distribution: TokenDistributionReport,
    embedding_port: DataPackEmbeddingPort,
    batch_size: int,
    experiment_kind: DiagnosticExperimentKind,
    model_id: str,
    model_revision: str,
    provider: str,
    device: str,
    gpu_telemetry: GpuTelemetryPort,
    per_record_tokens: Sequence[int] | None = None,
) -> EmbeddingExperimentResult:
    gpu_telemetry.reset_peak_memory()
    profiler = _BatchProfiler(batch_latencies=[])
    started = time.perf_counter()
    batch_index = 0
    for start_index, batch in iter_embedding_slices(semantic_texts, batch_size=batch_size):
        batch_texts = list(batch)
        batch_started = time.perf_counter()
        inference_started = time.perf_counter()
        embedding_port.embed_batch(batch_texts)
        inference_seconds = time.perf_counter() - inference_started
        batch_seconds = time.perf_counter() - batch_started
        input_tokens = _sum_batch_tokens(
            per_record_tokens,
            start_index=start_index,
            record_count=len(batch_texts),
        )
        profiler.record_batch(
            batch_index=batch_index,
            batch_size=batch_size,
            record_count=len(batch_texts),
            input_tokens=input_tokens,
            batch_latency_seconds=batch_seconds,
            inference_latency_seconds=inference_seconds,
        )
        gpu_telemetry.sample_utilization_percent()
        batch_index += 1
    embedding_seconds = time.perf_counter() - started
    cuda_environment = gpu_telemetry.environment()
    metrics = build_embedding_performance_metrics(
        model_id=model_id,
        model_revision=model_revision,
        provider=provider,
        device=device,
        token_distribution=token_distribution,
        batch_size=batch_size,
        batches_count=len(profiler.batch_latencies),
        embedding_seconds=embedding_seconds,
        gpu_name=cuda_environment.gpu_name,
        cuda_available=cuda_environment.cuda_available,
        peak_memory_mb=gpu_telemetry.peak_memory_mb(),
        average_gpu_utilization_percent=gpu_telemetry.average_utilization_percent(),
    )
    return EmbeddingExperimentResult(
        experiment_kind=experiment_kind,
        batch_size=batch_size,
        metrics=metrics,
        batch_latencies=tuple(profiler.batch_latencies),
    )


def run_embedding_diagnostics(
    *,
    dataset_path: Path,
    record_limit: int,
    qualification_id: str,
    production_batch_size: int | None = None,
    experiment: str | None = None,
    gpu_telemetry: GpuTelemetryPort | None = None,
) -> EmbeddingDiagnosticReport:
    validated_limit = validate_record_limit(record_limit)
    requested_experiments = resolve_requested_experiments(experiment)
    representation_only = (
        requested_experiments == (DiagnosticExperimentKind.REPRESENTATION_ONLY,)
    )
    ensure_embedding_provider_integrations_registered()
    profile = resolve_data_pack_build_execution_profile(PRODUCTION_LOCAL_GPU_PROFILE_ID)
    apply_data_pack_build_execution_profile(profile)
    baseline_batch_size = production_batch_size or profile.provider_batch_size
    rows = load_diagnostic_dataset_sample(dataset_path, record_limit=validated_limit)
    semantic_texts = derive_semantic_texts(rows)
    record_ids, source_refs = derive_record_identifiers(rows)
    embedding_configuration = load_vpi_embedding_configuration()
    model = embedding_configuration.model
    if model is None:
        msg = "embedding model is required for diagnostics"
        raise RuntimeError(msg)
    model_identity = resolve_embedding_model_identity(embedding_configuration.provider, model)
    token_counter_port = create_diagnostic_embedding_port(baseline_batch_size)
    token_counter = resolve_token_counter(token_counter_port)
    if token_counter is None:
        msg = "token counter is unavailable for the configured embedding provider"
        raise RuntimeError(msg)
    measurements = measure_record_representations(rows, semantic_texts, token_counter)
    token_distribution = build_token_distribution_report(
        measurements,
        record_ids=record_ids,
        source_refs=source_refs,
    )
    token_counter_port.close()
    per_record_tokens = tuple(measurement.token_count for measurement in measurements)

    execution_configuration = load_vpi_embedding_provider_execution_configuration()
    resolved_device = execution_configuration.device or "cpu"
    telemetry = gpu_telemetry or create_gpu_telemetry()

    if representation_only:
        baseline_metrics = build_embedding_performance_metrics(
            model_id=model,
            model_revision=model_identity.revision,
            provider=embedding_configuration.provider,
            device=resolved_device,
            token_distribution=token_distribution,
            batch_size=baseline_batch_size,
            batches_count=0,
            embedding_seconds=0.0,
            gpu_name=telemetry.environment().gpu_name,
            cuda_available=telemetry.environment().cuda_available,
            peak_memory_mb=None,
            average_gpu_utilization_percent=None,
        )
        classification = classify_embedding_bottleneck(
            token_distribution=token_distribution,
            baseline=baseline_metrics,
            batch_experiments=(),
        )
        return EmbeddingDiagnosticReport(
            qualification_id=qualification_id,
            record_limit=validated_limit,
            token_distribution=token_distribution,
            baseline=baseline_metrics,
            batch_experiments=(),
            classification=classification,
        )

    batch_results: list[EmbeddingExperimentResult] = []
    baseline_result: EmbeddingExperimentResult | None = None
    for experiment_kind in requested_experiments:
        batch_size = batch_size_for_experiment(
            experiment_kind,
            production_batch_size=baseline_batch_size,
        )
        experiment_port = create_diagnostic_embedding_port(batch_size)
        try:
            result = run_embedding_experiment(
                semantic_texts=semantic_texts,
                token_distribution=token_distribution,
                embedding_port=experiment_port,
                batch_size=batch_size,
                experiment_kind=experiment_kind,
                model_id=model,
                model_revision=model_identity.revision,
                provider=embedding_configuration.provider,
                device=resolved_device,
                gpu_telemetry=telemetry,
                per_record_tokens=per_record_tokens,
            )
        finally:
            experiment_port.close()
        if experiment_kind is DiagnosticExperimentKind.PRODUCTION_BASELINE:
            baseline_result = result
        batch_results.append(result)

    if baseline_result is None:
        baseline_result = next(
            (
                result
                for result in batch_results
                if result.experiment_kind is DiagnosticExperimentKind.PRODUCTION_BASELINE
            ),
            batch_results[0],
        )

    classification = classify_embedding_bottleneck(
        token_distribution=token_distribution,
        baseline=baseline_result.metrics,
        batch_experiments=tuple(result.metrics for result in batch_results),
    )
    return EmbeddingDiagnosticReport(
        qualification_id=qualification_id,
        record_limit=validated_limit,
        token_distribution=token_distribution,
        baseline=baseline_result.metrics,
        batch_experiments=tuple(batch_results),
        classification=classification,
    )


def _sum_batch_tokens(
    per_record_tokens: Sequence[int] | None,
    *,
    start_index: int,
    record_count: int,
) -> int:
    if per_record_tokens is None:
        return 0
    end_index = start_index + record_count
    return sum(per_record_tokens[start_index:end_index])
