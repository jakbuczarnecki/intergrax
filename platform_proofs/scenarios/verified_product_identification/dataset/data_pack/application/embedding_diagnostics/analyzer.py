"""Embedding diagnostic execution and controlled experiment orchestration."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from intergrax.rag.embedding.registry.execution_config import EmbeddingProviderExecutionConfig

from platform_proofs.scenarios.verified_product_identification.application.catalog.derive_search_representation import (
    build_source_record_ref,
    derive_search_representation,
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
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.build_performance import (
    DataPackBuildPerformanceMonitor,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.contracts import (
    BatchExperimentResult,
    BatchLatencyMeasurement,
    CudaEnvironmentSnapshot,
    EmbeddingDiagnosticReport,
    RecordRepresentationMeasurement,
    TokenDistributionReport,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.metrics import (
    batch_sizes_for_experiments,
    build_embedding_performance_metrics,
    build_token_distribution_report,
    classify_embedding_bottleneck,
    validate_record_limit,
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


class TokenCounterPort(Protocol):
    def count_tokens(self, text: str) -> int: ...


@dataclass(slots=True)
class _GpuUtilizationSampler:
    samples: list[float]

    def record(self) -> None:
        utilization = _query_gpu_utilization_percent()
        if utilization is not None:
            self.samples.append(utilization)

    def average(self) -> float | None:
        if not self.samples:
            return None
        return sum(self.samples) / len(self.samples)


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

    return _FunctionTokenCounter(count_tokens=count_tokens)


class _FunctionTokenCounter:
    def __init__(self, count_tokens: Callable[[str], int]) -> None:
        self._count_tokens = count_tokens

    def count_tokens(self, text: str) -> int:
        return self._count_tokens(text)


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
        measurements.append(
            RecordRepresentationMeasurement(
                global_row_index=row.global_row_index,
                semantic_text_length=len(semantic_text),
                character_count=len(semantic_text),
                token_count=token_counter.count_tokens(semantic_text),
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
    model_id: str,
    model_revision: str,
    provider: str,
    device: str,
    cuda_environment: CudaEnvironmentSnapshot,
) -> BatchExperimentResult:
    _reset_cuda_peak_memory()
    performance_monitor = DataPackBuildPerformanceMonitor()
    gpu_sampler = _GpuUtilizationSampler(samples=[])
    batch_latencies: list[BatchLatencyMeasurement] = []
    started = time.perf_counter()
    batch_index = 0
    for _start, batch in iter_embedding_slices(semantic_texts, batch_size=batch_size):
        batch_started = time.perf_counter()
        inference_started = time.perf_counter()
        embedding_port.embed_batch(list(batch))
        inference_seconds = time.perf_counter() - inference_started
        batch_seconds = time.perf_counter() - batch_started
        performance_monitor.sample()
        gpu_sampler.record()
        batch_latencies.append(
            BatchLatencyMeasurement(
                batch_index=batch_index,
                batch_size=batch_size,
                record_count=len(batch),
                batch_latency_seconds=batch_seconds,
                inference_latency_seconds=inference_seconds,
            )
        )
        batch_index += 1
    embedding_seconds = time.perf_counter() - started
    snapshot = performance_monitor.snapshot()
    metrics = build_embedding_performance_metrics(
        model_id=model_id,
        model_revision=model_revision,
        provider=provider,
        device=device,
        token_distribution=token_distribution,
        batch_size=batch_size,
        batches_count=len(batch_latencies),
        embedding_seconds=embedding_seconds,
        gpu_name=cuda_environment.gpu_name,
        cuda_available=cuda_environment.cuda_available,
        peak_memory_mb=snapshot.peak_vram_mb,
        average_gpu_utilization_percent=gpu_sampler.average(),
    )
    return BatchExperimentResult(
        batch_size=batch_size,
        metrics=metrics,
        batch_latencies=tuple(batch_latencies),
    )


def run_embedding_diagnostics(
    *,
    dataset_path: Path,
    record_limit: int,
    qualification_id: str,
    production_batch_size: int | None = None,
) -> EmbeddingDiagnosticReport:
    validated_limit = validate_record_limit(record_limit)
    ensure_embedding_provider_integrations_registered()
    profile = resolve_data_pack_build_execution_profile(PRODUCTION_LOCAL_GPU_PROFILE_ID)
    apply_data_pack_build_execution_profile(profile)
    baseline_batch_size = production_batch_size or profile.provider_batch_size
    rows = load_diagnostic_dataset_sample(dataset_path, record_limit=validated_limit)
    semantic_texts = derive_semantic_texts(rows)
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
    token_distribution = build_token_distribution_report(measurements)
    token_counter_port.close()

    cuda_environment = _resolve_cuda_environment()
    execution_configuration = load_vpi_embedding_provider_execution_configuration()
    resolved_device = execution_configuration.device or "cpu"

    baseline_port = create_diagnostic_embedding_port(baseline_batch_size)
    baseline_result = run_embedding_experiment(
        semantic_texts=semantic_texts,
        token_distribution=token_distribution,
        embedding_port=baseline_port,
        batch_size=baseline_batch_size,
        model_id=model,
        model_revision=model_identity.revision,
        provider=embedding_configuration.provider,
        device=resolved_device,
        cuda_environment=cuda_environment,
    )
    baseline_port.close()

    batch_results: list[BatchExperimentResult] = []
    for batch_size in batch_sizes_for_experiments(baseline_batch_size):
        if batch_size == baseline_batch_size:
            batch_results.append(baseline_result)
            continue
        experiment_port = create_diagnostic_embedding_port(batch_size)
        try:
            batch_results.append(
                run_embedding_experiment(
                    semantic_texts=semantic_texts,
                    token_distribution=token_distribution,
                    embedding_port=experiment_port,
                    batch_size=batch_size,
                    model_id=model,
                    model_revision=model_identity.revision,
                    provider=embedding_configuration.provider,
                    device=resolved_device,
                    cuda_environment=cuda_environment,
                )
            )
        finally:
            experiment_port.close()

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


def _resolve_cuda_environment() -> CudaEnvironmentSnapshot:
    try:
        import torch
    except ImportError:
        return CudaEnvironmentSnapshot(cuda_available=False, gpu_name=None)
    if not torch.cuda.is_available():
        return CudaEnvironmentSnapshot(cuda_available=False, gpu_name=None)
    return CudaEnvironmentSnapshot(
        cuda_available=True,
        gpu_name=torch.cuda.get_device_name(0),
    )


def _reset_cuda_peak_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def _query_gpu_utilization_percent() -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    first_line = completed.stdout.strip().splitlines()
    if not first_line:
        return None
    try:
        return float(first_line[0].strip())
    except ValueError:
        return None
