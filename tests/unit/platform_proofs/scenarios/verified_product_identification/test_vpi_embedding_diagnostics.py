"""Unit tests for VPI embedding diagnostics layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics import (
    CudaEnvironmentSnapshot,
    EmbeddingBottleneckCase,
    build_embedding_performance_metrics,
    build_token_distribution_report,
    classify_embedding_bottleneck,
    derive_semantic_texts,
    load_diagnostic_dataset_sample,
    measure_record_representations,
    percentile,
    run_embedding_experiment,
    validate_record_limit,
    write_embedding_diagnostic_report,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.contracts import (
    EmbeddingDiagnosticClassification,
    EmbeddingDiagnosticReport,
    RecordRepresentationMeasurement,
    TokenDistributionReport,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_resumable_builder_test_support import (
    FakeDataPackEmbeddingPort,
    write_tiny_selected_dataset,
)

pytestmark = pytest.mark.unit


class _FixedTokenCounter:
    def __init__(self, token_count: int) -> None:
        self._token_count = token_count

    def count_tokens(self, text: str) -> int:
        return self._token_count + len(text) // 100


def test_percentile_interpolates_between_values() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.50) == 2.5


def test_validate_record_limit_rejects_values_above_cap() -> None:
    with pytest.raises(ValueError, match="record_limit must be <= 100"):
        validate_record_limit(101)


def test_build_token_distribution_report_aggregates_measurements() -> None:
    measurements = (
        RecordRepresentationMeasurement(
            global_row_index=0,
            semantic_text_length=100,
            character_count=100,
            token_count=50,
        ),
        RecordRepresentationMeasurement(
            global_row_index=1,
            semantic_text_length=200,
            character_count=200,
            token_count=150,
        ),
    )
    report = build_token_distribution_report(measurements)
    assert report.record_count == 2
    assert report.total_tokens == 200
    assert report.max_tokens == 150


def test_classify_representation_case_when_token_p95_high() -> None:
    token_distribution = TokenDistributionReport(
        record_count=10,
        total_tokens=25000,
        average_tokens=2500.0,
        p50_tokens=2400.0,
        p95_tokens=2500.0,
        p99_tokens=2600.0,
        max_tokens=2700,
        semantic_text_length_avg=10000.0,
        semantic_text_length_p95=12000.0,
        record_measurements=(),
    )
    baseline = _metrics_for_classification(token_distribution, records_per_second=0.5)
    classification = classify_embedding_bottleneck(
        token_distribution=token_distribution,
        baseline=baseline,
        batch_experiments=(baseline,),
    )
    assert classification.case is EmbeddingBottleneckCase.REPRESENTATION_OPTIMIZATION


def test_classify_provider_case_when_tokens_reasonable_and_throughput_low() -> None:
    token_distribution = _reasonable_token_distribution()
    baseline = _metrics_for_classification(token_distribution, records_per_second=0.5)
    classification = classify_embedding_bottleneck(
        token_distribution=token_distribution,
        baseline=baseline,
        batch_experiments=(baseline,),
    )
    assert classification.case is EmbeddingBottleneckCase.PROVIDER_OPTIMIZATION


def test_classify_batch_tuning_case_when_throughput_scales_with_batch_size() -> None:
    token_distribution = _reasonable_token_distribution()
    baseline = _metrics_for_classification(token_distribution, records_per_second=1.0, batch_size=16)
    improved = _metrics_for_classification(token_distribution, records_per_second=2.0, batch_size=64)
    classification = classify_embedding_bottleneck(
        token_distribution=token_distribution,
        baseline=baseline,
        batch_experiments=(baseline, improved),
    )
    assert classification.case is EmbeddingBottleneckCase.BATCH_TUNING


def test_load_and_derive_semantic_texts_from_tiny_dataset(tmp_path: Path) -> None:
    dataset_path, _manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=5)
    rows = load_diagnostic_dataset_sample(dataset_path, record_limit=5)
    semantic_texts = derive_semantic_texts(rows)
    assert len(rows) == 5
    assert len(semantic_texts) == 5
    assert all(text.strip() for text in semantic_texts)


def test_measure_record_representations_counts_tokens(tmp_path: Path) -> None:
    dataset_path, _manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=3)
    rows = load_diagnostic_dataset_sample(dataset_path, record_limit=3)
    semantic_texts = derive_semantic_texts(rows)
    measurements = measure_record_representations(
        rows,
        semantic_texts,
        _FixedTokenCounter(token_count=10),
    )
    assert len(measurements) == 3
    assert all(measurement.token_count >= 10 for measurement in measurements)


def test_run_embedding_experiment_with_fake_port(tmp_path: Path) -> None:
    dataset_path, _manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=4)
    rows = load_diagnostic_dataset_sample(dataset_path, record_limit=4)
    semantic_texts = derive_semantic_texts(rows)
    measurements = measure_record_representations(
        rows,
        semantic_texts,
        _FixedTokenCounter(token_count=5),
    )
    token_distribution = build_token_distribution_report(measurements)
    result = run_embedding_experiment(
        semantic_texts=semantic_texts,
        token_distribution=token_distribution,
        embedding_port=FakeDataPackEmbeddingPort(),
        batch_size=2,
        model_id="BAAI/bge-m3",
        model_revision="rev",
        provider="hf",
        device="cpu",
        cuda_environment=CudaEnvironmentSnapshot(cuda_available=False, gpu_name=None),
    )
    assert result.metrics.batches_count == 2
    assert result.metrics.records_per_second > 0.0
    assert len(result.batch_latencies) == 2


def test_write_embedding_diagnostic_report_creates_files(tmp_path: Path) -> None:
    token_distribution = _reasonable_token_distribution()
    baseline = _metrics_for_classification(token_distribution, records_per_second=1.0)
    report = EmbeddingDiagnosticReport(
        qualification_id="unit-diagnostics",
        record_limit=10,
        token_distribution=token_distribution,
        baseline=baseline,
        batch_experiments=(),
        classification=EmbeddingDiagnosticClassification(
            case=EmbeddingBottleneckCase.UNDETERMINED,
            conclusion="unit test",
            recommended_next_task="none",
        ),
    )
    json_path, summary_path = write_embedding_diagnostic_report(tmp_path, report)
    assert json_path.is_file()
    assert summary_path.is_file()
    assert "classification" in json_path.read_text(encoding="utf-8")


def _reasonable_token_distribution() -> TokenDistributionReport:
    measurements = (
        RecordRepresentationMeasurement(
            global_row_index=0,
            semantic_text_length=400,
            character_count=400,
            token_count=120,
        ),
        RecordRepresentationMeasurement(
            global_row_index=1,
            semantic_text_length=450,
            character_count=450,
            token_count=140,
        ),
    )
    return build_token_distribution_report(measurements)


def _metrics_for_classification(
    token_distribution: TokenDistributionReport,
    *,
    records_per_second: float,
    batch_size: int = 16,
):
    return build_embedding_performance_metrics(
        model_id="BAAI/bge-m3",
        model_revision="rev",
        provider="hf",
        device="cpu",
        token_distribution=token_distribution,
        batch_size=batch_size,
        batches_count=1,
        embedding_seconds=token_distribution.record_count / records_per_second,
        gpu_name=None,
        cuda_available=False,
        peak_memory_mb=None,
        average_gpu_utilization_percent=None,
    )
