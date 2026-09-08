"""Unit tests for VPI embedding diagnostics layer."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics import (
    CudaEnvironmentSnapshot,
    DiagnosticExperimentKind,
    EmbeddingBottleneckCase,
    EmbeddingDiagnosticReport,
    NoOpGpuTelemetry,
    RecordRepresentationMeasurement,
    TokenPercentileBucket,
    assign_token_percentile_bucket,
    build_diagnostic_report_json_document,
    build_embedding_performance_metrics,
    build_token_distribution_report,
    build_token_statistics,
    classify_embedding_bottleneck,
    derive_semantic_texts,
    load_diagnostic_dataset_sample,
    measure_record_representations,
    percentile,
    resolve_requested_experiments,
    run_embedding_experiment,
    serialize_diagnostic_report_json,
    validate_record_limit,
    write_embedding_diagnostic_report,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.contracts import (
    EmbeddingDiagnosticClassification,
    TokenDistributionReport,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_resumable_builder_test_support import (
    FakeDataPackEmbeddingPort,
    write_tiny_selected_dataset,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DIAGNOSTICS_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/dataset/data_pack/application/embedding_diagnostics"
)
_FORBIDDEN_CORE_MODULES = (
    _DIAGNOSTICS_ROOT / "contracts.py",
    _DIAGNOSTICS_ROOT / "metrics.py",
    _DIAGNOSTICS_ROOT / "ports.py",
    _DIAGNOSTICS_ROOT / "serialization.py",
    _DIAGNOSTICS_ROOT / "telemetry.py",
)


class _FixedTokenCounter:
    def __init__(self, token_count: int) -> None:
        self._token_count = token_count

    def count_tokens(self, text: str) -> int:
        return self._token_count + len(text) // 100


class _FailingTokenCounter:
    def count_tokens(self, text: str) -> int:
        raise RuntimeError("provider tokenizer unavailable")


def test_percentile_interpolates_between_values() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.50) == 2.5


def test_build_token_statistics_includes_required_percentiles() -> None:
    stats = build_token_statistics([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    assert stats.count == 10
    assert stats.minimum == 10
    assert stats.maximum == 100
    assert stats.p50 == 55.0
    assert stats.p90 == 91.0
    assert stats.p95 == 95.5
    assert stats.p99 == 99.1


def test_validate_record_limit_rejects_values_above_cap() -> None:
    with pytest.raises(ValueError, match="record_limit must be <= 100"):
        validate_record_limit(101)


def test_build_token_distribution_report_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="measurements must not be empty"):
        build_token_distribution_report(())


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
    assert report.statistics.maximum == 150
    assert len(report.diagnostic_samples) == 2


def test_large_token_sample_assigns_extreme_bucket() -> None:
    measurements = tuple(
        RecordRepresentationMeasurement(
            global_row_index=index,
            semantic_text_length=length,
            character_count=length,
            token_count=length,
        )
        for index, length in enumerate((100, 200, 300, 400, 5000))
    )
    report = build_token_distribution_report(measurements)
    extreme = report.diagnostic_samples[-1]
    assert extreme.token_percentile_bucket is TokenPercentileBucket.AT_OR_ABOVE_P99
    assert report.statistics.maximum == 5000


def test_measure_record_representations_isolates_provider_failure(tmp_path: Path) -> None:
    dataset_path, _manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=1)
    rows = load_diagnostic_dataset_sample(dataset_path, record_limit=1)
    semantic_texts = derive_semantic_texts(rows)
    with pytest.raises(RuntimeError, match="token counting failed"):
        measure_record_representations(rows, semantic_texts, _FailingTokenCounter())


def test_noop_gpu_telemetry_does_not_crash_without_cuda() -> None:
    telemetry = NoOpGpuTelemetry()
    environment = telemetry.environment()
    assert environment.cuda_available is False
    telemetry.reset_peak_memory()
    assert telemetry.peak_memory_mb() is None
    assert telemetry.sample_utilization_percent() is None
    assert telemetry.average_utilization_percent() is None


def test_classify_representation_case_when_token_p95_high() -> None:
    token_distribution = _token_distribution_with_p95(2500.0)
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
        experiment_kind=DiagnosticExperimentKind.PRODUCTION_BASELINE,
        model_id="BAAI/bge-m3",
        model_revision="rev",
        provider="hf",
        device="cpu",
        gpu_telemetry=NoOpGpuTelemetry(),
        per_record_tokens=tuple(measurement.token_count for measurement in measurements),
    )
    assert result.metrics.batches_count == 2
    assert result.metrics.records_per_second > 0.0
    assert len(result.batch_latencies) == 2
    assert result.batch_latencies[0].input_tokens > 0


def test_json_serialization_round_trip(tmp_path: Path) -> None:
    report = _sample_report()
    payload = serialize_diagnostic_report_json(report)
    parsed = json.loads(payload)
    assert parsed["qualification_id"] == "unit-diagnostics"
    assert parsed["token_distribution"]["statistics"]["p90"] == parsed["token_distribution"]["statistics"]["p90"]


def test_deterministic_json_document_for_fixed_report() -> None:
    report = _sample_report()
    first = serialize_diagnostic_report_json(report)
    second = serialize_diagnostic_report_json(report)
    assert first == second
    assert build_diagnostic_report_json_document(report).qualification_id == "unit-diagnostics"


def test_write_embedding_diagnostic_report_creates_files(tmp_path: Path) -> None:
    report = _sample_report()
    json_path, summary_path = write_embedding_diagnostic_report(tmp_path, report)
    assert json_path.is_file()
    assert summary_path.is_file()
    assert "classification" in json_path.read_text(encoding="utf-8")


def test_resolve_requested_experiments_supports_aliases() -> None:
    assert resolve_requested_experiments("D") == (
        DiagnosticExperimentKind.REPRESENTATION_ONLY,
    )
    assert resolve_requested_experiments("full") == (
        DiagnosticExperimentKind.PRODUCTION_BASELINE,
        DiagnosticExperimentKind.BATCH_32,
        DiagnosticExperimentKind.BATCH_64,
    )


def test_assign_token_percentile_bucket_respects_statistics() -> None:
    stats = build_token_statistics([10, 20, 30, 40, 50])
    assert (
        assign_token_percentile_bucket(15, statistics_snapshot=stats)
        is TokenPercentileBucket.BELOW_P50
    )


def test_core_modules_have_no_forbidden_contract_patterns() -> None:
    forbidden_fragments = (
        "dict[str, Any]",
        ": Any",
        "dict[str, object]",
        "Mapping[str, object]",
    )
    for module_path in _FORBIDDEN_CORE_MODULES:
        source = module_path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{fragment} found in {module_path.name}"


def test_core_modules_have_no_forbidden_imports() -> None:
    forbidden_roots = (
        "psycopg",
        "qdrant",
        "docker",
        "sentence_transformers",
    )
    for module_path in _FORBIDDEN_CORE_MODULES:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
        violations = sorted(
            name
            for name in imported
            if any(root in name for root in forbidden_roots)
        )
        assert violations == [], f"{module_path.name} imports forbidden roots: {violations}"


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


def _token_distribution_with_p95(p95_value: float) -> TokenDistributionReport:
    report = _reasonable_token_distribution()
    stats = report.statistics
    return TokenDistributionReport(
        record_count=report.record_count,
        total_tokens=report.total_tokens,
        statistics=type(stats)(
            count=stats.count,
            minimum=stats.minimum,
            mean=stats.mean,
            p50=stats.p50,
            p90=stats.p90,
            p95=p95_value,
            p99=stats.p99,
            maximum=stats.maximum,
        ),
        semantic_text_length_avg=report.semantic_text_length_avg,
        semantic_text_length_p95=report.semantic_text_length_p95,
        record_measurements=report.record_measurements,
        diagnostic_samples=report.diagnostic_samples,
    )


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


def _sample_report() -> EmbeddingDiagnosticReport:
    token_distribution = _reasonable_token_distribution()
    baseline = _metrics_for_classification(token_distribution, records_per_second=1.0)
    return EmbeddingDiagnosticReport(
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
