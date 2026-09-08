"""Unit tests for VPI Data Pack performance profiling layer."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.performance.contracts import (
    PipelinePhase,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.performance.metrics import (
    build_performance_report,
    build_shard_performance_metrics,
    utc_now,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.performance.profiler import (
    PipelineProfiler,
    create_pipeline_profiler,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.performance.sinks import (
    write_performance_evidence,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder import (
    DataPackBuildConfig,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    resolve_data_pack_paths,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_resumable_builder_test_support import (
    FakeDataPackEmbeddingPort,
    patch_canonical_model_identity,
    run_resumable_data_pack_build_with_fake_policy,
    write_tiny_selected_dataset,
)

pytestmark = pytest.mark.unit


def test_pipeline_profiler_accumulates_exclusive_phase_timings() -> None:
    profiler = PipelineProfiler(enabled=True)
    with profiler.measure(PipelinePhase.READ):
        time.sleep(0.01)
    with profiler.measure(PipelinePhase.DERIVE):
        time.sleep(0.02)
    assert profiler.seconds(PipelinePhase.READ) >= 0.01
    assert profiler.seconds(PipelinePhase.DERIVE) >= 0.02


def test_noop_profiler_has_zero_overhead_timings() -> None:
    profiler = create_pipeline_profiler(enabled=False)
    with profiler.measure(PipelinePhase.EMBEDDING_INFERENCE):
        time.sleep(0.01)
    assert profiler.seconds(PipelinePhase.EMBEDDING_INFERENCE) == 0.0


def test_build_shard_metrics_computes_throughput() -> None:
    profiler = PipelineProfiler(enabled=True)
    with profiler.measure(PipelinePhase.TOTAL):
        with profiler.measure(PipelinePhase.EMBEDDING_INFERENCE):
            time.sleep(0.05)
    profiler.increment("embedding_calls", 4)
    profiler.increment("embedding_records", 64)
    started = utc_now()
    completed = utc_now()
    metrics = build_shard_performance_metrics(
        profiler,
        shard_ordinal=1,
        record_count=64,
        model_id="BAAI/bge-m3",
        device="cuda",
        batch_size=16,
        started_at=started,
        completed_at=completed,
    )
    assert metrics.embedding_calls == 4
    assert metrics.embedding_records == 64
    assert metrics.embedding_records_per_second > 0.0
    assert metrics.total_seconds >= 0.05


def test_write_performance_evidence_creates_json_and_summary(tmp_path: Path) -> None:
    profiler = PipelineProfiler(enabled=True)
    with profiler.measure(PipelinePhase.VALIDATION):
        time.sleep(0.01)
    with profiler.measure(PipelinePhase.TOTAL):
        pass
    metrics = build_shard_performance_metrics(
        profiler,
        shard_ordinal=1,
        record_count=10,
        model_id="BAAI/bge-m3",
        device="cpu",
        batch_size=16,
        started_at=utc_now(),
        completed_at=utc_now(),
    )
    report = build_performance_report((metrics,), qualification_id="test-qualification")
    json_path, summary_path = write_performance_evidence(tmp_path, report)
    assert json_path.is_file()
    assert summary_path.is_file()
    assert "dominant_phase" in json_path.read_text(encoding="utf-8")
    assert "Timing breakdown" in summary_path.read_text(encoding="utf-8")


def test_resumable_build_with_performance_profile_emits_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=50)
    output_root = tmp_path / "pack"
    evidence_dir = tmp_path / "performance"
    report = run_resumable_data_pack_build_with_fake_policy(
        DataPackBuildConfig(
            output_root=output_root,
            dataset_path=dataset_path,
            dataset_manifest_path=manifest_path,
            shard_size=25,
            max_records=50,
            start_fresh=True,
            enable_performance_profile=True,
            performance_output_dir=evidence_dir,
            performance_qualification_id="unit-performance",
        ),
        embedding_port=FakeDataPackEmbeddingPort(),
    )
    assert report.performance_report is not None
    assert len(report.performance_report.shard_metrics) == 2
    assert (evidence_dir / "performance-report.json").is_file()
    assert (evidence_dir / "PERFORMANCE_SUMMARY.md").is_file()
    assert report.finalized is True
    assert report.status is DataPackStatus.READY


def test_performance_profile_disabled_preserves_default_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=10)
    output_root = tmp_path / "pack"
    report = run_resumable_data_pack_build_with_fake_policy(
        DataPackBuildConfig(
            output_root=output_root,
            dataset_path=dataset_path,
            dataset_manifest_path=manifest_path,
            shard_size=10,
            max_records=10,
            start_fresh=True,
            enable_performance_profile=False,
        ),
        embedding_port=FakeDataPackEmbeddingPort(),
    )
    assert report.performance_report is None
    assert report.finalized is True
