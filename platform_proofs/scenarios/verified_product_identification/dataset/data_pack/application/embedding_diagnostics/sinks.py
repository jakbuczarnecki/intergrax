"""Machine-readable and human-readable embedding diagnostic evidence sinks."""

from __future__ import annotations

import json
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.contracts import (
    BatchExperimentResult,
    BatchLatencyMeasurement,
    EmbeddingDiagnosticReport,
    EmbeddingPerformanceMetrics,
    RecordRepresentationMeasurement,
    TokenDistributionReport,
)


def write_embedding_diagnostic_report(
    output_dir: Path,
    report: EmbeddingDiagnosticReport,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "embedding-diagnostic-report.json"
    summary_path = output_dir / "EMBEDDING_DIAGNOSTIC_SUMMARY.md"
    json_path.write_text(
        json.dumps(_report_to_json(report), indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(_report_to_markdown(report), encoding="utf-8")
    return json_path, summary_path


def _report_to_json(report: EmbeddingDiagnosticReport) -> dict[str, object]:
    return {
        "qualification_id": report.qualification_id,
        "record_limit": report.record_limit,
        "token_distribution": _token_distribution_to_json(report.token_distribution),
        "baseline": _metrics_to_json(report.baseline),
        "batch_experiments": [
            {
                "batch_size": experiment.batch_size,
                "metrics": _metrics_to_json(experiment.metrics),
                "batch_latencies": [
                    _batch_latency_to_json(latency) for latency in experiment.batch_latencies
                ],
            }
            for experiment in report.batch_experiments
        ],
        "classification": {
            "case": report.classification.case.value,
            "conclusion": report.classification.conclusion,
            "recommended_next_task": report.classification.recommended_next_task,
        },
    }


def _token_distribution_to_json(report: TokenDistributionReport) -> dict[str, object]:
    return {
        "record_count": report.record_count,
        "total_tokens": report.total_tokens,
        "average_tokens": round(report.average_tokens, 3),
        "p50_tokens": round(report.p50_tokens, 3),
        "p95_tokens": round(report.p95_tokens, 3),
        "p99_tokens": round(report.p99_tokens, 3),
        "max_tokens": report.max_tokens,
        "semantic_text_length_avg": round(report.semantic_text_length_avg, 3),
        "semantic_text_length_p95": round(report.semantic_text_length_p95, 3),
        "records": [
            _record_measurement_to_json(measurement)
            for measurement in report.record_measurements
        ],
    }


def _metrics_to_json(metrics: EmbeddingPerformanceMetrics) -> dict[str, object]:
    return {
        "identity": {
            "model_id": metrics.model_id,
            "model_revision": metrics.model_revision,
            "provider": metrics.provider,
            "device": metrics.device,
        },
        "dataset": {
            "record_count": metrics.record_count,
            "semantic_text_length_avg": round(metrics.semantic_text_length_avg, 3),
            "semantic_text_length_p95": round(metrics.semantic_text_length_p95, 3),
        },
        "tokens": {
            "total_tokens": metrics.total_tokens,
            "average_tokens": round(metrics.average_tokens, 3),
            "p50_tokens": round(metrics.p50_tokens, 3),
            "p95_tokens": round(metrics.p95_tokens, 3),
            "p99_tokens": round(metrics.p99_tokens, 3),
            "max_tokens": metrics.max_tokens,
        },
        "execution": {
            "batch_size": metrics.batch_size,
            "batches_count": metrics.batches_count,
            "embedding_seconds": round(metrics.embedding_seconds, 3),
            "records_per_second": round(metrics.records_per_second, 3),
            "tokens_per_second": round(metrics.tokens_per_second, 3),
        },
        "hardware": {
            "gpu_name": metrics.gpu_name,
            "cuda_available": metrics.cuda_available,
            "peak_memory_mb": (
                round(metrics.peak_memory_mb, 3) if metrics.peak_memory_mb is not None else None
            ),
            "average_gpu_utilization_percent": (
                round(metrics.average_gpu_utilization_percent, 3)
                if metrics.average_gpu_utilization_percent is not None
                else None
            ),
        },
    }


def _record_measurement_to_json(
    measurement: RecordRepresentationMeasurement,
) -> dict[str, object]:
    return {
        "global_row_index": measurement.global_row_index,
        "semantic_text_length": measurement.semantic_text_length,
        "character_count": measurement.character_count,
        "token_count": measurement.token_count,
    }


def _batch_latency_to_json(latency: BatchLatencyMeasurement) -> dict[str, object]:
    return {
        "batch_index": latency.batch_index,
        "batch_size": latency.batch_size,
        "record_count": latency.record_count,
        "batch_latency_seconds": round(latency.batch_latency_seconds, 6),
        "inference_latency_seconds": round(latency.inference_latency_seconds, 6),
    }


def _report_to_markdown(report: EmbeddingDiagnosticReport) -> str:
    token_report = report.token_distribution
    baseline = report.baseline
    lines = [
        "# Embedding Diagnostic Summary",
        "",
        f"- Qualification ID: `{report.qualification_id}`",
        f"- Records analyzed: {report.record_limit}",
        f"- Bottleneck case: **{report.classification.case.value}**",
        f"- Conclusion: {report.classification.conclusion}",
        f"- Recommended next task: {report.classification.recommended_next_task}",
        "",
        "## Token distribution",
        "",
        f"- Total tokens: {token_report.total_tokens}",
        f"- Average tokens: {token_report.average_tokens:.1f}",
        f"- p50 / p95 / p99 / max: {token_report.p50_tokens:.0f} / "
        f"{token_report.p95_tokens:.0f} / {token_report.p99_tokens:.0f} / {token_report.max_tokens}",
        f"- Semantic text length avg / p95: {token_report.semantic_text_length_avg:.0f} / "
        f"{token_report.semantic_text_length_p95:.0f}",
        "",
        "## Baseline execution",
        "",
        f"- Model: `{baseline.model_id}` ({baseline.provider})",
        f"- Device: `{baseline.device}`",
        f"- Batch size: {baseline.batch_size}",
        f"- Embedding seconds: {baseline.embedding_seconds:.3f}",
        f"- Throughput: {baseline.records_per_second:.3f} records/sec, "
        f"{baseline.tokens_per_second:.1f} tokens/sec",
        f"- Peak VRAM: {baseline.peak_memory_mb}",
        "",
        "## Batch experiments",
        "",
        "| Batch size | Records/sec | Tokens/sec | Peak VRAM (MB) | GPU util % |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for experiment in report.batch_experiments:
        metrics = experiment.metrics
        gpu_util = (
            f"{metrics.average_gpu_utilization_percent:.1f}"
            if metrics.average_gpu_utilization_percent is not None
            else "n/a"
        )
        peak_vram = (
            f"{metrics.peak_memory_mb:.1f}" if metrics.peak_memory_mb is not None else "n/a"
        )
        lines.append(
            f"| {metrics.batch_size} | {metrics.records_per_second:.3f} | "
            f"{metrics.tokens_per_second:.1f} | {peak_vram} | {gpu_util} |"
        )
    return "\n".join(lines)
