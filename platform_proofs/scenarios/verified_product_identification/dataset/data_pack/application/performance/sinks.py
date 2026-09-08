"""Machine-readable and human-readable performance evidence sinks."""

from __future__ import annotations

import json
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.performance.contracts import (
    PerformanceReport,
    ShardPerformanceMetrics,
)


def write_performance_evidence(
    output_dir: Path,
    report: PerformanceReport,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "performance-report.json"
    summary_path = output_dir / "PERFORMANCE_SUMMARY.md"
    payload = _report_to_json(report)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_path.write_text(_report_to_markdown(report), encoding="utf-8")
    return json_path, summary_path


def _report_to_json(report: PerformanceReport) -> dict[str, object]:
    shards = [_shard_to_json(metrics) for metrics in report.shard_metrics]
    return {
        "qualification_id": report.qualification_id,
        "dominant_phase": report.dominant_phase,
        "dominant_phase_seconds": round(report.dominant_phase_seconds, 3),
        "shards": shards,
    }


def _shard_to_json(metrics: ShardPerformanceMetrics) -> dict[str, object]:
    return {
        "shard": metrics.shard_ordinal,
        "records": metrics.record_count,
        "read_seconds": round(metrics.read_seconds, 3),
        "derive_seconds": round(metrics.derive_seconds, 3),
        "tokenize_seconds": round(metrics.tokenize_seconds, 3),
        "embedding_seconds": round(metrics.embedding_seconds, 3),
        "embedding_batch_seconds": round(metrics.embedding_batch_seconds, 3),
        "write_seconds": round(metrics.write_seconds, 3),
        "checksum_seconds": round(metrics.checksum_seconds, 3),
        "source_identity_seconds": round(metrics.source_identity_seconds, 3),
        "validation_seconds": round(metrics.validation_seconds, 3),
        "state_update_seconds": round(metrics.state_update_seconds, 3),
        "total_seconds": round(metrics.total_seconds, 3),
        "shard_records_per_second": round(metrics.shard_records_per_second, 3),
        "embedding": {
            "model_id": metrics.model_id,
            "device": metrics.device,
            "batch_size": metrics.batch_size,
            "embedding_calls": metrics.embedding_calls,
            "embedding_records": metrics.embedding_records,
            "embedding_records_per_second": round(metrics.embedding_records_per_second, 3),
        },
        "system": {
            "process_id": metrics.process_id,
            "started_at": metrics.started_at,
            "completed_at": metrics.completed_at,
        },
    }


def _report_to_markdown(report: PerformanceReport) -> str:
    lines = [
        "# Data Pack Build Performance Summary",
        "",
        f"- Qualification ID: `{report.qualification_id}`",
        f"- Dominant bottleneck: **{report.dominant_phase}** "
        f"({report.dominant_phase_seconds:.1f}s)",
        "",
    ]
    for metrics in report.shard_metrics:
        lines.extend(
            [
                f"## Shard {metrics.shard_ordinal}",
                "",
                f"- Records: {metrics.record_count}",
                f"- Total: {metrics.total_seconds:.1f}s "
                f"({metrics.shard_records_per_second:.3f} records/sec)",
                "",
                "### Timing breakdown",
                "",
                "| Phase | Seconds | % of total |",
                "| --- | ---: | ---: |",
            ]
        )
        total = metrics.total_seconds if metrics.total_seconds > 0 else 1.0
        breakdown = [
            ("read", metrics.read_seconds),
            ("derive", metrics.derive_seconds),
            ("tokenize", metrics.tokenize_seconds),
            ("embedding", metrics.embedding_seconds),
            ("embedding_batch", metrics.embedding_batch_seconds),
            ("parquet_write", metrics.write_seconds),
            ("checksum", metrics.checksum_seconds),
            ("source_identity", metrics.source_identity_seconds),
            ("validation", metrics.validation_seconds),
            ("state_update", metrics.state_update_seconds),
        ]
        for phase_name, seconds in breakdown:
            pct = 100.0 * seconds / total
            lines.append(f"| {phase_name} | {seconds:.1f} | {pct:.1f}% |")
        lines.extend(
            [
                "",
                "### Embedding throughput",
                "",
                f"- Model: `{metrics.model_id}`",
                f"- Device: `{metrics.device}`",
                f"- Batch size: {metrics.batch_size}",
                f"- Calls: {metrics.embedding_calls}",
                f"- Records embedded: {metrics.embedding_records}",
                f"- Throughput: {metrics.embedding_records_per_second:.3f} records/sec",
                "",
            ]
        )
    lines.append("## Next optimization areas")
    lines.append("")
    lines.append(_recommendation_for_phase(report.dominant_phase))
    return "\n".join(lines)


def _recommendation_for_phase(phase: str) -> str:
    recommendations = {
        "embedding": (
            "A. Embedding throughput: tune batch size, CUDA utilization, and provider batching."
        ),
        "tokenize": (
            "A. Embedding throughput: tokenizer overhead is dominant; consider batch size tuning "
            "or provider-side tokenization caching."
        ),
        "derive": (
            "B. Text processing: reduce unnecessary transformations or cache semantic derivation."
        ),
        "parquet_write": (
            "C. IO: tune parquet row groups, compression, and write batching."
        ),
        "checksum": (
            "C. IO: checksum generation is dominant; consider deferred or incremental checksums."
        ),
        "validation": (
            "C. IO / validation: reduce validation frequency or scope for temp shards."
        ),
        "state_update": (
            "D. Lifecycle: reduce excessive filesystem operations during state persistence."
        ),
        "read": (
            "C. IO: dataset reading is dominant; review parquet row-group access patterns."
        ),
        "embedding_batch": (
            "A. Embedding throughput: batch loop overhead is significant; increase provider batch size."
        ),
        "source_identity": (
            "B. Text processing / identity: source-ref digest computation is dominant."
        ),
    }
    return recommendations.get(
        phase,
        "Review full timing breakdown before selecting an optimization target.",
    )
