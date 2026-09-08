"""Machine-readable and human-readable embedding diagnostic evidence sinks."""

from __future__ import annotations

from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.contracts import (
    EmbeddingDiagnosticReport,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.ports import (
    DiagnosticSinkPort,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.serialization import (
    serialize_diagnostic_report_json,
)


class FileDiagnosticSink(DiagnosticSinkPort):
    def write_report(
        self,
        output_dir: Path,
        report: EmbeddingDiagnosticReport,
    ) -> tuple[Path, Path]:
        return write_embedding_diagnostic_report(output_dir, report)


def write_embedding_diagnostic_report(
    output_dir: Path,
    report: EmbeddingDiagnosticReport,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "embedding-diagnostic-report.json"
    summary_path = output_dir / "EMBEDDING_DIAGNOSTIC_SUMMARY.md"
    json_path.write_text(serialize_diagnostic_report_json(report), encoding="utf-8")
    summary_path.write_text(_report_to_markdown(report), encoding="utf-8")
    return json_path, summary_path


def _report_to_markdown(report: EmbeddingDiagnosticReport) -> str:
    token_report = report.token_distribution
    baseline = report.baseline
    stats = token_report.statistics
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
        f"- Average tokens: {stats.mean:.1f}",
        f"- min / p50 / p90 / p95 / p99 / max: {stats.minimum:.0f} / "
        f"{stats.p50:.0f} / {stats.p90:.0f} / {stats.p95:.0f} / {stats.p99:.0f} / "
        f"{stats.maximum}",
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
        "| Experiment | Batch size | Records/sec | Tokens/sec | Peak VRAM (MB) | GPU util % |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
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
            f"| {experiment.experiment_kind.value} | {metrics.batch_size} | "
            f"{metrics.records_per_second:.3f} | {metrics.tokens_per_second:.1f} | "
            f"{peak_vram} | {gpu_util} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic answers",
            "",
            f"1. Average input length: {stats.mean:.1f} tokens "
            f"({token_report.semantic_text_length_avg:.0f} characters).",
            (
                f"2. Extreme records present: "
                f"{'yes' if stats.maximum > TOKEN_EXTREME_THRESHOLD else 'no'} "
                f"(max={stats.maximum})."
            ),
            f"3. Dominant bottleneck classification: {report.classification.case.value}.",
            f"4. Recommended next action: {report.classification.recommended_next_task}.",
        ]
    )
    return "\n".join(lines)


TOKEN_EXTREME_THRESHOLD = 2000
