"""JSON and Markdown reporting for bounded CUDA batch throughput qualification."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.contracts import (
    BoundedCudaBatchThroughputReport,
)


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    msg = f"unsupported type for bounded CUDA qualification JSON: {type(value)!r}"
    raise TypeError(msg)


def bounded_cuda_report_to_json(report: BoundedCudaBatchThroughputReport) -> str:
    return json.dumps(asdict(report), indent=2, default=_json_default, sort_keys=True)


def write_bounded_cuda_report_json(path: Path, report: BoundedCudaBatchThroughputReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bounded_cuda_report_to_json(report), encoding="utf-8")


def _format_bytes(value: int) -> str:
    gib = value / (1024 ** 3)
    return f"{gib:.2f} GiB"


def render_bounded_cuda_markdown_report(report: BoundedCudaBatchThroughputReport) -> str:
    lines: list[str] = [
        "# Bounded CUDA Batch Throughput Report",
        "",
        f"Task: {report.task_id}",
        f"Status: {report.status.value}",
        "",
        "## CUDA environment",
        "",
        f"- Python: {report.preflight.python_version}",
        f"- Torch: {report.preflight.torch_version}",
        f"- CUDA runtime: {report.preflight.cuda_runtime_version}",
        f"- GPU: {report.preflight.gpu_name}",
        f"- Total VRAM: {_format_bytes(report.preflight.gpu_total_memory_bytes or 0)}",
        (
            "- Initial free VRAM: "
            f"{_format_bytes(report.preflight.gpu_free_memory_bytes_before_load or 0)}"
        ),
        "",
        "## Model",
        "",
        f"- Provider: {report.provider}",
        f"- Model: {report.model}",
        f"- Revision: {report.revision}",
        f"- Dimension: {report.dimension}",
        f"- Model load count: {report.model_load_count}",
        "",
        "## Policy",
        "",
        f"- Policy version: {report.policy_version}",
        f"- Token budget: {report.token_budget}",
        "- Queries changed: NO",
        "",
        "## Sample",
        "",
        f"- Dataset: {report.dataset_path}",
        f"- Record count: {report.record_count}",
        f"- Selection: {report.selection_method} / real",
        "",
        "## Token profile",
        "",
        f"- Average: {report.token_profile.average_tokens_per_record:.1f}",
        f"- P50: {report.token_profile.p50_tokens:.1f}",
        f"- P95: {report.token_profile.p95_tokens:.1f}",
        f"- Max: {report.token_profile.max_tokens}",
        "",
        "## Results",
        "",
        (
            "| batch | seconds | rec/s | tokens/s | avg ms/rec | peak allocated | "
            "peak reserved | headroom | safe |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for item in report.batch_measurements:
        lines.append(
            "| "
            f"{item.batch_size} | "
            f"{item.wall_clock_embedding_seconds:.3f} | "
            f"{item.records_per_second:.2f} | "
            f"{item.tokens_per_second:.0f} | "
            f"{item.average_milliseconds_per_record:.1f} | "
            f"{_format_bytes(item.peak_cuda_allocated_bytes)} | "
            f"{_format_bytes(item.peak_cuda_reserved_bytes)} | "
            f"{item.vram_headroom_fraction:.1%} | "
            f"{'yes' if item.safe else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Batch 32",
            "",
            f"- Decision: {'RUN' if report.optional_batch_32.executed else 'SKIPPED'}",
            f"- Reason: {report.optional_batch_32.reason}",
            "",
            "## Selected production batch candidate",
            "",
            f"- Batch: {report.production_batch_selection.batch_size}",
            f"- Why: {report.production_batch_selection.rationale}",
            "",
            "## Full 3,770,377 projection (embedding-only)",
            "",
            "| batch | rec/s | projected hours | peak VRAM | safe |",
            "|---:|---:|---:|---:|:---:|",
        ]
    )
    for projection in report.projections:
        measurement = next(
            (
                item
                for item in report.batch_measurements
                if item.batch_size == projection.batch_size
            ),
            None,
        )
        peak = (
            _format_bytes(measurement.peak_cuda_allocated_bytes)
            if measurement is not None
            else "N/A"
        )
        lines.append(
            "| "
            f"{projection.batch_size} | "
            f"{projection.records_per_second:.2f} | "
            f"{projection.projected_hours:.1f} | "
            f"{peak} | "
            f"{'yes' if projection.safe else 'no'} |"
        )
    if report.winner_projection is not None:
        lines.extend(
            [
                "",
                "## Winner projected embedding time",
                "",
                f"- Hours: {report.winner_projection.projected_hours:.1f}",
                f"- Days: {report.winner_projection.projected_days:.1f}",
                "",
                f"Projection only: {'YES' if report.projection_only else 'NO'}",
            ]
        )
    lines.extend(
        [
            "",
            "## Resource stability",
            "",
            f"- OOM: {'YES' if report.oom_observed else 'NO'}",
            f"- System instability: {'YES' if report.system_instability else 'NO'}",
            (
                "- Persistent VRAM growth: "
                f"{'YES' if report.persistent_vram_growth else 'NO'}"
            ),
        ]
    )
    if report.known_gaps:
        lines.extend(["", "## Known gaps", ""])
        for gap in report.known_gaps:
            lines.append(f"- {gap}")
    return "\n".join(lines) + "\n"


def write_bounded_cuda_markdown_report(
    path: Path,
    report: BoundedCudaBatchThroughputReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_bounded_cuda_markdown_report(report), encoding="utf-8")
