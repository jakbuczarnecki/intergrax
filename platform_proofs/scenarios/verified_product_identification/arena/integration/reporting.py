"""Arena report serialization."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    EmbeddingArenaReport,
)


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    msg = f"unsupported type for arena JSON: {type(value)!r}"
    raise TypeError(msg)


def arena_report_to_json(report: EmbeddingArenaReport) -> str:
    return json.dumps(asdict(report), indent=2, default=_json_default, sort_keys=True)


def write_arena_report(path: Path, report: EmbeddingArenaReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(arena_report_to_json(report), encoding="utf-8")


def render_arena_summary_markdown(report: EmbeddingArenaReport) -> str:
    lines = [
        "# VPI Embedding Model Arena Summary",
        "",
        f"- Arena version: `{report.arena_version}`",
        f"- Sample version: `{report.sample_manifest.version}`",
        f"- Query benchmark: `{report.query_benchmark_version}`",
        f"- Query cases: {len(report.query_cases)}",
        f"- Decision: `{report.decision.value}`",
        f"- Rationale: {report.decision_rationale}",
        f"- 5C4C finalists: {', '.join(report.finalists_for_5c4c) or 'none'}",
        "",
        "## Candidates",
        "",
        "| candidate | verdict | stage C rps | Recall@10 | speedup |",
        "|---|---|---:|---:|---:|",
    ]
    for result in report.candidate_results:
        rps = result.stage_c.throughput_records_per_second if result.stage_c else None
        recall = result.quality_metrics.recall_at_10 if result.quality_metrics else None
        speedup = (
            result.speedup_estimate.speedup_vs_baseline if result.speedup_estimate else None
        )
        lines.append(
            f"| {result.candidate_id} | {result.verdict.value} | "
            f"{rps or 'n/a'} | {recall if recall is not None else 'n/a'} | "
            f"{speedup if speedup is not None else 'n/a'} |"
        )
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
    return "\n".join(lines) + "\n"


def write_arena_summary(path: Path, report: EmbeddingArenaReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_arena_summary_markdown(report), encoding="utf-8")
