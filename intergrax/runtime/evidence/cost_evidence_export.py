# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Cost evidence CLI rendering and artifact export (EVID-COST-03)."""

from __future__ import annotations

from pathlib import Path

from intergrax.runtime.evidence.cost_evidence_contracts import (
    COST_EVIDENCE_OUTPUT_DIR,
    COST_EVIDENCE_REPORT_JSON,
    COST_EVIDENCE_REPORT_MARKDOWN,
    CostEvidenceArtifactRef,
    CostEvidenceCheckResult,
    CostEvidenceReport,
)
from intergrax.runtime.evidence.cost_evidence_runner import COST_EVIDENCE_OPERATOR_NOTE

DEFAULT_COST_EVIDENCE_OUTPUT_DIR = Path(COST_EVIDENCE_OUTPUT_DIR)


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _format_basis_list(result: CostEvidenceCheckResult) -> str:
    return ", ".join(basis.value for basis in result.basis)


def _format_result_cli(result: CostEvidenceCheckResult) -> str:
    header = (
        f"- {result.check_id}: {result.status.value} "
        f"[{_format_basis_list(result)}]"
    )
    detail = (
        f"  {result.title} — {result.message}"
        if result.message
        else f"  {result.title}"
    )
    return f"{header}\n{detail}"


def _format_artifact_cli_line(artifact_ref: CostEvidenceArtifactRef) -> str:
    if artifact_ref.description:
        return (
            f"- {artifact_ref.kind.value}: {artifact_ref.path} "
            f"({artifact_ref.description})"
        )
    return f"- {artifact_ref.kind.value}: {artifact_ref.path}"


def format_cost_evidence_cli(report: CostEvidenceReport) -> str:
    """Render a compact operator-facing cost evidence summary for terminal output."""
    lines = [
        report.title,
        f"Status: {report.status.value}",
        f"Report ID: {report.report_id}",
        f"Summary: {report.summary}",
        f"Note: {COST_EVIDENCE_OPERATOR_NOTE}",
        "",
        "Results:",
    ]
    for result in report.results:
        lines.append(_format_result_cli(result))

    lines.append("")
    if report.artifact_refs:
        lines.append("Artifacts:")
        for artifact_ref in report.artifact_refs:
            lines.append(_format_artifact_cli_line(artifact_ref))
    else:
        lines.append("Artifacts: none")

    return "\n".join(lines)


def _format_artifact_markdown_row(artifact_ref: CostEvidenceArtifactRef) -> str:
    kind = artifact_ref.kind.value
    path = _escape_markdown_cell(artifact_ref.path)
    description = _escape_markdown_cell(artifact_ref.description)
    return f"| {kind} | {path} | {description} |"


def format_cost_evidence_markdown(report: CostEvidenceReport) -> str:
    """Render a human-readable Markdown summary of cost evidence results."""
    lines = [
        "# Cost Evidence",
        "",
        f"- **Status:** {report.status.value}",
        f"- **Report ID:** {report.report_id}",
        f"- **Summary:** {_escape_markdown_cell(report.summary)}",
        f"- **Operator note:** {COST_EVIDENCE_OPERATOR_NOTE}",
        "",
        "## Results",
        "",
        "| Check ID | Kind | Status | Basis | Title | Message |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in report.results:
        basis = _escape_markdown_cell(_format_basis_list(result))
        title = _escape_markdown_cell(result.title)
        message = _escape_markdown_cell(result.message)
        lines.append(
            f"| {result.check_id} | {result.check_kind.value} | {result.status.value} | "
            f"{basis} | {title} | {message} |"
        )

    lines.extend(["", "## Artifacts", ""])
    if report.artifact_refs:
        lines.extend(
            [
                "| Kind | Path | Description |",
                "| --- | --- | --- |",
            ]
        )
        for artifact_ref in report.artifact_refs:
            lines.append(_format_artifact_markdown_row(artifact_ref))
    else:
        lines.append("_No cost evidence artifacts referenced._")

    lines.append("")
    return "\n".join(lines)


def write_cost_evidence_report(
    report: CostEvidenceReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write ``report.json`` and ``report.md`` under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / COST_EVIDENCE_REPORT_JSON
    md_path = output_dir / COST_EVIDENCE_REPORT_MARKDOWN
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(format_cost_evidence_markdown(report), encoding="utf-8")
    return json_path, md_path
