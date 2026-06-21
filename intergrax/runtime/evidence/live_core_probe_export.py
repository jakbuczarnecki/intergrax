# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Live core probe CLI rendering and artifact export (EVID-CORE-FU-01 · EVID-CORE-FU-01C)."""

from __future__ import annotations

from pathlib import Path

from intergrax.runtime.evidence.live_core_probe_contracts import (
    LiveCoreProbeArtifactRef,
    LiveCoreProbeReport,
    LiveCoreProbeResult,
)
from intergrax.runtime.evidence.live_core_probe_runner import LIVE_CORE_PROBE_OPERATOR_NOTE

DEFAULT_LIVE_CORE_PROBE_OUTPUT_DIR = Path("build/evidence/live_core_probes")

LIVE_CORE_PROBE_REPORT_JSON = "live_core_report.json"
LIVE_CORE_PROBE_REPORT_MARKDOWN = "live_core_report.md"


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _format_basis_list(result: LiveCoreProbeResult) -> str:
    return ", ".join(basis.value for basis in result.evidence_basis)


def _format_result_cli(result: LiveCoreProbeResult) -> str:
    header = (
        f"- {result.probe_kind.value}: {result.status.value} "
        f"[{_format_basis_list(result)}]"
    )
    detail = f"  {result.title} — {result.message}" if result.message else f"  {result.title}"
    return f"{header}\n{detail}"


def format_live_core_probe_cli(report: LiveCoreProbeReport) -> str:
    """Render a compact operator-facing live core probe summary for terminal output."""
    lines = [
        report.title,
        f"Status: {report.status.value}",
        f"Report ID: {report.report_id}",
        f"Summary: {report.summary}",
        f"Note: {LIVE_CORE_PROBE_OPERATOR_NOTE}",
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


def _format_artifact_cli_line(artifact_ref: LiveCoreProbeArtifactRef) -> str:
    if artifact_ref.description:
        return f"- {artifact_ref.kind.value}: {artifact_ref.path} ({artifact_ref.description})"
    return f"- {artifact_ref.kind.value}: {artifact_ref.path}"


def _format_artifact_markdown_row(artifact_ref: LiveCoreProbeArtifactRef) -> str:
    kind = artifact_ref.kind.value
    path = _escape_markdown_cell(artifact_ref.path)
    description = _escape_markdown_cell(artifact_ref.description)
    return f"| {kind} | {path} | {description} |"


def format_live_core_probe_markdown(report: LiveCoreProbeReport) -> str:
    """Render a human-readable Markdown summary of live core probe results."""
    lines = [
        "# Selected Live Tier-0 Probes",
        "",
        f"- **Status:** {report.status.value}",
        f"- **Report ID:** {report.report_id}",
        f"- **Summary:** {_escape_markdown_cell(report.summary)}",
        f"- **Operator note:** {LIVE_CORE_PROBE_OPERATOR_NOTE}",
        "",
        "## Results",
        "",
        "| Probe | Status | Evidence basis | Title | Message |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in report.results:
        basis = _escape_markdown_cell(_format_basis_list(result))
        title = _escape_markdown_cell(result.title)
        message = _escape_markdown_cell(result.message)
        lines.append(
            f"| {result.probe_kind.value} | {result.status.value} | {basis} | "
            f"{title} | {message} |"
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
        lines.append("_No live core probe artifacts referenced._")

    lines.append("")
    return "\n".join(lines)


def write_live_core_probe_report(
    report: LiveCoreProbeReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write ``live_core_report.json`` and ``live_core_report.md`` under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / LIVE_CORE_PROBE_REPORT_JSON
    md_path = output_dir / LIVE_CORE_PROBE_REPORT_MARKDOWN
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(format_live_core_probe_markdown(report), encoding="utf-8")
    return json_path, md_path
