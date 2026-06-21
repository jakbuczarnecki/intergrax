# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Evidence posture CLI rendering and artifact export (HEP Band 2ae · EVID-POSTURE-04)."""

from __future__ import annotations

from pathlib import Path

from intergrax.runtime.evidence.evidence_posture_contracts import (
    EvidencePostureArtifactRef,
    EvidencePostureSummary,
    EvidenceSignal,
)

DEFAULT_POSTURE_OUTPUT_DIR = Path("build/evidence/posture")

POSTURE_OPERATOR_NOTE = (
    "Read-only evidence posture derived from existing artifacts. "
    "Does not execute doctor, pytest, certify core, trace export, live runtime probes, "
    "RuntimeEventBus, trace store, or provider calls."
)


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _format_signal_cli(signal: EvidenceSignal) -> str:
    header = f"- {signal.kind.value}: {signal.status.value} [{signal.basis.value}]"
    detail = f"  {signal.title} — {signal.message}" if signal.message else f"  {signal.title}"
    return f"{header}\n{detail}"


def format_evidence_posture_cli(summary: EvidencePostureSummary) -> str:
    """Render a compact operator-facing posture summary for terminal output."""
    lines = [
        summary.title,
        f"Level: {summary.level.value}",
        f"Posture ID: {summary.posture_id}",
        f"Summary: {summary.summary}",
        f"Note: {POSTURE_OPERATOR_NOTE}",
        "",
        "Signals:",
    ]
    for signal in summary.signals:
        lines.append(_format_signal_cli(signal))

    lines.append("")
    if summary.artifact_refs:
        lines.append("Artifacts:")
        for artifact_ref in summary.artifact_refs:
            lines.append(f"- {artifact_ref.kind.value}: {artifact_ref.path}")
    else:
        lines.append("Artifacts: none")

    return "\n".join(lines)


def _format_artifact_markdown_row(artifact_ref: EvidencePostureArtifactRef) -> str:
    kind = artifact_ref.kind.value
    path = _escape_markdown_cell(artifact_ref.path)
    description = _escape_markdown_cell(artifact_ref.description)
    return f"| {kind} | {path} | {description} |"


def format_evidence_posture_markdown(summary: EvidencePostureSummary) -> str:
    """Render a human-readable Markdown summary of evidence posture."""
    lines = [
        "# Intergrax Evidence Posture",
        "",
        f"- **Level:** {summary.level.value}",
        f"- **Posture ID:** {summary.posture_id}",
        f"- **Summary:** {_escape_markdown_cell(summary.summary)}",
        f"- **Operator note:** {POSTURE_OPERATOR_NOTE}",
        "",
        "## Signals",
        "",
        "| Kind | Status | Basis | Title | Message |",
        "| --- | --- | --- | --- | --- |",
    ]
    for signal in summary.signals:
        title = _escape_markdown_cell(signal.title)
        message = _escape_markdown_cell(signal.message)
        lines.append(
            f"| {signal.kind.value} | {signal.status.value} | {signal.basis.value} | "
            f"{title} | {message} |"
        )

    lines.extend(["", "## Artifacts", ""])
    if summary.artifact_refs:
        lines.extend(
            [
                "| Kind | Path | Description |",
                "| --- | --- | --- |",
            ]
        )
        for artifact_ref in summary.artifact_refs:
            lines.append(_format_artifact_markdown_row(artifact_ref))
    else:
        lines.append("_No posture artifacts referenced._")

    lines.append("")
    return "\n".join(lines)


def write_evidence_posture(
    summary: EvidencePostureSummary,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write ``posture.json`` and ``posture.md`` under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "posture.json"
    md_path = output_dir / "posture.md"
    json_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(format_evidence_posture_markdown(summary), encoding="utf-8")
    return json_path, md_path
