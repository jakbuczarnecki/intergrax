# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Trace timeline CLI rendering and artifact export (HEP Band 2ae · EVID-TRACE-04)."""

from __future__ import annotations

from pathlib import Path

from intergrax.runtime.evidence.core_certification_spec import CORE_CERTIFICATION_EVIDENCE_KIND
from intergrax.runtime.evidence.trace_timeline_contracts import TraceTimeline, TraceTimelineEvent
from intergrax.runtime.evidence.trace_timeline_facets import TraceTimelineEventFacets

DEFAULT_TRACE_EVIDENCE_OUTPUT_DIR = Path("build/evidence/trace")

TRACE_TIMELINE_OPERATOR_NOTE = (
    "Derived from certification report.json only. "
    f"Evidence basis: {CORE_CERTIFICATION_EVIDENCE_KIND}. "
    "Not live runtime trace, RuntimeEventBus, or persisted trace store."
)


def _format_event_facets(facets: TraceTimelineEventFacets | None) -> str:
    if facets is None:
        return ""
    parts: list[str] = []
    if facets.policy is not None:
        parts.append(f"policy={facets.policy.outcome.value}")
    if facets.budget is not None:
        parts.append(f"budget={facets.budget.status.value}")
    if facets.hitl is not None:
        parts.append(f"hitl={facets.hitl.status.value}")
    if facets.evidence is not None:
        parts.append(f"evidence={facets.evidence.origin.value}")
    if facets.scenario_lifecycle is not None:
        parts.append(f"phase={facets.scenario_lifecycle.phase.value}")
    return f" [{', '.join(parts)}]" if parts else ""


def _format_event_line(event: TraceTimelineEvent) -> str:
    scenario = f" ({event.scenario_id})" if event.scenario_id else ""
    severity = f" [{event.severity.value}]" if event.severity.value != "info" else ""
    facets = _format_event_facets(event.facets)
    message = f" - {event.message}" if event.message else ""
    return (
        f"[{event.sequence:>3}] {event.kind.value}{scenario}{severity}{facets}\n"
        f"      {event.title}{message}"
    )


def format_trace_timeline_cli(timeline: TraceTimeline) -> str:
    """Render a compact operator-facing timeline for terminal output."""
    lines = [
        f"Trace timeline: {timeline.title}",
        f"Evidence basis: {CORE_CERTIFICATION_EVIDENCE_KIND}",
        f"Note: {TRACE_TIMELINE_OPERATOR_NOTE}",
        f"Run ID: {timeline.timeline_id}",
        f"Kind: {timeline.kind.value}",
    ]
    if timeline.source_report_path:
        lines.append(f"Source report: {timeline.source_report_path}")
    lines.extend(["", "Events:", ""])
    for event in timeline.events:
        lines.append(_format_event_line(event))
    return "\n".join(lines)


def format_trace_timeline_markdown(timeline: TraceTimeline) -> str:
    """Render a human-readable Markdown summary of the timeline."""
    lines = [
        "# Intergrax Trace Evidence Timeline",
        "",
        f"- **Title:** {timeline.title}",
        f"- **Timeline ID:** {timeline.timeline_id}",
        f"- **Kind:** {timeline.kind.value}",
        f"- **Evidence basis:** {CORE_CERTIFICATION_EVIDENCE_KIND}",
        f"- **Operator note:** {TRACE_TIMELINE_OPERATOR_NOTE}",
        f"- **Generated:** {timeline.generated_at.isoformat()}",
    ]
    if timeline.source_report_path:
        lines.append(f"- **Source report:** {timeline.source_report_path}")
    lines.extend(
        [
            "",
            "## Events",
            "",
            "| Seq | Kind | Scenario | Severity | Title |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for event in timeline.events:
        scenario = event.scenario_id or ""
        title = event.title.replace("|", "\\|")
        message = event.message.replace("|", "\\|")
        cell = f"{title} - {message}" if message else title
        lines.append(
            f"| {event.sequence} | {event.kind.value} | {scenario} | "
            f"{event.severity.value} | {cell} |"
        )

    facet_lines: list[str] = []
    for event in timeline.events:
        if event.facets is None:
            continue
        summary = _format_event_facets(event.facets).strip()
        if summary:
            facet_lines.append(
                f"- **{event.sequence}** `{event.kind.value}`{summary}"
            )
    if facet_lines:
        lines.extend(["", "## Facets", ""])
        lines.extend(facet_lines)

    lines.append("")
    return "\n".join(lines)


def write_trace_timeline(
    timeline: TraceTimeline,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write ``timeline.json`` and ``timeline.md`` under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "timeline.json"
    md_path = output_dir / "timeline.md"
    json_path.write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(format_trace_timeline_markdown(timeline), encoding="utf-8")
    return json_path, md_path
