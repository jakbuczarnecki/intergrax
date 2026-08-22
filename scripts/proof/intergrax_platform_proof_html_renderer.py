# © Artur Czarnecki. All rights reserved.

"""Generic self-contained HTML renderer for Platform Proof evidence (PP-REPORT-3).

``PlatformProofEvidence`` → ``report.html`` projection. Presentation only — not
independent truth. Domain-specific sections are reserved via ``_render_domain_extensions``.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Callable

from scripts.proof.intergrax_platform_proof_evidence import (
    PLATFORM_PROOF_EVIDENCE_SCHEMA_VERSION,
    ArchitectureEdgeEvidence,
    ArchitectureEvidence,
    ConclusionEvidence,
    DatasetEnvironmentEvidence,
    DomainExtensionEvidence,
    EnvironmentEvidence,
    EvaluatorCheckEvidence,
    EvaluatorSummaryEvidence,
    EvidenceEdge,
    EvidenceGraphEvidence,
    EvidenceNode,
    FailureEvidence,
    FinalOutputEvidence,
    ParticipantEvidence,
    PlatformProofEvidence,
    ProofClaimEvidence,
    ProofEvidenceExecutionStatus,
    ProofExecutionStep,
    ProofIdentityEvidence,
    ProofStepExecutionStatus,
    ProvenanceEvidence,
    ReportSafeField,
    ReportSafePayload,
    ReportSafeScalar,
    ReportSafeText,
    ReportSafeVisibility,
    ReproductionEvidence,
    ScenarioEvidence,
    ToolsSqlInvestigationExtension,
    proof_authored_report_safe_text,
)

PLATFORM_PROOF_REPORT_SCHEMA_VERSION = "intergrax.platform_proof_report.v1"
PLATFORM_PROOF_REPORT_STANDARD_VERSION = "v1"
PLATFORM_PROOF_HTML_RENDERER_VERSION = "1.0.0"

REPORT_FILENAME = "report.html"

_EXTERNAL_STYLESHEET_RE = re.compile(
    r'<link[^>]+rel=["\']stylesheet["\'][^>]*href=["\']https?://',
    re.IGNORECASE,
)
_EXTERNAL_SCRIPT_RE = re.compile(
    r'<script[^>]+src=["\']https?://',
    re.IGNORECASE,
)
_REMOTE_IMPORT_RE = re.compile(r"@import\s+url\(\s*['\"]?https?://", re.IGNORECASE)
_REMOTE_IMAGE_RE = re.compile(
    r'<img[^>]+src=["\']https?://',
    re.IGNORECASE,
)


class PlatformProofReportRenderError(Exception):
    """Raised when report-safe rendering cannot proceed safely."""


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _render_report_safe_text(value: ReportSafeText) -> str:
    if value.visibility == ReportSafeVisibility.REDACTED:
        return "[REDACTED]"
    if value.visibility == ReportSafeVisibility.SUMMARY_ONLY:
        raise PlatformProofReportRenderError(
            "ReportSafeText with SUMMARY_ONLY visibility cannot be rendered as full text"
        )
    return value.text


def _render_report_safe_scalar(value: ReportSafeScalar) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _render_report_safe_field_value(
    value: ReportSafeScalar | ReportSafeText | tuple[ReportSafeScalar | ReportSafeText, ...],
) -> str:
    if isinstance(value, ReportSafeText):
        return _render_report_safe_text(value)
    if isinstance(value, tuple):
        parts = [
            _render_report_safe_field_value(item) for item in value
        ]
        return ", ".join(parts)
    if isinstance(value, str):
        raise PlatformProofReportRenderError(
            "raw string in ReportSafeField value is not allowed for rendering"
        )
    return _render_report_safe_scalar(value)


def _render_report_safe_field(field: ReportSafeField) -> str:
    if field.visibility == ReportSafeVisibility.REDACTED:
        return "[REDACTED]"
    if field.visibility == ReportSafeVisibility.SUMMARY_ONLY:
        return "[summary only]"
    if field.value is None:
        return "—"
    return _render_report_safe_field_value(field.value)


def _render_report_safe_payload(payload: ReportSafePayload) -> str:
    if payload.visibility == ReportSafeVisibility.REDACTED:
        return "<p class=\"muted\">[REDACTED]</p>"
    parts: list[str] = []
    summary = _render_report_safe_text(payload.summary).strip()
    if summary:
        parts.append(f"<p class=\"payload-summary\">{_escape(summary)}</p>")
    if payload.fields:
        rows = []
        for field in payload.fields:
            rows.append(
                "<tr>"
                f"<th scope=\"row\">{_escape(field.name)}</th>"
                f"<td><pre class=\"payload-value\">{_escape(_render_report_safe_field(field))}</pre></td>"
                "</tr>"
            )
        parts.append(
            "<table class=\"payload-table\"><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
    if not parts:
        return "<p class=\"muted\">—</p>"
    return "".join(parts)


def _status_badge(status: ProofEvidenceExecutionStatus) -> str:
    labels = {
        ProofEvidenceExecutionStatus.PASS: ("PASS", "status-pass", "✓"),
        ProofEvidenceExecutionStatus.FAIL: ("FAIL", "status-fail", "✗"),
        ProofEvidenceExecutionStatus.BLOCKED: ("BLOCKED", "status-blocked", "⊘"),
        ProofEvidenceExecutionStatus.CRASH: ("CRASH", "status-crash", "⚠"),
    }
    label, css_class, icon = labels[status]
    return (
        f'<span class="status-badge {css_class}" role="status" aria-label="{label}">'
        f'<span class="status-icon" aria-hidden="true">{icon}</span>'
        f"<span class=\"status-label\">{_escape(label)}</span>"
        "</span>"
    )


def _step_status_badge(status: ProofStepExecutionStatus) -> str:
    mapping = {
        ProofStepExecutionStatus.OK: ("ok", "step-ok"),
        ProofStepExecutionStatus.FAIL: ("fail", "step-fail"),
        ProofStepExecutionStatus.SKIPPED: ("skipped", "step-skipped"),
    }
    label, css = mapping[status]
    return f'<span class="step-status {css}">{_escape(label)}</span>'


def _css() -> str:
    return """
:root {
  --bg: #f4f6f8;
  --surface: #ffffff;
  --text: #1a2332;
  --muted: #5c6b7a;
  --border: #d8dee6;
  --accent: #2563eb;
  --pass: #15803d;
  --pass-bg: #ecfdf3;
  --fail: #b91c1c;
  --fail-bg: #fef2f2;
  --blocked: #b45309;
  --blocked-bg: #fffbeb;
  --crash: #7f1d1d;
  --crash-bg: #fef2f2;
  --card-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  --max-width: 960px;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: var(--sans);
  color: var(--text);
  background: var(--bg);
  line-height: 1.5;
}
main {
  max-width: var(--max-width);
  margin: 0 auto;
  padding: 1.5rem 1rem 3rem;
}
h1, h2, h3 { line-height: 1.25; margin: 0 0 0.75rem; }
h1 { font-size: 1.75rem; }
h2 {
  font-size: 1.25rem;
  margin-top: 2rem;
  padding-bottom: 0.35rem;
  border-bottom: 1px solid var(--border);
}
h3 { font-size: 1.05rem; margin-top: 1.25rem; }
p { margin: 0 0 0.75rem; }
ul, ol { margin: 0 0 0.75rem 1.25rem; }
.muted { color: var(--muted); }
.card-grid {
  display: grid;
  gap: 0.75rem;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  margin-bottom: 1rem;
}
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.9rem 1rem;
  box-shadow: var(--card-shadow);
}
.card-label {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--muted);
  margin-bottom: 0.25rem;
}
.card-value { font-weight: 600; word-break: break-word; }
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.85rem;
  border: 1px solid transparent;
}
.status-pass { color: var(--pass); background: var(--pass-bg); border-color: #86efac; }
.status-fail { color: var(--fail); background: var(--fail-bg); border-color: #fecaca; }
.status-blocked { color: var(--blocked); background: var(--blocked-bg); border-color: #fde68a; }
.status-crash { color: var(--crash); background: var(--crash-bg); border-color: #fca5a5; }
.executive-summary {
  background: var(--surface);
  border-left: 4px solid var(--accent);
  padding: 1rem 1.1rem;
  border-radius: 0 8px 8px 0;
  margin-bottom: 1.25rem;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92rem;
  background: var(--surface);
}
th, td {
  border: 1px solid var(--border);
  padding: 0.5rem 0.6rem;
  vertical-align: top;
  text-align: left;
}
th { background: #f8fafc; }
tbody tr:nth-child(even) { background: #fbfdff; }
.trace-desktop { display: block; }
.trace-mobile { display: none; }
.trace-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.75rem;
  margin-bottom: 0.75rem;
}
.trace-card dt {
  font-size: 0.75rem;
  text-transform: uppercase;
  color: var(--muted);
  margin-top: 0.5rem;
}
.trace-card dd { margin: 0.15rem 0 0; }
pre, .payload-value {
  font-family: var(--mono);
  font-size: 0.82rem;
  white-space: pre-wrap;
  word-break: break-word;
  background: #f8fafc;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.5rem 0.6rem;
  margin: 0;
  max-height: 14rem;
  overflow: auto;
}
.evidence-id {
  font-family: var(--mono);
  font-size: 0.78rem;
  background: #eef2ff;
  border: 1px solid #c7d2fe;
  border-radius: 4px;
  padding: 0.05rem 0.3rem;
}
.ground-truth-panel {
  display: grid;
  gap: 0.75rem;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}
.ground-truth-panel section {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.75rem;
  background: var(--surface);
}
.ground-truth-panel h3 { margin-top: 0; font-size: 0.95rem; }
.verdict-panel {
  border: 2px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  background: var(--surface);
}
.verdict-panel.pass { border-color: #86efac; }
.verdict-panel.fail { border-color: #fecaca; }
.final-output-panel {
  border: 1px dashed var(--border);
  border-radius: 8px;
  padding: 1rem;
  background: #fafbfc;
}
.failure-panel {
  border-left: 4px solid var(--fail);
  background: var(--fail-bg);
  padding: 0.9rem 1rem;
  border-radius: 0 8px 8px 0;
}
.arch-diagram, .graph-diagram {
  width: 100%;
  max-width: 100%;
  height: auto;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
}
.section-empty {
  color: var(--muted);
  font-style: italic;
  padding: 0.5rem 0;
}
.step-ok { color: var(--pass); font-weight: 600; }
.step-fail { color: var(--fail); font-weight: 600; }
.step-skipped { color: var(--muted); font-weight: 600; }
@media (max-width: 720px) {
  .trace-desktop { display: none; }
  .trace-mobile { display: block; }
}
@media print {
  body { background: #fff; }
  main { max-width: none; padding: 0; }
  .card, .trace-card, table { break-inside: avoid; }
  pre { max-height: none; overflow: visible; }
}
""".strip()


def _render_executive_summary(evidence: PlatformProofEvidence) -> str:
    status = evidence.execution.status.value
    claim = evidence.claim.claim
    limitation = evidence.limitations[0] if evidence.limitations else "See limitations section."
    evaluator_line = "No evaluator summary recorded."
    if evidence.evaluator is not None:
        evaluator_line = (
            "Evaluator passed."
            if evidence.evaluator.passed
            else "Evaluator did not pass."
        )
        if evidence.evaluator.failure_reasons:
            evaluator_line += f" Reasons: {', '.join(evidence.evaluator.failure_reasons)}."
    strongest = "execution timeline"
    if evidence.evidence_graph.nodes:
        strongest = f"evidence graph ({evidence.evidence_graph.nodes[0].evidence_id})"
    paragraphs = [
        (
            f"This report documents proof {_escape(evidence.proof_identity.proof_id)} "
            f"({_escape(evidence.proof_identity.title)}) in domain "
            f"{_escape(evidence.proof_identity.domain)}."
        ),
        (
            f"Execution result: {_escape(status)} on platform "
            f"{_escape(evidence.execution.platform)} at revision "
            f"{_escape(evidence.proof_identity.source_revision)}."
        ),
        f"Claim under test: {_escape(claim)}",
        f"Primary evaluator conclusion: {_escape(evaluator_line)}",
        f"Strongest evidence pointer: {_escape(strongest)}.",
        f"Primary limitation: {_escape(limitation)}",
    ]
    return "<div class=\"executive-summary\">" + "".join(f"<p>{p}</p>" for p in paragraphs) + "</div>"


def _render_claim(claim: ProofClaimEvidence) -> str:
    parts = [
        f"<p><strong>Claim:</strong> {_escape(claim.claim)}</p>",
        f"<p><strong>User relevance:</strong> {_escape(claim.user_relevance)}</p>",
        "<h3>Success criteria</h3><ul>"
        + "".join(f"<li>{_escape(item)}</li>" for item in claim.success_criteria)
        + "</ul>",
        "<h3>Falsification criteria</h3><ul>"
        + "".join(f"<li>{_escape(item)}</li>" for item in claim.falsification_criteria)
        + "</ul>",
    ]
    return "".join(parts)


def _render_excluded_claims(claim: ProofClaimEvidence) -> str:
    if not claim.excluded_claims:
        return '<p class="section-empty">No excluded claims declared.</p>'
    return "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in claim.excluded_claims) + "</ul>"


def _render_architecture_diagram(architecture: ArchitectureEvidence) -> str:
    participants = architecture.participants
    if not participants:
        return '<p class="section-empty">No architecture participants.</p>'
    row_height = 56
    width = 640
    height = max(120, len(participants) * row_height + 40)
    nodes: list[str] = []
    for index, participant in enumerate(participants):
        y = 30 + index * row_height
        nodes.append(
            f'<rect x="40" y="{y - 20}" width="220" height="40" rx="6" '
            f'fill="#f8fafc" stroke="#94a3b8"/>'
            f'<text x="50" y="{y + 5}" font-size="12" fill="#1e293b">'
            f"{_escape(participant.name)}</text>"
            f'<text x="280" y="{y + 5}" font-size="11" fill="#64748b">'
            f"{_escape(participant.participant_class.value)}</text>"
        )
    edges_svg: list[str] = []
    id_to_index = {p.participant_id: i for i, p in enumerate(participants)}
    for edge in architecture.edges:
        from_index = id_to_index.get(edge.from_participant)
        to_index = id_to_index.get(edge.to_participant)
        if from_index is None or to_index is None:
            continue
        y1 = 30 + from_index * row_height + 20
        y2 = 30 + to_index * row_height - 20
        mid_x = width - 80
        edges_svg.append(
            f'<line x1="{mid_x}" y1="{y1}" x2="{mid_x}" y2="{y2}" '
            f'stroke="#64748b" marker-end="url(#arrow)"/>'
            f'<text x="{mid_x + 8}" y="{(y1 + y2) // 2}" font-size="10" fill="#475569">'
            f"{_escape(edge.relationship)}</text>"
        )
    return (
        f'<svg class="arch-diagram" viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="arch-diagram-title">'
        '<title id="arch-diagram-title">Architecture under proof</title>'
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" '
        'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#64748b"/></marker></defs>'
        + "".join(nodes)
        + "".join(edges_svg)
        + "</svg>"
    )


def _render_participants(participants: tuple[ParticipantEvidence, ...]) -> str:
    if not participants:
        return '<p class="section-empty">No participants recorded.</p>'
    rows = []
    for participant in participants:
        rows.append(
            "<tr>"
            f"<td>{_escape(participant.name)}</td>"
            f"<td>{_escape(participant.implementation)}</td>"
            f"<td>{_escape(participant.version_or_model)}</td>"
            f"<td>{_escape(participant.role)}</td>"
            f"<td>{_escape(participant.participant_class.value)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Component</th><th>Implementation</th><th>Version</th><th>Role</th><th>Class</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_dataset(dataset: DatasetEnvironmentEvidence) -> str:
    items = [
        ("Dataset ID", dataset.dataset_id),
        ("Version", dataset.dataset_version),
        ("Row count", str(dataset.row_count)),
        ("Fingerprint", dataset.fingerprint_sha256),
    ]
    if dataset.seed is not None:
        items.append(("Seed", str(dataset.seed)))
    if dataset.infrastructure_identity:
        items.append(("Infrastructure", dataset.infrastructure_identity))
    if dataset.access_mode:
        items.append(("Access mode", dataset.access_mode))
    cards = "".join(
        f'<div class="card"><div class="card-label">{_escape(label)}</div>'
        f'<div class="card-value">{_escape(value)}</div></div>'
        for label, value in items
    )
    ground_truth = ""
    if dataset.ground_truth_checks or dataset.information_exposed_to_model:
        gt_items = "".join(f"<li>{_escape(item)}</li>" for item in dataset.ground_truth_checks)
        model_items = "".join(
            f"<li>{_escape(item)}</li>" for item in dataset.information_exposed_to_model
        )
        ground_truth = (
            '<div class="ground-truth-panel">'
            "<section><h3>Ground truth known to proof</h3>"
            f"<ul>{gt_items or '<li class=\"muted\">None declared</li>'}</ul></section>"
            "<section><h3>Information available to model</h3>"
            f"<ul>{model_items or '<li class=\"muted\">None declared</li>'}</ul></section>"
            "</div>"
        )
    return f'<div class="card-grid">{cards}</div>{ground_truth}'


def _render_environment(environment: EnvironmentEvidence) -> str:
    if environment.dataset is None:
        return '<p class="section-empty">No environment or dataset context recorded.</p>'
    return _render_dataset(environment.dataset)


def _render_scenario_overview(scenarios: tuple[ScenarioEvidence, ...]) -> str:
    if not scenarios:
        return '<p class="section-empty">No scenarios executed.</p>'
    blocks: list[str] = []
    for scenario in scenarios:
        metrics = ", ".join(f"{m.name}={m.value}" for m in scenario.metrics) or "—"
        blocks.append(
            f"<article class=\"card\"><h3>{_escape(scenario.scenario_id)} — "
            f"{_escape(scenario.title)}</h3>"
            f"<p><strong>Question:</strong> {_escape(scenario.question)}</p>"
            f"<p><strong>Expected:</strong> {_escape(scenario.expected_behavior)}</p>"
            f"<p><strong>Falsification:</strong> {_escape(scenario.falsification_condition)}</p>"
            f"<p><strong>Result:</strong> {_status_badge(scenario.execution_status)}</p>"
            f"<p><strong>Metrics:</strong> {_escape(metrics)}</p>"
            "</article>"
        )
    return '<div class="card-grid">' + "".join(blocks) + "</div>"


def _render_trace_step_row(step: ProofExecutionStep) -> str:
    basis = " ".join(f'<span class="evidence-id">{_escape(i)}</span>' for i in step.evidence_basis_ids) or "—"
    created = " ".join(f'<span class="evidence-id">{_escape(i)}</span>' for i in step.evidence_created_ids) or "—"
    input_html = _render_report_safe_payload(step.input) if step.input else "<span class=\"muted\">—</span>"
    observation_html = (
        _render_report_safe_payload(step.observation)
        if step.observation
        else "<span class=\"muted\">—</span>"
    )
    return (
        "<tr>"
        f"<td>{step.step_index}</td>"
        f"<td>{_escape(_render_report_safe_text(step.purpose))}</td>"
        f"<td>{basis}</td>"
        f"<td>{_escape(_render_report_safe_text(step.action))}</td>"
        f"<td>{input_html}</td>"
        f"<td>{observation_html}</td>"
        f"<td>{created}</td>"
        f"<td>{_step_status_badge(step.status)}</td>"
        "</tr>"
    )


def _render_trace_step_card(step: ProofExecutionStep) -> str:
    basis = ", ".join(step.evidence_basis_ids) or "—"
    created = ", ".join(step.evidence_created_ids) or "—"
    return (
        f'<div class="trace-card"><h3>Step {step.step_index}</h3>'
        f"<dl>"
        f"<dt>Purpose</dt><dd>{_escape(_render_report_safe_text(step.purpose))}</dd>"
        f"<dt>Evidence basis</dt><dd>{_escape(basis)}</dd>"
        f"<dt>Action</dt><dd>{_escape(_render_report_safe_text(step.action))}</dd>"
        f"<dt>Input</dt><dd>{_render_report_safe_payload(step.input) if step.input else '—'}</dd>"
        f"<dt>Observation</dt><dd>{_render_report_safe_payload(step.observation) if step.observation else '—'}</dd>"
        f"<dt>Evidence created</dt><dd>{_escape(created)}</dd>"
        f"<dt>Status</dt><dd>{_step_status_badge(step.status)}</dd>"
        "</dl></div>"
    )


def _collect_trace_steps(evidence: PlatformProofEvidence) -> tuple[ProofExecutionStep, ...]:
    steps: list[ProofExecutionStep] = []
    for scenario in evidence.scenarios:
        steps.extend(scenario.steps)
    steps.sort(key=lambda step: step.step_index)
    return tuple(steps)


def _render_execution_timeline(evidence: PlatformProofEvidence) -> str:
    steps = _collect_trace_steps(evidence)
    if not steps:
        return '<p class="section-empty">No execution steps recorded.</p>'
    table_rows = "".join(_render_trace_step_row(step) for step in steps)
    cards = "".join(_render_trace_step_card(step) for step in steps)
    return (
        '<div class="trace-desktop"><table><thead><tr>'
        "<th>Step</th><th>Purpose</th><th>Evidence basis</th><th>Action</th>"
        "<th>Input</th><th>Observation</th><th>Evidence created</th><th>Status</th>"
        "</tr></thead><tbody>"
        f"{table_rows}</tbody></table></div>"
        f'<div class="trace-mobile">{cards}</div>'
    )


def _render_evidence_graph(graph: EvidenceGraphEvidence) -> str:
    if not graph.nodes:
        return '<p class="section-empty">No evidence graph nodes.</p>'
    node_rows = []
    for node in graph.nodes:
        node_rows.append(
            "<tr>"
            f"<td><span class=\"evidence-id\">{_escape(node.evidence_id)}</span></td>"
            f"<td>{_escape(node.kind.value)}</td>"
            f"<td>{_escape(node.label)}</td>"
            f"<td>{_escape(_render_report_safe_text(node.summary))}</td>"
            f"<td>{_escape(node.producing_step_id or '—')}</td>"
            "</tr>"
        )
    edge_rows = []
    for edge in graph.edges:
        target = edge.to_evidence_id or edge.to_step_id or "—"
        edge_rows.append(
            "<tr>"
            f"<td>{_escape(edge.from_evidence_id)}</td>"
            f"<td>{_escape(target)}</td>"
            f"<td>{_escape(edge.relationship.value)}</td>"
            "</tr>"
        )
    diagram_nodes = []
    for index, node in enumerate(graph.nodes[:8]):
        y = 30 + index * 36
        diagram_nodes.append(
            f'<rect x="20" y="{y - 14}" width="200" height="28" rx="4" fill="#eef2ff" stroke="#c7d2fe"/>'
            f'<text x="28" y="{y + 4}" font-size="11">{_escape(node.label)}</text>'
        )
    diagram_height = max(80, len(graph.nodes[:8]) * 36 + 20)
    diagram = (
        f'<svg class="graph-diagram" viewBox="0 0 260 {diagram_height}" role="img" '
        f'aria-labelledby="graph-diagram-title">'
        '<title id="graph-diagram-title">Evidence graph</title>'
        + "".join(diagram_nodes)
        + "</svg>"
    )
    return (
        diagram
        + "<table><thead><tr><th>ID</th><th>Kind</th><th>Label</th><th>Summary</th><th>Step</th>"
        "</tr></thead><tbody>"
        + "".join(node_rows)
        + "</tbody></table>"
        + "<h3>Relationships</h3>"
        + (
            "<table><thead><tr><th>From</th><th>To</th><th>Relationship</th></tr></thead><tbody>"
            + "".join(edge_rows)
            + "</tbody></table>"
            if edge_rows
            else '<p class="section-empty">No graph edges.</p>'
        )
    )


def _render_final_output(final_output: FinalOutputEvidence | None) -> str:
    if final_output is None or not final_output.present:
        return '<p class="section-empty">No final output recorded.</p>'
    if not final_output.report_safe:
        return '<p class="section-empty">Final output not marked report-safe.</p>'
    content = _render_report_safe_text(final_output.content)
    basis = ", ".join(final_output.evidence_basis_ids) or "—"
    return (
        '<div class="final-output-panel">'
        "<h3>Model / proof final output</h3>"
        f"<pre>{_escape(content)}</pre>"
        f"<p class=\"muted\">Evidence basis: {_escape(basis)}</p>"
        "<p class=\"muted\">This section is not the evaluator verdict.</p>"
        "</div>"
    )


def _render_evaluator_checks(checks: tuple[EvaluatorCheckEvidence, ...]) -> str:
    if not checks:
        return "<p class=\"muted\">No explicit checks recorded.</p>"
    rows = []
    for check in checks:
        result = "pass" if check.passed else "fail"
        evidence_refs = ", ".join(check.evidence_ids) or "—"
        rows.append(
            "<tr>"
            f"<td>{_escape(check.label)}</td>"
            f"<td>{_escape(result)}</td>"
            f"<td>{_escape(_render_report_safe_text(check.explanation))}</td>"
            f"<td>{_escape(evidence_refs)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Check</th><th>Result</th><th>Explanation</th><th>Evidence</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_evaluator(evaluator: EvaluatorSummaryEvidence | None) -> str:
    if evaluator is None:
        return '<p class="section-empty">No evaluator summary recorded.</p>'
    css = "pass" if evaluator.passed else "fail"
    reasons = ""
    if evaluator.failure_reasons:
        reasons = "<p><strong>Failure reasons:</strong> " + _escape(
            ", ".join(evaluator.failure_reasons)
        ) + "</p>"
    return (
        f'<div class="verdict-panel {css}">'
        "<h3>Evaluator verdict</h3>"
        f"<p><strong>Overall:</strong> {'passed' if evaluator.passed else 'not passed'}</p>"
        f"{reasons}"
        f"{_render_evaluator_checks(evaluator.checks)}"
        "</div>"
    )


def _render_failure(failure: FailureEvidence | None, status: ProofEvidenceExecutionStatus) -> str:
    if status == ProofEvidenceExecutionStatus.PASS or failure is None:
        return '<p class="section-empty">No failure analysis — execution passed.</p>'
    milestones = "".join(f"<li>{_escape(item)}</li>" for item in failure.completed_milestones)
    skipped = "".join(f"<li>{_escape(item)}</li>" for item in failure.skipped_not_reached)
    diagnostic = ""
    if failure.safe_diagnostic is not None:
        diagnostic = (
            "<p><strong>Diagnostic:</strong> "
            f"{_escape(_render_report_safe_text(failure.safe_diagnostic))}</p>"
        )
    return (
        '<div class="failure-panel">'
        f"<p><strong>Classification:</strong> {_escape(failure.classification.value)}</p>"
        f"<p><strong>Boundary:</strong> {_escape(failure.boundary or '—')}</p>"
        f"<p><strong>Message:</strong> {_escape(_render_report_safe_text(failure.message))}</p>"
        f"<p><strong>Failed milestone:</strong> {_escape(failure.failed_milestone or '—')}</p>"
        f"<h3>Completed milestones</h3><ul>{milestones or '<li class=\"muted\">None</li>'}</ul>"
        f"<h3>Not reached</h3><ul>{skipped or '<li class=\"muted\">None</li>'}</ul>"
        f"{diagnostic}"
        "</div>"
    )


def _render_limitations(limitations: tuple[str, ...]) -> str:
    if not limitations:
        return '<p class="section-empty">No execution-specific limitations declared.</p>'
    return "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in limitations) + "</ul>"


def _render_conclusion(conclusion: ConclusionEvidence) -> str:
    parts: list[str] = []
    if conclusion.supported_conclusions:
        parts.append(
            "<h3>Supported</h3><ul>"
            + "".join(f"<li>{_escape(item)}</li>" for item in conclusion.supported_conclusions)
            + "</ul>"
        )
    if conclusion.unsupported_conclusions:
        parts.append(
            "<h3>Not supported</h3><ul>"
            + "".join(f"<li>{_escape(item)}</li>" for item in conclusion.unsupported_conclusions)
            + "</ul>"
        )
    if conclusion.open_questions:
        parts.append(
            "<h3>Open questions</h3><ul>"
            + "".join(f"<li>{_escape(item)}</li>" for item in conclusion.open_questions)
            + "</ul>"
        )
    if not parts:
        return '<p class="section-empty">No conclusion synthesis recorded.</p>'
    return "".join(parts)


def _render_reproduction(reproduction: ReproductionEvidence) -> str:
    prereq = "".join(f"<li>{_escape(item)}</li>" for item in reproduction.prerequisites) or (
        "<li class=\"muted\">None declared</li>"
    )
    env_names = ", ".join(reproduction.required_env_variable_names) or "—"
    fingerprint = reproduction.dataset_fingerprint_sha256 or "—"
    return (
        f"<p><strong>Source revision:</strong> {_escape(reproduction.source_revision)}</p>"
        f"<p><strong>Profile env vars:</strong> {_escape(env_names)}</p>"
        f"<p><strong>Dataset fingerprint:</strong> {_escape(fingerprint)}</p>"
        "<h3>Prerequisites</h3>"
        f"<ul>{prereq}</ul>"
        "<h3>Command</h3>"
        f"<pre>{_escape(reproduction.command)}</pre>"
    )


def _render_provenance(provenance: ProvenanceEvidence, identity: ProofIdentityEvidence) -> str:
    return (
        "<table><tbody>"
        f"<tr><th scope=\"row\">proof_id</th><td>{_escape(provenance.proof_id)}</td></tr>"
        f"<tr><th scope=\"row\">execution_id</th><td>{_escape(provenance.execution_id)}</td></tr>"
        f"<tr><th scope=\"row\">source revision</th><td>{_escape(provenance.source_revision)}</td></tr>"
        f"<tr><th scope=\"row\">evidence schema</th><td>{_escape(provenance.evidence_schema_version)}</td></tr>"
        f"<tr><th scope=\"row\">report schema</th><td>{_escape(PLATFORM_PROOF_REPORT_SCHEMA_VERSION)}</td></tr>"
        f"<tr><th scope=\"row\">report standard</th><td>{_escape(PLATFORM_PROOF_REPORT_STANDARD_VERSION)}</td></tr>"
        f"<tr><th scope=\"row\">renderer version</th><td>{_escape(PLATFORM_PROOF_HTML_RENDERER_VERSION)}</td></tr>"
        f"<tr><th scope=\"row\">evidence checksum</th><td>{_escape(provenance.evidence_checksum or '—')}</td></tr>"
        f"<tr><th scope=\"row\">artifact identity</th><td>{_escape(provenance.artifact_identity)}</td></tr>"
        f"<tr><th scope=\"row\">generated at</th><td>{_escape(provenance.generated_at.isoformat())}</td></tr>"
        f"<tr><th scope=\"row\">execution profile</th><td>{_escape(identity.execution_profile.value)}</td></tr>"
        "</tbody></table>"
    )


def _render_domain_extension_scalar_summary(
    extension: ToolsSqlInvestigationExtension,
) -> str:
    """Bounded generic summary — no raw SQL strings."""
    items = [
        ("Extension ID", extension.extension_id),
        ("Successful tool calls", str(extension.successful_tool_calls)),
        ("Investigation proof steps", str(extension.investigation_proof_step_count)),
        ("Stop reason", extension.stop_reason or "—"),
    ]
    if extension.follow_up_has_valid_basis is not None:
        items.append(
            ("Follow-up valid basis", "yes" if extension.follow_up_has_valid_basis else "no")
        )
    rows = "".join(
        f"<tr><th scope=\"row\">{_escape(label)}</th><td>{_escape(value)}</td></tr>"
        for label, value in items
    )
    return (
        "<h3>Domain-specific evidence</h3>"
        "<table><tbody>"
        f"{rows}"
        "</tbody></table>"
    )


def _render_domain_extensions(domain_extension: DomainExtensionEvidence) -> str:
    """Reserved hook for future domain section renderers (PP-REPORT-4+)."""
    if domain_extension.tools is not None:
        return _render_domain_extension_scalar_summary(domain_extension.tools)
    return ""


def _render_report_identity(evidence: PlatformProofEvidence) -> str:
    identity = evidence.proof_identity
    execution = evidence.execution
    cards = (
        f'<div class="card"><div class="card-label">Proof ID</div>'
        f'<div class="card-value">{_escape(identity.proof_id)}</div></div>'
        f'<div class="card"><div class="card-label">Domain</div>'
        f'<div class="card-value">{_escape(identity.domain)}</div></div>'
        f'<div class="card"><div class="card-label">Status</div>'
        f'<div class="card-value">{_status_badge(execution.status)}</div></div>'
        f'<div class="card"><div class="card-label">Revision</div>'
        f'<div class="card-value">{_escape(identity.source_revision)}</div></div>'
        f'<div class="card"><div class="card-label">Platform</div>'
        f'<div class="card-value">{_escape(execution.platform)}</div></div>'
        f'<div class="card"><div class="card-label">Started</div>'
        f'<div class="card-value">{_escape(execution.started_at.isoformat())}</div></div>'
    )
    return (
        f"<h1>{_escape(identity.title)}</h1>"
        f'<p class="muted">Platform Proof Report · standard {_escape(PLATFORM_PROOF_REPORT_STANDARD_VERSION)}</p>'
        f'<div class="card-grid">{cards}</div>'
    )


def render_platform_proof_report(evidence: PlatformProofEvidence) -> str:
    """Render self-contained HTML for the given typed evidence."""
    sections: list[tuple[str, str, Callable[[], str]]] = [
        ("report-identity", "Report identity", lambda: _render_report_identity(evidence)),
        ("executive-summary", "Executive summary", lambda: _render_executive_summary(evidence)),
        ("claim-under-test", "Claim under test", lambda: _render_claim(evidence.claim)),
        ("excluded-claims", "What this proof does not prove", lambda: _render_excluded_claims(evidence.claim)),
        (
            "architecture-under-proof",
            "Architecture under proof",
            lambda: _render_architecture_diagram(evidence.architecture),
        ),
        ("participants", "Participants / components", lambda: _render_participants(evidence.participants)),
        ("data-environment", "Data / environment", lambda: _render_environment(evidence.environment)),
        ("scenario-overview", "Scenario overview", lambda: _render_scenario_overview(evidence.scenarios)),
        ("execution-timeline", "Execution timeline", lambda: _render_execution_timeline(evidence)),
        ("evidence-graph", "Evidence graph", lambda: _render_evidence_graph(evidence.evidence_graph)),
        ("final-output", "Final output", lambda: _render_final_output(evidence.final_output)),
        ("evaluator-verdict", "Evaluator verdict", lambda: _render_evaluator(evidence.evaluator)),
        (
            "failure-analysis",
            "Failure analysis",
            lambda: _render_failure(evidence.failure, evidence.execution.status),
        ),
        ("limitations", "Limitations", lambda: _render_limitations(evidence.limitations)),
        ("conclusion", "Conclusion", lambda: _render_conclusion(evidence.conclusion)),
        ("reproduction", "Reproduction", lambda: _render_reproduction(evidence.reproduction)),
        ("provenance", "Provenance", lambda: _render_provenance(evidence.provenance, evidence.proof_identity)),
    ]
    body_parts: list[str] = []
    for section_id, title, renderer in sections:
        body_parts.append(
            f'<section id="{section_id}" aria-labelledby="{section_id}-heading">'
            f'<h2 id="{section_id}-heading">{_escape(title)}</h2>'
            f"{renderer()}"
            "</section>"
        )
    domain_html = _render_domain_extensions(evidence.domain_extension)
    if domain_html:
        body_parts.append(
            f'<section id="domain-extension" aria-labelledby="domain-extension-heading">'
            f"{domain_html}</section>"
        )
    document = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Platform Proof Report — {_escape(evidence.proof_identity.proof_id)}</title>\n"
        f"<style>\n{_css()}\n</style>\n"
        "</head>\n<body>\n<main>\n"
        + "\n".join(body_parts)
        + "\n</main>\n</body>\n</html>\n"
    )
    return document


def write_platform_proof_report(
    evidence: PlatformProofEvidence,
    *,
    output_path: Path,
) -> Path:
    """Write self-contained report HTML to ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_platform_proof_report(evidence), encoding="utf-8")
    return output_path


def assert_no_external_report_dependencies(html_content: str) -> None:
    """Test helper — detect renderer-emitted remote dependencies."""
    if _EXTERNAL_STYLESHEET_RE.search(html_content):
        raise PlatformProofReportRenderError("external stylesheet reference detected")
    if _EXTERNAL_SCRIPT_RE.search(html_content):
        raise PlatformProofReportRenderError("external script reference detected")
    if _REMOTE_IMPORT_RE.search(html_content):
        raise PlatformProofReportRenderError("remote CSS import detected")
    if _REMOTE_IMAGE_RE.search(html_content):
        raise PlatformProofReportRenderError("remote image reference detected")
