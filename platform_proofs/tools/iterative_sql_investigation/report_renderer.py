# © Artur Czarnecki. All rights reserved.
# Intergrax platform proof — TOOLS-ITERATIVE-SQL-INVESTIGATION (PP-REPORT-4).

"""TOOLS specialized HTML report renderer — reference domain proof report."""

from __future__ import annotations

from pathlib import Path

from scripts.proof.intergrax_platform_proof_evidence import (
    EvidenceGraphEvidence,
    PlatformProofEvidence,
    ProofEvidenceExecutionStatus,
    ProofExecutionStep,
    ReportSafeVisibility,
    ScenarioEvidence,
    ToolsSqlInvestigationExtension,
)
from scripts.proof.intergrax_platform_proof_html_renderer import (
    REPORT_FILENAME,
    RenderedReportSection,
    escape_report_html,
    render_evidence_id_badge,
    render_platform_proof_report,
    render_report_safe_payload,
    render_report_safe_text,
    render_step_status_badge,
)

_TOOLS_CSS = """
.tools-overview-grid { margin-bottom: 1rem; }
.tools-timeline { display: flex; flex-direction: column; gap: 0.5rem; }
.tools-timeline-step {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.75rem 1rem;
  background: var(--surface);
}
.tools-timeline-step.tool-call { border-left: 4px solid var(--accent); }
.tools-timeline-arrow {
  text-align: center;
  color: var(--muted);
  font-size: 0.85rem;
  padding: 0.15rem 0;
}
.tools-flow-diagram { margin: 0.75rem 0; }
.tools-scenario-flow {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.9rem 1rem;
  background: var(--surface);
  margin-bottom: 0.75rem;
}
.tools-scenario-flow h4 { margin: 0 0 0.5rem; }
.tools-scenario-flow.pass { border-left: 4px solid var(--pass); }
.tools-scenario-flow.fail { border-left: 4px solid var(--fail); }
.tools-verdict-banner {
  border-radius: 8px;
  padding: 0.85rem 1rem;
  margin-bottom: 1rem;
  font-weight: 600;
}
.tools-verdict-banner.pass { background: var(--pass-bg); color: var(--pass); border: 1px solid #86efac; }
.tools-verdict-banner.fail { background: var(--fail-bg); color: var(--fail); border: 1px solid #fecaca; }
.tools-verdict-banner.blocked { background: var(--blocked-bg); color: var(--blocked); border: 1px solid #fde68a; }
.tools-verdict-banner.crash { background: var(--crash-bg); color: var(--crash); border: 1px solid #fca5a5; }
.tools-ground-truth {
  display: grid;
  gap: 0.75rem;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  margin: 0.75rem 0;
}
.tools-ground-truth section {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.75rem;
  background: var(--surface);
}
.tools-ground-truth .proof-panel { border-left: 4px solid #b45309; }
.tools-ground-truth .model-panel { border-left: 4px solid var(--accent); }
.tools-tool-table { font-size: 0.88rem; }
.tools-tool-table td pre { max-height: 8rem; }
""".strip()

_SCENARIO_FLOW_LABELS: dict[str, tuple[str, ...]] = {
    "A": (
        "Region delay comparison",
        "Absolute delayed counts",
        "Normalize by volume",
        "Hub / service slice",
        "Anomaly isolated",
        "Exclusion / baseline check",
    ),
    "B": (
        "Observed weight–delay correlation",
        "Segmented control checks",
        "Alternative explanations",
        "Refuse unsupported causality",
    ),
    "C": (
        "Staffing question posed",
        "Staffing evidence unavailable in schema",
        "Bounded limitation / refusal",
        "No fabricated cause",
    ),
}


def _tools_extension(evidence: PlatformProofEvidence) -> ToolsSqlInvestigationExtension | None:
    return evidence.domain_extension.tools


def _llm_participant_label(evidence: PlatformProofEvidence) -> str:
    for participant in evidence.participants:
        if participant.participant_id == "llm-provider":
            return f"{participant.implementation}/{participant.version_or_model}"
    return "—"


def _render_overview_cards(evidence: PlatformProofEvidence) -> str:
    tools = _tools_extension(evidence)
    dataset = evidence.environment.dataset
    fingerprint = dataset.fingerprint_sha256 if dataset else "—"
    row_count = str(dataset.row_count) if dataset else "—"
    cards: list[str] = []
    if tools is not None:
        cards.extend(
            [
                ("Successful tool calls", str(tools.successful_tool_calls)),
                ("Investigation proof steps", str(tools.investigation_proof_step_count)),
                ("Stop reason", tools.stop_reason or "—"),
                (
                    "Follow-up evidence basis",
                    "valid"
                    if tools.follow_up_has_valid_basis
                    else ("invalid" if tools.follow_up_has_valid_basis is False else "—"),
                ),
            ]
        )
    cards.extend(
        [
            ("Scenarios executed", str(len(evidence.scenarios))),
            (
                "Evaluator",
                "PASS" if evidence.evaluator and evidence.evaluator.passed else "FAIL",
            ),
            ("Model / provider", _llm_participant_label(evidence)),
            ("PostgreSQL", "docker fixture"),
            ("Dataset fingerprint", fingerprint[:16] + "…" if len(fingerprint) > 16 else fingerprint),
            ("Dataset rows", row_count),
        ]
    )
    return (
        '<div class="card-grid tools-overview-grid">'
        + "".join(
            f'<div class="card"><div class="card-label">{escape_report_html(label)}</div>'
            f'<div class="card-value">{escape_report_html(value)}</div></div>'
            for label, value in cards
        )
        + "</div>"
    )


def _render_topology_diagram(evidence: PlatformProofEvidence) -> str:
    """Inline SVG showing the iterative tool feedback loop."""
    participants = {p.participant_id: p.name for p in evidence.participants}
    llm = participants.get("llm-provider", "LLM / planner")
    runtime = participants.get("intergrax-runtime", "Bounded tool loop")
    sql_tool = participants.get("sql-tool", "SQL tool")
    postgres = participants.get("postgres-fixture", "PostgreSQL")
    width, height = 520, 220
    boxes = [
        (20, 20, llm),
        (20, 70, runtime),
        (20, 120, sql_tool),
        (20, 170, postgres),
    ]
    nodes = []
    for x, y, label in boxes:
        nodes.append(
            f'<rect x="{x}" y="{y}" width="200" height="36" rx="6" fill="#f8fafc" stroke="#94a3b8"/>'
            f'<text x="{x + 8}" y="{y + 22}" font-size="11" fill="#1e293b">'
            f"{escape_report_html(label)}</text>"
        )
    forward = (
        '<line x1="220" y1="38" x2="280" y2="38" stroke="#64748b" marker-end="url(#tools-arrow)"/>'
        '<line x1="280" y1="38" x2="280" y2="188" stroke="#64748b"/>'
        '<line x1="280" y1="88" x2="220" y2="88" stroke="#64748b" marker-end="url(#tools-arrow)"/>'
        '<line x1="280" y1="138" x2="220" y2="138" stroke="#64748b" marker-end="url(#tools-arrow)"/>'
        '<line x1="280" y1="188" x2="220" y2="188" stroke="#64748b" marker-end="url(#tools-arrow)"/>'
        '<text x="290" y="60" font-size="10" fill="#475569">plan</text>'
        '<text x="290" y="110" font-size="10" fill="#475569">invoke</text>'
        '<text x="290" y="160" font-size="10" fill="#475569">read SQL</text>'
    )
    feedback = (
        '<path d="M 320 188 C 400 188, 400 38, 320 38" fill="none" stroke="#2563eb" '
        'stroke-width="2" marker-end="url(#tools-arrow-blue)"/>'
        '<text x="360" y="115" font-size="10" fill="#2563eb">observation → next action</text>'
    )
    return (
        f'<svg class="arch-diagram tools-flow-diagram" viewBox="0 0 {width} {height}" '
        f'role="img" aria-labelledby="tools-topology-title">'
        '<title id="tools-topology-title">Tool execution topology with feedback loop</title>'
        '<defs>'
        '<marker id="tools-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
        '<path d="M0,0 L6,3 L0,6 Z" fill="#64748b"/></marker>'
        '<marker id="tools-arrow-blue" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
        '<path d="M0,0 L6,3 L0,6 Z" fill="#2563eb"/></marker>'
        "</defs>"
        + "".join(nodes)
        + forward
        + feedback
        + "</svg>"
    )


def _render_step_chain_arrow(basis_ids: tuple[str, ...]) -> str:
    if not basis_ids:
        return ""
    badges = " ".join(render_evidence_id_badge(item) for item in basis_ids)
    return (
        f'<div class="tools-timeline-arrow">↓ uses {badges}</div>'
    )


def _render_investigation_timeline(evidence: PlatformProofEvidence) -> str:
    if not evidence.scenarios:
        return '<p class="section-empty">No investigation steps recorded.</p>'
    blocks: list[str] = []
    for scenario in evidence.scenarios:
        if not scenario.steps:
            continue
        blocks.append(
            f'<h3>Scenario {escape_report_html(scenario.scenario_id)}</h3>'
        )
        for step in scenario.steps:
            blocks.append(_render_timeline_step(step))
            if step.evidence_basis_ids:
                blocks.append(_render_step_chain_arrow(step.evidence_basis_ids))
    if not blocks:
        return '<p class="section-empty">No tool-call steps in scenarios.</p>'
    return '<div class="tools-timeline">' + "".join(blocks) + "</div>"


def _render_timeline_step(step: ProofExecutionStep) -> str:
    is_tool = step.tool_invocation is not None
    css = "tools-timeline-step tool-call" if is_tool else "tools-timeline-step"
    basis = (
        " ".join(render_evidence_id_badge(item) for item in step.evidence_basis_ids)
        or "—"
    )
    created = (
        " ".join(render_evidence_id_badge(item) for item in step.evidence_created_ids)
        or "—"
    )
    observation = (
        render_report_safe_payload(step.observation)
        if step.observation
        else '<span class="muted">—</span>'
    )
    tool_line = ""
    if step.tool_invocation is not None:
        tool_line = (
            f"<p><strong>Tool:</strong> "
            f"{escape_report_html(step.tool_invocation.tool_id)} "
            f"({escape_report_html(step.tool_invocation.call_id)})</p>"
        )
    return (
        f'<article class="{css}">'
        f"<h4>Step {step.step_index + 1}</h4>"
        f"<p><strong>Purpose:</strong> {escape_report_html(render_report_safe_text(step.purpose))}</p>"
        f"<p><strong>Evidence basis:</strong> {basis}</p>"
        f"<p><strong>Action:</strong> {escape_report_html(render_report_safe_text(step.action))}</p>"
        f"{tool_line}"
        f"<p><strong>Input:</strong> "
        f"{render_report_safe_payload(step.input) if step.input else '<span class=\"muted\">—</span>'}</p>"
        f"<p><strong>Observation:</strong> {observation}</p>"
        f"<p><strong>Evidence created:</strong> {created}</p>"
        f"<p><strong>Status:</strong> {render_step_status_badge(step.status)}</p>"
        "</article>"
    )


def _render_dependency_graph_svg(graph: EvidenceGraphEvidence) -> str:
    if not graph.nodes:
        return ""
    node_height = 32
    width = 480
    height = max(80, len(graph.nodes) * node_height + 40)
    nodes_svg: list[str] = []
    id_to_y: dict[str, int] = {}
    for index, node in enumerate(graph.nodes[:12]):
        y = 24 + index * node_height
        id_to_y[node.evidence_id] = y
        short = node.evidence_id if len(node.evidence_id) <= 22 else node.evidence_id[:19] + "…"
        nodes_svg.append(
            f'<rect x="20" y="{y - 12}" width="200" height="24" rx="4" fill="#eef2ff" stroke="#c7d2fe"/>'
            f'<text x="28" y="{y + 4}" font-size="10">{escape_report_html(short)}</text>'
        )
    edges_svg: list[str] = []
    for edge in graph.edges[:20]:
        from_y = id_to_y.get(edge.from_evidence_id)
        to_id = edge.to_evidence_id or edge.to_step_id
        if from_y is None or to_id is None:
            continue
        to_y = id_to_y.get(to_id, from_y + node_height)
        rel = edge.relationship.value.replace("_", " ")
        edges_svg.append(
            f'<line x1="220" y1="{from_y}" x2="300" y2="{to_y}" stroke="#94a3b8" '
            f'marker-end="url(#dep-arrow)"/>'
            f'<text x="308" y="{(from_y + to_y) // 2}" font-size="9" fill="#64748b">'
            f"{escape_report_html(rel)}</text>"
        )
    return (
        f'<svg class="graph-diagram" viewBox="0 0 {width} {height}" role="img" '
        f'aria-labelledby="tools-dep-graph-title">'
        '<title id="tools-dep-graph-title">Evidence dependency graph</title>'
        '<defs><marker id="dep-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" '
        'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8"/></marker></defs>'
        + "".join(nodes_svg)
        + "".join(edges_svg)
        + "</svg>"
    )


def _render_dependency_graph(evidence: PlatformProofEvidence) -> str:
    graph = evidence.evidence_graph
    if not graph.nodes:
        return '<p class="section-empty">No evidence dependency graph recorded.</p>'
    edge_rows = []
    for edge in graph.edges:
        target = edge.to_evidence_id or edge.to_step_id or "—"
        edge_rows.append(
            "<tr>"
            f"<td>{render_evidence_id_badge(edge.from_evidence_id)}</td>"
            f"<td>{escape_report_html(target)}</td>"
            f"<td>{escape_report_html(edge.relationship.value)}</td>"
            "</tr>"
        )
    return (
        _render_dependency_graph_svg(graph)
        + "<table><thead><tr><th>From</th><th>To</th><th>Relationship</th></tr></thead><tbody>"
        + "".join(edge_rows)
        + "</tbody></table>"
    )


def _render_tool_call_table(evidence: PlatformProofEvidence) -> str:
    rows: list[str] = []
    for scenario in evidence.scenarios:
        for step in scenario.steps:
            inv = step.tool_invocation
            if inv is None:
                continue
            sql_preview = "—"
            if inv.safe_arguments is not None:
                for field in inv.safe_arguments.fields:
                    if field.name == "sql" and field.visibility == ReportSafeVisibility.REPORT_SAFE:
                        if field.value is not None and hasattr(field.value, "text"):
                            sql_preview = field.value.text[:200]
            duration = "—"
            rows.append(
                "<tr>"
                f"<td>{escape_report_html(inv.call_id)}</td>"
                f"<td>{escape_report_html(inv.tool_id)}</td>"
                f"<td>{escape_report_html(render_report_safe_text(step.purpose))}</td>"
                f"<td>{' '.join(render_evidence_id_badge(i) for i in step.evidence_basis_ids) or '—'}</td>"
                f"<td><pre>{escape_report_html(sql_preview)}</pre></td>"
                f"<td>{escape_report_html(render_report_safe_text(inv.output_summary) if inv.output_summary else '—')}</td>"
                f"<td>{'pass' if inv.success else 'fail'}</td>"
                f"<td>{duration}</td>"
                f"<td>{' '.join(render_evidence_id_badge(i) for i in step.evidence_created_ids) or '—'}</td>"
                "</tr>"
            )
    if not rows:
        return '<p class="section-empty">No tool invocations recorded.</p>'
    return (
        '<table class="tools-tool-table"><thead><tr>'
        "<th>Call ID</th><th>Tool</th><th>Purpose</th><th>Evidence basis</th>"
        "<th>SQL preview</th><th>Output</th><th>Result</th><th>Duration</th><th>Evidence</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _render_scenario_flow_svg(labels: tuple[str, ...], *, accent: str = "#2563eb") -> str:
    width = 400
    height = 28 * len(labels) + 20
    parts: list[str] = []
    for index, label in enumerate(labels):
        y = 16 + index * 28
        parts.append(
            f'<rect x="20" y="{y - 10}" width="360" height="22" rx="4" fill="#f8fafc" stroke="#d8dee6"/>'
            f'<text x="28" y="{y + 4}" font-size="10" fill="#1e293b">{escape_report_html(label)}</text>'
        )
        if index < len(labels) - 1:
            parts.append(
                f'<line x1="200" y1="{y + 12}" x2="200" y2="{y + 28}" stroke="{accent}" '
                f'marker-end="url(#scenario-arrow)"/>'
            )
    return (
        f'<svg class="graph-diagram" viewBox="0 0 {width} {height}" role="img">'
        '<defs><marker id="scenario-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" '
        'orient="auto"><path d="M0,0 L5,3 L0,6 Z" fill="#2563eb"/></marker></defs>'
        + "".join(parts)
        + "</svg>"
    )


def _scenario_semantic_check(scenario: ScenarioEvidence, check_id: str) -> bool | None:
    if scenario.evaluator is None:
        return None
    for check in scenario.evaluator.checks:
        if check.check_id == check_id:
            return check.passed
    return None


def _render_scenario_visualizations(evidence: PlatformProofEvidence) -> str:
    if not evidence.scenarios:
        return '<p class="section-empty">No scenarios to visualize.</p>'
    blocks: list[str] = []
    for scenario in evidence.scenarios:
        sid = scenario.scenario_id.upper()
        passed = scenario.execution_status == ProofEvidenceExecutionStatus.PASS
        css = "pass" if passed else "fail"
        flow_labels = _SCENARIO_FLOW_LABELS.get(sid, (scenario.expected_behavior,))
        semantic_note = ""
        if sid == "A":
            sem = _scenario_semantic_check(scenario, "scenario_a_semantics")
            if sem is not None:
                semantic_note = (
                    "<p><strong>Outcome:</strong> progressive narrowing to supported North segment; "
                    f"volume-only rejected — check {'passed' if sem else 'failed'}.</p>"
                )
        elif sid == "B":
            sem = _scenario_semantic_check(scenario, "scenario_b_semantics")
            if sem is not None:
                semantic_note = (
                    "<p><strong>Outcome:</strong> correlation examined with controls; "
                    f"unsupported causality refused — check {'passed' if sem else 'failed'}.</p>"
                )
        elif sid == "C":
            sem = _scenario_semantic_check(scenario, "scenario_c_semantics")
            if sem is not None:
                semantic_note = (
                    "<p><strong>Outcome:</strong> missing staffing evidence acknowledged; "
                    f"no fabricated cause — check {'passed' if sem else 'failed'}.</p>"
                )
        blocks.append(
            f'<article class="tools-scenario-flow {css}">'
            f"<h4>Scenario {escape_report_html(sid)} — {escape_report_html(scenario.title)}</h4>"
            f"<p>{escape_report_html(scenario.question)}</p>"
            f"{_render_scenario_flow_svg(flow_labels)}"
            f"{semantic_note}"
            f"<p class=\"muted\"><strong>Falsification:</strong> "
            f"{escape_report_html(scenario.falsification_condition)}</p>"
            "</article>"
        )
    return "".join(blocks)


def _render_ground_truth_panels(evidence: PlatformProofEvidence) -> str:
    dataset = evidence.environment.dataset
    if dataset is None:
        return '<p class="section-empty">No dataset ground-truth separation recorded.</p>'
    gt_items = "".join(f"<li>{escape_report_html(item)}</li>" for item in dataset.ground_truth_checks)
    model_items = "".join(
        f"<li>{escape_report_html(item)}</li>" for item in dataset.information_exposed_to_model
    )
    return (
        '<div class="tools-ground-truth">'
        '<section class="proof-panel"><h3>Ground truth known to proof</h3>'
        f"<ul>{gt_items or '<li class=\"muted\">None declared</li>'}</ul>"
        "<p class=\"muted\">Seeded anomaly truth and verification stats — not given to the model.</p>"
        "</section>"
        '<section class="model-panel"><h3>Information available to model</h3>'
        f"<ul>{model_items or '<li class=\"muted\">None declared</li>'}</ul>"
        "<p class=\"muted\">Only investigation questions and bounded SQL observations.</p>"
        "</section>"
        "</div>"
    )


def _render_evaluator_falsification(evidence: PlatformProofEvidence) -> str:
    status = evidence.execution.status
    banner_css = {
        ProofEvidenceExecutionStatus.PASS: "pass",
        ProofEvidenceExecutionStatus.FAIL: "fail",
        ProofEvidenceExecutionStatus.BLOCKED: "blocked",
        ProofEvidenceExecutionStatus.CRASH: "crash",
    }[status]
    if status == ProofEvidenceExecutionStatus.PASS:
        headline = "WHY THIS PROOF PASSED"
        detail = "All scenario evaluator checks passed under bounded proof conditions."
        if evidence.evaluator and evidence.evaluator.checks:
            passed_labels = [c.label for c in evidence.evaluator.checks if c.passed]
            if passed_labels:
                detail = "Checks passed: " + "; ".join(passed_labels) + "."
    elif status == ProofEvidenceExecutionStatus.BLOCKED:
        headline = "WHY THIS PROOF WAS BLOCKED"
        detail = (
            render_report_safe_text(evidence.failure.message)
            if evidence.failure
            else "Execution blocked."
        )
    elif status == ProofEvidenceExecutionStatus.CRASH:
        headline = "WHY THIS PROOF CRASHED"
        detail = (
            render_report_safe_text(evidence.failure.message)
            if evidence.failure
            else "Runtime crash before completion."
        )
    else:
        headline = "WHY THIS PROOF FAILED"
        reasons = evidence.evaluator.failure_reasons if evidence.evaluator else ()
        detail = "; ".join(reasons) if reasons else "Evaluator checks did not pass."
    banner = (
        f'<div class="tools-verdict-banner {banner_css}">'
        f"<p>{escape_report_html(headline)}</p>"
        f"<p style=\"font-weight:400\">{escape_report_html(detail)}</p>"
        "</div>"
    )
    scenario_blocks: list[str] = []
    for scenario in evidence.scenarios:
        if scenario.evaluator is None:
            continue
        checks_html = []
        for check in scenario.evaluator.checks:
            result = "pass" if check.passed else "fail"
            refs = ", ".join(check.evidence_ids) or "—"
            checks_html.append(
                "<tr>"
                f"<td>{escape_report_html(check.label)}</td>"
                f"<td>{escape_report_html(result)}</td>"
                f"<td>{escape_report_html(render_report_safe_text(check.explanation))}</td>"
                f"<td>{escape_report_html(refs)}</td>"
                "</tr>"
            )
        scenario_blocks.append(
            f"<h3>Scenario {escape_report_html(scenario.scenario_id)} evaluator</h3>"
            "<table><thead><tr><th>Check</th><th>Result</th><th>Explanation</th><th>Evidence</th>"
            "</tr></thead><tbody>"
            + "".join(checks_html)
            + "</tbody></table>"
        )
    return banner + "".join(scenario_blocks)


def _build_domain_sections(evidence: PlatformProofEvidence) -> tuple[RenderedReportSection, ...]:
    return (
        RenderedReportSection(
            section_id="tools-investigation-overview",
            title="Investigation overview",
            html=_render_overview_cards(evidence),
        ),
        RenderedReportSection(
            section_id="tools-execution-topology",
            title="Tool execution topology",
            html=_render_topology_diagram(evidence),
        ),
        RenderedReportSection(
            section_id="tools-ground-truth-separation",
            title="Ground truth vs model information",
            html=_render_ground_truth_panels(evidence),
        ),
        RenderedReportSection(
            section_id="tools-investigation-timeline",
            title="Iterative investigation timeline",
            html=_render_investigation_timeline(evidence),
        ),
        RenderedReportSection(
            section_id="tools-evidence-dependency",
            title="Evidence dependency chain",
            html=_render_dependency_graph(evidence),
        ),
        RenderedReportSection(
            section_id="tools-tool-call-detail",
            title="Tool call detail",
            html=_render_tool_call_table(evidence),
        ),
        RenderedReportSection(
            section_id="tools-scenario-visualizations",
            title="Scenario outcome visualization",
            html=_render_scenario_visualizations(evidence),
        ),
        RenderedReportSection(
            section_id="tools-evaluator-falsification",
            title="Evaluator and falsification summary",
            html=_render_evaluator_falsification(evidence),
        ),
    )


def render_tools_sql_investigation_report(evidence: PlatformProofEvidence) -> str:
    """Render full TOOLS proof report using generic template + domain sections."""
    return render_platform_proof_report(
        evidence,
        domain_sections=_build_domain_sections(evidence),
        extra_css=_TOOLS_CSS,
    )


def write_tools_sql_investigation_report(
    evidence: PlatformProofEvidence,
    *,
    output_path: Path | None = None,
    run_directory: Path | None = None,
) -> Path:
    """Write report.html from typed evidence."""
    if output_path is None:
        if run_directory is None:
            raise ValueError("output_path or run_directory required")
        output_path = run_directory / REPORT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_tools_sql_investigation_report(evidence), encoding="utf-8")
    return output_path
