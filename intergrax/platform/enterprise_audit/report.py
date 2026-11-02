# © Artur Czarnecki. All rights reserved.

"""JSON and Markdown audit report builders."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import (
    AuditExecutionManifest,
    AuditReport,
    CheckRunOutcome,
)


def _finding_dict(finding) -> dict:
    raw = asdict(finding)
    raw["severity"] = finding.severity.value
    raw["category"] = finding.category.value
    if finding.drift is not None:
        raw["drift"] = finding.drift.value
    else:
        raw.pop("drift", None)
    return raw


def build_report_json(report: AuditReport, manifest: AuditExecutionManifest) -> dict:
    return {
        "status": report.status.value,
        "exit_code": int(report.exit_code),
        "audit_mode": report.audit_mode.value,
        "current_sha": report.current_sha,
        "baseline_sha": report.baseline_sha,
        "limitations": list(report.limitations),
        "coverage_matrix_summary": report.coverage_matrix_summary,
        "findings": [_finding_dict(f) for f in report.findings],
        "execution_manifest": {
            "schema_version": manifest.schema_version,
            "audit_mode": manifest.audit_mode.value,
            "qualification_sha": manifest.qualification_sha,
            "audit_execution_sha": manifest.audit_execution_sha,
            "checks_planned": manifest.checks_planned,
            "checks_executed": manifest.checks_executed,
            "checks_skipped": manifest.checks_skipped,
            "coverage_label": manifest.coverage_label,
        },
    }


def write_reports(
    output_dir: Path,
    report: AuditReport,
    manifest: AuditExecutionManifest,
    outcomes: tuple[CheckRunOutcome, ...],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "enterprise_audit_report.json"
    md_path = output_dir / "enterprise_audit_report.md"
    payload = build_report_json(report, manifest)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report, manifest), encoding="utf-8")
    detail_path = output_dir / "enterprise_audit_outcomes.json"
    detail_path.write_text(
        json.dumps([asdict(o) for o in outcomes], indent=2) + "\n",
        encoding="utf-8",
    )
    return json_path, md_path


def render_markdown(report: AuditReport, manifest: AuditExecutionManifest) -> str:
    lines = [
        "# Enterprise audit report",
        "",
        f"- **Status:** {report.status.value}",
        f"- **Exit code:** {int(report.exit_code)}",
        f"- **Mode:** {report.audit_mode.value}",
        f"- **Current SHA:** `{report.current_sha}`",
        f"- **Baseline SHA:** `{report.baseline_sha or 'none'}`",
        f"- **Checks planned/executed/skipped:** "
        f"{manifest.checks_planned}/{manifest.checks_executed}/{manifest.checks_skipped}",
        "",
        "## Qualification limitations",
        "",
    ]
    for item in report.limitations:
        lines.append(f"- {item}")
    lines.extend(["", "## Findings", ""])
    if not report.findings:
        lines.append("_No findings._")
    else:
        for finding in report.findings:
            drift = f" ({finding.drift.value})" if finding.drift else ""
            lines.append(f"- **{finding.finding_id}** [{finding.severity.value}]{drift}: {finding.message}")
    lines.append("")
    return "\n".join(lines)
