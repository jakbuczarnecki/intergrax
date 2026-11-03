# © Artur Czarnecki. All rights reserved.

"""JSON and Markdown audit report builders."""

from __future__ import annotations

import json
import os
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
        "certifying_run": report.certifying_run,
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
            "certifying_run": manifest.certifying_run,
            "run_status": manifest.run_status,
            "check_manifest_digest": manifest.check_manifest_digest,
        },
    }


def write_reports(
    output_dir: Path,
    report: AuditReport,
    manifest: AuditExecutionManifest,
    outcomes: tuple[CheckRunOutcome, ...],
    *,
    extra_artifacts: dict | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "enterprise_audit_report.json"
    md_path = output_dir / "enterprise_audit_report.md"
    payload = build_report_json(report, manifest)
    md_body = render_markdown(report, manifest)
    detail_path = output_dir / "enterprise_audit_outcomes.json"
    detail_body = json.dumps([asdict(o) for o in outcomes], indent=2) + "\n"
    _atomic_write(json_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_write(md_path, md_body)
    _atomic_write(detail_path, detail_body)
    if extra_artifacts:
        for name, data in extra_artifacts.items():
            path = output_dir / name
            _atomic_write(path, json.dumps(data, indent=2, sort_keys=True) + "\n")
    return json_path, md_path


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


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
