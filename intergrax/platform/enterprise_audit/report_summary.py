# © Artur Czarnecki. All rights reserved.

"""Console summary after enterprise audit run (C4-D)."""

from __future__ import annotations

import sys
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import AuditExecutionManifest, AuditReport, AuditSeverity


def print_run_summary(
    report: AuditReport,
    manifest: AuditExecutionManifest,
    *,
    output_dir: Path,
    duration_ms: int = 0,
    certifying_run: bool = True,
) -> None:
    blocking = {AuditSeverity.BLOCKER, AuditSeverity.ERROR}
    failed = sum(
        1
        for o in manifest.outcomes
        if o.executed and (o.exit_code != 0 or any(f.severity in blocking for f in o.findings))
    )
    top = sorted(
        report.findings,
        key=lambda f: (
            0 if f.severity is AuditSeverity.BLOCKER else 1 if f.severity is AuditSeverity.ERROR else 2,
            f.finding_id,
        ),
    )[:5]
    json_path = output_dir / "enterprise_audit_report.json"
    md_path = output_dir / "enterprise_audit_report.md"
    bundle = output_dir / "evidence_manifest.v1.json"
    print("--- Enterprise audit summary ---", file=sys.stdout)
    print(f"Mode: {report.audit_mode.value}  certifying_run={certifying_run}", file=sys.stdout)
    print(f"Git SHA: {report.current_sha}", file=sys.stdout)
    print(f"Status: {report.status.value}  exit_code={int(report.exit_code)}", file=sys.stdout)
    counts = f"{manifest.checks_executed}/{manifest.checks_skipped}/{manifest.checks_planned}"
    print(f"Checks executed/skipped/planned: {counts}", file=sys.stdout)
    print(f"Checks failed (executed): {failed}", file=sys.stdout)
    print(f"Coverage label: {manifest.coverage_label}", file=sys.stdout)
    if duration_ms:
        print(f"Duration ms: {duration_ms}", file=sys.stdout)
    if top:
        print("Top findings:", file=sys.stdout)
        for f in top:
            print(f"  - [{f.severity.value}] {f.finding_id}: {f.message[:120]}", file=sys.stdout)
    print(f"Report JSON: {json_path}", file=sys.stdout)
    print(f"Report Markdown: {md_path}", file=sys.stdout)
    if bundle.is_file():
        print(f"Evidence bundle: {output_dir}", file=sys.stdout)
    if report.audit_mode.value == "FAST":
        print("Note: FAST is not full enterprise certification.", file=sys.stdout)
