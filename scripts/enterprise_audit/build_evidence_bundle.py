#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Build closed evidence bundle directory from audit output (C050)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from intergrax.platform.enterprise_audit.evidence import build_evidence_manifest
from intergrax.platform.enterprise_audit.validators import verify_closed_evidence_bundle


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qualification-sha", required=True)
    args = parser.parse_args()
    report_path = args.output_dir / "enterprise_audit_report.json"
    if not report_path.is_file():
        print("missing enterprise_audit_report.json", file=sys.stderr)
        return 2
    import json

    from intergrax.platform.enterprise_audit.contracts import AuditExecutionManifest, AuditMode

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    em = payload.get("execution_manifest", {})
    manifest = AuditExecutionManifest(
        audit_mode=AuditMode(em.get("audit_mode", "FULL")),
        qualification_sha=str(em.get("qualification_sha", args.qualification_sha)),
        audit_execution_sha=str(em.get("audit_execution_sha", args.qualification_sha)),
        checks_planned=int(em.get("checks_planned", 0)),
        checks_executed=int(em.get("checks_executed", 0)),
        checks_skipped=int(em.get("checks_skipped", 0)),
        coverage_label=str(em.get("coverage_label", "PARTIAL")),
        certifying_run=bool(em.get("certifying_run", True)),
    )
    build_evidence_manifest(
        args.output_dir,
        run_id=args.output_dir.name,
        qualification_sha=args.qualification_sha,
        execution_manifest=manifest,
    )
    errors = verify_closed_evidence_bundle(args.output_dir)
    if errors:
        print("; ".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
