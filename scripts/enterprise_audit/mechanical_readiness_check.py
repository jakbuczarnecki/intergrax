#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Mechanical B6 readiness checks for ENT-AUDIT-X (no PASS declaration)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    root = _repo_root()
    checks: list[tuple[str, bool, str]] = []
    plan_rel = (
        "docs/project/maintainers/qualification/ent_audit_x/"
        "ENT_AUDIT_X_FINAL_CORRECTIVE_IMPLEMENTATION_AND_B6_READINESS_PLAN.md"
    )
    checks.append(("corrective_plan_doc", (root / plan_rel).is_file(), "corrective master plan missing"))
    checks.append(("audit_bat", (root / "scripts/enterprise_audit/audit.bat").is_file(), "audit.bat missing"))
    checks.append(("audit_sh", (root / "scripts/enterprise_audit/audit.sh").is_file(), "audit.sh missing"))
    checks.append(
        (
            "pr_workflow",
            (root / ".github/workflows/enterprise-audit.yml").is_file(),
            "enterprise-audit.yml missing",
        )
    )
    checks.append(
        (
            "evidence_catalog",
            (root / "intergrax/platform/enterprise_audit/data/evidence_artifact_catalog.v1.json").is_file(),
            "evidence catalog missing",
        )
    )
    runbook = root / "docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md"
    checks.append(
        (
            "runbook_branch_protection",
            runbook.is_file() and "branch protection" in runbook.read_text(encoding="utf-8").lower(),
            "runbook branch protection section missing",
        )
    )
    dossier = root / "docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md"
    checks.append(("protocol_03_dossier", dossier.is_file(), "Protocol 03 dossier missing"))

    report = {"ready_for_independent_b6_mechanical": True, "checks": []}
    exit_code = 0
    for name, ok, detail in checks:
        report["checks"].append({"name": name, "ok": ok, "detail": detail if not ok else ""})
        if not ok:
            report["ready_for_independent_b6_mechanical"] = False
            exit_code = 1
            print(f"FAIL {name}: {detail}", file=sys.stderr)
        else:
            print(f"OK {name}", file=sys.stdout)

    out_dir = root / ".tmp" / "session" / "ent-audit-x-mechanical-readiness"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mechanical_readiness_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
