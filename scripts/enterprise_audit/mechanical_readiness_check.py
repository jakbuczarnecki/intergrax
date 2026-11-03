#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Mechanical B6 readiness checks for ENT-AUDIT-X (fail-closed; no PASS declaration)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_head(root: Path) -> str:
    proc = _run_git(root, "rev-parse", "HEAD")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git rev-parse failed")
    return proc.stdout.strip()


def _git_clean(root: Path) -> bool:
    proc = _run_git(root, "status", "--porcelain")
    return proc.returncode == 0 and proc.stdout.strip() == ""


def _parse_ladder_pin(ladder_text: str, role: str) -> str | None:
    for line in ladder_text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] != role:
            continue
        sha = cells[1].strip().strip("`")
        if re.fullmatch(r"[0-9a-f]{40}", sha):
            return sha
    return None


def _latest_l3_report(root: Path) -> Path | None:
    proof_root = root / ".tmp" / "proof"
    if not proof_root.is_dir():
        return None
    candidates = sorted(proof_root.rglob("enterprise_audit_report.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def main() -> int:
    root = _repo_root()
    head = _git_head(root)
    ladder_path = root / "docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_SHA_LADDER.md"
    ladder_text = ladder_path.read_text(encoding="utf-8") if ladder_path.is_file() else ""
    tes_pin = _parse_ladder_pin(ladder_text, "TEST_EXECUTION_SHA")

    checks: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    add("implementation_sha_clean", _git_clean(root), "working tree must be clean for mechanical readiness")
    add("implementation_sha_head", bool(re.fullmatch(r"[0-9a-f]{40}", head)), "invalid HEAD sha")

    tes_ok = False
    tes_detail = "TEST_EXECUTION_SHA not pinned in ladder"
    if tes_pin:
        ancestor = _run_git(root, "merge-base", "--is-ancestor", tes_pin, "HEAD")
        tes_ok = ancestor.returncode == 0
        tes_detail = "" if tes_ok else f"{tes_pin} not ancestor of HEAD"
    add("test_execution_sha_pinned", tes_ok, tes_detail)

    l3_report_path = _latest_l3_report(root)
    l3_ok = False
    l3_detail = "no L3 enterprise_audit_report.json under .tmp/proof/"
    manifest_exec = 0
    manifest_planned = 0
    coverage_label = ""
    if l3_report_path and l3_report_path.is_file():
        try:
            report = json.loads(l3_report_path.read_text(encoding="utf-8"))
            manifest = report.get("execution_manifest", {})
            manifest_exec = int(manifest.get("checks_executed", 0))
            manifest_planned = int(manifest.get("checks_planned", 0))
            coverage_label = str(manifest.get("coverage_label", ""))
            l3_ok = manifest_exec == 85 and manifest_planned == 85 and coverage_label == "FULL"
            if not l3_ok:
                l3_detail = f"L3 manifest {manifest_exec}/{manifest_planned} coverage={coverage_label or '?'}"
            else:
                l3_detail = ""
        except (json.JSONDecodeError, TypeError, ValueError):
            l3_detail = "L3 report JSON invalid"
    add("full_85_85_executed", l3_ok, l3_detail)

    add("ten_13_executed", False, "TEN-* outcomes not verified @ TEST_EXECUTION_SHA (OPEN)")
    add("vbn_9_executed", False, "VBN-* outcomes not verified @ TEST_EXECUTION_SHA (OPEN)")
    add("vbn_domains_31", False, "vbn_domains.v1.json accounted==31 not proven @ TES (OPEN)")
    add("l1_l2_l3_logs", l3_report_path is not None, "pytest/L3 logs @ TEST_EXECUTION_SHA missing")
    add("adv_31_fault_suite", False, "ADV31 fault injection not qualified @ TES (OPEN)")
    add("evidence_integrity", False, "closed-world bundle @ EVIDENCE_PUBLICATION_SHA missing")
    add("sha_ancestry_documented", tes_ok, "merge-base TEST_EXECUTION_SHA→HEAD required")
    frz_data = root / "intergrax/platform/enterprise_audit/data"
    add("frz_mappings_complete", frz_data.is_dir(), "FRZ artifact path missing")
    add("pr_nightly_ci_proof", False, "CI run URLs + artifacts not pinned in dossier (OPEN)")
    add("bat_sh_acceptance", False, "C063 launcher acceptance not qualified @ TES (OPEN)")
    dossier = root / "docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md"
    add("dossier_complete", dossier.is_file(), "Protocol 03 dossier missing")
    add("external_blockers_listed", True, "")

    ready = all(ok for _, ok, _ in checks)
    report = {
        "formally_certified": False,
        "technical_ready_for_independent_b6_mechanical": ready,
        "implementation_sha": head,
        "test_execution_sha": tes_pin,
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
    }

    out_dir = root / ".tmp" / "session" / "ent-audit-x-mechanical-readiness"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mechanical_readiness_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    exit_code = 0 if ready else 1
    for name, ok, detail in checks:
        if ok:
            print(f"OK {name}", file=sys.stdout)
        else:
            print(f"FAIL {name}: {detail}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
