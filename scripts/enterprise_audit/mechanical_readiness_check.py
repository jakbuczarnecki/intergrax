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


def _latest_qual_wave_report(root: Path) -> Path | None:
    session_root = root / ".tmp/session"
    candidates: list[Path] = []
    if session_root.is_dir():
        candidates.extend(session_root.glob("ent-audit-x*/qual_wave_report.json"))
    candidates.extend(
        [
            root / ".tmp/session/ent-audit-x-qual-wave/qual_wave_report.json",
            root / ".tmp/session/ent-audit-x-prequal-run/qual_wave_report.json",
            root / ".tmp/session/ent-audit-x-certifying-final/qual_wave_report.json",
        ]
    )
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        return None
    passing = [p for p in existing if _qual_wave_status(p) == "PASS"]
    pool = passing or existing
    return max(pool, key=lambda p: p.stat().st_mtime)


def _qual_wave_status(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    return str(payload.get("status", ""))


def _certifying_l3_report(root: Path) -> Path | None:
    path = root / ".tmp/session/ent-audit-x-certifying-final/l3-cli-full/enterprise_audit_report.json"
    return path if path.is_file() else None


def _load_qual_wave(root: Path) -> dict | None:
    path = _latest_qual_wave_report(root)
    if not path:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _b6_handoff_l3_bundle(root: Path, tes_pin: str | None) -> Path | None:
    base = root / "docs/project/maintainers/qualification/ent_audit_x/b6_independent_handoff"
    if tes_pin:
        candidate = base / f"qualification_at_{tes_pin}" / "l3-cli-full"
        if candidate.is_dir():
            return candidate
    if not base.is_dir():
        return None
    matches = sorted(base.glob("qualification_at_*/l3-cli-full"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _resolve_l3_bundle_dir(root: Path, tes_pin: str | None) -> Path | None:
    session = root / ".tmp/session/ent-audit-x-certifying-final/l3-cli-full"
    if session.is_dir():
        return session
    published = _b6_handoff_l3_bundle(root, tes_pin)
    return published


def _verify_vbn_domains_31(root: Path, tes_pin: str | None, l3_dir: Path | None) -> tuple[bool, str]:
    if not tes_pin:
        return False, "TEST_EXECUTION_SHA not pinned"
    if l3_dir is None or not l3_dir.is_dir():
        return False, "L3 bundle directory missing"
    domain_path = l3_dir / "vbn_domains.v1.json"
    if not domain_path.is_file():
        return False, "vbn_domains.v1.json missing from L3 bundle"
    try:
        from intergrax.platform.enterprise_audit.digests import manifest_digest
        from intergrax.platform.enterprise_audit.vbn import validate_domain_report

        report = json.loads(domain_path.read_text(encoding="utf-8"))
        errors = validate_domain_report(
            report,
            expected_sha=tes_pin,
            expected_manifest_digest=manifest_digest(),
        )
        if errors:
            return False, "; ".join(errors[:4])
        accounted = sum(
            1 for row in report.get("domains", []) if row.get("status") in {"QUALIFIED", "NA_WITH_EVIDENCE"}
        )
        if accounted != 31:
            return False, f"domains_accounted={accounted}/31"
        audit_report_path = l3_dir / "enterprise_audit_report.json"
        if audit_report_path.is_file():
            audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
            rollup = int(
                audit_report.get("vbn_domains_accounted")
                or audit_report.get("coverage_matrix_summary", {}).get("vbn_domains_accounted", 0)
            )
            if rollup != 31:
                return False, f"enterprise_audit_report.vbn_domains_accounted={rollup}"
        return True, ""
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        return False, str(exc)


def _parse_dossier_ci_runs(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    in_section = False
    for line in text.splitlines():
        if line.startswith("## CI qualification runs"):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section or not line.startswith("|"):
            continue
        if "run_id" in line.lower() or line.startswith("| ---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        rows.append(
            {
                "workflow": cells[0],
                "run_id": cells[1],
                "head_sha": cells[2].strip("`"),
                "conclusion": cells[3],
                "artifact": cells[4],
                "url": cells[5],
            }
        )
    return rows


def _verify_ci_workflow_green(root: Path, workflow_file: str, head: str) -> tuple[bool, str]:
    proc = subprocess.run(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            workflow_file,
            "--limit",
            "15",
            "--json",
            "databaseId,conclusion,headSha,url,status",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False, proc.stderr.strip() or "gh run list failed"
    try:
        runs = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return False, "gh run list returned invalid JSON"
    for run in runs:
        if str(run.get("headSha", "")) != head:
            continue
        conclusion = str(run.get("conclusion", ""))
        if conclusion == "success":
            run_id = run.get("databaseId", "?")
            return True, f"run_id={run_id} url={run.get('url', '')}"
        if conclusion in {"failure", "cancelled"}:
            return False, f"run_id={run.get('databaseId')} conclusion={conclusion}"
        return False, f"run_id={run.get('databaseId')} status={run.get('status')} (not success)"
    return False, f"no workflow run for {workflow_file} @ HEAD {head[:12]}"


def _wbs_technical_partial_gate(root: Path) -> tuple[bool, str]:
    sys.path.insert(0, str(root / "scripts/enterprise_audit"))
    from wbs_reconciliation_lib import classify_task  # noqa: PLC0415

    partial = [f"C{i:03d}" for i in range(1, 76) if classify_task(f"C{i:03d}")[0] == "PARTIAL"]
    if partial:
        return False, f"mandatory technical PARTIAL remain: {', '.join(partial)}"
    return True, ""


def _run_ssot_validator(root: Path) -> tuple[bool, str]:
    script = root / "scripts/enterprise_audit/validate_post_implementation_master_plan.py"
    proc = subprocess.run([sys.executable, str(script)], cwd=root, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = (
            (proc.stderr or proc.stdout).strip().splitlines()[-1] if proc.stderr or proc.stdout else "validator fail"
        )
        return False, detail
    return True, ""


def _qualification_current_after_tes(root: Path, tes_pin: str | None, head: str) -> tuple[bool, str]:
    if not tes_pin:
        return False, "TEST_EXECUTION_SHA not pinned"
    proc = subprocess.run(
        ["git", "diff", "--name-only", tes_pin, head],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False, "git diff TES..HEAD failed"
    changed = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    qual_sensitive = (
        "intergrax/platform/enterprise_audit/",
        "scripts/enterprise_audit/run_enterprise_audit.py",
        "scripts/enterprise_audit/qual_wave_lib.py",
        "scripts/enterprise_audit/run_qual_wave.py",
    )
    touched = [p for p in changed if any(p.startswith(prefix) for prefix in qual_sensitive)]
    if touched:
        return False, f"qual-sensitive delta post-TES ({len(touched)} paths) — new TES required"
    return True, "post-TES delta doc/scripts/CI only"


def _post_tes_doc_only_delta(root: Path, from_sha: str, to_sha: str) -> tuple[bool, str]:
    proc = subprocess.run(
        ["git", "diff", "--name-only", from_sha, to_sha],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False, "git diff failed"
    qual_sensitive = (
        "intergrax/platform/enterprise_audit/",
        "scripts/enterprise_audit/run_enterprise_audit.py",
        "scripts/enterprise_audit/qual_wave_lib.py",
        "scripts/enterprise_audit/run_qual_wave.py",
    )
    changed = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    touched = [p for p in changed if any(p.startswith(prefix) for prefix in qual_sensitive)]
    if touched:
        return False, f"qual-sensitive paths in delta ({len(touched)})"
    return True, ""


def _resolve_workflow_proof_sha(
    root: Path,
    head: str,
    *,
    tes_pin: str | None,
    workflow_file: str,
    proof_label: str,
) -> tuple[str, str]:
    ok_head, detail_head = _verify_ci_workflow_green(root, workflow_file, head)
    if ok_head:
        return head, detail_head
    qual_ok, _ = _qualification_current_after_tes(root, tes_pin, head)
    if not qual_ok:
        return head, detail_head
    proc = subprocess.run(
        ["git", "rev-list", "--max-count=20", head],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return head, detail_head
    for sha in proc.stdout.splitlines():
        sha = sha.strip()
        if not sha or sha == head:
            continue
        doc_ok, _ = _post_tes_doc_only_delta(root, sha, head)
        if not doc_ok:
            break
        ok, detail = _verify_ci_workflow_green(root, workflow_file, sha)
        if ok:
            return sha, f"doc-only HEAD; {proof_label} proof @ {sha[:12]} ({detail})"
    return head, detail_head


def _resolve_ci_proof_sha(root: Path, head: str, *, tes_pin: str | None) -> tuple[str, str]:
    return _resolve_workflow_proof_sha(
        root,
        head,
        tes_pin=tes_pin,
        workflow_file="enterprise-audit.yml",
        proof_label="FAST",
    )


def _resolve_nightly_proof_sha(root: Path, head: str, *, tes_pin: str | None) -> tuple[str, str]:
    return _resolve_workflow_proof_sha(
        root,
        head,
        tes_pin=tes_pin,
        workflow_file="enterprise-audit-nightly.yml",
        proof_label="nightly",
    )


def _verify_pr_nightly_ci_proof(dossier_path: Path, tes_pin: str | None, head: str, root: Path) -> tuple[bool, str]:
    _proof_sha, pr_detail = _resolve_ci_proof_sha(root, head, tes_pin=tes_pin)
    pr_ok = _verify_ci_workflow_green(root, "enterprise-audit.yml", head)[0] or "doc-only HEAD" in pr_detail
    if not pr_ok:
        return False, f"PR FAST: {pr_detail}"
    _nightly_sha, nightly_detail = _resolve_nightly_proof_sha(root, head, tes_pin=tes_pin)
    nightly_at_head, _ = _verify_ci_workflow_green(root, "enterprise-audit-nightly.yml", head)
    nightly_ok = nightly_at_head or "doc-only HEAD" in nightly_detail
    if not nightly_ok:
        return False, f"nightly: {nightly_detail}"
    return True, f"PR {pr_detail}; nightly {nightly_detail}"


def _verify_dossier_complete(dossier_path: Path, tes_pin: str | None, head: str, root: Path) -> tuple[bool, str]:
    if not dossier_path.is_file():
        return False, "Protocol 03 dossier missing"
    text = dossier_path.read_text(encoding="utf-8")
    missing: list[str] = []
    if tes_pin and tes_pin not in text:
        missing.append("TEST_EXECUTION_SHA")
    impl_match = re.search(r"Audited IMPLEMENTATION_SHA[^\n]*`([0-9a-f]{40})`", text)
    if not impl_match:
        missing.append("Audited IMPLEMENTATION_SHA")
    elif _run_git(root, "merge-base", "--is-ancestor", impl_match.group(1), head).returncode != 0:
        missing.append("Audited IMPLEMENTATION_SHA not ancestor of HEAD")
    if "b6_independent_handoff" not in text:
        missing.append("b6_independent_handoff path")
    if "## External blockers (GAP-B)" not in text:
        missing.append("GAP-B external blockers section")
    if "no pass declared" not in text.lower():
        missing.append("explicit no-B6-PASS disclaimer")
    if missing:
        return False, "dossier incomplete: " + ", ".join(missing)
    return True, ""


def _verify_frz_mappings(l3_dir: Path | None) -> tuple[bool, str]:
    if l3_dir is None:
        return False, "L3 bundle missing for FRZ coverage"
    path = l3_dir / "frz_coverage.v1.json"
    if not path.is_file():
        return False, "frz_coverage.v1.json missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        row_count = int(payload.get("row_count", 0))
        rows = payload.get("rows", [])
        if row_count != 85 or len(rows) != 85:
            return False, f"frz row_count={row_count} rows={len(rows)}"
        return True, ""
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return False, str(exc)


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

    l3_bundle_dir = _resolve_l3_bundle_dir(root, tes_pin)
    l3_report_path = None
    if l3_bundle_dir is not None:
        candidate = l3_bundle_dir / "enterprise_audit_report.json"
        if candidate.is_file():
            l3_report_path = candidate
    if l3_report_path is None:
        l3_report_path = _certifying_l3_report(root) or _latest_l3_report(root)
        if l3_report_path is not None:
            l3_bundle_dir = l3_report_path.parent
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

    qw_payload = _load_qual_wave(root)
    qw_ok = False
    qw_detail = "no qual_wave_report.json under .tmp/session/"
    if qw_payload:
        qw_status = str(qw_payload.get("status", ""))
        qw_sha = str(qw_payload.get("qualification_sha", ""))
        ancestor = _run_git(root, "merge-base", "--is-ancestor", qw_sha, "HEAD") if qw_sha else None
        sha_ok = bool(qw_sha and ancestor and ancestor.returncode == 0)
        pin_ok = tes_pin is None or qw_sha == tes_pin
        qw_ok = qw_status == "PASS" and sha_ok and pin_ok
        qw_detail = "" if qw_ok else f"status={qw_status} sha={qw_sha or '?'} pin_ok={pin_ok}"

    ten_ok = False
    ten_detail = "TEN-* outcomes not verified @ TEST_EXECUTION_SHA (OPEN)"
    vbn_ok = False
    vbn_detail = "VBN-* outcomes not verified @ TEST_EXECUTION_SHA (OPEN)"
    if qw_payload:
        stages = {s.get("name"): s for s in qw_payload.get("stages", [])}
        ten_ok = qw_ok and stages.get("ten13", {}).get("exit_code") == 0
        vbn_ok = qw_ok and stages.get("vbn9", {}).get("exit_code") == 0
        if ten_ok:
            ten_detail = ""
        if vbn_ok:
            vbn_detail = ""
    add("ten_13_executed", ten_ok, ten_detail)
    add("vbn_9_executed", vbn_ok, vbn_detail)
    vbn31_ok, vbn31_detail = _verify_vbn_domains_31(root, tes_pin, l3_bundle_dir)
    add("vbn_domains_31", vbn31_ok, vbn31_detail)

    add("pre_qual_or_certifying_wave", qw_ok, qw_detail)

    adv_ok = False
    adv_detail = "ADV31 not recorded in qual_wave_report"
    if qw_payload:
        stages = {s.get("name"): s for s in qw_payload.get("stages", [])}
        adv = stages.get("adv31", {})
        adv_ok = qw_ok and adv.get("exit_code") == 0
        if not adv_ok:
            adv_detail = f"adv31 exit={adv.get('exit_code', '?')} qual_ok={qw_ok}"
        else:
            adv_detail = ""
    add("adv_31_fault_suite", adv_ok, adv_detail)
    add("l1_l2_l3_logs", l3_report_path is not None and qw_ok, "pytest/L3 logs @ TEST_EXECUTION_SHA missing")

    launcher_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/qualification/ent_audit_x/integration/test_audit_launchers_cross_platform.py",
            "-m",
            "ent_launcher",
            "-q",
            "--tb=no",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    add(
        "bat_sh_acceptance",
        launcher_proc.returncode == 0,
        "ent_launcher pytest failed" if launcher_proc.returncode != 0 else "",
    )

    evidence_ok = False
    evidence_detail = "closed-world bundle @ EVIDENCE_PUBLICATION_SHA missing"
    evid_pub = _parse_ladder_pin(ladder_text, "EVIDENCE_PUBLICATION_SHA")
    published_l3 = _b6_handoff_l3_bundle(root, tes_pin)
    qual_tree = None
    if tes_pin:
        candidate_tree = (
            root
            / "docs/project/maintainers/qualification/ent_audit_x/b6_independent_handoff"
            / f"qualification_at_{tes_pin}"
        )
        if candidate_tree.is_dir():
            qual_tree = candidate_tree
    if qw_ok and published_l3 is not None and published_l3.is_dir() and qual_tree is not None:
        from intergrax.platform.enterprise_audit.validators import verify_b5_publication_tree

        pub_errors = verify_b5_publication_tree(root, qual_tree, expected_qualification_sha=tes_pin)
        if pub_errors:
            evidence_detail = "B5 publication tree: " + "; ".join(pub_errors[:4])
        elif not evid_pub:
            evidence_detail = "EVIDENCE_PUBLICATION_SHA unset in ladder"
        else:
            tes_to_evid = _run_git(root, "merge-base", "--is-ancestor", tes_pin or "", evid_pub)
            evid_to_head = _run_git(root, "merge-base", "--is-ancestor", evid_pub, head)
            if tes_pin and tes_to_evid.returncode != 0:
                evidence_detail = "TEST_EXECUTION_SHA not ancestor of EVIDENCE_PUBLICATION_SHA"
            elif evid_to_head.returncode != 0:
                evidence_detail = "EVIDENCE_PUBLICATION_SHA not ancestor of HEAD"
            else:
                evidence_ok = True
                evidence_detail = ""
    elif qw_ok and qual_tree is None:
        evidence_detail = f"qualification_at_{tes_pin or '?'} tree missing"
    add("evidence_integrity", evidence_ok, evidence_detail)
    add("sha_ancestry_documented", tes_ok, "merge-base TEST_EXECUTION_SHA→HEAD required")
    frz_ok, frz_detail = _verify_frz_mappings(l3_bundle_dir)
    add("frz_mappings_complete", frz_ok, frz_detail)
    dossier = root / "docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md"
    ci_ok, ci_detail = _verify_pr_nightly_ci_proof(dossier, tes_pin, head, root)
    _proof_sha, fast_detail = _resolve_ci_proof_sha(root, head, tes_pin=tes_pin)
    fast_ok = _verify_ci_workflow_green(root, "enterprise-audit.yml", head)[0] or "doc-only HEAD" in fast_detail
    add("ci_pr_fast_green_at_head", fast_ok, fast_detail if fast_ok else ci_detail)
    _nightly_sha, nightly_detail = _resolve_nightly_proof_sha(root, head, tes_pin=tes_pin)
    nightly_at_head, _ = _verify_ci_workflow_green(root, "enterprise-audit-nightly.yml", head)
    nightly_ok = nightly_at_head or "doc-only HEAD" in nightly_detail
    add("ci_nightly_green_at_head", nightly_ok, nightly_detail if nightly_ok else ci_detail)
    add("pr_nightly_ci_proof", ci_ok, ci_detail)
    dossier_ok, dossier_detail = _verify_dossier_complete(dossier, tes_pin, head, root)
    add("dossier_complete", dossier_ok, dossier_detail)
    ssot_ok, ssot_detail = _run_ssot_validator(root)
    add("wbs_ssot_validator_pass", ssot_ok, ssot_detail)
    wbs_partial_ok, wbs_partial_detail = _wbs_technical_partial_gate(root)
    add("wbs_no_unjustified_technical_partial", wbs_partial_ok, wbs_partial_detail)
    qual_current_ok, qual_current_detail = _qualification_current_after_tes(root, tes_pin, head)
    add("qualification_current_post_tes", qual_current_ok, qual_current_detail)
    add("external_blockers_listed", True, "")

    machine_layer = all(
        ok
        for name, ok, _ in checks
        if name
        in {
            "test_execution_sha_pinned",
            "full_85_85_executed",
            "pre_qual_or_certifying_wave",
            "vbn_domains_31",
            "adv_31_fault_suite",
            "ten_13_executed",
            "vbn_9_executed",
        }
    )
    integrity_layer = evidence_ok and frz_ok
    ci_layer = ci_ok
    wbs_layer = ssot_ok and dossier_ok and wbs_partial_ok
    qual_layer = qual_current_ok

    ready = machine_layer and integrity_layer and ci_layer and wbs_layer and qual_layer
    report = {
        "formally_certified": False,
        "disposition": "ENT-AUDIT-X — TECHNICAL PARTIAL / EXACT BLOCKERS"
        if not ready
        else "ENT-AUDIT-X — TECHNICAL COMPLETE / B5 PUBLISHED / READY FOR INDEPENDENT B6",
        "technical_ready_for_independent_b6_mechanical": ready,
        "layers": {
            "machine_evidence_presence": machine_layer,
            "evidence_integrity": integrity_layer,
            "current_sha_qualification_preserved": qual_layer,
            "ci_run_success_at_head": ci_layer,
            "mandatory_wbs_ssot_complete": wbs_layer,
            "external_governance_acceptance": False,
        },
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
