from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import AuditMode
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner


def test_l3_real_fast_executes_mandatory_floor(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=False)
    _, manifest, code = runner.run(
        AuditMode.FAST,
        output_dir=tmp_path / "l3-fast-real",
        check_ids_filter=frozenset({"ENT-A-BND-01"}),
    )
    assert manifest.checks_executed == 1
    assert code.value in {0, 1, 2, 3}


def test_l3_clean_checkout_full_dry_plan(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=True)
    _, manifest, code = runner.run(AuditMode.FULL, output_dir=tmp_path / "l3-full")
    assert manifest.checks_planned == 85
    assert code.value in {0, 2}


def test_l3_real_full_single_check_subprocess(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=False)
    _, manifest, _ = runner.run(
        AuditMode.FULL,
        output_dir=tmp_path / "l3-full-one",
        check_ids_filter=frozenset({"ENT-A-BND-01"}),
    )
    assert manifest.checks_executed == 1


def test_l3_freeze_requires_baseline_or_bootstrap(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=True, require_clean_repo=False)
    _, _, code = runner.run(
        AuditMode.FREEZE,
        output_dir=tmp_path / "l3-freeze",
        bootstrap_baseline=True,
    )
    assert code.value in {0, 2}


def test_l3_freeze_real_single_check(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=False, require_clean_repo=False)
    _, manifest, code = runner.run(
        AuditMode.FREEZE,
        output_dir=tmp_path / "l3-freeze-one",
        check_ids_filter=frozenset({"ENT-A-BND-01"}),
        bootstrap_baseline=True,
    )
    assert manifest.checks_executed == 1
    assert code.value in {0, 1, 2, 3}


def test_l3_evidence_manifest_roundtrip(tmp_path: Path) -> None:
    from intergrax.platform.enterprise_audit.evidence import verify_evidence_manifest

    root = Path(__file__).resolve().parents[4]
    out = tmp_path / "bundle"
    runner = EnterpriseAuditRunner(root, dry_run=True)
    runner.run(AuditMode.FAST, output_dir=out, write_evidence_bundle=True)
    assert verify_evidence_manifest(out)


def test_l3_vbn_31_domains_replay_honest(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts/enterprise_audit/vbn_domain_report.py"
    out = tmp_path / "vbn_domains.v1.json"
    completed = subprocess.run(
        [sys.executable, str(script), "--output", str(out)],
        cwd=root,
        check=False,
    )
    assert completed.returncode == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["domains_total"] == 31
    assert len(payload["domains"]) == 31
    statuses = {d["status"] for d in payload["domains"]}
    assert statuses <= {"QUALIFIED", "NA_WITH_EVIDENCE", "GAP", "PARTIAL"}
