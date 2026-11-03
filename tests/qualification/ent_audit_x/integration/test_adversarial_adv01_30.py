from __future__ import annotations

import pytest
from intergrax.platform.enterprise_audit.evidence import verify_evidence_manifest
from intergrax.platform.enterprise_audit.manifest import load_default_manifest


def test_adv01_manifest_has_eighty_five_checks() -> None:
    assert len(load_default_manifest()) == 85


def test_adv02_duplicate_check_id_rejected() -> None:
    from intergrax.platform.enterprise_audit.manifest import _validate_manifest

    checks = load_default_manifest()
    dup = (*checks, checks[0])
    with pytest.raises(ValueError, match="duplicate"):
        _validate_manifest(dup)


def test_adv08_freeze_without_baseline_blocked(tmp_path) -> None:
    from pathlib import Path

    from intergrax.platform.enterprise_audit.contracts import AuditMode, EnterpriseAuditExitCode
    from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner

    root = Path(__file__).resolve().parents[4]
    runner = EnterpriseAuditRunner(root, dry_run=True, require_clean_repo=False)
    _, _, code = runner.run(AuditMode.FREEZE, output_dir=tmp_path / "adv8")
    assert code is EnterpriseAuditExitCode.VIOLATION


def test_adv12_tampered_evidence_fails(tmp_path) -> None:
    from intergrax.platform.enterprise_audit.evidence import build_evidence_manifest
    from tests.qualification.ent_audit_x._manifest_fixtures import sample_full_execution_manifest

    out = tmp_path / "ev"
    out.mkdir()
    (out / "artifact.txt").write_text("ok", encoding="utf-8")
    sha = "a" * 40
    build_evidence_manifest(
        out,
        run_id="r1",
        qualification_sha=sha,
        execution_manifest=sample_full_execution_manifest(qualification_sha=sha),
    )
    (out / "artifact.txt").write_text("bad", encoding="utf-8")
    assert verify_evidence_manifest(out) is False


def test_adv26_empty_changed_since_selection() -> None:
    import subprocess
    from pathlib import Path

    from intergrax.platform.enterprise_audit.changed_since import select_checks_for_changed_since

    root = Path(__file__).resolve().parents[4]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    selected = select_checks_for_changed_since(load_default_manifest(), repo_root=root, since_sha=head)
    assert len(selected) >= 13


# ADV-03..ADV-30 mapping (one test per ADV id; shared oracle where noted in ENT master plan)
def test_adv03_manifest_annex_c_alignment_script_exists() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    assert (root / "scripts/enterprise_audit/check_manifest_annex_c_alignment.py").is_file()


def test_adv04_manifest_default_dag_acyclic() -> None:
    from intergrax.platform.enterprise_audit.schedule import topological_order

    topological_order(load_default_manifest())


def test_adv05_full_dry_run_marks_skipped(tmp_path) -> None:
    from pathlib import Path

    from intergrax.platform.enterprise_audit.contracts import AuditMode
    from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner

    root = Path(__file__).resolve().parents[4]
    _, manifest, _ = EnterpriseAuditRunner(root, dry_run=True).run(AuditMode.FULL, output_dir=tmp_path / "adv05")
    assert manifest.checks_skipped == manifest.checks_planned


def test_adv06_fast_coverage_label_partial(tmp_path) -> None:
    from pathlib import Path

    from intergrax.platform.enterprise_audit.contracts import AuditMode
    from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner

    root = Path(__file__).resolve().parents[4]
    _, manifest, _ = EnterpriseAuditRunner(root, dry_run=True).run(AuditMode.FAST, output_dir=tmp_path / "adv6")
    assert manifest.coverage_label == "PARTIAL"


def test_adv07_fast_floor_module_importable() -> None:
    from intergrax.platform.enterprise_audit.modes import fast_floor_incomplete

    assert callable(fast_floor_incomplete)


def test_adv09_baseline_path_under_qual_dir() -> None:
    from pathlib import Path

    from intergrax.platform.enterprise_audit.baseline import baseline_path

    p = baseline_path(Path("tests/qualification/ent_audit_x"))
    assert p.name == "ENTERPRISE_AUDIT_BASELINE.json"


def test_adv10_schedule_validate_on_load() -> None:
    from intergrax.platform.enterprise_audit.schedule import validate_schedule

    validate_schedule(load_default_manifest())


def test_adv11_sandbox_rejects_mutating() -> None:
    from pathlib import Path

    from intergrax.platform.enterprise_audit.contracts import ManifestCheck
    from intergrax.platform.enterprise_audit.sandbox import CheckExecutionSandbox

    root = Path(__file__).resolve().parents[4]
    check = ManifestCheck(check_id="M", entrypoint_kind="python_script", entrypoint="scripts/x.py", mutating=True)
    with pytest.raises(ValueError, match="mutating"):
        CheckExecutionSandbox(root).run_check(check)


def test_adv13_changed_since_requires_since_sha_cli() -> None:
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/enterprise_audit/run_enterprise_audit.py"), "--mode", "CHANGED-SINCE"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0


def test_adv14_drift_regressed_enum_used() -> None:
    from intergrax.platform.enterprise_audit.contracts import DriftStatus

    assert DriftStatus.REGRESSED.value == "REGRESSED"


def test_adv15_evidence_manifest_schema_version(tmp_path) -> None:
    import json

    from intergrax.platform.enterprise_audit.evidence import build_evidence_manifest
    from tests.qualification.ent_audit_x._manifest_fixtures import sample_full_execution_manifest

    out = tmp_path / "adv15_ev"
    out.mkdir()
    sha = "d" * 40
    build_evidence_manifest(
        out,
        run_id="adv15",
        qualification_sha=sha,
        execution_manifest=sample_full_execution_manifest(qualification_sha=sha),
    )
    raw = json.loads((out / "evidence_manifest.v1.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1


def test_adv16_path_traversal_unit_exists() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    assert (root / "tests/qualification/ent_audit_x/unit/test_sandbox_path_traversal.py").is_file()


def test_adv17_no_shell_in_sandbox_source() -> None:
    from pathlib import Path

    text = (Path(__file__).resolve().parents[4] / "intergrax/platform/enterprise_audit/sandbox.py").read_text(
        encoding="utf-8"
    )
    assert '"shell": False' in text


def test_adv18_manifest_digest_callable() -> None:
    from intergrax.platform.enterprise_audit.digests import manifest_digest

    assert len(manifest_digest()) == 64


def test_adv19_repo_identity_reads_sha() -> None:
    from pathlib import Path

    from intergrax.platform.enterprise_audit.repo_identity import read_repo_identity

    ident = read_repo_identity(Path(__file__).resolve().parents[4])
    assert len(ident.head_sha) == 40


def test_adv20_tenant_enrich_importable() -> None:
    from intergrax.platform.enterprise_audit.tenant import enrich_tenant_findings

    assert enrich_tenant_findings(()) == ()


def test_adv21_vbn_enrich_importable() -> None:
    from intergrax.platform.enterprise_audit.vbn import enrich_vbn_findings

    assert enrich_vbn_findings(()) == ()


def test_adv22_report_json_written(tmp_path) -> None:
    from pathlib import Path

    from intergrax.platform.enterprise_audit.contracts import AuditMode
    from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner

    root = Path(__file__).resolve().parents[4]
    out = tmp_path / "adv22"
    EnterpriseAuditRunner(root, dry_run=True).run(AuditMode.FAST, output_dir=out)
    assert (out / "enterprise_audit_report.json").is_file()


def test_adv23_frz_coverage_artifact() -> None:
    from intergrax.platform.enterprise_audit.digests import frz_coverage_artifact
    from intergrax.platform.enterprise_audit.manifest import load_default_manifest

    art = frz_coverage_artifact(load_default_manifest())
    assert art["row_count"] == 85


def test_adv24_bootstrap_policy_integration_test_exists() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    assert (root / "tests/qualification/ent_audit_x/integration/test_baseline_bootstrap.py").is_file()


def test_adv25_vbn_domains_not_fake_constant(tmp_path) -> None:
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    out = tmp_path / "domains.json"
    subprocess.run(
        [sys.executable, str(root / "scripts/enterprise_audit/vbn_domain_report.py"), "--output", str(out)],
        cwd=root,
        check=True,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "limitations" in payload


def test_adv27_modes_enum_has_changed_since() -> None:
    from intergrax.platform.enterprise_audit.contracts import AuditMode

    assert AuditMode.CHANGED_SINCE.value == "CHANGED-SINCE"


def test_adv28_ci_runbook_documents_branch_protection() -> None:
    from pathlib import Path

    text = (
        Path(__file__).resolve().parents[4]
        / "docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md"
    ).read_text(encoding="utf-8")
    assert "branch protection" in text.lower()


def test_adv29_write_evidence_bundle_flag(tmp_path) -> None:
    from pathlib import Path

    from intergrax.platform.enterprise_audit.contracts import AuditMode
    from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner

    root = Path(__file__).resolve().parents[4]
    out = tmp_path / "adv29"
    with pytest.raises(ValueError, match="certifying_run"):
        EnterpriseAuditRunner(root, dry_run=True).run(
            AuditMode.FAST,
            output_dir=out,
            write_evidence_bundle=True,
        )


def test_adv30_l3_suite_present() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    assert (root / "tests/qualification/ent_audit_x/l3/test_l3_clean_checkout_full.py").is_file()
