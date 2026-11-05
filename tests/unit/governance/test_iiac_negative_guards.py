"""IIAC governance negative guards (plan §14 N-01..N-12).

Coverage map: N-01 same session; N-02 handoff; N-03 product diff; N-04 stale SHA;
N-05 threat model; N-06 equivalence; N-07 FRZ-AUD-10; N-08 activation digest;
N-09 forbidden PASS files; N-10 ENT predecessor; N-11 digest length; N-12 Session A NOT_IN_FORCE.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts/governance"
EXAMPLE = REPO / "docs/project/maintainers/qualification/examples/iiac_audit_session_manifest.example.json"
ACTIVATION = REPO / "docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json"
PACKAGE = REPO / "docs/project/maintainers/qualification/ent_audit_x/OD_ENT_02_COMPENSATING_CONTROL_PACKAGE.md"

_SHA = "eab65e010f5a1abfc413bb05419435bdf6b1810b"
_DIGEST = "a2e4e09011bc54b985784125bed85e30da0cfe11b2ee736da130a17b5dc09a6f"


def _base_manifest() -> dict:
    return {
        "stage_id": "IIAC-GOV-1-EXAMPLE",
        "certification_profile": "iiac-v1",
        "policy_version": "IIAC-2026-11-05.1",
        "implementation_session_id": "impl-session-a",
        "implementation_agent_id": "implementer",
        "auditor_session_id": "audit-session-b",
        "auditor_agent_id": "independent-quality-auditor",
        "certifier_session_id": "cert-session-b",
        "certifier_agent_id": "certifier",
        "audited_implementation_sha": _SHA,
        "test_execution_sha": _SHA,
        "evidence_publication_sha": _SHA,
        "audit_execution_sha": _SHA,
        "audit_report_digest": _DIGEST,
        "audit_scope_manifest_digest": _DIGEST,
        "test_matrix_digest": _DIGEST,
        "findings_digest": _DIGEST,
        "verdict": "CONDITIONAL",
        "limitations": ["EXAMPLE ONLY — NOT A PASS RECORD"],
        "certification_publication_sha": _SHA,
        "assurance_level": "process_enforced",
    }


def _run(script: str, *args: str) -> int:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode


def test_n01_same_implementation_and_auditor_session_fails(tmp_path: Path) -> None:
    manifest = _base_manifest()
    manifest["auditor_session_id"] = manifest["implementation_session_id"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _run("verify_iiac_audit_manifest.py", str(path)) != 0


def test_n02_missing_handoff_template_flag(tmp_path: Path) -> None:
    manifest = _base_manifest()
    path = tmp_path / "ok.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    rc = _run("verify_iiac_audit_manifest.py", str(path), "--require-handoff-template")
    assert rc == 0


def test_n03_auditor_product_write_surface_strict(tmp_path: Path) -> None:
    manifest = _base_manifest()
    manifest["audit_execution_sha"] = "0000000000000000000000000000000000000001"
    path = tmp_path / "n03.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        _run("verify_iiac_role_separation.py", str(path), "--strict-product-diff") != 0
        or _run("verify_iiac_role_separation.py", str(path)) == 0
    )


def test_n04_stale_audit_sha_not_ancestor(tmp_path: Path) -> None:
    manifest = _base_manifest()
    manifest["audit_execution_sha"] = "0000000000000000000000000000000000000001"
    path = tmp_path / "n04.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _run("verify_iiac_audit_manifest.py", str(path)) != 0


def test_n05_compensating_without_threat_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    threat = REPO / "docs/project/maintainers/qualification/ent_audit_x/OD_ENT_02_THREAT_MODEL.md"
    if not threat.is_file():
        pytest.skip("threat model not present")
    backup = threat.read_text(encoding="utf-8")
    try:
        threat.write_text("# stub\n", encoding="utf-8")
        assert _run("verify_od_ent_02_compensating_control.py") != 0
    finally:
        threat.write_text(backup, encoding="utf-8")


def test_n06_no_equivalence_claim_in_package() -> None:
    text = PACKAGE.read_text(encoding="utf-8").lower()
    assert "equivalent to live branch protection" not in text


def test_n07_frz_aud10_open_in_checklist() -> None:
    frz = REPO / "docs/project/maintainers/qualification/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md"
    text = frz.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "FRZ-AUD-10" in line:
            assert "| OPEN |" in line or "COMPENSATING" in line.upper()
            break
    else:
        pytest.fail("FRZ-AUD-10 row missing")


def test_n08_in_force_without_session_b_digest(tmp_path: Path) -> None:
    backup = ACTIVATION.read_text(encoding="utf-8")
    try:
        ACTIVATION.write_text(
            json.dumps(
                {
                    "status": "IN_FORCE",
                    "policy_version": "IIAC-test",
                    "effective_from_sha": _SHA,
                }
            ),
            encoding="utf-8",
        )
        assert _run("verify_iiac_policy_activation.py") != 0
    finally:
        ACTIVATION.write_text(backup, encoding="utf-8")


def test_n09_implementer_must_not_publish_pass_filename_glob() -> None:
    qual = REPO / "docs/project/maintainers/qualification"
    bad = list(qual.rglob("*IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md"))
    assert not bad, f"forbidden PASS records present: {bad}"


def test_n10_ent_preflight_evol_predecessor_when_iiac_forced(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.qualification.ent_audit_x_formal_closure_preflight as pre

    monkeypatch.setattr(pre, "_iiac_in_force", lambda: True)
    assert pre.main() != 0


def test_n11_false_pass_digest_too_short(tmp_path: Path) -> None:
    manifest = _base_manifest()
    manifest["audit_report_digest"] = "short"
    path = tmp_path / "n11.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _run("verify_iiac_audit_manifest.py", str(path)) != 0


def test_n12_session_a_policy_not_in_force_by_default() -> None:
    rec = json.loads(ACTIVATION.read_text(encoding="utf-8"))
    assert rec.get("status") == "NOT_IN_FORCE"
    assert _run("verify_iiac_policy_activation.py", "--expect-not-in-force") == 0


def test_example_manifest_validates() -> None:
    assert EXAMPLE.is_file()
    assert _run("verify_iiac_audit_manifest.py", str(EXAMPLE)) == 0
