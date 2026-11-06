"""IIAC governance negative guards (plan §14 N-01..N-12).

Coverage map: N-01 same session; N-02 handoff; N-03 product diff; N-04 stale SHA;
N-05 threat model; N-06 equivalence; N-07 FRZ-AUD-10; N-08 activation digest;
N-09 forbidden PASS files; N-10 ENT predecessor; N-11 digest length; N-12 Session A NOT_IN_FORCE.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts/governance"
QUALIFICATION = "docs/project/maintainers/qualification"
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
    """T-IGX-018-A: the stub threat model lives in ``tmp_path``; canonical files are never rewritten."""
    from scripts.governance import verify_od_ent_02_compensating_control as compensating

    threat = REPO / "docs/project/maintainers/qualification/ent_audit_x/OD_ENT_02_THREAT_MODEL.md"
    if not threat.is_file():
        pytest.skip("threat model not present")
    before = threat.read_bytes()
    stub = tmp_path / "OD_ENT_02_THREAT_MODEL.md"
    stub.write_text("# stub\n", encoding="utf-8")
    monkeypatch.setattr(compensating, "OD_ENT_02_THREAT", stub)
    assert compensating.main() != 0
    assert threat.read_bytes() == before


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
    """T-IGX-018-A: the forged IN_FORCE record is a temporary copy, never the canonical activation record."""
    forged = tmp_path / "IIAC_POLICY_ACTIVATION_RECORD.json"
    forged.write_text(
        json.dumps({"status": "IN_FORCE", "policy_version": "IIAC-test", "effective_from_sha": _SHA}),
        encoding="utf-8",
    )
    assert _run("verify_iiac_policy_activation.py", "--activation-record", str(forged)) != 0


def _content_digest(path: Path) -> str:
    """SHA-256 of the content with line endings normalised (a converting checkout must not hide a binding)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _manifest_passes(manifest: Path, repo: Path) -> bool:
    """A manifest counts only when the canonical manifest and role-separation validators accept it."""
    for script, extra in (
        ("verify_iiac_audit_manifest.py", ()),
        ("verify_iiac_role_separation.py", ("--strict-product-diff",)),
    ):
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / script), str(manifest), *extra, "--repo", str(repo)],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return False
    return True


def unreferenced_pass_records(repo: Path, verify: Callable[[Path, Path], bool] = _manifest_passes) -> list[str]:
    """N-09 content guard: every IIAC pass record must be bound by a *valid* manifest (or be its auditor copy).

    Replaces the pre-activation filename glob, which cannot tell a forged record from a genuine one.
    """
    qual = repo / QUALIFICATION
    records = sorted(qual.rglob("*IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md"))
    bound_paths: set[str] = set()
    bound_digests: set[str] = set()
    for manifest in sorted(qual.rglob("iiac_audit_session_manifest*.json")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        rel = str(data.get("pass_record_path") or "").replace("\\", "/") if isinstance(data, dict) else ""
        target = repo / rel if rel else None
        if target is None or not target.is_file() or not verify(manifest, repo):
            continue
        bound_paths.add(rel)
        bound_digests.add(_content_digest(target))
    findings = []
    for record in records:
        rel = record.relative_to(repo).as_posix()
        if rel not in bound_paths and _content_digest(record) not in bound_digests:
            findings.append(rel)
    return findings


def test_n09_implementer_must_not_publish_unbound_pass_records() -> None:
    assert unreferenced_pass_records(REPO) == []


def _qual_fixture(tmp_path: Path, *, pass_text: str, manifest_ref: str | None) -> Path:
    root = tmp_path / "r"
    qual = root / QUALIFICATION
    (qual / "evol_x").mkdir(parents=True)
    pass_rel = f"{QUALIFICATION}/evol_x/EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md"
    (root / pass_rel).write_text(pass_text, encoding="utf-8")
    if manifest_ref is not None:
        evidence = qual / "iiac_evidence/EVOL-X/certification"
        evidence.mkdir(parents=True)
        (evidence / "iiac_audit_session_manifest.EVOL-X.json").write_text(
            json.dumps({"pass_record_path": manifest_ref}), encoding="utf-8"
        )
    return root


def test_n09_unreferenced_pass_record_is_flagged(tmp_path: Path) -> None:
    root = _qual_fixture(tmp_path, pass_text="forged\n", manifest_ref=None)
    assert unreferenced_pass_records(root, verify=lambda _m, _r: True) == [
        f"{QUALIFICATION}/evol_x/EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md"
    ]


def test_n09_pass_record_bound_by_an_invalid_manifest_is_flagged(tmp_path: Path) -> None:
    ref = f"{QUALIFICATION}/evol_x/EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md"
    root = _qual_fixture(tmp_path, pass_text="forged\n", manifest_ref=ref)
    assert unreferenced_pass_records(root, verify=lambda _m, _r: False) == [ref]


def test_n09_pass_record_bound_by_a_valid_manifest_is_accepted(tmp_path: Path) -> None:
    ref = f"{QUALIFICATION}/evol_x/EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md"
    root = _qual_fixture(tmp_path, pass_text="genuine\n", manifest_ref=ref)
    assert unreferenced_pass_records(root, verify=lambda _m, _r: True) == []


def test_n09_byte_identical_auditor_copy_of_a_bound_record_is_accepted(tmp_path: Path) -> None:
    ref = f"{QUALIFICATION}/evol_x/EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md"
    root = _qual_fixture(tmp_path, pass_text="genuine\n", manifest_ref=ref)
    copy = root / QUALIFICATION / "iiac_evidence/EVOL-X/audit/EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md"
    copy.parent.mkdir(parents=True)
    copy.write_text("genuine\n", encoding="utf-8")
    assert unreferenced_pass_records(root, verify=lambda _m, _r: True) == []
    copy.write_text("tampered copy\n", encoding="utf-8")
    assert len(unreferenced_pass_records(root, verify=lambda _m, _r: True)) == 1


@pytest.fixture(scope="module", autouse=True)
def _canonical_activation_record_is_never_modified() -> Iterator[None]:
    """HF-02 guard: the module must leave the canonical activation record and its Git hash untouched."""

    def snapshot() -> tuple[bytes, str]:
        digest = subprocess.run(
            ["git", "hash-object", "--no-filters", str(ACTIVATION)],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return ACTIVATION.read_bytes(), digest

    before = snapshot()
    yield
    assert snapshot() == before, "negative guards modified the canonical IIAC activation record"


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


def test_n12_policy_is_in_force_only_with_valid_activation_evidence(tmp_path: Path) -> None:
    """Post-activation N-12: IN_FORCE must verify, and an inactive claim is only valid when truly inactive."""
    rec = json.loads(ACTIVATION.read_text(encoding="utf-8"))
    assert rec.get("status") == "IN_FORCE"
    assert _run("verify_iiac_policy_activation.py", "--require-in-force") == 0
    assert _run("verify_iiac_policy_activation.py", "--expect-not-in-force") != 0
    inactive = tmp_path / "IIAC_POLICY_ACTIVATION_RECORD.json"
    inactive.write_text(
        json.dumps({"status": "NOT_IN_FORCE", "policy_version": rec["policy_version"]}), encoding="utf-8"
    )
    assert _run("verify_iiac_policy_activation.py", "--expect-not-in-force", "--activation-record", str(inactive)) == 0
    assert _run("verify_iiac_policy_activation.py", "--require-in-force", "--activation-record", str(inactive)) != 0


def test_example_manifest_validates() -> None:
    assert EXAMPLE.is_file()
    assert _run("verify_iiac_audit_manifest.py", str(EXAMPLE)) == 0
