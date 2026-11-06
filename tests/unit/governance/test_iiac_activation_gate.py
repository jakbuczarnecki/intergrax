"""ACT-01 adversarial matrix for the IIAC activation gate.

Fixtures are built in a temporary git repository. They are not certification evidence.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/governance/verify_iiac_policy_activation.py"
CANONICAL = REPO / "docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json"
REAUDIT_REPORT = (
    REPO
    / "docs/project/maintainers/qualification/iiac_evidence/IIAC-GOV-1/reaudit"
    / "IIAC_SESSION_B_F01_F03_REAUDIT_REPORT.md"
)
HISTORICAL_REPORT = (
    REPO
    / "docs/project/maintainers/qualification/iiac_evidence/IIAC-GOV-1/historical_session_b"
    / "IIAC_SESSION_B_INDEPENDENT_GOVERNANCE_AUDIT_REPORT.md"
)
POLICY = "IIAC-2026-11-05.1"
REPORT_DIGEST = "62b93665a97024337638e966ca87cbe37f32e1b4c9c795bef46e4c1097aefbde"
HISTORICAL_DIGEST = "56c9b078e42e1408799e37a2535aa61879e31021864214e46dfa165fb41fded1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "core.autocrlf=false", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    proc = subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "-c",
            "user.email=iiac-fixture@example.com",
            "-c",
            "user.name=iiac-fixture",
            "commit",
            "-m",
            message,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git commit failed")
    return _git(repo, "rev-parse", "HEAD")


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _dump(path: Path, payload: dict) -> bytes:
    raw = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    _write(path, raw)
    return raw


def _build_repo(tmp: Path) -> dict[str, str]:
    repo = tmp / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "development")
    _write(repo / "governance.txt", b"governance-implementation\n")
    _dump(
        repo / "activation/IIAC_POLICY_ACTIVATION_RECORD.json",
        {"status": "NOT_IN_FORCE", "policy_version": POLICY},
    )
    governance = _commit(repo, "governance implementation")
    _write(repo / "corrective.txt", b"corrective-implementation\n")
    corrective = _commit(repo, "corrective implementation")
    _write(repo / "published/certification_marker.txt", b"certifier publication slot\n")
    certification = _commit(repo, "certification publication")

    report = b"independent audit report fixture\n"
    scope = b'{"stage_id":"IIAC-GOV-1"}\n'
    matrix = b"uv run pytest tests/unit/governance/test_iiac_activation_gate.py\nRESULT: fixture\n"
    findings = {
        "verdict": "PASS",
        "findings": [
            {"id": "F01", "original_severity": "BLOCKER", "reaudit_disposition": "RESOLVED"},
            {"id": "F02", "original_severity": "LOW", "reaudit_disposition": "RESOLVED"},
        ],
    }
    findings_raw = _dump(repo / "evidence/findings.json", findings)
    _write(repo / "evidence/report.md", report)
    _write(repo / "evidence/scope.json", scope)
    _write(repo / "evidence/matrix.txt", matrix)
    report_digest = _sha256(report)
    handoff = (
        f"Policy version: {POLICY}\n"
        "implementation_session_id: impl-session\n"
        "auditor_session_id: audit-session\n"
        "certifier_session_id: cert-session\n"
    ).encode()
    _write(repo / "evidence/handoff.md", handoff)
    signature = f"Approved IIAC policy activation {POLICY}: operator @ 2026-11-05T00:00:00+00:00"
    operator = (
        "Decision ID: FIXTURE-001\n"
        f"Policy version: {POLICY}\n"
        "Decision scope: IIAC policy activation\n"
        "Decision status: AUTHORIZED\n"
        f"Audit report digest: {report_digest}\n"
        f"{signature}\n"
    ).encode()
    _write(repo / "evidence/IIAC_OD_ACTIVATION_FIXTURE-001.md", operator)
    manifest = {
        "stage_id": "IIAC-GOV-1",
        "certification_profile": "iiac-v1",
        "policy_version": POLICY,
        "implementation_session_id": "impl-session",
        "implementation_agent_id": "implementer",
        "auditor_session_id": "audit-session",
        "auditor_agent_id": "independent-quality-auditor",
        "certifier_session_id": "cert-session",
        "certifier_agent_id": "certification-governance-agent",
        "audited_implementation_sha": corrective,
        "test_execution_sha": corrective,
        "evidence_publication_sha": corrective,
        "audit_execution_sha": corrective,
        "audit_report_digest": report_digest,
        "audit_scope_manifest_digest": _sha256(scope),
        "test_matrix_digest": _sha256(matrix),
        "findings_digest": _sha256(findings_raw),
        "verdict": "PASS",
        "limitations": ["synthetic fixture — not production certification"],
        "certification_publication_sha": certification,
        "assurance_level": "process_enforced",
        "handoff_record_path": "evidence/handoff.md",
    }
    manifest_raw = _dump(repo / "evidence/manifest.json", manifest)
    evidence = _commit(repo, "evidence publication")
    record = {
        "policy_version": POLICY,
        "status": "IN_FORCE",
        "effective_from_sha": None,
        "policy_activation_sha": None,
        "governance_implementation_sha": governance,
        "corrective_implementation_sha": corrective,
        "audit_execution_sha": corrective,
        "evidence_publication_sha": evidence,
        "certification_publication_sha": certification,
        "session_b_verdict": "PASS",
        "session_b_governance_audit_report_digest": report_digest,
        "session_b_manifest_digest": _sha256(manifest_raw),
        "session_b_manifest_path": "evidence/manifest.json",
        "audit_report_path": "evidence/report.md",
        "audit_scope_path": "evidence/scope.json",
        "test_matrix_path": "evidence/matrix.txt",
        "findings_path": "evidence/findings.json",
        "handoff_record_path": "evidence/handoff.md",
        "handoff_digest": _sha256(handoff),
        "operator_decision_path": "evidence/IIAC_OD_ACTIVATION_FIXTURE-001.md",
        "operator_decision_id": "FIXTURE-001",
        "operator_activation_line": signature,
    }
    _dump(repo / "activation/IIAC_POLICY_ACTIVATION_RECORD.json", record)
    phase1 = _commit(repo, "activation phase 1")
    record["effective_from_sha"] = phase1
    record["policy_activation_sha"] = phase1
    _dump(repo / "activation/IIAC_POLICY_ACTIVATION_RECORD.json", record)
    head = _commit(repo, "activation phase 2")
    return {"repo": str(repo), "head": head, "phase1": phase1, "corrective": corrective, "governance": governance}


def _run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo",
            str(repo),
            "--activation-record",
            str(repo / "activation/IIAC_POLICY_ACTIVATION_RECORD.json"),
            "--evidence-root",
            str(repo),
            *args,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def _load(repo: Path) -> dict:
    return json.loads((repo / "activation/IIAC_POLICY_ACTIVATION_RECORD.json").read_text(encoding="utf-8"))


def _save_and_commit(repo: Path, record: dict, message: str) -> None:
    _dump(repo / "activation/IIAC_POLICY_ACTIVATION_RECORD.json", record)
    _commit(repo, message)


def _retie_manifest(repo: Path, record: dict, manifest: dict) -> None:
    raw = _dump(repo / record["session_b_manifest_path"], manifest)
    record["session_b_manifest_digest"] = _sha256(raw)


def _live(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def test_act_n01_inactive_policy_fails_default_and_require(tmp_path: Path) -> None:
    """T-IGX-018-A: the negative guard now runs on a NOT_IN_FORCE fixture (the canonical policy is IN_FORCE)."""
    repo = Path(_build_repo(tmp_path)["repo"])
    _save_and_commit(repo, {"status": "NOT_IN_FORCE", "policy_version": POLICY}, "deactivate fixture policy")
    default = _run(repo)
    required = _run(repo, "--require-in-force")
    inactive = _run(repo, "--expect-not-in-force")
    assert default.returncode != 0
    assert "ACT-N01" in default.stderr
    assert required.returncode != 0
    assert "ACT-N01" in required.stderr
    assert inactive.returncode == 0, inactive.stderr


def test_canonical_policy_is_in_force_and_verifies() -> None:
    """Post-activation contract: the canonical record verifies under --require-in-force and not as inactive."""
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    assert canonical["status"] == "IN_FORCE"
    required = _live("--require-in-force")
    assert required.returncode == 0, required.stderr
    assert _live().returncode == 0
    assert _live("--expect-not-in-force").returncode != 0


def test_positive_fixture_passes_without_touching_the_canonical_policy(tmp_path: Path) -> None:
    before = CANONICAL.read_bytes()
    built = _build_repo(tmp_path)
    repo = Path(built["repo"])
    default = _run(repo)
    required = _run(repo, "--require-in-force")
    assert default.returncode == 0, default.stderr
    assert required.returncode == 0, required.stderr
    assert CANONICAL.read_bytes() == before


def test_act_n02_arbitrary_digest(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    record = _load(repo)
    record["session_b_governance_audit_report_digest"] = "deadbeefdeadbeef"
    _save_and_commit(repo, record, "bad digest")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N02" in result.stderr


def test_act_n03_missing_manifest(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    record = _load(repo)
    record["session_b_manifest_path"] = "evidence/missing.json"
    _save_and_commit(repo, record, "missing manifest")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N03" in result.stderr


def test_act_n04_verdict_fail(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    record = _load(repo)
    manifest = json.loads((repo / record["session_b_manifest_path"]).read_text(encoding="utf-8"))
    manifest["verdict"] = "FAIL"
    _retie_manifest(repo, record, manifest)
    record["session_b_verdict"] = "FAIL"
    _save_and_commit(repo, record, "verdict fail")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N04" in result.stderr


def test_act_n05_verdict_conditional(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    record = _load(repo)
    manifest = json.loads((repo / record["session_b_manifest_path"]).read_text(encoding="utf-8"))
    manifest["verdict"] = "CONDITIONAL"
    _retie_manifest(repo, record, manifest)
    _save_and_commit(repo, record, "verdict conditional")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N05" in result.stderr


def test_act_n06_missing_operator_decision(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    record = _load(repo)
    record["operator_decision_path"] = "evidence/IIAC_OD_ACTIVATION_MISSING.md"
    record["operator_activation_line"] = None
    _save_and_commit(repo, record, "missing operator")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N06" in result.stderr


def test_act_n07_fabricated_operator_string(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    record = _load(repo)
    fabricated = b"Approved: operator Session A @ 2026-11-05T13:00:00+01:00 (IIAC-GOVERNANCE-REFORM)\n"
    _write(repo / record["operator_decision_path"], fabricated)
    record["operator_activation_line"] = "Approved"
    _save_and_commit(repo, record, "fabricated operator")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N07" in result.stderr


def test_act_n08_digest_mismatch(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    record = _load(repo)
    record["session_b_governance_audit_report_digest"] = "ab" * 32
    _save_and_commit(repo, record, "digest mismatch")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N08" in result.stderr


def test_act_n09_wrong_audit_sha(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    record = _load(repo)
    manifest = json.loads((repo / record["session_b_manifest_path"]).read_text(encoding="utf-8"))
    manifest["audited_implementation_sha"] = record["governance_implementation_sha"]
    _retie_manifest(repo, record, manifest)
    _save_and_commit(repo, record, "wrong audit sha")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N09" in result.stderr


def test_act_n10_self_certifying_agent(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    record = _load(repo)
    manifest = json.loads((repo / record["session_b_manifest_path"]).read_text(encoding="utf-8"))
    manifest["auditor_session_id"] = manifest["implementation_session_id"]
    manifest["auditor_agent_id"] = manifest["implementation_agent_id"]
    _retie_manifest(repo, record, manifest)
    handoff = (repo / record["handoff_record_path"]).read_bytes()
    handoff = handoff.replace(b"auditor_session_id: audit-session", b"auditor_session_id: impl-session")
    _write(repo / record["handoff_record_path"], handoff)
    record["handoff_digest"] = _sha256(handoff)
    _save_and_commit(repo, record, "self certifying")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N10" in result.stderr


def test_act_n11_stale_governance_change(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    _write(repo / "scripts/governance/verify_iiac_policy_activation.py", b"changed-after-audit\n")
    _commit(repo, "stale governance edit")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N11" in result.stderr


def test_act_n12_effective_sha_pre_activation_blob(tmp_path: Path) -> None:
    built = _build_repo(tmp_path)
    repo = Path(built["repo"])
    record = _load(repo)
    record["effective_from_sha"] = built["governance"]
    record["policy_activation_sha"] = built["governance"]
    _save_and_commit(repo, record, "pre-activation sha")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N12" in result.stderr


def test_act_n12_self_referential_head(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    record = _load(repo)
    record["effective_from_sha"] = _git(repo, "rev-parse", "HEAD")
    _dump(repo / "activation/IIAC_POLICY_ACTIVATION_RECORD.json", record)
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N12" in result.stderr
    assert "self-referential" in result.stderr


def test_act_n13_missing_report_bytes(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    record = _load(repo)
    (repo / record["audit_report_path"]).unlink()
    _commit(repo, "delete report")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N13" in result.stderr


def test_act_n14_path_traversal(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    record = _load(repo)
    record["audit_report_path"] = "../outside.md"
    _save_and_commit(repo, record, "traversal")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N14" in result.stderr


def test_act_n14_symlink_escape(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    outside = tmp_path / "outside.md"
    outside.write_bytes(b"outside\n")
    link = repo / "evidence" / "linked-report.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    record = _load(repo)
    record["audit_report_path"] = "evidence/linked-report.md"
    record["session_b_governance_audit_report_digest"] = _sha256(b"outside\n")
    _save_and_commit(repo, record, "symlink")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N14" in result.stderr


def test_act_n15_open_blocker(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    record = _load(repo)
    findings = {
        "findings": [
            {"id": "F01", "original_severity": "BLOCKER", "reaudit_disposition": "OPEN"},
        ]
    }
    raw = _dump(repo / record["findings_path"], findings)
    manifest = json.loads((repo / record["session_b_manifest_path"]).read_text(encoding="utf-8"))
    manifest["findings_digest"] = _sha256(raw)
    _retie_manifest(repo, record, manifest)
    _save_and_commit(repo, record, "open blocker")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N15" in result.stderr


def test_act_n16_policy_version_mismatch(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    record = _load(repo)
    record["policy_version"] = "IIAC-1999-01-01.1"
    manifest = json.loads((repo / record["session_b_manifest_path"]).read_text(encoding="utf-8"))
    manifest["policy_version"] = "IIAC-1999-01-01.1"
    _retie_manifest(repo, record, manifest)
    _save_and_commit(repo, record, "policy mismatch")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N16" in result.stderr


def test_act_n17_incomplete_provenance(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    record = _load(repo)
    manifest = json.loads((repo / record["session_b_manifest_path"]).read_text(encoding="utf-8"))
    manifest["certifier_session_id"] = ""
    _retie_manifest(repo, record, manifest)
    _save_and_commit(repo, record, "missing certifier")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N17" in result.stderr


def test_act_n18_roadmap_closure_conflict(tmp_path: Path) -> None:
    repo = Path(_build_repo(tmp_path)["repo"])
    record = _load(repo)
    record["stage_closures"] = {"EVOL-X": "CLOSED", "ENT-AUDIT-X": "CLOSED", "ARCH-FREEZE": "CLOSED"}
    _save_and_commit(repo, record, "stage closure")
    result = _run(repo, "--require-in-force")
    assert result.returncode != 0
    assert "ACT-N18" in result.stderr


def test_historical_partial_bytes_remain_distinct() -> None:
    assert REAUDIT_REPORT.is_file()
    assert HISTORICAL_REPORT.is_file()
    assert _sha256(REAUDIT_REPORT.read_bytes()) == REPORT_DIGEST
    assert _sha256(HISTORICAL_REPORT.read_bytes()) == HISTORICAL_DIGEST
    for path, digest in (
        (REAUDIT_REPORT, REPORT_DIGEST),
        (HISTORICAL_REPORT, HISTORICAL_DIGEST),
    ):
        rel = path.relative_to(REPO).as_posix()
        blob = subprocess.run(
            ["git", "cat-file", "blob", f"HEAD:{rel}"],
            cwd=REPO,
            capture_output=True,
            check=True,
        )
        assert _sha256(blob.stdout) == digest
    historical = json.loads(
        (
            REPO
            / "docs/project/maintainers/qualification/iiac_evidence/IIAC-GOV-1/historical_session_b"
            / "iiac_audit_session_manifest.IIAC-GOV-1.json"
        ).read_text(encoding="utf-8")
    )
    reaudit = json.loads(
        (
            REPO
            / "docs/project/maintainers/qualification/iiac_evidence/IIAC-GOV-1/reaudit"
            / "iiac_audit_session_manifest.IIAC-GOV-1.json"
        ).read_text(encoding="utf-8")
    )
    assert historical["verdict"] == "CONDITIONAL"
    assert reaudit["verdict"] == "PASS"
    assert historical["audited_implementation_sha"] == "2ae1014116d353bc7742985e62f2bccc2bfac798"
    assert reaudit["audited_implementation_sha"] == "5d6b44b05f90ecf420d45d01659ea64565b52848"


def test_canonical_evidence_root_override_is_rejected(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--require-in-force",
            "--evidence-root",
            str(tmp_path),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "evidence-root override is forbidden" in result.stderr


def test_expect_not_in_force_rejects_contradictory_pass(tmp_path: Path) -> None:
    repo = tmp_path / "inactive"
    repo.mkdir()
    _git(repo, "init", "-b", "development")
    record = {
        "status": "NOT_IN_FORCE",
        "policy_version": POLICY,
        "session_b_verdict": "PASS",
        "operator_activation_line": "Approved",
    }
    _dump(repo / "activation/IIAC_POLICY_ACTIVATION_RECORD.json", record)
    _commit(repo, "contradictory inactive")
    result = _run(repo)
    inactive = _run(repo, "--expect-not-in-force")
    assert result.returncode != 0
    assert inactive.returncode != 0
