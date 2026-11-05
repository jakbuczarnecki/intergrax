# © Artur Czarnecki. All rights reserved.

"""EVOL-X B-4 corrective: two-gate closure preflight adversarial matrix (T-IGX-017-A).

Every fixture is an isolated temporary git repository that mirrors the canonical relative
layout. Fixture bytes are synthetic; they are not EVOL-X certification evidence and are never
written into the canonical repository.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from scripts.qualification import evol_x_closure_publication_preflight as pf

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/qualification/evol_x_closure_publication_preflight.py"
POLICY = "IIAC-2026-11-05.1"

IMPL_SESSION = "HISTORICAL_UNRECORDED__FIXTURE_EXCEPTION"
AUDITOR_SESSION = "FIXTURE-AUDIT-SESSION__claude-code-aaaa"
CERTIFIER_SESSION = "FIXTURE-CERT-SESSION__claude-code-bbbb"

AUDITOR_LIMITATIONS = [
    "LIM-01 Selection scope is EVOL_CONSERVATIVE_FULL only (not PLATFORM_CONSERVATIVE_FULL/T4)",
    "LIM-02 AUTOMATION_FIRST_SCOPED: EVOL-M-09H NOT_MEASURED; M-09A proves tool-side attribution only",
    "LIM-03 TESTED_BY test ownership is direct-import only (F-C-03)",
    "LIM-04 EVOL-M-03 measured on self-surface arcs under oracle O-X",
    "LIM-05 FRZ-EVOL-07 is qualification-layer only; production ActivationService.rollback is not gated",
    "LIM-06 Frozen-core registry covers 9 surfaces",
    "LIM-07 FRZ-EVOL-10 scenario-wave audit belongs to ENT-AUDIT-X",
]
CERTIFIED_LIMITATIONS = [
    *AUDITOR_LIMITATIONS[:2],
    AUDITOR_LIMITATIONS[2] + "; no claim of complete test-impact coverage",
    *AUDITOR_LIMITATIONS[3:],
    "F-C-02 FRZ-EVOL-06 bundle scope_footnote is empty",
    "Evidence-only publication: does not close EVOL-X",
]
DISPOSITIONS = {
    "FRZ-EVOL-01": "ELIGIBLE_PASS_SCOPED(LIM-03)",
    "FRZ-EVOL-02": "ELIGIBLE_PASS",
    "FRZ-EVOL-03": "ELIGIBLE_PASS",
    "FRZ-EVOL-04": "ELIGIBLE_PASS",
    "FRZ-EVOL-05": "ELIGIBLE_PASS",
    "FRZ-EVOL-06": "ELIGIBLE_PASS_SCOPED(LIM-01,LIM-03,LIM-04; footnote: EVOL_CONSERVATIVE_FULL scope only)",
    "FRZ-EVOL-07": "ELIGIBLE_PASS_EVOL_X_SCOPED(LIM-05; production rollback enforcement is NOT claimed)",
    "FRZ-EVOL-08": "ELIGIBLE_PASS_SCOPED(LIM-02; M-09H not measured)",
    "FRZ-EVOL-09": "ELIGIBLE_PASS_SCOPED(LIM-06)",
    "FRZ-EVOL-10": "ELIGIBLE_PASS_SCOPED(LIM-07)",
}
AUDITOR_FRZ = {k: ("PASS — SCOPED" if "SCOPED" in v else "PASS") for k, v in DISPOSITIONS.items()}
PROMOTED_ROWS = {
    "FRZ-EVOL-01": ("PASS — SCOPED", "LIM-03"),
    "FRZ-EVOL-02": ("PASS", "—"),
    "FRZ-EVOL-03": ("PASS", "—"),
    "FRZ-EVOL-04": ("PASS", "—"),
    "FRZ-EVOL-05": ("PASS", "—"),
    "FRZ-EVOL-06": ("PASS — SCOPED", "LIM-01, LIM-03, LIM-04; EVOL_CONSERVATIVE_FULL scope only"),
    "FRZ-EVOL-07": (
        "PASS (EVOL-X SCOPED)",
        "LIM-05; qualification-layer only; production rollback enforcement NOT claimed",
    ),
    "FRZ-EVOL-08": ("PASS — SCOPED", "LIM-02; change-time diagnostics; M-09H not measured"),
    "FRZ-EVOL-09": ("PASS — SCOPED", "LIM-06"),
    "FRZ-EVOL-10": ("PASS — SCOPED", "LIM-07"),
}


# --------------------------------------------------------------------------- git fixture helpers


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-c", "core.autocrlf=false", *args], cwd=repo, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=b4-fixture@example.com", "-c", "user.name=b4-fixture", "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _write(repo: Path, rel: str, data: bytes | str) -> bytes:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _dump(repo: Path, rel: str, payload: dict[str, Any]) -> bytes:
    return _write(repo, rel, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _load(repo: Path, rel: str) -> dict[str, Any]:
    return json.loads((repo / rel).read_text(encoding="utf-8"))


def _index(repo: Path, rel_dir: str, rel_index: str) -> bytes:
    root = repo / rel_dir
    files = {
        p.relative_to(root).as_posix(): {"sha256": _sha256(p.read_bytes()), "bytes": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != Path(rel_index).name
    }
    return _dump(repo, rel_index, {"schema_version": "iiac_evidence_index.v1", "stage_id": "EVOL-X", "files": files})


def _reindex_cert(repo: Path) -> None:
    _index(repo, pf.CERT_DIR, pf.REL_CERT_INDEX)


# --------------------------------------------------------------------------- fixture builders


def _build_activation(repo: Path) -> None:
    """Minimal valid IIAC activation chain (mirrors tests/unit/governance/test_iiac_activation_gate.py)."""
    _write(repo, "iiac_gov/governance.txt", "governance-implementation\n")
    _dump(repo, pf.REL_ACTIVATION, {"status": "NOT_IN_FORCE", "policy_version": POLICY})
    governance = _commit(repo, "iiac governance implementation")
    _write(repo, "iiac_gov/corrective.txt", "corrective-implementation\n")
    corrective = _commit(repo, "iiac corrective implementation")
    _write(repo, "iiac_gov/certification_marker.txt", "certifier publication slot\n")
    certification = _commit(repo, "iiac certification publication")
    report = _write(repo, "iiac_gov/report.md", "independent governance audit fixture\n")
    scope = _write(repo, "iiac_gov/scope.json", '{"stage_id":"IIAC-GOV-1"}\n')
    matrix = _write(repo, "iiac_gov/matrix.txt", "fixture matrix\n")
    findings = _dump(repo, "iiac_gov/findings.json", {"findings": [{"id": "F01", "severity": "LOW"}]})
    handoff = _write(
        repo,
        "iiac_gov/handoff.md",
        f"Policy version: {POLICY}\nimplementation_session_id: gov-impl\nauditor_session_id: gov-audit\n",
    )
    signature = f"Approved IIAC policy activation {POLICY}: operator @ 2026-11-05T00:00:00+00:00"
    _write(
        repo,
        "iiac_gov/IIAC_OD_ACTIVATION_FIXTURE.md",
        "Decision ID: FIXTURE-OD\n"
        f"Policy version: {POLICY}\n"
        "Decision scope: IIAC policy activation\n"
        "Decision status: AUTHORIZED\n"
        f"Audit report digest: {_sha256(report)}\n"
        f"{signature}\n",
    )
    manifest = _dump(
        repo,
        "iiac_gov/manifest.json",
        {
            "stage_id": "IIAC-GOV-1",
            "certification_profile": "iiac-v1",
            "policy_version": POLICY,
            "implementation_session_id": "gov-impl",
            "implementation_agent_id": "implementer",
            "auditor_session_id": "gov-audit",
            "auditor_agent_id": "independent-quality-auditor",
            "certifier_session_id": "gov-cert",
            "certifier_agent_id": "certification-governance-agent",
            "audited_implementation_sha": corrective,
            "test_execution_sha": corrective,
            "evidence_publication_sha": corrective,
            "audit_execution_sha": corrective,
            "audit_report_digest": _sha256(report),
            "audit_scope_manifest_digest": _sha256(scope),
            "test_matrix_digest": _sha256(matrix),
            "findings_digest": _sha256(findings),
            "verdict": "PASS",
            "limitations": ["synthetic fixture"],
            "certification_publication_sha": certification,
            "assurance_level": "process_enforced",
        },
    )
    evidence = _commit(repo, "iiac evidence publication")
    record: dict[str, Any] = {
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
        "session_b_governance_audit_report_digest": _sha256(report),
        "session_b_manifest_digest": _sha256(manifest),
        "session_b_manifest_path": "iiac_gov/manifest.json",
        "audit_report_path": "iiac_gov/report.md",
        "audit_scope_path": "iiac_gov/scope.json",
        "test_matrix_path": "iiac_gov/matrix.txt",
        "findings_path": "iiac_gov/findings.json",
        "handoff_record_path": "iiac_gov/handoff.md",
        "handoff_digest": _sha256(handoff),
        "operator_decision_path": "iiac_gov/IIAC_OD_ACTIVATION_FIXTURE.md",
        "operator_decision_id": "FIXTURE-OD",
        "operator_activation_line": signature,
    }
    _dump(repo, pf.REL_ACTIVATION, record)
    phase1 = _commit(repo, "iiac activation phase 1")
    record["effective_from_sha"] = phase1
    record["policy_activation_sha"] = phase1
    _dump(repo, pf.REL_ACTIVATION, record)
    _commit(repo, "iiac activation phase 2")


def _ladder(tes: str, final: str, internal_b6: str, b6: str | None = None) -> str:
    b6_cell = f"`{b6}`" if b6 else "—"
    return (
        "﻿# EVOL-X — SHA ladder (fixture)\n\n"
        "| Role | SHA | Notes |\n|------|-----|-------|\n"
        f"| TEST_EXECUTION_SHA | `{tes}` | fixture TES |\n"
        f"| EVIDENCE_FINAL_PUBLICATION_SHA | `{final}` | fixture FINAL |\n"
        f"| B6_ACCEPTANCE_PUBLICATION_SHA | {b6_cell} | {'IIAC B6' if b6 else '**PENDING.**'} |\n"
        f"| INTERNAL_B6_ACCEPTANCE_PUBLICATION_SHA | `{internal_b6}` | historical internal record |\n"
        "| INTERNAL_CLOSURE_STATUS | **INTERNALLY CLOSED** | not formal |\n\n"
        "## Superseded (historical)\n\n| Role | SHA | Notes |\n|---|---|---|\n"
        "| TEST_EXECUTION_SHA | `0123456789abcdef0123456789abcdef01234567` | superseded |\n"
    )


def _register(decided: bool) -> str:
    def row(od: str, status: str, date: str) -> str:
        return f"| **{od}** | {status} | {date} | fixture |\n"

    closing = ("**OPERATOR APPROVED**", "2026-11-06") if decided else ("**PENDING**", "—")
    return (
        "# EVOL-X register (fixture)\n\n| ID | Status | Signed date | Notes |\n|---|---|---|---|\n"
        + row("OD-EVOL-05b", "SIGNED / APPROVED", "2026-11-01")
        + row("OD-EVOL-10", "SIGNED / APPROVED", "2026-11-01")
        + row("OD-EVOL-11", "SIGNED / APPROVED (auditor identity NOT NAMED)", "2026-11-01")
        + row("OD-EVOL-13", "**OPERATOR APPROVED**", "2026-11-01")
        + row("OD-EVOL-12", *closing)
        + row("OD-EVOLX-08", *closing)
    )


def _checklist(promoted: dict[str, tuple[str, str]] | None = None) -> str:
    lines = [
        "# Freeze checklist (fixture)\n",
        "| ID | Area | Criterion | Owner | Status | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for frz_id, requirement in pf.FRZ_EVOL_CANONICAL_REQUIREMENTS.items():
        status, notes = (promoted or {}).get(frz_id, ("OPEN", "—"))
        lines.append(f"| {frz_id} | Enterprise evolvability | {requirement} | **EVOL-X** | {status} | {notes} |")
    return "\n".join(lines) + "\n"


def _governance(**overrides: Any) -> dict[str, Any]:
    base = {
        "schema_version": "evol_x_auto_b6_governance_status.v1",
        "ecap_amendment_id": "ECAP-P4-EVOL-SCOPE-AUTO-B6",
        "normative_effect": "NOT_IN_FORCE",
        "b6_acceptance_publication_sha": None,
        "evol_x_closed": False,
    }
    base.update(overrides)
    return base


def _pass_record(tes: str, final: str) -> str:
    rows = "".join(f"| {k} | {v} | fixture |\n" for k, v in AUDITOR_FRZ.items())
    return (
        "# EVOL-X — IIAC independent audit PASS record (fixture)\n\n"
        f"Stage `EVOL-X` · policy `{POLICY}` · manifest verdict `PASS_SCOPED`\n"
        "Auditor role `independent-quality-auditor`\n"
        f"TES `{tes}` · FINAL `{final}`\n\n"
        "| FRZ_ID | Auditor result | Limitation |\n|---|---|---|\n" + rows
    )


def _handoff() -> str:
    return (
        "# IIAC session handoff record — EVOL-X certification (fixture)\n\n"
        f"**Policy:** `{POLICY}`\n\n## Operator attestation\n\n"
        f"- [x] `implementation_session_id`: `{IMPL_SESSION}` (approved historical exception)\n"
        f"- [x] `auditor_session_id`: `{AUDITOR_SESSION}`\n"
        f"- [x] `certifier_session_id`: `{CERTIFIER_SESSION}`\n\n"
        "## Operator signature line\n\n`Approved handoff: operator @ 2026-11-05T20:41:40+01:00`\n"
    )


def _build_base(repo: Path) -> dict[str, str]:
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "development")
    _build_activation(repo)

    for rel in (
        "testing_support/change_safety/impact.py",
        "tests/qualification/evol_x/test_fixture_scope.py",
        "tests/fixtures/evol_x/s1.json",
        "scripts/qualification/evol_x_p03_evidence.py",
        "scripts/qualification/evol_x_vl3_manifest.py",
    ):
        _write(repo, rel, f"# fixture {rel}\n")
    tes = _commit(repo, "EVOL-X TES (fixture)")

    _write(repo, f"{pf.EVOL_DIR}/evol_x_p03_independent_replay/MANIFEST.json", '{"fixture": true}\n')
    _write(repo, pf.REL_LADDER, _ladder(tes, "0" * 40, tes))
    _dump(repo, pf.REL_GOVERNANCE, _governance())
    _write(repo, pf.REL_REGISTER, _register(decided=False))
    _write(repo, pf.REL_FRZ_CHECKLIST, _checklist())
    final = _commit(repo, "EVOL-X FINAL evidence (fixture)")
    _write(repo, pf.REL_LADDER, _ladder(tes, final, tes))
    final_bound = _commit(repo, "EVOL-X ladder binds FINAL (fixture)")

    report = _write(repo, pf.REL_AUDIT_REPORT, "# Independent audit dossier (fixture)\n")
    scope = _write(repo, pf.REL_AUDIT_SCOPE, json.dumps({"stage_id": "EVOL-X", "in_scope": ["fixture"]}) + "\n")
    matrix = _write(repo, pf.REL_TEST_MATRIX, "fixture test matrix 1/1\n")
    findings = _dump(
        repo,
        pf.REL_FINDINGS,
        {
            "stage_id": "EVOL-X",
            "audited_implementation_sha": tes,
            "summary": {"PROVEN_ARCHITECTURAL_DEFECT": 0, "IN_SCOPE_BLOCKER": 0, "blocking_for_scoped_pass": 0},
            "findings": [
                {"id": "F-C-03", "class": "TRACKED_FREEZE_DEBT", "severity": "MEDIUM"},
                {"id": "F-C-08", "class": "ENVIRONMENT_TEST_ISSUE", "severity": "PROCEDURAL"},
            ],
        },
    )
    pass_record = _pass_record(tes, final)
    _write(repo, pf.REL_PASS_RECORD, pass_record)
    _write(repo, pf.REL_AUDIT_PASS_COPY, pass_record)
    _write(repo, f"{pf.AUDIT_DIR}/evidence/probes/probe_fixture.py", "x=1;y  =  2\n")
    draft = {
        "stage_id": "EVOL-X",
        "certification_profile": "iiac-v1",
        "policy_version": POLICY,
        "implementation_session_id": "UNRECORDED",
        "implementation_agent_id": "implementer",
        "auditor_session_id": AUDITOR_SESSION,
        "auditor_agent_id": "independent-quality-auditor",
        "certifier_session_id": "PENDING_CERTIFIER",
        "certifier_agent_id": "certification-governance-agent",
        "audited_implementation_sha": tes,
        "test_execution_sha": tes,
        "evidence_publication_sha": final,
        "audit_execution_sha": final,
        "audit_report_digest": _sha256(report),
        "audit_scope_manifest_digest": _sha256(scope),
        "test_matrix_digest": _sha256(matrix),
        "findings_digest": _sha256(findings),
        "verdict": "PASS_SCOPED",
        "limitations": AUDITOR_LIMITATIONS,
        "certification_publication_sha": "PENDING_CERTIFIER",
    }
    _dump(repo, pf.REL_AUDITOR_DRAFT, draft)
    index_raw = _index(repo, pf.AUDIT_DIR, pf.REL_AUDIT_INDEX)
    audit_pub = _commit(repo, "P1 publish independent audit evidence (fixture)")

    manifest = {
        **draft,
        "implementation_session_id": IMPL_SESSION,
        "certifier_session_id": CERTIFIER_SESSION,
        "limitations": CERTIFIED_LIMITATIONS,
        "certification_publication_sha": audit_pub,
        "assurance_level": "process_enforced",
        "handoff_record_path": pf.REL_CERT_HANDOFF,
        "audit_session_head_sha": final_bound,
        "audit_publication_sha": audit_pub,
        "audit_report_path": pf.REL_AUDIT_REPORT,
        "pass_record_path": pf.REL_PASS_RECORD,
        "evidence_index_path": pf.REL_AUDIT_INDEX,
    }
    manifest.pop("manifest_status", None)
    cert = {
        "schema_version": "iiac-certification-record.v1",
        "stage_id": "EVOL-X",
        "policy_version": POLICY,
        "independent_audit_verdict": "PASS_SCOPED",
        "certification_verdict": "PASS_SCOPED_EVIDENCE_PUBLICATION_ONLY",
        "stage_closure_status": "NOT_CLOSED",
        "b6_acceptance": "NOT_ISSUED",
        **{k: manifest[k] for k in ("audited_implementation_sha", "test_execution_sha", "evidence_publication_sha")},
        "audit_execution_sha": final,
        "audit_session_head_sha": final_bound,
        "audit_publication_sha": audit_pub,
        "certification_publication_sha": audit_pub,
        "b6_acceptance_publication_sha": None,
        "accepted_audit_report_digest": _sha256(report),
        "findings_digest": _sha256(findings),
        "audit_evidence_index_digest": _sha256(index_raw),
        "pass_record_digest": _sha256(pass_record.encode("utf-8")),
        "manifest_path": pf.REL_MANIFEST,
        "certifier_session_id": CERTIFIER_SESSION,
        "certifier_agent_id": "certification-governance-agent",
        "implementation_session_id": IMPL_SESSION,
        "frz_evol_dispositions": DISPOSITIONS,
        "open_blockers_for_closure": ["B-5 OD-EVOL-12 / OD-EVOLX-08 authority confirmation or signed rows"],
    }
    _dump(repo, pf.REL_MANIFEST, manifest)
    _dump(repo, pf.REL_CERT_RECORD, cert)
    _write(repo, pf.REL_CERT_HANDOFF, _handoff())
    _reindex_cert(repo)
    cert_pub = _commit(repo, "P2 publish certification evidence (fixture)")

    manifest["certification_publication_sha"] = cert_pub
    cert["certification_publication_sha"] = cert_pub
    _dump(repo, pf.REL_MANIFEST, manifest)
    _dump(repo, pf.REL_CERT_RECORD, cert)
    _reindex_cert(repo)
    head = _commit(repo, "Bind certification publication SHA (fixture)")
    return {"tes": tes, "final": final, "audit_pub": audit_pub, "cert_pub": cert_pub, "head": head}


def _publish_b6(repo: Path) -> str:
    """Two-phase B6 acceptance: phase 1 publishes, phase 2 binds the SHA (non-self-referential)."""
    _dump(
        repo,
        pf.REL_GOVERNANCE,
        _governance(formal_b6_status="ACCEPTED", b6_acceptance_policy_version=POLICY),
    )
    cert = _load(repo, pf.REL_CERT_RECORD)
    cert["b6_acceptance"] = "ISSUED"
    _dump(repo, pf.REL_CERT_RECORD, cert)
    _reindex_cert(repo)
    return _commit(repo, "P3 B6 acceptance phase 1 (fixture)")


def _close(repo: Path, built: dict[str, str]) -> str:
    b6 = _publish_b6(repo)
    _dump(
        repo,
        pf.REL_GOVERNANCE,
        _governance(formal_b6_status="ACCEPTED", b6_acceptance_policy_version=POLICY, b6_acceptance_publication_sha=b6),
    )
    _write(repo, pf.REL_LADDER, _ladder(built["tes"], built["final"], built["tes"], b6=b6))
    cert = _load(repo, pf.REL_CERT_RECORD)
    cert["b6_acceptance_publication_sha"] = b6
    cert["open_blockers_for_closure"] = []
    _dump(repo, pf.REL_CERT_RECORD, cert)
    _reindex_cert(repo)
    _write(repo, pf.REL_FRZ_CHECKLIST, _checklist(PROMOTED_ROWS))
    _write(repo, pf.REL_REGISTER, _register(decided=True))
    _commit(repo, "P3 bind B6 + P4 FRZ dispositions + operator decisions (fixture)")
    return b6


@pytest.fixture(scope="module")
def base_repo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, str]]:
    root = tmp_path_factory.mktemp("evol_x_b4_base") / "repo"
    built = _build_base(root)
    return root, built


@pytest.fixture(scope="module")
def closed_repo(tmp_path_factory: pytest.TempPathFactory, base_repo: tuple[Path, dict[str, str]]):
    src, built = base_repo
    root = tmp_path_factory.mktemp("evol_x_b4_closed") / "repo"
    shutil.copytree(src, root)
    b6 = _close(root, built)
    return root, {**built, "b6": b6}


@pytest.fixture()
def repo(tmp_path: Path, base_repo: tuple[Path, dict[str, str]]) -> tuple[Path, dict[str, str]]:
    src, built = base_repo
    dst = tmp_path / "repo"
    shutil.copytree(src, dst)
    return dst, built


@pytest.fixture()
def closed(tmp_path: Path, closed_repo) -> tuple[Path, dict[str, str]]:
    src, built = closed_repo
    dst = tmp_path / "repo"
    shutil.copytree(src, dst)
    return dst, built


def _eval(repo: Path) -> pf.PreflightResult:
    return pf.evaluate(repo, evidence_root=repo)


def _codes(errors: list[str]) -> set[str]:
    return {e.split(" ", 1)[0] for e in errors}


def _mutate_json(repo: Path, rel: str, change: Callable[[dict[str, Any]], None], *, reindex: bool = False) -> None:
    data = _load(repo, rel)
    change(data)
    _dump(repo, rel, data)
    if reindex:
        _reindex_cert(repo)
    _commit(repo, f"fixture mutation of {rel}")


def _assert_a_fails(repo: Path, code: str) -> pf.PreflightResult:
    result = _eval(repo)
    assert not result.gate_a_pass, "Gate A must fail closed"
    assert not result.gate_b_pass, "Gate B must fail closed"
    assert code in _codes(result.gate_a), result.gate_a
    return result


# --------------------------------------------------------------------------- positives


def test_valid_b6_readiness_without_formal_closure(repo) -> None:
    root, _ = repo
    result = _eval(root)
    assert result.mode == "IIAC"
    assert result.gate_a == []
    assert not result.gate_b_pass
    codes = _codes(result.gate_b)
    assert {"B-01", "B-02", "B-06", "B-07", "B-09"} <= codes
    assert "B-00" not in codes


def test_valid_formal_closure_fixture_with_complete_evidence(closed) -> None:
    root, _ = closed
    result = _eval(root)
    assert result.gate_a == []
    assert result.gate_b == []
    assert result.gate_b_pass


def test_cli_reports_both_gates_and_selects_exit_code(repo) -> None:
    root, _ = repo

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--repo", str(root), "--evidence-root", str(root), *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )

    readiness = run("--gate", "b6-readiness")
    closure = run()
    assert readiness.returncode == 0, readiness.stderr
    assert "Gate A - B6 acceptance readiness: PASS" in readiness.stdout
    assert "Gate B - formal EVOL-X closure readiness: FAIL" in readiness.stdout
    assert "FIXTURE (not canonical evidence)" in readiness.stdout
    assert closure.returncode == 1
    payload = json.loads(run("--json").stdout)
    assert payload["gate_a_pass"] is True and payload["gate_b_pass"] is False


# --------------------------------------------------------------------------- policy


def test_policy_claiming_in_force_without_valid_activation_fails_both_gates(repo) -> None:
    root, _ = repo
    _mutate_json(root, pf.REL_ACTIVATION, lambda r: r.update(effective_from_sha="f" * 40))
    result = _eval(root)
    assert result.mode == "INVALID"
    assert "A-01" in _codes(result.gate_a)
    assert not result.gate_b_pass


def test_policy_not_in_force_uses_legacy_path_and_rejects_iiac_evidence(repo) -> None:
    root, _ = repo
    _dump(root, pf.REL_ACTIVATION, {"status": "NOT_IN_FORCE", "policy_version": POLICY})
    _commit(root, "deactivate fixture policy")
    result = _eval(root)
    assert result.mode == "LEGACY"
    assert {"L-00", "L-01", "L-05"} <= _codes(result.gate_a)
    assert not result.gate_b_pass


def test_dual_activation_of_superseded_ecap_p4_rejected(repo) -> None:
    root, _ = repo
    _mutate_json(root, pf.REL_GOVERNANCE, lambda g: g.update(normative_effect="IN_FORCE"))
    _assert_a_fails(root, "A-02")


def test_canonical_repository_rejects_evidence_root_override() -> None:
    result = pf.evaluate(REPO, evidence_root=REPO)
    assert result.mode == "INVALID" and not result.gate_a_pass and not result.gate_b_pass


# --------------------------------------------------------------------------- pass record / manifest


def test_missing_pass_record(repo) -> None:
    root, _ = repo
    (root / pf.REL_PASS_RECORD).unlink()
    _commit(root, "drop pass record")
    _assert_a_fails(root, "A-14")


def test_empty_pass_record(repo) -> None:
    root, _ = repo
    _write(root, pf.REL_PASS_RECORD, b"")
    _commit(root, "empty pass record")
    _assert_a_fails(root, "A-14")


def test_forged_pass_record_with_plausible_text(repo) -> None:
    root, built = repo
    forged = _pass_record(built["tes"], built["final"]).replace("PASS — SCOPED", "PASS")
    _write(root, pf.REL_PASS_RECORD, forged)
    _commit(root, "forge pass record")
    result = _assert_a_fails(root, "A-14")
    assert any("digest" in e or "auditor copy" in e for e in result.gate_a)


def test_missing_manifest(repo) -> None:
    root, _ = repo
    (root / pf.REL_MANIFEST).unlink()
    _commit(root, "drop manifest")
    _assert_a_fails(root, "A-03")


def test_malformed_manifest(repo) -> None:
    root, _ = repo
    _write(root, pf.REL_MANIFEST, "{not json")
    _commit(root, "malformed manifest")
    _assert_a_fails(root, "A-03")


def test_schema_invalid_manifest(repo) -> None:
    root, _ = repo
    _mutate_json(root, pf.REL_MANIFEST, lambda m: m.pop("findings_digest"), reindex=True)
    _assert_a_fails(root, "A-04")


def test_wrong_stage_id(repo) -> None:
    root, _ = repo
    _mutate_json(root, pf.REL_MANIFEST, lambda m: m.update(stage_id="ENT-AUDIT-X"), reindex=True)
    _assert_a_fails(root, "A-05")


def test_wrong_certification_profile_and_policy(repo) -> None:
    root, _ = repo
    _mutate_json(root, pf.REL_MANIFEST, lambda m: m.update(policy_version="IIAC-2099-01-01.9"), reindex=True)
    _assert_a_fails(root, "A-05")


def test_wrong_sha_not_a_commit(repo) -> None:
    root, _ = repo
    _mutate_json(root, pf.REL_MANIFEST, lambda m: m.update(audit_publication_sha="a" * 40), reindex=True)
    _assert_a_fails(root, "A-10")


def test_wrong_sha_breaks_ancestry_and_auditor_binding(repo) -> None:
    root, built = repo
    _mutate_json(root, pf.REL_MANIFEST, lambda m: m.update(audited_implementation_sha=built["final"]), reindex=True)
    result = _assert_a_fails(root, "A-07")
    assert "A-11" in _codes(result.gate_a) or "A-10" in _codes(result.gate_a)


def test_wrong_digest_consistently_reindexed(repo) -> None:
    root, _ = repo
    _mutate_json(root, pf.REL_MANIFEST, lambda m: m.update(audit_report_digest="b" * 64), reindex=True)
    _assert_a_fails(root, "A-12")


def test_tampered_audit_evidence_bytes(repo) -> None:
    root, _ = repo
    _write(root, pf.REL_AUDIT_REPORT, "# rewritten dossier\n")
    _commit(root, "rewrite dossier")
    result = _assert_a_fails(root, "A-12")
    assert "A-13" in _codes(result.gate_a)


def test_uncommitted_evidence_edit_rejected(repo) -> None:
    root, _ = repo
    _write(root, pf.REL_CERT_HANDOFF, _handoff() + "\nlate edit\n")
    _assert_a_fails(root, "A-16")


def test_unindexed_evidence_rejected(repo) -> None:
    root, _ = repo
    _write(root, f"{pf.AUDIT_DIR}/injected.md", "injected after publication\n")
    _commit(root, "inject evidence")
    _assert_a_fails(root, "A-13")


# --------------------------------------------------------------------------- roles / staleness


def test_self_certified_audit_session(repo) -> None:
    root, _ = repo
    _mutate_json(root, pf.REL_MANIFEST, lambda m: m.update(certifier_session_id=AUDITOR_SESSION), reindex=True)
    _assert_a_fails(root, "A-09")


def test_self_certified_agent_identity(repo) -> None:
    root, _ = repo
    _mutate_json(root, pf.REL_MANIFEST, lambda m: m.update(implementation_agent_id="independent-quality-auditor"))
    _assert_a_fails(root, "A-09")


def test_stale_audit_after_scope_change(repo) -> None:
    root, _ = repo
    _write(root, "testing_support/change_safety/impact.py", "# changed after the audit\n")
    _commit(root, "post-audit change to audited scope")
    _assert_a_fails(root, "A-11")


def test_stale_audit_after_ladder_moves_tes(repo) -> None:
    root, built = repo
    _write(root, pf.REL_LADDER, _ladder(built["audit_pub"], built["final"], built["tes"]))
    _commit(root, "ladder points at a new TES")
    _assert_a_fails(root, "A-11")


def test_unresolved_blocking_finding(repo) -> None:
    root, _ = repo
    findings = _load(root, pf.REL_FINDINGS)
    findings["findings"].append({"id": "F-C-99", "class": "IN_SCOPE_BLOCKER", "severity": "HIGH"})
    _dump(root, pf.REL_FINDINGS, findings)
    _commit(root, "add blocking finding")
    _assert_a_fails(root, "A-15")


def test_unattested_handoff(repo) -> None:
    root, _ = repo
    _write(root, pf.REL_CERT_HANDOFF, _handoff().replace("2026-11-05T20:41:40+01:00", "________________"))
    _reindex_cert(root)
    _commit(root, "unsign handoff")
    _assert_a_fails(root, "A-16")


def test_open_b4_blocker_holds_gate_a(repo) -> None:
    root, _ = repo
    _mutate_json(
        root,
        pf.REL_CERT_RECORD,
        lambda c: c["open_blockers_for_closure"].append("B-4 corrective preflight re-audit pending"),
        reindex=True,
    )
    _assert_a_fails(root, "A-18")


# --------------------------------------------------------------------------- scope widening


def test_silent_promotion_pass_scoped_to_pass(repo) -> None:
    root, _ = repo

    def widen(m: dict[str, Any]) -> None:
        m["verdict"] = "PASS"

    _mutate_json(root, pf.REL_MANIFEST, widen, reindex=True)
    _assert_a_fails(root, "A-07")


@pytest.mark.parametrize(
    ("lim_id", "replacement"),
    [
        ("LIM-02", "LIM-02 AUTOMATION_FIRST_SCOPED: EVOL-M-09H MEASURED"),
        ("LIM-03", "LIM-03 TESTED_BY ownership complete platform-wide"),
        ("LIM-04", "LIM-04 EVOL-M-03 measured on all platform arcs"),
        ("LIM-05", "LIM-05 production ActivationService.rollback enforcement demonstrated"),
    ],
)
def test_scope_widening_by_rewriting_limitation(repo, lim_id: str, replacement: str) -> None:
    root, _ = repo

    def widen(m: dict[str, Any]) -> None:
        m["limitations"] = [replacement if x.startswith(lim_id) else x for x in m["limitations"]]

    _mutate_json(root, pf.REL_MANIFEST, widen, reindex=True)
    result = _assert_a_fails(root, "A-07")
    assert "A-08" in _codes(result.gate_a)


def test_scope_widening_by_dropping_limitation(repo) -> None:
    root, _ = repo
    _mutate_json(
        root,
        pf.REL_MANIFEST,
        lambda m: m.update(limitations=[x for x in m["limitations"] if not x.startswith("LIM-05")]),
        reindex=True,
    )
    _assert_a_fails(root, "A-08")


@pytest.mark.parametrize(
    ("frz_id", "disposition"),
    [
        ("FRZ-EVOL-07", "ELIGIBLE_PASS"),
        ("FRZ-EVOL-08", "ELIGIBLE_PASS_SCOPED(LIM-02)"),
        ("FRZ-EVOL-01", "ELIGIBLE_PASS"),
    ],
)
def test_scope_widening_in_certified_frz_disposition(repo, frz_id: str, disposition: str) -> None:
    root, _ = repo
    _mutate_json(
        root, pf.REL_CERT_RECORD, lambda c: c["frz_evol_dispositions"].update({frz_id: disposition}), reindex=True
    )
    _assert_a_fails(root, "A-08")


# --------------------------------------------------------------------------- legacy evidence


def test_historical_legacy_record_used_as_iiac_pass_record(repo) -> None:
    root, _ = repo
    legacy = f"{pf.EVOL_DIR}/EVOL_X_INTERNAL_B6_ACCEPTANCE_RECORD.md"
    _write(root, legacy, "historical internal B6 acceptance\n")
    _mutate_json(root, pf.REL_MANIFEST, lambda m: m.update(pass_record_path=legacy), reindex=True)
    _assert_a_fails(root, "A-14")


def test_historical_sha_used_as_b6_acceptance(closed) -> None:
    root, built = closed

    def legacy(g: dict[str, Any]) -> None:
        g["b6_acceptance_publication_sha"] = built["tes"]

    _mutate_json(root, pf.REL_GOVERNANCE, legacy)
    cert = _load(root, pf.REL_CERT_RECORD)
    cert["b6_acceptance_publication_sha"] = built["tes"]
    _dump(root, pf.REL_CERT_RECORD, cert)
    _reindex_cert(root)
    _write(root, pf.REL_LADDER, _ladder(built["tes"], built["final"], built["tes"], b6=built["tes"]))
    _commit(root, "point B6 at a historical SHA")
    result = _eval(root)
    assert result.gate_a == []
    assert "B-03" in _codes(result.gate_b)
    assert not result.gate_b_pass


# --------------------------------------------------------------------------- Gate B negatives


def test_missing_b6_publication_sha_at_formal_closure(closed) -> None:
    root, built = closed
    _mutate_json(root, pf.REL_GOVERNANCE, lambda g: g.update(b6_acceptance_publication_sha=None))
    _write(root, pf.REL_LADDER, _ladder(built["tes"], built["final"], built["tes"]))
    _commit(root, "unset B6")
    result = _eval(root)
    assert result.gate_a_pass
    assert {"B-01", "B-02"} <= _codes(result.gate_b)
    assert not result.gate_b_pass


def test_self_referential_b6_sha(closed) -> None:
    root, built = closed
    head = _git(root, "rev-parse", "HEAD")
    _mutate_json(root, pf.REL_GOVERNANCE, lambda g: g.update(b6_acceptance_publication_sha=head))
    _write(root, pf.REL_LADDER, _ladder(built["tes"], built["final"], built["tes"], b6=head))
    _commit(root, "rebind B6 to previous HEAD")
    # B6 now points at a commit whose governance blob already carries a SHA (not a phase-1 publication).
    result = _eval(root)
    assert "B-04" in _codes(result.gate_b) or "B-03" in _codes(result.gate_b)
    assert not result.gate_b_pass


def test_missing_required_operator_decisions(closed) -> None:
    root, _ = closed
    _write(root, pf.REL_REGISTER, _register(decided=False))
    _commit(root, "OD-EVOL-12 / OD-EVOLX-08 pending")
    result = _eval(root)
    assert result.gate_a_pass
    errors = [e for e in result.gate_b if e.startswith("B-07")]
    assert any("OD-EVOL-12" in e for e in errors) and any("OD-EVOLX-08" in e for e in errors)
    assert not result.gate_b_pass


def test_od_evol_11_named_auditor_is_not_required_under_iiac(closed) -> None:
    root, _ = closed
    register = (root / pf.REL_REGISTER).read_text(encoding="utf-8")
    assert "NOT NAMED" in register
    assert _eval(root).gate_b_pass


@pytest.mark.parametrize(
    ("frz_id", "status", "notes", "fragment"),
    [
        (
            "FRZ-EVOL-07",
            "PASS",
            "LIM-05; qualification-layer only; production rollback enforcement NOT claimed",
            "falsely",
        ),
        ("FRZ-EVOL-08", "PASS — SCOPED", "LIM-02; change-time diagnostics", "M-09H"),
        ("FRZ-EVOL-06", "PASS — SCOPED", "LIM-01, LIM-04; EVOL_CONSERVATIVE_FULL scope only", "LIM-03"),
        ("FRZ-EVOL-03", "OPEN", "—", "no individual PASS"),
    ],
)
def test_frz_row_falsely_promoted_or_missing(closed, frz_id: str, status: str, notes: str, fragment: str) -> None:
    root, _ = closed
    _write(root, pf.REL_FRZ_CHECKLIST, _checklist({**PROMOTED_ROWS, frz_id: (status, notes)}))
    _commit(root, f"tamper {frz_id}")
    result = _eval(root)
    assert result.gate_a_pass
    assert any(e.startswith("B-06") and frz_id in e and fragment in e for e in result.gate_b), result.gate_b


def test_mass_promotion_without_canonical_requirement_text(closed) -> None:
    root, _ = closed
    text = (root / pf.REL_FRZ_CHECKLIST).read_text(encoding="utf-8")
    _write(root, pf.REL_FRZ_CHECKLIST, text.replace("explicit (in-flight instances)", "explicit"))
    _commit(root, "rewrite criterion")
    assert "B-06" in _codes(_eval(root).gate_b)


def test_open_blocker_holds_formal_closure(closed) -> None:
    root, _ = closed
    _mutate_json(
        root,
        pf.REL_CERT_RECORD,
        lambda c: c.update(open_blockers_for_closure=["B-5 OD-EVOLX-08 confirmation"]),
        reindex=True,
    )
    result = _eval(root)
    assert result.gate_a_pass
    assert "B-09" in _codes(result.gate_b)


def test_gate_a_pass_never_implies_gate_b(repo) -> None:
    root, _ = repo
    result = _eval(root)
    assert result.gate_a_pass and not result.gate_b_pass


# --------------------------------------------------------------------------- canonical repository


def test_canonical_repository_formal_closure_fails_closed_while_b6_unpublished() -> None:
    governance = json.loads((REPO / pf.REL_GOVERNANCE).read_text(encoding="utf-8"))
    if governance.get("b6_acceptance_publication_sha"):
        pytest.skip("B6 acceptance published; canonical pre-B6 assertion no longer applies")
    result = pf.evaluate(REPO)
    assert result.mode == "IIAC", result.gate_a
    assert not result.gate_b_pass
    cert = json.loads((REPO / pf.REL_CERT_RECORD).read_text(encoding="utf-8"))
    if cert.get("open_blockers_for_closure"):
        # Evidence validates; only the certifier's open blockers hold B6 readiness.
        assert _codes(result.gate_a) <= {"A-18"}, result.gate_a
    rows = pf.parse_frz_rows((REPO / pf.REL_FRZ_CHECKLIST).read_text(encoding="utf-8"))
    assert all(_codes([c[4]]) == {"OPEN"} for frz in rows.values() for c in frz)
