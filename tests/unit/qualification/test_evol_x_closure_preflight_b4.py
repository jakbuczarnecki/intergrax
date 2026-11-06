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
from datetime import UTC, datetime, timedelta
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


B4_TEXT = "B-4 legacy closure preflight requires independent corrective implementation and re-audit"
B5_TEXT = "B-5 OD-EVOL-12 / OD-EVOLX-08 authority confirmation or signed rows"
P3_TEXT = "B6 acceptance (P3), FRZ promotion (P4) and formal closure (P5) not authorized in this publication"
REAUDIT_DIR = f"{pf.EVIDENCE_ROOT_REL}/b4_delta_reaudit"
REAUDIT_INDEX = f"{REAUDIT_DIR}/EVIDENCE_INDEX.json"
REAUDIT_MANIFEST = f"{REAUDIT_DIR}/iiac_audit_session_manifest.EVOL-X.b4_reaudit.json"
REAUDIT_IMPL_SESSION = "FIXTURE-B4-CORRECTIVE-IMPL-SESSION"
REAUDIT_AUDITOR_SESSION = "FIXTURE-B4-REAUDIT-SESSION__claude-code-cccc"
REAUDIT_CERT_SESSION = "FIXTURE-B4-RECERT-SESSION__claude-code-dddd"
P3_DECISION_ID = "P3-FIXTURE-AUTH"
P3_RECORD = f"{pf.OPERATOR_DIR}/OD_P3_B6_AUTHORIZATION_FIXTURE.md"
OD12_RECORD = f"{pf.OPERATOR_DIR}/OD_EVOL_12_FIXTURE.md"
ODX08_RECORD = f"{pf.OPERATOR_DIR}/OD_EVOLX_08_FIXTURE.md"
OD12_SCOPE = "FRZ-EVOL-07, FRZ-EVOL-08 and FRZ-EVOL-10 scoped promotion wording"
ODX08_SCOPE = "EVOL-X formal closure after B6 acceptance and FRZ promotion"
B6_DECISION_ID = "B6-ACCEPT-EVOL-X-FIXTURE"
B6_RECORD_NAME = "EVOL_X_IIAC_B6_ACCEPTANCE_RECORD.md"


def _gitattributes(*, pass_record_rule: bool = True) -> str:
    """Fixture attributes derived from the canonical ``.gitattributes`` so the real rules are exercised."""
    lines = []
    for line in (REPO / ".gitattributes").read_text(encoding="utf-8").splitlines():
        if line.strip().endswith("-text") and "qualification" in line:
            if "EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD" in line and not pass_record_rule:
                continue
            lines.append(line.strip())
    lines.append("iiac_gov/** -text")
    return "\n".join(lines) + "\n"


def _now_iso(delta: timedelta | None = None) -> str:
    return (datetime.now(UTC) + (delta or timedelta())).replace(microsecond=0).isoformat()


def _operator_record(
    decision_id: str,
    scope: str,
    *,
    status: str = "AUTHORIZED",
    stamp: str | None = None,
    signed: bool = True,
    extra: dict[str, str] | None = None,
) -> str:
    signature = (
        f"Approved {decision_id}: operator @ {stamp or _now_iso()}"
        if signed
        else "Operator signature: ________________"
    )
    lines = [
        f"# Operator decision record {decision_id} (fixture)",
        "",
        f"Decision ID: {decision_id}",
        f"Stage: {pf.STAGE_ID}",
        f"Policy version: {POLICY}",
        f"Decision scope: {scope}",
        f"Decision status: {status}",
        *[f"{k}: {v}" for k, v in (extra or {}).items()],
        "",
        signature,
        "",
    ]
    return "\n".join(lines)


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
    _write(repo, ".gitattributes", _gitattributes())
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


def _register(decided: bool, notes: dict[str, str] | None = None, *, closing_status: str | None = None) -> str:
    def row(od: str, status: str, date: str) -> str:
        return f"| **{od}** | {status} | {date} | {(notes or {}).get(od, 'fixture')} |\n"

    closing = (closing_status or "**OPERATOR APPROVED**", _now_iso()[:10]) if decided else ("**PENDING**", "—")
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
        "open_blockers_for_closure": [B4_TEXT, B5_TEXT, P3_TEXT],
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
    built = {"tes": tes, "final": final, "audit_pub": audit_pub, "cert_pub": cert_pub, "head": head}
    built.update(_resolve_a_blockers(repo, built))
    return built


def _reaudit_manifest(corrective: str, report: bytes, scope: bytes, matrix: bytes, findings: bytes) -> dict[str, Any]:
    return {
        "stage_id": "EVOL-X",
        "certification_profile": "iiac-v1",
        "policy_version": POLICY,
        "implementation_session_id": REAUDIT_IMPL_SESSION,
        "implementation_agent_id": "implementer",
        "auditor_session_id": REAUDIT_AUDITOR_SESSION,
        "auditor_agent_id": "independent-quality-auditor",
        "certifier_session_id": REAUDIT_CERT_SESSION,
        "certifier_agent_id": "certification-governance-agent",
        "audited_implementation_sha": corrective,
        "test_execution_sha": corrective,
        "evidence_publication_sha": corrective,
        "audit_execution_sha": corrective,
        "certification_publication_sha": corrective,
        "audit_report_digest": _sha256(report),
        "audit_scope_manifest_digest": _sha256(scope),
        "test_matrix_digest": _sha256(matrix),
        "findings_digest": _sha256(findings),
        "verdict": "PASS",
        "limitations": ["Delta re-audit of RA-B4-01..04 only"],
        "assurance_level": "process_enforced",
        "audit_report_path": f"{REAUDIT_DIR}/REAUDIT_DOSSIER.md",
        "findings_path": f"{REAUDIT_DIR}/REAUDIT_FINDINGS.json",
        "handoff_record_path": f"{REAUDIT_DIR}/IIAC_SESSION_HANDOFF_RECORD_EVOL_X_B4_REAUDIT.md",
    }


def _reaudit_handoff() -> str:
    return (
        "# IIAC session handoff record — B-4 delta re-audit (fixture)\n\n"
        f"**Policy:** `{POLICY}`\n\n## Operator attestation\n\n"
        f"- [x] `implementation_session_id`: `{REAUDIT_IMPL_SESSION}`\n"
        f"- [x] `auditor_session_id`: `{REAUDIT_AUDITOR_SESSION}`\n"
        f"- [x] `certifier_session_id`: `{REAUDIT_CERT_SESSION}`\n\n"
        "## Operator signature line\n\n`Approved handoff: operator @ 2026-11-06T09:00:00+01:00`\n"
    )


def _b4_entry(corrective: str, closure: str, cert_pub: str, manifest_raw: bytes, index_raw: bytes) -> dict[str, Any]:
    manifest = json.loads(manifest_raw)
    return {
        "blocker_id": "B-4",
        "status": "RESOLVED_WITH_EVIDENCE",
        "baseline_text_sha256": _sha256(B4_TEXT.encode()),
        "opened_at_sha": cert_pub,
        "resolution_kind": "INDEPENDENT_REAUDIT",
        "closure_publication_sha": closure,
        "resolved_implementation_sha": corrective,
        "closure_evidence_index_path": REAUDIT_INDEX,
        "closure_evidence_index_digest": _sha256(index_raw),
        "reaudit_manifest_path": REAUDIT_MANIFEST,
        "reaudit_manifest_digest": _sha256(manifest_raw),
        "certifier_disposition": {
            "disposition": "ACCEPTED",
            "certifier_agent_id": "certification-governance-agent",
            "certifier_session_id": REAUDIT_CERT_SESSION,
            "policy_version": POLICY,
            "audit_report_digest": manifest["audit_report_digest"],
            "findings_digest": manifest["findings_digest"],
        },
    }


def _operator_entry(
    blocker_id: str, baseline: str, cert_pub: str, closure: str, items: list[dict[str, str]]
) -> dict[str, Any]:
    return {
        "blocker_id": blocker_id,
        "status": "RESOLVED_WITH_EVIDENCE",
        "baseline_text_sha256": _sha256(baseline.encode()),
        "opened_at_sha": cert_pub,
        "resolution_kind": "OPERATOR_AUTHORIZATION",
        "closure_publication_sha": closure,
        "closure_evidence": items,
    }


def _resolve_a_blockers(repo: Path, built: dict[str, str]) -> dict[str, str]:
    """Independent B-4 re-audit + operator P3 authorization, published and then recorded by the certifier."""
    _write(repo, "scripts/fixture_b4_corrective.py", "# fixture corrective implementation\n")
    corrective = _commit(repo, "B-4 corrective implementation (fixture)")
    report = _write(repo, f"{REAUDIT_DIR}/REAUDIT_DOSSIER.md", "# Delta re-audit dossier (fixture)\n")
    scope = _write(repo, f"{REAUDIT_DIR}/audit_scope.json", '{"stage_id":"EVOL-X"}\n')
    matrix = _write(repo, f"{REAUDIT_DIR}/test_matrix.txt", "fixture re-audit matrix\n")
    findings = _dump(
        repo,
        f"{REAUDIT_DIR}/REAUDIT_FINDINGS.json",
        {
            "stage_id": "EVOL-X",
            "audited_implementation_sha": corrective,
            "summary": {"PROVEN_ARCHITECTURAL_DEFECT": 0, "IN_SCOPE_BLOCKER": 0, "blocking_for_scoped_pass": 0},
            "findings": [
                {"id": f"RA-B4-0{n}", "severity": "HIGH", "disposition": "RESOLVED", "blocks": ["B-4 acceptance"]}
                for n in range(1, 5)
            ],
        },
    )
    manifest_raw = _dump(repo, REAUDIT_MANIFEST, _reaudit_manifest(corrective, report, scope, matrix, findings))
    _write(repo, f"{REAUDIT_DIR}/IIAC_SESSION_HANDOFF_RECORD_EVOL_X_B4_REAUDIT.md", _reaudit_handoff())
    index_raw = _index(repo, REAUDIT_DIR, REAUDIT_INDEX)
    p3_raw = _write(repo, P3_RECORD, _operator_record(P3_DECISION_ID, "B6 acceptance authorization for EVOL-X"))
    closure = _commit(repo, "B-4 re-audit evidence + P3 operator authorization (fixture)")
    cert = _load(repo, pf.REL_CERT_RECORD)
    cert["open_blockers_for_closure"] = [B5_TEXT]
    cert["resolved_blockers_for_closure"] = [
        _b4_entry(corrective, closure, built["cert_pub"], manifest_raw, index_raw),
        _operator_entry(
            "B6-P3",
            P3_TEXT,
            built["cert_pub"],
            closure,
            [{"decision_id": P3_DECISION_ID, "path": P3_RECORD, "sha256": _sha256(p3_raw)}],
        ),
    ]
    _dump(repo, pf.REL_CERT_RECORD, cert)
    _reindex_cert(repo)
    resolved_head = _commit(repo, "Certifier records B-4 and P3 resolutions (fixture)")
    return {"corrective": corrective, "reaudit_closure": closure, "resolved_head": resolved_head}


def _b6_record_text(repo: Path, **overrides: str) -> str:
    manifest = _load(repo, pf.REL_MANIFEST)
    fields = {
        "Accepted verdict": manifest["verdict"],
        "Accepted limitations": ", ".join(sorted(pf._limitation_map(manifest["limitations"]))),
        "Independent audit manifest digest": _sha256((repo / pf.REL_MANIFEST).read_bytes()),
        "Audit report digest": manifest["audit_report_digest"],
        "Findings digest": manifest["findings_digest"],
        "Audited implementation SHA": manifest["audited_implementation_sha"],
        "Certification publication SHA": manifest["certification_publication_sha"],
        "Certifier session": manifest["certifier_session_id"],
        "Certifier agent": "certification-governance-agent",
    }
    control = {"scope", "status", "stamp", "signed", "decision_id"}
    fields.update({k: v for k, v in overrides.items() if k not in control})
    return _operator_record(
        overrides.get("decision_id", B6_DECISION_ID),
        overrides.get("scope", "B6 acceptance for EVOL-X (PASS_SCOPED, evidence-bound)"),
        status=overrides.get("status", "AUTHORIZED"),
        stamp=overrides.get("stamp"),
        signed=overrides.get("signed", "yes") == "yes",
        extra=fields,
    )


def _publish_b6(repo: Path, *, with_record: bool = True, **record_overrides: str) -> str:
    """Two-phase B6 acceptance: phase 1 publishes (record + flags), phase 2 binds the SHA (non-self-referential)."""
    _dump(
        repo,
        pf.REL_GOVERNANCE,
        _governance(formal_b6_status="ACCEPTED", b6_acceptance_policy_version=POLICY),
    )
    cert = _load(repo, pf.REL_CERT_RECORD)
    cert["b6_acceptance"] = "ISSUED"
    if with_record:
        record = _write(repo, f"{pf.B6_DIR}/{B6_RECORD_NAME}", _b6_record_text(repo, **record_overrides))
        _index(repo, pf.B6_DIR, pf.REL_B6_INDEX)
        cert["b6_acceptance_record_path"] = pf.REL_B6_RECORD
        cert["b6_acceptance_record_digest"] = _sha256(record)
    _dump(repo, pf.REL_CERT_RECORD, cert)
    _reindex_cert(repo)
    return _commit(repo, "P3 B6 acceptance phase 1 (fixture)")


def _bind_b6(repo: Path, built: dict[str, str], b6: str) -> None:
    """Phase 2: bind the published B6 SHA into governance, ladder and the certification record."""
    _dump(
        repo,
        pf.REL_GOVERNANCE,
        _governance(formal_b6_status="ACCEPTED", b6_acceptance_policy_version=POLICY, b6_acceptance_publication_sha=b6),
    )
    _write(repo, pf.REL_LADDER, _ladder(built["tes"], built["final"], built["tes"], b6=b6))
    cert = _load(repo, pf.REL_CERT_RECORD)
    cert["b6_acceptance_publication_sha"] = b6
    _dump(repo, pf.REL_CERT_RECORD, cert)
    _reindex_cert(repo)


def _publish_decisions(repo: Path, *, od12_scope: str = OD12_SCOPE) -> dict[str, Any]:
    """OD-EVOL-12 / OD-EVOLX-08 operator records (a separate operator act after B6 acceptance)."""
    od12 = _write(repo, OD12_RECORD, _operator_record("OD-EVOL-12", od12_scope))
    odx08 = _write(repo, ODX08_RECORD, _operator_record("OD-EVOLX-08", ODX08_SCOPE))
    sha = _commit(repo, "OD-EVOL-12 / OD-EVOLX-08 operator decision records (fixture)")
    return {"sha": sha, "od12": od12, "odx08": odx08}


def _decision_notes(decisions: dict[str, Any]) -> dict[str, str]:
    base = "../iiac_evidence/EVOL-X/operator_decisions"
    return {
        "OD-EVOL-12": f"[record]({base}/{Path(OD12_RECORD).name}) sha256:{_sha256(decisions['od12'])}",
        "OD-EVOLX-08": f"[record]({base}/{Path(ODX08_RECORD).name}) sha256:{_sha256(decisions['odx08'])}",
    }


def _finish_close(repo: Path, built: dict[str, str], b6: str, decisions: dict[str, Any]) -> None:
    _bind_b6(repo, built, b6)
    cert = _load(repo, pf.REL_CERT_RECORD)
    cert["open_blockers_for_closure"] = []
    cert["resolved_blockers_for_closure"].append(
        _operator_entry(
            "B-5",
            B5_TEXT,
            built["cert_pub"],
            decisions["sha"],
            [
                {"decision_id": "OD-EVOL-12", "path": OD12_RECORD, "sha256": _sha256(decisions["od12"])},
                {"decision_id": "OD-EVOLX-08", "path": ODX08_RECORD, "sha256": _sha256(decisions["odx08"])},
            ],
        )
    )
    _dump(repo, pf.REL_CERT_RECORD, cert)
    _reindex_cert(repo)
    _write(repo, pf.REL_FRZ_CHECKLIST, _checklist(PROMOTED_ROWS))
    _write(repo, pf.REL_REGISTER, _register(decided=True, notes=_decision_notes(decisions)))
    _commit(repo, "P3 bind B6 + P4 FRZ dispositions + operator decisions (fixture)")


def _close(repo: Path, built: dict[str, str]) -> str:
    b6 = _publish_b6(repo)
    decisions = _publish_decisions(repo)
    _finish_close(repo, built, b6, decisions)
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
        lambda c: c.update(
            open_blockers_for_closure=["B-5 OD-EVOLX-08 confirmation"],
            resolved_blockers_for_closure=[e for e in c["resolved_blockers_for_closure"] if e["blocker_id"] != "B-5"],
        ),
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


# =========================================================================== RA-B4 corrective (T-IGX-017-B)
#
# Independent re-audit findings RA-B4-01..04. Every case below runs against an isolated temporary
# git fixture; the canonical activation record and the original auditor evidence are never touched.


def _clone(src: Path, dst: Path, *, autocrlf: bool) -> Path:
    """A fresh checkout of the fixture with the given ``core.autocrlf`` (Git for Windows default: true)."""
    flag = "true" if autocrlf else "false"
    proc = subprocess.run(
        ["git", "clone", "-q", "-c", f"core.autocrlf={flag}", "--no-hardlinks", str(src), str(dst)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return dst


def _b(errors: list[str], code: str) -> list[str]:
    return [e for e in errors if e.startswith(code)]


def _resolved(root: Path, blocker_id: str) -> dict[str, Any]:
    cert = _load(root, pf.REL_CERT_RECORD)
    return next(e for e in cert["resolved_blockers_for_closure"] if e["blocker_id"] == blocker_id)


def _edit_resolution(root: Path, blocker_id: str, change: Callable[[dict[str, Any]], None]) -> None:
    def apply(cert: dict[str, Any]) -> None:
        change(next(e for e in cert["resolved_blockers_for_closure"] if e["blocker_id"] == blocker_id))

    _mutate_json(root, pf.REL_CERT_RECORD, apply, reindex=True)


# --------------------------------------------------------------------------- RA-B4-01 fresh checkout / CRLF


def test_ra_b4_01_pass_record_is_protected_from_eol_conversion_in_canonical_attributes() -> None:
    proc = subprocess.run(
        ["git", "check-attr", "text", "--", pf.REL_PASS_RECORD, pf.REL_AUDIT_PASS_COPY, pf.REL_ACTIVATION],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("text: unset") == 3, proc.stdout


def test_ra_b4_01_gitattributes_rule_is_narrow() -> None:
    rules = [
        line.strip()
        for line in (REPO / ".gitattributes").read_text(encoding="utf-8").splitlines()
        if "EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD" in line
    ]
    assert rules == [f"{pf.REL_PASS_RECORD} -text"]


def test_ra_b4_01_clean_lf_checkout_passes(closed, tmp_path: Path) -> None:
    src, _ = closed
    clone = _clone(src, tmp_path / "l", autocrlf=False)
    result = _eval(clone)
    assert result.gate_a == [] and result.gate_b == [], (result.gate_a, result.gate_b)


def test_ra_b4_01_clean_crlf_checkout_with_text_attribute_passes(closed, tmp_path: Path) -> None:
    src, _ = closed
    clone = _clone(src, tmp_path / "c", autocrlf=True)
    # The attribute keeps the digest-bound pass record at its blob bytes even on a converting checkout ...
    assert b"\r" not in (clone / pf.REL_PASS_RECORD).read_bytes()
    # ... while unprotected text files really are converted (the fixture reproduces the original condition).
    assert b"\r\n" in (clone / pf.REL_LADDER).read_bytes()
    result = _eval(clone)
    assert result.gate_a == [] and result.gate_b == [], (result.gate_a, result.gate_b)


def test_ra_b4_01_clean_crlf_checkout_without_attribute_passes_via_blob_contract(closed, tmp_path: Path) -> None:
    src, _ = closed
    bare = tmp_path / "na"
    shutil.copytree(src, bare)
    _write(bare, ".gitattributes", _gitattributes(pass_record_rule=False))
    _commit(bare, "drop the pass record attribute")
    clone = _clone(bare, tmp_path / "nc", autocrlf=True)
    on_disk = (clone / pf.REL_PASS_RECORD).read_bytes()
    assert b"\r\n" in on_disk, "fixture must reproduce the RA-B4-01 CRLF materialisation"
    assert on_disk != (clone / pf.REL_AUDIT_PASS_COPY).read_bytes()
    result = _eval(clone)
    assert result.gate_a == [] and result.gate_b == [], (result.gate_a, result.gate_b)


def test_ra_b4_01_raw_comparison_would_fail_but_canonical_bytes_hold(closed, tmp_path: Path) -> None:
    """Guard against a vacuous fixture: the raw working-tree bytes differ, the canonical bytes do not."""
    src, _ = closed
    bare = tmp_path / "na"
    shutil.copytree(src, bare)
    _write(bare, ".gitattributes", _gitattributes(pass_record_rule=False))
    _commit(bare, "drop the pass record attribute")
    clone = _clone(bare, tmp_path / "nc", autocrlf=True)
    assert (clone / pf.REL_PASS_RECORD).read_bytes() != (clone / pf.REL_AUDIT_PASS_COPY).read_bytes()
    ctx = pf.Context(clone, clone / pf.REL_ACTIVATION, clone, head=_git(clone, "rev-parse", "HEAD"))
    assert pf._read_bytes(ctx, pf.REL_PASS_RECORD) == (clone / pf.REL_AUDIT_PASS_COPY).read_bytes()


@pytest.mark.parametrize("autocrlf", [False, True])
def test_ra_b4_01_modified_committed_evidence_fails_in_every_checkout(closed, tmp_path: Path, autocrlf: bool) -> None:
    src, _ = closed
    clone = _clone(src, tmp_path / "k", autocrlf=autocrlf)
    _write(clone, pf.REL_PASS_RECORD, (clone / pf.REL_PASS_RECORD).read_text(encoding="utf-8") + "\nforged line\n")
    _commit(clone, "modify the committed pass record")
    result = _eval(clone)
    assert not result.gate_a_pass
    assert _b(result.gate_a, "A-14"), result.gate_a


@pytest.mark.parametrize("autocrlf", [False, True])
def test_ra_b4_01_uncommitted_edit_fails_even_when_it_is_only_in_the_working_tree(
    closed, tmp_path: Path, autocrlf: bool
) -> None:
    src, _ = closed
    clone = _clone(src, tmp_path / "k", autocrlf=autocrlf)
    path = clone / pf.REL_PASS_RECORD
    path.write_bytes(path.read_bytes().replace(b"PASS", b"FAIL", 1))
    result = _eval(clone)
    assert not result.gate_a_pass
    assert any("differs from HEAD" in e or "differs from the byte-preserved" in e for e in _b(result.gate_a, "A-14"))


def test_ra_b4_01_crlf_inside_a_text_protected_path_is_not_normalised(closed, tmp_path: Path) -> None:
    src, _ = closed
    clone = _clone(src, tmp_path / "k", autocrlf=False)
    path = clone / pf.REL_PASS_RECORD
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    result = _eval(clone)
    assert not result.gate_a_pass
    assert _b(result.gate_a, "A-14")


def test_ra_b4_01_wrong_digest_fails(closed) -> None:
    root, _ = closed
    _mutate_json(root, pf.REL_CERT_RECORD, lambda c: c.update(pass_record_digest="c" * 64), reindex=True)
    result = _eval(root)
    assert any("digest" in e for e in _b(result.gate_a, "A-14")), result.gate_a


def test_ra_b4_01_byte_contract_unit_semantics(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    _git(root, "init", "-b", "development")
    _write(root, ".gitattributes", "frozen.txt -text\n")
    _write(root, "plain.txt", "a\nb\n")
    _write(root, "frozen.txt", "a\nb\n")
    _write(root, "mixed.txt", "a\r\nb\n")
    head = _commit(root, "seed")
    ctx = pf.Context(root, root / "x.json", None, head=head)
    blob = b"a\nb\n"
    assert pf._worktree_matches_blob(ctx, "plain.txt", blob, blob)
    assert pf._worktree_matches_blob(ctx, "plain.txt", b"a\r\nb\r\n", blob)  # converting path
    assert not pf._worktree_matches_blob(ctx, "frozen.txt", b"a\r\nb\r\n", blob)  # -text path: exact only
    assert not pf._worktree_matches_blob(ctx, "plain.txt", b"a\r\nb\n", blob)  # partial conversion is tampering
    assert not pf._worktree_matches_blob(ctx, "plain.txt", b"a\r\nc\r\n", blob)  # content change
    assert not pf._worktree_matches_blob(ctx, "mixed.txt", b"a\r\r\nb\r\n", b"a\r\nb\n")  # blob already holds CR


# --------------------------------------------------------------------------- RA-B4-02 blocker lifecycle


def test_ra_b4_02_genuine_independently_accepted_resolutions_pass_gate_a(repo) -> None:
    root, _ = repo
    ledger_cert = _load(root, pf.REL_CERT_RECORD)
    assert {e["blocker_id"] for e in ledger_cert["resolved_blockers_for_closure"]} == {"B-4", "B6-P3"}
    result = _eval(root)
    assert result.gate_a == []
    assert not result.gate_b_pass


def test_ra_b4_02_open_blocker_retained_fails(repo) -> None:
    root, _ = repo

    def reopen(c: dict[str, Any]) -> None:
        c["resolved_blockers_for_closure"] = [e for e in c["resolved_blockers_for_closure"] if e["blocker_id"] != "B-4"]
        c["open_blockers_for_closure"].append(B4_TEXT)

    _mutate_json(root, pf.REL_CERT_RECORD, reopen, reindex=True)
    result = _assert_a_fails(root, "A-18")
    assert any("open certification blocker" in e and "B-4" in e for e in result.gate_a)


def test_ra_b4_02_open_and_resolved_at_once_fails(repo) -> None:
    root, _ = repo
    _mutate_json(root, pf.REL_CERT_RECORD, lambda c: c["open_blockers_for_closure"].append(B4_TEXT), reindex=True)
    result = _assert_a_fails(root, "A-18")
    assert any("open and resolved at once" in e for e in result.gate_a)


@pytest.mark.parametrize("blocker_id", ["B-4", "B6-P3"])
def test_ra_b4_02_blocker_deleted_without_closure_evidence_fails(repo, blocker_id: str) -> None:
    root, _ = repo

    def delete(c: dict[str, Any]) -> None:
        c["resolved_blockers_for_closure"] = [
            e for e in c["resolved_blockers_for_closure"] if e["blocker_id"] != blocker_id
        ]

    _mutate_json(root, pf.REL_CERT_RECORD, delete, reindex=True)
    result = _assert_a_fails(root, "A-18")
    assert any("disappeared without closure evidence" in e and blocker_id in e for e in result.gate_a)


def test_ra_b4_02_all_blockers_deleted_from_a_real_baseline_fails_both_gates(repo) -> None:
    """The auditor's reproduction: deleting the lines (here: the whole ledger) must not manufacture readiness."""
    root, _ = repo

    def wipe(c: dict[str, Any]) -> None:
        c["open_blockers_for_closure"] = []
        c.pop("resolved_blockers_for_closure")

    _mutate_json(root, pf.REL_CERT_RECORD, wipe, reindex=True)
    result = _assert_a_fails(root, "A-18")
    assert len(_b(result.gate_a, "A-18")) >= 3
    assert "B-09" in _codes(result.gate_b)


def test_ra_b4_02_gate_b_only_blocker_deleted_fails_both_gates(repo) -> None:
    root, _ = repo
    _mutate_json(root, pf.REL_CERT_RECORD, lambda c: c.update(open_blockers_for_closure=[]), reindex=True)
    result = _assert_a_fails(root, "A-18")
    assert any("B-5" in e and "disappeared" in e for e in result.gate_a)
    assert any(e.startswith("B-09") and "B-5" in e for e in result.gate_b)


def test_ra_b4_02_missing_or_null_blocker_field_fails(repo) -> None:
    root, _ = repo
    field = "open_blockers_for_closure"
    _mutate_json(root, pf.REL_CERT_RECORD, lambda c: c.pop(field), reindex=True)
    _assert_a_fails(root, "A-18")
    _mutate_json(root, pf.REL_CERT_RECORD, lambda c: c.update({field: None}), reindex=True)
    result = _assert_a_fails(root, "A-18")
    assert any("missing or not a list" in e for e in result.gate_a)


def test_ra_b4_02_blocker_renamed_without_lineage_fails(repo) -> None:
    root, _ = repo

    def rename(c: dict[str, Any]) -> None:
        for entry in c["resolved_blockers_for_closure"]:
            if entry["blocker_id"] == "B-4":
                entry["blocker_id"] = "B-4-renamed"
        c["open_blockers_for_closure"] = [x.replace("B-5", "B-55") for x in c["open_blockers_for_closure"]]

    _mutate_json(root, pf.REL_CERT_RECORD, rename, reindex=True)
    result = _assert_a_fails(root, "A-18")
    assert any("B-4 disappeared" in e for e in result.gate_a)
    assert any("does not match any blocker open" in e for e in result.gate_a)
    assert any("B-5 disappeared" in e for e in result.gate_a)


def test_ra_b4_02_status_text_alone_is_not_closure(repo) -> None:
    root, _ = repo

    def downgrade(c: dict[str, Any]) -> None:
        c["resolved_blockers_for_closure"] = [
            {"blocker_id": "B-4", "status": "RESOLVED"},
            c["resolved_blockers_for_closure"][1],
        ]

    _mutate_json(root, pf.REL_CERT_RECORD, downgrade, reindex=True)
    result = _assert_a_fails(root, "A-18")
    assert any("status must be RESOLVED_WITH_EVIDENCE" in e for e in result.gate_a)


def _reclassify_as_resolved_only_in_text(c: dict[str, Any]) -> None:
    c["open_blockers_for_closure"].append("B-4 RESOLVED legacy closure preflight")


def test_ra_b4_02_reclassified_text_does_not_clear_the_blocker(repo) -> None:
    root, _ = repo
    _mutate_json(root, pf.REL_CERT_RECORD, _reclassify_as_resolved_only_in_text, reindex=True)
    _assert_a_fails(root, "A-18")


_FAKE_EVIDENCE_CASES: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
    (
        "fake_closure_sha",
        lambda e: e.update(closure_publication_sha="d" * 40),
        "closure_publication_sha is not a commit",
    ),
    (
        "closure_sha_not_sha",
        lambda e: e.update(closure_publication_sha="HEAD"),
        "closure_publication_sha is not a commit",
    ),
    ("wrong_baseline_digest", lambda e: e.update(baseline_text_sha256="e" * 64), "baseline_text_sha256"),
    ("wrong_opened_at", lambda e: e.update(opened_at_sha="a" * 40), "opened_at_sha"),
    ("wrong_kind", lambda e: e.update(resolution_kind="OPERATOR_AUTHORIZATION"), "resolution_kind"),
    ("fake_index_digest", lambda e: e.update(closure_evidence_index_digest="f" * 64), "closure_evidence_index_digest"),
    ("fake_manifest_digest", lambda e: e.update(reaudit_manifest_digest="f" * 64), "reaudit_manifest_digest"),
    ("missing_index_path", lambda e: e.pop("closure_evidence_index_path"), "closure_evidence_index_path"),
    (
        "evidence_inside_certification_dir",
        lambda e: e.update(closure_evidence_index_path=pf.REL_CERT_INDEX, reaudit_manifest_path=pf.REL_MANIFEST),
        "own directory",
    ),
    (
        "evidence_path_traversal",
        lambda e: e.update(reaudit_manifest_path=f"{REAUDIT_DIR}/../../../x.json"),
        "path traversal",
    ),
    (
        "implementation_not_a_descendant",
        lambda e: e.update(resolved_implementation_sha="a" * 40),
        "resolved_implementation_sha",
    ),
    ("disposition_not_accepted", lambda e: e["certifier_disposition"].update(disposition="PENDING"), "ACCEPTED"),
    (
        "disposition_certifier_mismatch",
        lambda e: e["certifier_disposition"].update(certifier_session_id="SOMEONE-ELSE"),
        "certifier_disposition session",
    ),
    (
        "disposition_wrong_authority",
        lambda e: e["certifier_disposition"].update(certifier_agent_id="independent-quality-auditor"),
        "certifier_agent_id",
    ),
    (
        "disposition_digest_drift",
        lambda e: e["certifier_disposition"].update(findings_digest="0" * 64),
        "findings_digest",
    ),
]


@pytest.mark.parametrize(("name", "change", "fragment"), _FAKE_EVIDENCE_CASES, ids=[c[0] for c in _FAKE_EVIDENCE_CASES])
def test_ra_b4_02_fake_closure_evidence_fails(repo, name: str, change, fragment: str) -> None:
    root, _ = repo
    _edit_resolution(root, "B-4", change)
    result = _assert_a_fails(root, "A-18")
    assert any(fragment in e for e in _b(result.gate_a, "A-18")), (name, result.gate_a)


def test_ra_b4_02_closure_published_before_the_certification_fails(repo) -> None:
    root, built = repo
    _edit_resolution(root, "B-4", lambda e: e.update(closure_publication_sha=built["tes"]))
    result = _assert_a_fails(root, "A-18")
    assert any("after the certification publication" in e for e in result.gate_a)


def _rewrite_reaudit(root: Path, change: Callable[[dict[str, Any]], None], *, handoff: str | None = None) -> None:
    """Forge re-audit evidence *consistently* (manifest, index, ledger digests) so only semantics can reject it."""
    manifest = _load(root, REAUDIT_MANIFEST)
    change(manifest)
    manifest_raw = _dump(root, REAUDIT_MANIFEST, manifest)
    if handoff is not None:
        _write(root, manifest["handoff_record_path"], handoff)
    index_raw = _index(root, REAUDIT_DIR, REAUDIT_INDEX)
    closure = _commit(root, "forged re-audit evidence")
    cert = _load(root, pf.REL_CERT_RECORD)
    entry = next(e for e in cert["resolved_blockers_for_closure"] if e["blocker_id"] == "B-4")
    entry["closure_publication_sha"] = closure
    entry["closure_evidence_index_digest"] = _sha256(index_raw)
    entry["reaudit_manifest_digest"] = _sha256(manifest_raw)
    entry["certifier_disposition"]["certifier_session_id"] = manifest["certifier_session_id"]
    _dump(root, pf.REL_CERT_RECORD, cert)
    _reindex_cert(root)
    _commit(root, "forged ledger")


_FORGED_REAUDIT_CASES: list[tuple[str, Callable[[dict[str, Any]], None], str]] = [
    ("verdict_fail", lambda m: m.update(verdict="FAIL"), "not an independent acceptance"),
    ("verdict_conditional", lambda m: m.update(verdict="CONDITIONAL"), "not an independent acceptance"),
    ("still_a_draft", lambda m: m.update(manifest_status="AUDITOR_DRAFT_NOT_A_PASS"), "still a draft"),
    ("auditor_equals_certifier", lambda m: m.update(certifier_session_id=m["auditor_session_id"]), "distinct"),
    ("implementer_equals_auditor", lambda m: m.update(implementation_session_id=m["auditor_session_id"]), "distinct"),
    ("certifier_pending", lambda m: m.update(certifier_session_id="PENDING_CERTIFIER"), "PENDING"),
    ("auditor_agent_is_implementer", lambda m: m.update(auditor_agent_id="implementer"), "auditor_agent_id"),
    ("certifier_agent_is_auditor", lambda m: m.update(certifier_agent_id="independent-quality-auditor"), "distinct"),
    ("wrong_stage", lambda m: m.update(stage_id="ENT-AUDIT-X"), "stage_id"),
    ("wrong_policy", lambda m: m.update(policy_version="IIAC-2099-01-01.9"), "policy_version"),
    (
        "audited_sha_not_a_commit",
        lambda m: m.update(audited_implementation_sha="0" * 40),
        "audited_implementation_sha is not a commit",
    ),
    (
        "audited_sha_not_the_corrective",
        lambda m: m.update(
            audited_implementation_sha="{}",
            test_execution_sha="{}",
            evidence_publication_sha="{}",
            audit_execution_sha="{}",
        ),
        "contain the resolved corrective",
    ),
]


@pytest.mark.parametrize(
    ("name", "change", "fragment"), _FORGED_REAUDIT_CASES, ids=[c[0] for c in _FORGED_REAUDIT_CASES]
)
def test_ra_b4_02_forged_independent_acceptance_fails(repo, name: str, change, fragment: str) -> None:
    root, built = repo

    def forge(manifest: dict[str, Any]) -> None:
        change(manifest)
        for key, value in list(manifest.items()):
            if value == "{}":
                manifest[key] = built["tes"]

    _rewrite_reaudit(root, forge)
    result = _assert_a_fails(root, "A-18")
    assert any(fragment in e for e in _b(result.gate_a, "A-18")), (name, _b(result.gate_a, "A-18"))


def test_ra_b4_02_unresolved_reaudit_finding_blocks_closure(repo) -> None:
    root, _ = repo
    findings = _load(root, f"{REAUDIT_DIR}/REAUDIT_FINDINGS.json")
    findings["findings"][0]["disposition"] = "OPEN"
    _dump(root, f"{REAUDIT_DIR}/REAUDIT_FINDINGS.json", findings)
    _rewrite_reaudit(
        root, lambda m: m.update(findings_digest=_sha256((root / f"{REAUDIT_DIR}/REAUDIT_FINDINGS.json").read_bytes()))
    )
    result = _eval(root)
    assert not result.gate_a_pass
    assert any("unresolved blocking finding" in e for e in _b(result.gate_a, "A-18")), result.gate_a


def test_ra_b4_02_unsigned_reaudit_handoff_blocks_closure(repo) -> None:
    root, _ = repo
    _rewrite_reaudit(
        root, lambda m: None, handoff=_reaudit_handoff().replace("2026-11-06T09:00:00+01:00", "________________")
    )
    result = _eval(root)
    assert not result.gate_a_pass
    assert any("handoff" in e for e in _b(result.gate_a, "A-18")), result.gate_a


def test_ra_b4_02_tampered_reaudit_evidence_after_publication_fails(repo) -> None:
    root, _ = repo
    _write(root, f"{REAUDIT_DIR}/REAUDIT_DOSSIER.md", "# rewritten after publication\n")
    _commit(root, "tamper")
    result = _assert_a_fails(root, "A-18")
    assert any("digest mismatch" in e or "differs from the bytes committed" in e for e in result.gate_a)


def test_ra_b4_02_unindexed_reaudit_evidence_fails(repo) -> None:
    root, _ = repo
    _write(root, f"{REAUDIT_DIR}/injected.md", "injected\n")
    _commit(root, "inject")
    result = _assert_a_fails(root, "A-18")
    assert any("unindexed evidence" in e for e in result.gate_a)


@pytest.mark.parametrize(
    ("name", "record", "fragment"),
    [
        (
            "unsigned",
            _operator_record(P3_DECISION_ID, "B6 acceptance authorization for EVOL-X", signed=False),
            "placeholder",
        ),
        (
            "not_authorized",
            _operator_record(P3_DECISION_ID, "B6 acceptance authorization for EVOL-X", status="NOT AUTHORIZED"),
            "not AUTHORIZED",
        ),
        (
            "pending",
            _operator_record(P3_DECISION_ID, "B6 acceptance authorization for EVOL-X", status="PENDING"),
            "not AUTHORIZED",
        ),
        ("wrong_scope", _operator_record(P3_DECISION_ID, "IIAC policy activation"), "Decision scope lacks"),
        (
            "future_signature",
            _operator_record(
                P3_DECISION_ID, "B6 acceptance authorization for EVOL-X", stamp=_now_iso(timedelta(days=30))
            ),
            "in the future",
        ),
    ],
)
def test_ra_b4_02_forged_p3_operator_authorization_fails(repo, name: str, record: str, fragment: str) -> None:
    root, _ = repo
    raw = _write(root, P3_RECORD, record)
    closure = _commit(root, "forged P3 record")
    cert = _load(root, pf.REL_CERT_RECORD)
    entry = next(e for e in cert["resolved_blockers_for_closure"] if e["blocker_id"] == "B6-P3")
    entry["closure_publication_sha"] = closure
    entry["closure_evidence"][0]["sha256"] = _sha256(raw)
    _dump(root, pf.REL_CERT_RECORD, cert)
    _reindex_cert(root)
    _commit(root, "forged P3 ledger")
    result = _assert_a_fails(root, "A-18")
    assert any(fragment in e for e in _b(result.gate_a, "A-18")), (name, result.gate_a)


def test_ra_b4_02_p3_record_cannot_be_an_od_evol_12_record(repo) -> None:
    root, _ = repo
    raw = _write(root, P3_RECORD, _operator_record("OD-EVOL-12", OD12_SCOPE + " B6 acceptance EVOL-X"))
    closure = _commit(root, "OD-EVOL-12 reused as P3 authorization")
    cert = _load(root, pf.REL_CERT_RECORD)
    entry = next(e for e in cert["resolved_blockers_for_closure"] if e["blocker_id"] == "B6-P3")
    entry.update(closure_publication_sha=closure)
    entry["closure_evidence"] = [{"decision_id": "OD-EVOL-12", "path": P3_RECORD, "sha256": _sha256(raw)}]
    _dump(root, pf.REL_CERT_RECORD, cert)
    _reindex_cert(root)
    _commit(root, "forged P3 ledger")
    result = _assert_a_fails(root, "A-18")
    assert any("cannot authorize" in e for e in result.gate_a)


def test_ra_b4_02_gate_a_pass_does_not_depend_on_the_b6_publication_sha(repo, tmp_path: Path, closed_repo) -> None:
    """Gate A holds both before B6 exists (``repo``) and after it is published (a copy of ``closed_repo``)."""
    root, _ = repo
    assert _eval(root).gate_a == []
    assert _load(root, pf.REL_GOVERNANCE)["b6_acceptance_publication_sha"] is None
    published = tmp_path / "published"
    shutil.copytree(closed_repo[0], published)
    assert _eval(published).gate_a == []


# --------------------------------------------------------------------------- RA-B4-03 operator decision parsing


@pytest.mark.parametrize(
    "status",
    [
        "NOT APPROVED",
        "**NOT APPROVED**",
        "UNAPPROVED",
        "PENDING",
        "**PENDING**",
        "REJECTED",
        "REJECTED - not approved",
        "DRAFT",
        "NOT SIGNED",
        "UNSIGNED",
        "",
        "   ",
        "—",
        "TBD",
        "NOT CONFIRMED",
        "CONFIRMED? NO",
        "REVOKED (previously APPROVED)",
        "APPROVED - WITHDRAWN",
        "APPROVED (SUPERSEDED)",
        "DECLINED",
        "DENIED",
        "SIGNED / APPROVED / NOT",
        "OPERATOR APPROVED — DIRECTION AND IMPLEMENTATION ONLY",
        "APPROVED IF REVIEW PASSES",
        "AWAITING SIGNATURE (APPROVED DRAFT)",
        "ΑPPROVED",  # Greek capital alpha: homoglyph of the Latin A
        "APPROVED​ WITHDRAWN",
    ],
)
def test_ra_b4_03_negated_or_ambiguous_status_is_not_a_decision(status: str) -> None:
    assert not pf.decision_recorded(status, "2026-11-01")
    assert pf.decision_status_errors(status, "2026-11-01")


@pytest.mark.parametrize(
    "status",
    [
        "SIGNED / APPROVED",
        "**OPERATOR APPROVED**",
        "OPERATOR APPROVED",
        "SIGNED",
        "APPROVED",
        "CONFIRMED",
        "signed / approved",
    ],
)
def test_ra_b4_03_canonical_positive_states_are_accepted(status: str) -> None:
    assert pf.decision_recorded(status, "2026-11-01")


@pytest.mark.parametrize(
    "date_text", ["", "—", "2026-13-45", "2026-11-1", "yesterday", "2999-01-01", "2026-11-01T10:00:00"]
)
def test_ra_b4_03_invalid_or_future_decision_date_is_rejected(date_text: str) -> None:
    assert not pf.decision_recorded("SIGNED / APPROVED", date_text)


def test_ra_b4_03_real_register_rows_still_classify_correctly() -> None:
    rows = pf.parse_register((REPO / pf.REL_REGISTER).read_text(encoding="utf-8"))
    for od_id in ("OD-EVOL-05b", "OD-EVOL-10", "OD-EVOL-13"):
        assert pf.decision_recorded(*rows[od_id]), od_id
    for od_id in ("OD-EVOL-12", "OD-EVOLX-08"):
        assert not pf.decision_recorded(*rows[od_id]), od_id
    # scope-limited approvals are never closing decisions
    for od_id in ("OD-EVOL-14", "OD-EVOL-15", "OD-EVOL-17"):
        assert not pf.decision_recorded(*rows[od_id]), od_id


@pytest.mark.parametrize("status", ["**NOT APPROVED**", "**PENDING**", "**REJECTED**", "**DRAFT**", "NOT SIGNED"])
def test_ra_b4_03_gate_b_rejects_negated_closing_decision(closed, status: str) -> None:
    root, built = closed
    decisions = {
        "od12": (root / OD12_RECORD).read_bytes(),
        "odx08": (root / ODX08_RECORD).read_bytes(),
    }
    _write(root, pf.REL_REGISTER, _register(decided=True, notes=_decision_notes(decisions), closing_status=status))
    _commit(root, "negated closing decisions")
    result = _eval(root)
    assert result.gate_a_pass
    b07 = _b(result.gate_b, "B-07")
    assert any("OD-EVOL-12" in e for e in b07) and any("OD-EVOLX-08" in e for e in b07), result.gate_b
    assert not result.gate_b_pass


def test_ra_b4_03_narrative_mention_of_approval_is_not_a_decision(closed) -> None:
    root, _ = closed
    pending = _register(decided=False)
    narrative = (
        "\n## History\n\nOD-EVOL-12 was APPROVED in an earlier draft; "
        "OPERATOR APPROVED OD-EVOLX-08 on 2026-11-01.\n\n"
        "| ID | Proposed decision text | Technical basis |\n|---|---|---|\n"
        "| **OD-EVOL-12** | OPERATOR APPROVED | 2026-11-01 |\n"
        "| **OD-EVOLX-08** | SIGNED / APPROVED | 2026-11-01 |\n"
    )
    _write(root, pf.REL_REGISTER, pending + narrative)
    _commit(root, "narrative approval")
    result = _eval(root)
    assert result.gate_a_pass
    b07 = _b(result.gate_b, "B-07")
    assert any("OD-EVOL-12" in e for e in b07) and any("OD-EVOLX-08" in e for e in b07)


def test_ra_b4_03_duplicate_status_rows_are_ambiguous(closed) -> None:
    root, _ = closed
    text = (root / pf.REL_REGISTER).read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| **OD-EVOL-12**"))
    _write(root, pf.REL_REGISTER, text.replace(row, row + "\n" + row.replace("OPERATOR APPROVED", "PENDING")))
    _commit(root, "duplicate row")
    assert any("ambiguous" in e for e in _b(_eval(root).gate_b, "B-07"))


def test_ra_b4_03_row_outside_the_status_table_is_ignored(closed) -> None:
    root, _ = closed
    text = (root / pf.REL_REGISTER).read_text(encoding="utf-8")
    _write(
        root,
        pf.REL_REGISTER,
        text.replace("| ID | Status | Signed date | Notes |", "| Decision | Outcome | Date | Notes |"),
    )
    _commit(root, "renamed status table")
    assert any("not recorded in the register status table" in e for e in _b(_eval(root).gate_b, "B-07"))


def test_ra_b4_03_wrong_decision_scope_fails(closed) -> None:
    root, _ = closed
    bad = _operator_record("OD-EVOL-12", "IIAC policy activation")
    raw = _write(root, OD12_RECORD, bad)
    _commit(root, "wrong scope")
    decisions = {"od12": raw, "odx08": (root / ODX08_RECORD).read_bytes()}
    _write(root, pf.REL_REGISTER, _register(decided=True, notes=_decision_notes(decisions)))
    _commit(root, "register binds the wrong-scope record")
    b07 = _b(_eval(root).gate_b, "B-07")
    assert any("Decision scope lacks" in e for e in b07), b07


@pytest.mark.parametrize(
    ("name", "mutate", "fragment"),
    [
        ("wrong_id", lambda t: t.replace("Decision ID: OD-EVOL-12", "Decision ID: OD-EVOL-99"), "Decision ID"),
        ("unsigned", lambda t: t.replace(t.splitlines()[-1], "Operator signature: ________________"), "placeholder"),
        ("not_authorized", lambda t: t.replace("AUTHORIZED", "NOT AUTHORIZED"), "not AUTHORIZED"),
        (
            "signature_for_other_decision",
            lambda t: t.replace("Approved OD-EVOL-12:", "Approved OD-EVOLX-08:"),
            "signature line is for",
        ),
        (
            "no_signature",
            lambda t: "\n".join(line for line in t.splitlines() if not line.startswith("Approved ")),
            "signature line",
        ),
        (
            "signature_only_in_a_code_block",
            lambda t: "\n".join(
                ["```", *[line for line in t.splitlines() if line.startswith("Approved ")], "```"]
                + [line for line in t.splitlines() if not line.startswith("Approved ")]
            ),
            "signature line",
        ),
        ("wrong_policy", lambda t: t.replace(POLICY, "IIAC-2099-01-01.9"), "Policy version"),
    ],
)
def test_ra_b4_03_unverifiable_decision_record_fails(closed, name: str, mutate, fragment: str) -> None:
    root, _ = closed
    raw = _write(root, OD12_RECORD, mutate((root / OD12_RECORD).read_text(encoding="utf-8")))
    decisions = {"od12": raw, "odx08": (root / ODX08_RECORD).read_bytes()}
    _write(root, pf.REL_REGISTER, _register(decided=True, notes=_decision_notes(decisions)))
    _commit(root, f"forged decision record {name}")
    b07 = _b(_eval(root).gate_b, "B-07")
    assert any(fragment in e for e in b07), (name, b07)


@pytest.mark.parametrize(
    ("name", "notes", "fragment"),
    [
        ("no_record_cited", {"OD-EVOL-12": "fixture"}, "exactly one decision record"),
        (
            "digest_mismatch",
            {"OD-EVOL-12": "[r](../iiac_evidence/EVOL-X/operator_decisions/OD_EVOL_12_FIXTURE.md) sha256:" + "a" * 64},
            "digest does not match",
        ),
        (
            "missing_record",
            {"OD-EVOL-12": "[r](../iiac_evidence/EVOL-X/operator_decisions/NOPE.md) sha256:" + "a" * 64},
            "is missing",
        ),
    ],
)
def test_ra_b4_03_register_must_bind_the_decision_record(
    closed, name: str, notes: dict[str, str], fragment: str
) -> None:
    root, _ = closed
    decisions = {"od12": (root / OD12_RECORD).read_bytes(), "odx08": (root / ODX08_RECORD).read_bytes()}
    merged = {**_decision_notes(decisions), **notes}
    _write(root, pf.REL_REGISTER, _register(decided=True, notes=merged))
    _commit(root, f"register binding {name}")
    b07 = _b(_eval(root).gate_b, "B-07")
    assert any(fragment in e for e in b07), (name, b07)


def test_ra_b4_03_uncommitted_decision_record_fails(closed) -> None:
    root, _ = closed
    path = root / OD12_RECORD
    path.write_bytes(path.read_bytes() + b"late edit\n")
    assert any("OD-EVOL-12" in e for e in _b(_eval(root).gate_b, "B-07"))


def test_ra_b4_03_genuine_authorized_decisions_pass(closed) -> None:
    root, _ = closed
    result = _eval(root)
    assert _b(result.gate_b, "B-07") == []
    assert result.gate_b_pass


def test_ra_b4_03_signature_provenance_limit_is_documented_in_the_gate() -> None:
    assert "not cryptographic" in (pf.__doc__ or "").lower()


# --------------------------------------------------------------------------- RA-B4-04 evidence-bound B6


def _flags_only_b6(root: Path, built: dict[str, str], *, bind: bool = True) -> str:
    b6 = _publish_b6(root, with_record=False)
    if bind:
        _bind_b6(root, built, b6)
        _commit(root, "bind flags-only B6")
    return b6


def test_ra_b4_04_flags_only_b6_fails_gate_b(repo) -> None:
    root, built = repo
    _flags_only_b6(root, built)
    result = _eval(root)
    assert result.gate_a == []
    b05 = _b(result.gate_b, "B-05")
    assert any("absent at B6_ACCEPTANCE_PUBLICATION_SHA" in e for e in b05), result.gate_b
    assert not result.gate_b_pass


def test_ra_b4_04_flag_flip_by_arbitrary_actor_cannot_close_the_gate(repo) -> None:
    """The auditor's G6 reproduction: flags + blockers cleared + FRZ rows + register edited, no authorisation."""
    root, built = repo
    _flags_only_b6(root, built)
    _mutate_json(root, pf.REL_CERT_RECORD, lambda c: c.update(open_blockers_for_closure=[]), reindex=True)
    _write(root, pf.REL_FRZ_CHECKLIST, _checklist(PROMOTED_ROWS))
    _write(root, pf.REL_REGISTER, _register(decided=True))
    _commit(root, "edit everything")
    result = _eval(root)
    assert not result.gate_a_pass, "deleting the B-5 blocker must not manufacture Gate A readiness"
    assert "A-18" in _codes(result.gate_a)
    assert {"B-05", "B-07"} <= _codes(result.gate_b)
    assert not result.gate_b_pass


def test_ra_b4_04_fake_b6_sha_fails(closed) -> None:
    root, built = closed
    _mutate_json(root, pf.REL_GOVERNANCE, lambda g: g.update(b6_acceptance_publication_sha="9" * 40))
    cert = _load(root, pf.REL_CERT_RECORD)
    cert["b6_acceptance_publication_sha"] = "9" * 40
    _dump(root, pf.REL_CERT_RECORD, cert)
    _reindex_cert(root)
    _write(root, pf.REL_LADDER, _ladder(built["tes"], built["final"], built["tes"], b6="9" * 40))
    _commit(root, "fabricated B6 SHA")
    result = _eval(root)
    assert result.gate_a_pass
    assert any("is not a commit" in e for e in _b(result.gate_b, "B-03")), result.gate_b
    assert not result.gate_b_pass


def test_ra_b4_04_b6_record_absent_at_the_pinned_sha_fails(repo) -> None:
    root, built = repo
    pinned = _flags_only_b6(root, built, bind=False)  # phase 1 without a record
    record = _write(root, f"{pf.B6_DIR}/{B6_RECORD_NAME}", _b6_record_text(root))
    _index(root, pf.B6_DIR, pf.REL_B6_INDEX)
    cert = _load(root, pf.REL_CERT_RECORD)
    cert.update(b6_acceptance_record_path=pf.REL_B6_RECORD, b6_acceptance_record_digest=_sha256(record))
    _dump(root, pf.REL_CERT_RECORD, cert)
    _reindex_cert(root)
    _commit(root, "record appears after the pinned B6 SHA")
    _bind_b6(root, built, pinned)
    _commit(root, "bind the earlier phase-1 commit")
    result = _eval(root)
    assert any("absent at B6_ACCEPTANCE_PUBLICATION_SHA" in e for e in _b(result.gate_b, "B-05")), result.gate_b
    assert not result.gate_b_pass


def test_ra_b4_04_historical_b6_substituted_fails(closed) -> None:
    root, built = closed
    legacy = f"{pf.EVOL_DIR}/EVOL_X_INTERNAL_B6_ACCEPTANCE_RECORD.md"
    raw = _write(root, legacy, "historical internal B6 acceptance\n")
    _mutate_json(
        root,
        pf.REL_CERT_RECORD,
        lambda c: c.update(b6_acceptance_record_path=legacy, b6_acceptance_record_digest=_sha256(raw)),
        reindex=True,
    )
    assert any("canonical B6 acceptance record path" in e for e in _b(_eval(root).gate_b, "B-05"))


def test_ra_b4_04_historical_sha_substituted_for_the_record_fails(closed) -> None:
    root, built = closed
    _mutate_json(root, pf.REL_GOVERNANCE, lambda g: g.update(b6_acceptance_publication_sha=built["tes"]))
    cert = _load(root, pf.REL_CERT_RECORD)
    cert["b6_acceptance_publication_sha"] = built["tes"]
    _dump(root, pf.REL_CERT_RECORD, cert)
    _reindex_cert(root)
    _write(root, pf.REL_LADDER, _ladder(built["tes"], built["final"], built["tes"], b6=built["tes"]))
    _commit(root, "historical SHA as B6")
    result = _eval(root)
    assert result.gate_a_pass
    assert {"B-03", "B-05"} <= _codes(result.gate_b)


def test_ra_b4_04_record_mutated_after_publication_fails(closed) -> None:
    root, _ = closed
    path = root / pf.REL_B6_RECORD
    raw = path.read_bytes().replace(b"Decision status: AUTHORIZED", b"Decision status: AUTHORIZED ")
    _write(root, pf.REL_B6_RECORD, raw)
    _index(root, pf.B6_DIR, pf.REL_B6_INDEX)
    _commit(root, "late edit of the B6 record")
    b05 = _b(_eval(root).gate_b, "B-05")
    assert any("differs from the bytes published" in e or "digest" in e for e in b05), b05


def test_ra_b4_04_manifest_changed_after_b6_publication_fails(closed) -> None:
    root, _ = closed
    _mutate_json(
        root, pf.REL_MANIFEST, lambda m: m["limitations"].append("LIM-99 added after B6 acceptance"), reindex=True
    )
    result = _eval(root)
    assert any("manifest changed after" in e for e in _b(result.gate_b, "B-05")), result.gate_b


@pytest.mark.parametrize(
    ("name", "overrides", "fragment"),
    [
        ("widened_verdict", {"Accepted verdict": "PASS"}, "Accepted verdict"),
        ("widened_limitations", {"Accepted limitations": "LIM-01"}, "Accepted limitations"),
        (
            "extra_limitation",
            {"Accepted limitations": "LIM-01, LIM-02, LIM-03, LIM-04, LIM-05, LIM-06, LIM-07, F-C-02, LIM-99"},
            "Accepted limitations",
        ),
        ("wrong_manifest_digest", {"Independent audit manifest digest": "a" * 64}, "manifest digest"),
        ("wrong_report_digest", {"Audit report digest": "b" * 64}, "Audit report digest"),
        ("wrong_findings_digest", {"Findings digest": "c" * 64}, "Findings digest"),
        ("wrong_audited_sha", {"Audited implementation SHA": "d" * 40}, "Audited implementation SHA"),
        ("wrong_certification_sha", {"Certification publication SHA": "e" * 40}, "Certification publication SHA"),
        ("foreign_certifier_session", {"Certifier session": "SOMEONE-ELSE"}, "Certifier session"),
        ("wrong_certifier_agent", {"Certifier agent": "implementer"}, "Certifier agent"),
        ("not_authorized", {"status": "NOT AUTHORIZED"}, "not AUTHORIZED"),
        ("pending", {"status": "PENDING"}, "not AUTHORIZED"),
        ("unsigned", {"signed": "no"}, "placeholder"),
        ("wrong_scope", {"scope": "IIAC policy activation"}, "Decision scope lacks"),
        ("future_signature", {"stamp": _now_iso(timedelta(days=30))}, "in the future"),
        ("stale_signature", {"stamp": "2020-01-01T00:00:00+00:00"}, "predates the certification"),
    ],
)
def test_ra_b4_04_b6_record_content_must_match_the_independent_audit(
    repo, name: str, overrides: dict[str, str], fragment: str
) -> None:
    root, built = repo
    b6 = _publish_b6(root, **overrides)
    decisions = _publish_decisions(root)
    _finish_close(root, built, b6, decisions)
    result = _eval(root)
    assert result.gate_a_pass
    assert any(fragment in e for e in _b(result.gate_b, "B-05")), (name, _b(result.gate_b, "B-05"))
    assert not result.gate_b_pass


def test_ra_b4_04_signature_post_dating_the_publication_commit_fails(repo) -> None:
    root, built = repo
    b6 = _publish_b6(root, stamp=_now_iso(timedelta(hours=6)))
    decisions = _publish_decisions(root)
    _finish_close(root, built, b6, decisions)
    result = _eval(root)
    assert any("post-dates the B6 acceptance publication" in e for e in _b(result.gate_b, "B-05")), result.gate_b


def test_ra_b4_04_b6_before_p3_operator_authorization_fails(repo) -> None:
    root, built = repo
    # Publish B6 on a branch point *before* the P3 authorization by pointing the ledger at a later authorization.
    b6 = _publish_b6(root)
    late = _write(root, P3_RECORD, _operator_record(P3_DECISION_ID, "B6 acceptance authorization for EVOL-X"))
    later = _commit(root, "P3 authorization published after B6 (out of order)")
    _edit_resolution(
        root,
        "B6-P3",
        lambda e: (
            e.update(closure_publication_sha=later),
            e["closure_evidence"][0].update(sha256=_sha256(late)),
        ),
    )
    decisions = _publish_decisions(root)
    _finish_close(root, built, b6, decisions)
    result = _eval(root)
    assert any("after the operator P3 authorization" in e for e in _b(result.gate_b, "B-05")), result.gate_b


def test_ra_b4_04_correctly_published_b6_with_verified_evidence_passes(closed) -> None:
    root, built = closed
    result = _eval(root)
    assert result.gate_a == [] and result.gate_b == []
    assert _load(root, pf.REL_CERT_RECORD)["b6_acceptance_record_path"] == pf.REL_B6_RECORD
    assert (root / pf.REL_B6_RECORD).is_file() and built["tes"]


def test_ra_b4_04_gate_a_pass_never_implies_gate_b_pass(repo) -> None:
    root, built = repo
    before = _eval(root)
    assert before.gate_a_pass and not before.gate_b_pass
    _flags_only_b6(root, built)
    result = _eval(root)
    assert result.gate_a_pass and not result.gate_b_pass


def test_ra_b4_04_scoped_limitations_remain_binding_after_b6(closed) -> None:
    root, _ = closed
    _mutate_json(
        root,
        pf.REL_MANIFEST,
        lambda m: m.update(limitations=[x for x in m["limitations"] if not x.startswith("LIM-05")]),
        reindex=True,
    )
    result = _eval(root)
    assert not result.gate_a_pass and "A-08" in _codes(result.gate_a)
    assert not result.gate_b_pass


def test_ra_b4_04_no_self_certification_or_evidence_drift_after_closure(closed) -> None:
    root, _ = closed
    _mutate_json(root, pf.REL_MANIFEST, lambda m: m.update(certifier_session_id=AUDITOR_SESSION), reindex=True)
    result = _eval(root)
    assert "A-09" in _codes(result.gate_a)
    assert not result.gate_b_pass


# --------------------------------------------------------------------------- canonical repository


def test_ra_b4_canonical_gate_a_reflects_unresolved_blockers_and_gate_b_requires_real_evidence() -> None:
    cert = json.loads((REPO / pf.REL_CERT_RECORD).read_text(encoding="utf-8"))
    governance = json.loads((REPO / pf.REL_GOVERNANCE).read_text(encoding="utf-8"))
    if governance.get("b6_acceptance_publication_sha") or cert.get("resolved_blockers_for_closure"):
        pytest.skip("blockers resolved / B6 published; canonical pre-closure assertion no longer applies")
    result = pf.evaluate(REPO)
    assert result.mode == "IIAC", result.gate_a
    assert _codes(result.gate_a) == {"A-18"}, result.gate_a
    assert sum("open certification blocker" in e for e in result.gate_a) == 2
    assert not result.gate_b_pass
    codes = _codes(result.gate_b)
    assert {"B-01", "B-06", "B-07", "B-09"} <= codes
    assert any("OD-EVOL-12" in e for e in _b(result.gate_b, "B-07"))
    assert any("OD-EVOLX-08" in e for e in _b(result.gate_b, "B-07"))
    assert pf.blocker_key(cert["open_blockers_for_closure"][0]) == "B-4"
    assert pf.blocker_key(cert["open_blockers_for_closure"][2]) == "B6-P3"
