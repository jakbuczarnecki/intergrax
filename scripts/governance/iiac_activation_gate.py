"""Fail-closed IIAC policy activation checks (ACT-01).

The canonical checkout stays inactive until a later certification session.
This module does not set ``IN_FORCE`` and does not accept Session A
implementation approval as an activation decision.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.governance.iiac_common import (
    REPO,
    git_merge_base_is_ancestor,
    git_rev_parse,
    product_paths_changed_between,
    validate_manifest_dict,
)

POLICY_VERSION = "IIAC-2026-11-05.1"
CANONICAL_RECORD = REPO / "docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json"
EVIDENCE_ROOT = REPO / "docs/project/maintainers/qualification/iiac_evidence"
QUALIFICATION_ROOT = REPO / "docs/project/maintainers/qualification"
SCRIPTS = REPO / "scripts/governance"

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_DIGEST64 = re.compile(r"^[0-9a-f]{64}$")
_SIGNATURE = re.compile(r"^Approved IIAC policy activation IIAC-2026-11-05\.1: operator @ ([0-9T:.+-Z]+)\s*$")
_CLOSURE = re.compile(r"(EVOL-X|ENT-AUDIT-X|ARCH-FREEZE).{0,80}\bCLOSED\b", re.IGNORECASE)

INACTIVE_STATUSES = frozenset({"NOT_IN_FORCE", "DRAFT", "PENDING"})
CLOSED_FINDING = frozenset({"RESOLVED", "CLOSED"})
BLOCKING_SEVERITY = frozenset({"BLOCKER", "CRITICAL"})
MATERIAL_GOVERNANCE_PATHS = frozenset(
    {
        "scripts/governance/verify_iiac_policy_activation.py",
        "scripts/governance/iiac_activation_gate.py",
        "scripts/governance/iiac_common.py",
        "scripts/governance/verify_iiac_audit_manifest.py",
        "scripts/governance/verify_iiac_role_separation.py",
        "docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md",
    }
)
BINDING_KEYS = (
    "policy_version",
    "session_b_governance_audit_report_digest",
    "session_b_verdict",
    "operator_decision_id",
    "governance_implementation_sha",
    "corrective_implementation_sha",
    "audit_execution_sha",
    "evidence_publication_sha",
    "certification_publication_sha",
    "session_b_manifest_digest",
)
REQUIRED_IN_FORCE = (
    "policy_version",
    "status",
    "effective_from_sha",
    "policy_activation_sha",
    "governance_implementation_sha",
    "corrective_implementation_sha",
    "audit_execution_sha",
    "evidence_publication_sha",
    "certification_publication_sha",
    "session_b_verdict",
    "session_b_governance_audit_report_digest",
    "session_b_manifest_digest",
    "session_b_manifest_path",
    "audit_report_path",
    "audit_scope_path",
    "test_matrix_path",
    "findings_path",
    "handoff_record_path",
    "handoff_digest",
    "operator_decision_path",
    "operator_decision_id",
    "operator_activation_line",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_show_bytes(repo: Path, sha: str, relpath: str) -> bytes:
    proc = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "show", f"{sha}:{relpath}"],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(detail or f"git show {sha}:{relpath} failed")
    return proc.stdout


def _posix(path: str) -> str:
    return str(path).replace("\\", "/").strip()


def _path_rejected(raw: str) -> str | None:
    text = _posix(raw)
    if not text:
        return "empty evidence path"
    if text.startswith("/") or text.startswith("//") or re.match(r"^[A-Za-z]:", text):
        return "absolute evidence path"
    if ".." in Path(text).parts:
        return "path traversal"
    return None


def _under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _has_symlink(path: Path, stop: Path) -> bool:
    current = path
    stop_resolved = stop.resolve()
    while True:
        if current.is_symlink():
            return True
        if current.resolve() == stop_resolved or current == current.parent:
            return False
        current = current.parent


def resolve_evidence_path(
    repo: Path,
    raw: str,
    *,
    kind: str,
    evidence_root: Path | None,
) -> Path:
    reason = _path_rejected(raw)
    if reason:
        raise ValueError(f"ACT-N14 {reason}: {raw}")
    rel = _posix(raw)
    if "historical_session_b" in rel and kind == "manifest":
        raise ValueError("historical PARTIAL evidence cannot be the active acceptance record")
    path = (repo / rel).resolve()
    if evidence_root is not None:
        root = evidence_root.resolve()
        if not _under_root(path, root):
            raise ValueError(f"ACT-N14 evidence path outside authorized root: {rel}")
        if _has_symlink(path, root):
            raise ValueError(f"ACT-N14 symlink evidence path: {rel}")
        return path

    name = Path(rel).name
    if kind in {"report", "manifest", "scope", "matrix", "findings"}:
        if not rel.startswith("docs/project/maintainers/qualification/iiac_evidence/"):
            raise ValueError(f"ACT-N14 artifact is outside iiac_evidence: {rel}")
        root = EVIDENCE_ROOT
    elif kind == "handoff":
        if not (name.startswith("IIAC_SESSION_HANDOFF_RECORD_") and name.endswith(".md")):
            raise ValueError(f"ACT-N14 handoff path is not a session record: {rel}")
        if name == "IIAC_SESSION_HANDOFF_RECORD.template.md":
            raise ValueError("ACT-N14 handoff template is not session provenance")
        root = QUALIFICATION_ROOT
    elif kind == "operator":
        upper = name.upper()
        if "PACKET" in upper or "DRAFT" in upper:
            raise ValueError("ACT-N06 unsigned activation packet is not an operator decision")
        if not (name.startswith("IIAC_OD_ACTIVATION_") and name.endswith(".md")):
            raise ValueError(f"ACT-N06 operator decision filename rejected: {rel}")
        root = QUALIFICATION_ROOT
    else:
        raise ValueError(f"unsupported evidence kind: {kind}")
    if not _under_root(path, root):
        raise ValueError(f"ACT-N14 evidence path outside authorized root: {rel}")
    if _has_symlink(path, root):
        raise ValueError(f"ACT-N14 symlink evidence path: {rel}")
    return path


def closure_conflicts(record: dict[str, Any], operator_text: str | None = None) -> list[str]:
    errors: list[str] = []
    closures = record.get("stage_closures")
    if isinstance(closures, dict):
        for key, value in closures.items():
            if str(value).upper() == "CLOSED":
                errors.append(f"ACT-N18 activation record closes {key}")
    blob = json.dumps(record, ensure_ascii=False)
    if operator_text:
        blob = blob + "\n" + operator_text
    for match in _CLOSURE.finditer(blob):
        errors.append(f"ACT-N18 activation text promotes {match.group(1)} to CLOSED")
    return errors


def contradictory_inactive_claims(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    status = str(record.get("status") or "").upper()
    if status == "IN_FORCE":
        errors.append("policy status is IN_FORCE")
    if record.get("effective_from_sha"):
        errors.append("inactive record carries effective_from_sha")
    verdict = str(record.get("session_b_verdict") or "").upper()
    line = str(record.get("operator_activation_line") or "").strip()
    if verdict == "PASS" and line:
        errors.append("PASS verdict combined with an operator line while inactive")
    errors.extend(closure_conflicts(record, None))
    return errors


def _sha_field(record: dict[str, Any], key: str, errors: list[str]) -> str:
    value = str(record.get(key) or "").strip()
    if not _SHA40.fullmatch(value):
        errors.append(f"{key} must be a 40-char lowercase git SHA")
        return ""
    return value


def _ancestor(repo: Path, ancestor: str, descendant: str, errors: list[str], label: str) -> None:
    if not ancestor or not descendant:
        return
    try:
        if not git_merge_base_is_ancestor(ancestor, descendant, repo=repo):
            errors.append(f"ACT-N09 {label}: {ancestor} is not an ancestor of {descendant}")
    except RuntimeError as exc:
        errors.append(str(exc))


def _read_json_file(path: Path) -> tuple[Any | None, str | None]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, f"unreadable {path.name}: {exc}"
    try:
        return json.loads(raw.decode("utf-8")), None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"malformed JSON {path.name}: {exc}"


def _verdict_error(verdict: str) -> str:
    if verdict == "FAIL":
        return "ACT-N04 independent verdict FAIL is not activation authority"
    if verdict == "CONDITIONAL":
        return "ACT-N05 verdict CONDITIONAL is not activation authority"
    if verdict in {"PARTIAL", "PASS_SCOPED"}:
        return f"historical or scoped verdict {verdict} cannot authorize activation"
    if not verdict:
        return "missing independent verdict"
    return f"unsupported verdict {verdict}"


def _operator_errors(record: dict[str, Any], text: str, report_digest: str) -> list[str]:
    errors: list[str] = []
    decision_id = str(record.get("operator_decision_id") or "").strip()
    line = str(record.get("operator_activation_line") or "").strip()
    if not decision_id:
        errors.append("ACT-N06 missing operator decision id")
    if "________________" in text or "NOT SIGNED" in text or "Decision status: PENDING" in text:
        errors.append("ACT-N06 operator decision is still pending")
    if "Decision status: AUTHORIZED" not in text:
        errors.append("ACT-N07 operator file has no AUTHORIZED activation decision")
    if f"Decision ID: {decision_id}" not in text:
        errors.append("ACT-N07 decision id is not in the operator record")
    if f"Policy version: {POLICY_VERSION}" not in text:
        errors.append("ACT-N16 operator record policy version mismatch")
    if "Decision scope: IIAC policy activation" not in text:
        errors.append("ACT-N07 decision scope is not IIAC policy activation")
    if f"Audit report digest: {report_digest}" not in text:
        errors.append("ACT-N07 operator record does not cite the independent audit digest")
    match = _SIGNATURE.fullmatch(line)
    if line in {"Approved", "Approved:"} or line.startswith("Approved: operator Session A"):
        errors.append("ACT-N07 Session A or bare Approved string is not activation authorization")
    elif not match:
        errors.append("ACT-N07 operator activation line is not an authorized decision line")
    elif line not in text.splitlines():
        errors.append("ACT-N07 activation line is absent from the operator record")
    if "IIAC-GOVERNANCE-REFORM" in text and "Decision status: AUTHORIZED" not in text:
        errors.append("ACT-N07 Session A implementation approval is not policy activation")
    return errors


def _findings_errors(document: dict[str, Any]) -> list[str]:
    findings = document.get("findings")
    if not isinstance(findings, list):
        return ["ACT-N15 findings register has no findings array"]
    errors: list[str] = []
    for item in findings:
        if not isinstance(item, dict):
            errors.append("ACT-N15 malformed finding entry")
            continue
        severity = str(item.get("original_severity") or item.get("severity") or "").upper()
        disposition = str(item.get("reaudit_disposition") or item.get("disposition") or "").upper()
        if severity in BLOCKING_SEVERITY and disposition not in CLOSED_FINDING:
            errors.append(f"ACT-N15 open {severity} finding {item.get('id')}")
    return errors


def _companion_errors(repo: Path, manifest_path: Path) -> list[str]:
    errors: list[str] = []
    calls = (
        ("verify_iiac_audit_manifest.py", (str(manifest_path), "--repo", str(repo))),
        (
            "verify_iiac_role_separation.py",
            (str(manifest_path), "--strict-product-diff", "--repo", str(repo)),
        ),
    )
    for script, args in calls:
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / script), *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            tail = detail[-1] if detail else "no output"
            errors.append(f"ACT-N10 companion {script} failed: {tail}")
    return errors


def _material_changes(repo: Path, audit_sha: str, head: str) -> list[str]:
    proc = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "diff", "--name-only", audit_sha, head],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return [proc.stderr.strip() or "git diff failed for stale-audit check"]
    changed: list[str] = []
    for line in proc.stdout.splitlines():
        rel = line.strip().replace("\\", "/")
        if rel in MATERIAL_GOVERNANCE_PATHS:
            changed.append(rel)
    return changed


def evaluate_activation(
    record: dict[str, Any],
    *,
    repo: Path,
    record_path: Path,
    evidence_root: Path | None,
    require_in_force: bool,
) -> list[str]:
    """Return activation errors. Empty means the selected mode passed."""
    status = str(record.get("status") or "").upper()
    if not require_in_force:
        if status not in INACTIVE_STATUSES:
            return [f"ACT-N01 --expect-not-in-force rejected status {status or 'missing'}"]
        return [f"contradictory active claim: {item}" for item in contradictory_inactive_claims(record)]

    if status != "IN_FORCE":
        return [f"ACT-N01 status {status or 'missing'} is not activation eligibility"]

    errors: list[str] = []
    for key in REQUIRED_IN_FORCE:
        if key not in record or record.get(key) in (None, ""):
            errors.append(f"missing activation field: {key}")
    if str(record.get("policy_version") or "") != POLICY_VERSION:
        errors.append(f"ACT-N16 unsupported policy version {record.get('policy_version')!r}; expected {POLICY_VERSION}")
    if str(record.get("session_b_verdict") or "").upper() != "PASS":
        errors.append(_verdict_error(str(record.get("session_b_verdict") or "").upper()))
    errors.extend(closure_conflicts(record, None))

    try:
        head = git_rev_parse("HEAD", repo=repo)
    except RuntimeError as exc:
        errors.append(str(exc))
        head = ""

    effective = _sha_field(record, "effective_from_sha", errors)
    activation_sha = _sha_field(record, "policy_activation_sha", errors)
    governance = _sha_field(record, "governance_implementation_sha", errors)
    corrective = _sha_field(record, "corrective_implementation_sha", errors)
    audit_sha = _sha_field(record, "audit_execution_sha", errors)
    evidence_sha = _sha_field(record, "evidence_publication_sha", errors)
    certification_sha = _sha_field(record, "certification_publication_sha", errors)

    if effective and head and effective == head:
        errors.append("ACT-N12 self-referential effective_from_sha equals HEAD")
    if effective and activation_sha and effective != activation_sha:
        errors.append("policy_activation_sha must equal effective_from_sha")
    _ancestor(repo, governance, corrective, errors, "governance implementation SHA")
    _ancestor(repo, corrective, audit_sha, errors, "corrective implementation SHA")
    _ancestor(repo, audit_sha, evidence_sha, errors, "audit execution SHA")
    _ancestor(repo, evidence_sha, effective, errors, "evidence publication SHA")
    _ancestor(repo, certification_sha, effective, errors, "certification publication SHA")
    if certification_sha and corrective and certification_sha == corrective:
        errors.append("certification_publication_sha must not equal the audited corrective SHA")

    if head and record_path.is_file():
        try:
            rel_record = record_path.resolve().relative_to(repo.resolve()).as_posix()
            committed = git_show_bytes(repo, head, rel_record)
            if sha256_bytes(record_path.read_bytes()) != sha256_bytes(committed):
                errors.append("worktree activation record does not match HEAD")
        except (RuntimeError, ValueError) as exc:
            errors.append(str(exc))

    if effective and head and effective != head:
        try:
            rel_record = record_path.resolve().relative_to(repo.resolve()).as_posix()
            phase1 = json.loads(git_show_bytes(repo, effective, rel_record).decode("utf-8"))
        except (RuntimeError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"ACT-N12 unreadable activation blob at effective_from_sha: {exc}")
            phase1 = None
        if isinstance(phase1, dict):
            if str(phase1.get("status") or "").upper() != "IN_FORCE":
                errors.append("ACT-N12 effective_from_sha points at a pre-activation blob")
            if phase1.get("effective_from_sha") not in (None, ""):
                errors.append("phase-1 activation blob must leave effective_from_sha empty")
            if str(phase1.get("policy_version") or "") != POLICY_VERSION:
                errors.append("ACT-N16 phase-1 policy version mismatch")
            for key in BINDING_KEYS:
                if phase1.get(key) != record.get(key):
                    errors.append(f"phase-1 blob {key} does not match the activation record")

    digest = str(record.get("session_b_governance_audit_report_digest") or "").strip()
    if not _DIGEST64.fullmatch(digest):
        errors.append("ACT-N02 arbitrary or invalid audit digest")

    artifacts: dict[str, Path] = {}
    for key, kind in (
        ("audit_report_path", "report"),
        ("session_b_manifest_path", "manifest"),
        ("audit_scope_path", "scope"),
        ("test_matrix_path", "matrix"),
        ("findings_path", "findings"),
        ("handoff_record_path", "handoff"),
        ("operator_decision_path", "operator"),
    ):
        raw = record.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            artifacts[kind] = resolve_evidence_path(repo, raw, kind=kind, evidence_root=evidence_root)
        except ValueError as exc:
            errors.append(str(exc))

    report_path = artifacts.get("report")
    report_bytes: bytes | None = None
    if report_path is None:
        errors.append("ACT-N13 missing audit report path")
    elif not report_path.is_file():
        errors.append("ACT-N13 missing audit report bytes")
    else:
        report_bytes = report_path.read_bytes()
        actual = sha256_bytes(report_bytes)
        if _DIGEST64.fullmatch(digest) and actual != digest:
            errors.append("ACT-N08 audit report digest mismatch")
        if evidence_sha:
            try:
                committed_report = git_show_bytes(repo, evidence_sha, _posix(str(record.get("audit_report_path"))))
                if report_bytes != committed_report:
                    errors.append("ACT-N11 audit report bytes differ from evidence publication SHA")
            except RuntimeError as exc:
                errors.append(f"ACT-N13 evidence publication is missing the audit report: {exc}")

    manifest_path = artifacts.get("manifest")
    manifest: dict[str, Any] | None = None
    if manifest_path is None:
        errors.append("ACT-N03 missing manifest path")
    elif not manifest_path.is_file():
        errors.append("ACT-N03 nonexistent manifest path")
    else:
        loaded, load_error = _read_json_file(manifest_path)
        if load_error or not isinstance(loaded, dict):
            errors.append(load_error or "ACT-N03 manifest is not a JSON object")
        else:
            manifest = loaded
            manifest_digest = sha256_bytes(manifest_path.read_bytes())
            recorded_manifest_digest = str(record.get("session_b_manifest_digest") or "")
            if not _DIGEST64.fullmatch(recorded_manifest_digest) or recorded_manifest_digest != manifest_digest:
                errors.append("ACT-N08 manifest digest mismatch")
            schema_errors: list[str] = []
            validate_manifest_dict(manifest, schema_errors)
            errors.extend(schema_errors)
            if str(manifest.get("stage_id") or "") != "IIAC-GOV-1":
                errors.append("manifest stage_id must be IIAC-GOV-1")
            if str(manifest.get("policy_version") or "") != POLICY_VERSION:
                errors.append("ACT-N16 manifest policy version mismatch")
            manifest_verdict = str(manifest.get("verdict") or "").upper()
            if manifest_verdict != "PASS":
                errors.append(_verdict_error(manifest_verdict))
            if str(manifest.get("audited_implementation_sha") or "") != corrective:
                errors.append("ACT-N09 manifest audited SHA does not match corrective implementation SHA")
            if str(manifest.get("audit_execution_sha") or "") != audit_sha:
                errors.append("ACT-N09 manifest audit execution SHA mismatch")
            if str(manifest.get("certification_publication_sha") or "") != certification_sha:
                errors.append("manifest certification_publication_sha mismatch")
            if manifest.get("implementation_session_id") == manifest.get("auditor_session_id"):
                errors.append("ACT-N10 self-certifying agent: implementer session equals auditor session")
            if manifest.get("implementation_agent_id") == manifest.get("auditor_agent_id"):
                errors.append("ACT-N10 self-certifying agent: implementer agent equals auditor agent")
            if not str(manifest.get("certifier_session_id") or "").strip():
                errors.append("ACT-N17 missing certifier session provenance")
            if not str(manifest.get("auditor_session_id") or "").strip():
                errors.append("ACT-N17 missing auditor session provenance")
            for field, kind in (
                ("audit_report_digest", "report"),
                ("audit_scope_manifest_digest", "scope"),
                ("test_matrix_digest", "matrix"),
                ("findings_digest", "findings"),
            ):
                artifact = artifacts.get(kind)
                recorded = str(manifest.get(field) or "")
                if artifact is None or not artifact.is_file():
                    errors.append(f"ACT-N13 missing bytes for {field}")
                    continue
                if sha256_bytes(artifact.read_bytes()) != recorded:
                    errors.append(f"ACT-N08 {field} mismatch")
            errors.extend(_companion_errors(repo, manifest_path))

    scope_path = artifacts.get("scope")
    matrix_path = artifacts.get("matrix")
    findings_path = artifacts.get("findings")
    if findings_path is None or not findings_path.is_file():
        errors.append("ACT-N15 missing findings register")
    else:
        loaded, load_error = _read_json_file(findings_path)
        if load_error or not isinstance(loaded, dict):
            errors.append(load_error or "ACT-N15 findings register is not a JSON object")
        else:
            errors.extend(_findings_errors(loaded))
    if scope_path is None or not scope_path.is_file():
        errors.append("ACT-N13 missing audit scope bytes")
    if matrix_path is None or not matrix_path.is_file():
        errors.append("ACT-N13 missing test matrix bytes")

    handoff_path = artifacts.get("handoff")
    if handoff_path is None or not handoff_path.is_file():
        errors.append("ACT-N17 missing session handoff record")
    else:
        handoff_bytes = handoff_path.read_bytes()
        handoff_digest = str(record.get("handoff_digest") or "")
        if not _DIGEST64.fullmatch(handoff_digest) or sha256_bytes(handoff_bytes) != handoff_digest:
            errors.append("ACT-N08 handoff digest mismatch")
        handoff_text = handoff_bytes.decode("utf-8", errors="replace")
        if POLICY_VERSION not in handoff_text:
            errors.append("ACT-N16 handoff policy version mismatch")
        if manifest is not None:
            impl = str(manifest.get("implementation_session_id") or "")
            auditor = str(manifest.get("auditor_session_id") or "")
            if impl and impl not in handoff_text:
                errors.append("ACT-N17 handoff omits implementation session id")
            if auditor and auditor not in handoff_text:
                errors.append("ACT-N17 handoff omits auditor session id")

    operator_path = artifacts.get("operator")
    if operator_path is None or not operator_path.is_file():
        errors.append("ACT-N06 missing operator activation decision record")
    else:
        operator_text = operator_path.read_text(encoding="utf-8")
        errors.extend(_operator_errors(record, operator_text, digest if _DIGEST64.fullmatch(digest) else ""))
        errors.extend(closure_conflicts(record, operator_text))

    if audit_sha and head:
        for rel in _material_changes(repo, audit_sha, head):
            errors.append(f"ACT-N11 stale audit after material governance change: {rel}")
        try:
            product_changes = product_paths_changed_between(audit_sha, head, repo=repo)
        except RuntimeError as exc:
            errors.append(str(exc))
            product_changes = []
        for rel in product_changes:
            errors.append(f"ACT-N11 product path changed after the audited SHA: {rel}")

    return errors
