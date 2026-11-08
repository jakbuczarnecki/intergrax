# © Artur Czarnecki. All rights reserved.

"""Baseline load/store for enterprise audit drift (candidate vs accepted FSM)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import (
    AuditCategory,
    AuditFinding,
    AuditSeverity,
    DriftStatus,
    ManifestCheck,
)

_SEVERITY_RANK = {
    AuditSeverity.INFO: 0,
    AuditSeverity.TRACKED_DEBT: 1,
    AuditSeverity.WARNING: 2,
    AuditSeverity.ENVIRONMENT: 3,
    AuditSeverity.ERROR: 4,
    AuditSeverity.BLOCKER: 5,
}

ACCEPTED_BASELINE_FILENAME = "ENTERPRISE_AUDIT_BASELINE.json"
CANDIDATE_BASELINE_FILENAME = "CANDIDATE_ENTERPRISE_AUDIT_BASELINE.json"
OD_ENT_01_DECISION_ID = "OD-ENT-01"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_LADDER_REL = Path("docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_SHA_LADDER.md")
_DOSSIER_REL = Path("docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md")
_OPERATOR_RECORDS_REL = Path("docs/project/maintainers/qualification/ent_audit_x/operator_records")
_LADDER_ROLE_KEYS = (
    "TEST_EXECUTION_SHA",
    "EVIDENCE_PUBLICATION_SHA",
    "B6_ACCEPTANCE_PUBLICATION_SHA",
    "ACCEPTED_BASELINE_SHA",
)

_QUAL_REL = Path("docs/project/maintainers/qualification")
IIAC_POLICY_ACTIVATION_REL = _QUAL_REL / "IIAC_POLICY_ACTIVATION_RECORD.json"
IIAC_STAGE_ID = "ENT-AUDIT-X"
IIAC_DECISIONS_REL = _QUAL_REL / "iiac_evidence" / IIAC_STAGE_ID / "operator_decisions"
IIAC_CERTIFICATION_REL = _QUAL_REL / "iiac_evidence" / IIAC_STAGE_ID / "certification"
DELEGATE_RECORD_ORIGIN = "operator-delegate"
_AGENT_ORIGINS = frozenset({"agent", "cursor", "automation"})
_SIGNATURE_KEYS = frozenset({"signature_line", "delegation_line"})
_ACCEPTING_VERDICTS = frozenset({"PASS", "PASS_SCOPED"})
_TZ_ISO = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})"
_DELEGATE_SIGNATURE_RE = re.compile(rf"^Approved (?P<decision>OD-ENT-\d{{2}}): operator-delegate @ (?P<ts>{_TZ_ISO})$")
_DELEGATION_LINE_RE = re.compile(r"^Delegation: (?P<path>[^@\s]+)@(?P<sha>[0-9a-f]{40})$")
_OPERATOR_DELEGATION_RE = re.compile(
    rf"^Approved delegation (?P<id>[^:\r\n]+?): operator @ (?P<ts>{_TZ_ISO})\s*$", re.MULTILINE
)
_DELEGATION_LABELS = {
    "delegation_id": "Delegation ID:",
    "policy_version": "Policy version:",
    "date": "Date (ISO-8601):",
    "decision_id": "Decision ID (decision to be signed):",
    "record_path": "Record path (repo-relative path of the record to be signed):",
    "signed_text_digest": "SHA-256 of the text to be signed:",
    "designated_session_id": "Designated session id:",
    "implementer_session_ids": "Implementer session id(s) of this stage",
    "expires": "Expires (ISO-8601):",
}
BlobReader = Callable[[str, str], str | None]


class BaselinePhase(StrEnum):
    NO_ACCEPTED = "NO_ACCEPTED"
    CANDIDATE_ONLY = "CANDIDATE_ONLY"
    ACCEPTED_ACTIVE = "ACCEPTED_ACTIVE"


@dataclass(frozen=True, slots=True)
class EnterpriseAuditBaseline:
    accepted_baseline_sha: str
    findings: dict[str, dict]
    schema_version: int = 1
    check_manifest_digest: str = ""


def baseline_path(default_dir: Path) -> Path:
    return default_dir / ACCEPTED_BASELINE_FILENAME


def candidate_baseline_path(default_dir: Path) -> Path:
    return default_dir / CANDIDATE_BASELINE_FILENAME


def assert_baseline_dir_jailed(baseline_dir: Path, repo_root: Path) -> Path:
    resolved = baseline_dir.resolve()
    qual_root = (repo_root / "tests" / "qualification" / "ent_audit_x").resolve()
    try:
        resolved.relative_to(qual_root)
    except ValueError as exc:
        raise ValueError("baseline_dir must stay under tests/qualification/ent_audit_x") from exc
    return resolved


def resolve_baseline_phase(baseline_dir: Path) -> BaselinePhase:
    accepted = baseline_path(baseline_dir)
    candidate = candidate_baseline_path(baseline_dir)
    if accepted.is_file():
        return BaselinePhase.ACCEPTED_ACTIVE
    if candidate.is_file():
        return BaselinePhase.CANDIDATE_ONLY
    return BaselinePhase.NO_ACCEPTED


def load_baseline(path: Path) -> EnterpriseAuditBaseline | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return EnterpriseAuditBaseline(
        accepted_baseline_sha=str(raw["accepted_baseline_sha"]),
        findings={k: v for k, v in raw.get("findings", {}).items()},
        schema_version=int(raw.get("schema_version", 1)),
        check_manifest_digest=str(raw.get("check_manifest_digest", "")),
    )


def load_candidate_baseline(path: Path) -> EnterpriseAuditBaseline | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return EnterpriseAuditBaseline(
        accepted_baseline_sha=str(raw.get("candidate_baseline_sha", raw.get("accepted_baseline_sha", ""))),
        findings={k: v for k, v in raw.get("findings", {}).items()},
        schema_version=int(raw.get("schema_version", 1)),
    )


def save_baseline(path: Path, baseline: EnterpriseAuditBaseline) -> None:
    """Write accepted baseline — operator promotion path only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": baseline.schema_version,
        "accepted_baseline_sha": baseline.accepted_baseline_sha,
        "check_manifest_digest": baseline.check_manifest_digest,
        "findings": baseline.findings,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_candidate_baseline(path: Path, baseline: EnterpriseAuditBaseline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": baseline.schema_version,
        "candidate_baseline_sha": baseline.accepted_baseline_sha,
        "check_manifest_digest": baseline.check_manifest_digest,
        "findings": baseline.findings,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_accepted_baseline_requires_operator(*, od_record_path: Path) -> None:
    """Fail closed when promotion attempted without operator OD-ENT-01 record."""
    verify_operator_decision_record(od_record_path, for_promotion=False)


def parse_sha_ladder_table(ladder_path: Path) -> dict[str, str | None]:
    """Parse ENT_AUDIT_X_SHA_LADDER.md role rows; unset/* → None."""
    if not ladder_path.is_file():
        raise ValueError("ENT_AUDIT_X_SHA_LADDER.md missing")
    roles: dict[str, str | None] = {}
    for line in ladder_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        role = cells[0].strip()
        if role not in _LADDER_ROLE_KEYS:
            continue
        sha_cell = cells[1].strip().strip("`")
        if sha_cell.lower().startswith("unset") or sha_cell.startswith("*"):
            roles[role] = None
        elif _SHA40.match(sha_cell):
            roles[role] = sha_cell
        else:
            roles[role] = None
    return roles


def _require_sha40(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA40.match(value):
        raise ValueError(f"missing or invalid ladder pin: {field}")
    return value


def _dossier_lists_shas(dossier_path: Path, *shas: str) -> bool:
    if not dossier_path.is_file():
        return False
    text = dossier_path.read_text(encoding="utf-8")
    return all(sha in text for sha in shas)


def signed_text_digest(record: dict) -> str:
    """SHA-256 of the text a delegation record binds: the OD record minus its two signature lines."""
    body = {key: value for key, value in record.items() if key not in _SIGNATURE_KEYS}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_blob_reader(repo_root: Path) -> BlobReader:
    """Read ``<sha>:<path>`` from git, so a delegation is checked as published, not as edited in the tree."""
    root = repo_root.resolve()

    def _read(sha: str, rel: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(root), "show", f"{sha}:{rel}"],
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        return completed.stdout.decode("utf-8", errors="replace")

    return _read


def _parse_instant(value: str, *, field: str, end_of_day: bool = False) -> datetime:
    text = value.strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            day = datetime.fromisoformat(text).replace(tzinfo=UTC)
            return day.replace(hour=23, minute=59, second=59) if end_of_day else day
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"delegate record: {field} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"delegate record: {field} must carry a timezone: {value!r}")
    return parsed


def _delegation_field(text: str, label: str) -> str | None:
    match = re.search(rf"{re.escape(label)}[^`\n]*`([^`\n]*)`", text)
    if match is None:
        return None
    value = match.group(1).strip()
    if not value or set(value) <= {"_"}:
        return None
    return value


def _delegate_signature_parts(raw: dict) -> tuple[re.Match[str], re.Match[str]]:
    signature = _DELEGATE_SIGNATURE_RE.match(str(raw.get("signature_line", "")).strip())
    if signature is None or signature.group("decision") != OD_ENT_01_DECISION_ID:
        raise ValueError(
            "delegate record: signature_line must be 'Approved OD-ENT-01: operator-delegate @ <ISO-8601 with timezone>'"
        )
    delegation = _DELEGATION_LINE_RE.match(str(raw.get("delegation_line", "")).strip())
    if delegation is None:
        raise ValueError(
            "delegate record: delegation_line must be 'Delegation: <repo-relative path>@<40-hex commit sha>'"
        )
    return signature, delegation


def _read_iiac_policy_version(repo_root: Path) -> str:
    path = repo_root / IIAC_POLICY_ACTIVATION_REL
    if not path.is_file():
        raise ValueError("IIAC policy activation record missing; delegated promotion blocked")
    record = json.loads(path.read_text(encoding="utf-8"))
    if str(record.get("status", "")).upper() != "IN_FORCE":
        raise ValueError("IIAC policy is not IN_FORCE; delegated promotion blocked")
    version = str(record.get("policy_version", "")).strip()
    if not version:
        raise ValueError("IIAC policy activation record has no policy_version")
    return version


def _verify_delegation(raw: dict, *, record_path: Path, repo_root: Path, blob_reader: BlobReader | None) -> None:
    from intergrax.platform.enterprise_audit.repo_identity import assert_git_ancestor

    signature, delegation = _delegate_signature_parts(raw)
    policy = _read_iiac_policy_version(repo_root)
    if str(raw.get("policy_version", "")) != policy:
        raise ValueError(f"delegate record: policy_version must equal the in-force IIAC policy {policy}")

    rel = Path(delegation.group("path"))
    decisions_prefix = IIAC_DECISIONS_REL.as_posix() + "/"
    if rel.is_absolute() or ".." in rel.parts or not rel.as_posix().startswith(decisions_prefix):
        raise ValueError(f"delegate record: delegation record must live under {decisions_prefix}")
    sha = delegation.group("sha")
    assert_git_ancestor(repo_root, ancestor_sha=sha)
    text = (blob_reader or git_blob_reader(repo_root))(sha, rel.as_posix())
    if text is None:
        raise ValueError(f"delegate record: delegation record not found at {rel.as_posix()}@{sha}")

    operator_line = _OPERATOR_DELEGATION_RE.search(text)
    if operator_line is None:
        raise ValueError(
            "delegate record: delegation record has no operator signature line "
            "'Approved delegation <id>: operator @ <ISO-8601>' (blank or template text is not a signature)"
        )
    fields = {name: _delegation_field(text, label) for name, label in _DELEGATION_LABELS.items()}
    missing = sorted(name for name, value in fields.items() if value is None)
    if missing:
        raise ValueError(f"delegate record: delegation record has blank or missing fields: {', '.join(missing)}")
    values = {name: str(value) for name, value in fields.items()}

    if operator_line.group("id").strip() != values["delegation_id"]:
        raise ValueError("delegate record: operator signature line names a different delegation id")
    if values["decision_id"] != OD_ENT_01_DECISION_ID:
        raise ValueError(
            f"delegate record: delegation covers decision {values['decision_id']}, not {OD_ENT_01_DECISION_ID}"
        )
    if values["policy_version"] != policy:
        raise ValueError("delegate record: delegation policy version differs from the in-force IIAC policy")
    if values["signed_text_digest"].lower() != signed_text_digest(raw):
        raise ValueError(
            "delegate record: SHA-256 of the signed text differs from the delegation (record changed after delegation)"
        )
    expected_path = record_path.resolve().relative_to(repo_root.resolve()).as_posix()
    if values["record_path"] != expected_path:
        raise ValueError(f"delegate record: delegation binds record path {values['record_path']}, not {expected_path}")

    designated = str(raw.get("designated_session_id", "")).strip()
    if not designated or designated != values["designated_session_id"]:
        raise ValueError("delegate record: designated_session_id differs from the delegation record")
    implementers = {item for item in re.split(r"[,\s]+", values["implementer_session_ids"]) if item}
    if designated in implementers:
        raise ValueError("delegate record: designated session is listed as an implementer session")

    _parse_instant(values["date"], field="delegation Date")
    signed_at = _parse_instant(signature.group("ts"), field="signature_line timestamp")
    operator_signed = _parse_instant(operator_line.group("ts"), field="operator delegation timestamp")
    expires = _parse_instant(values["expires"], field="delegation Expires", end_of_day=True)
    if not operator_signed <= signed_at <= expires:
        raise ValueError("delegate record: signature timestamp is outside the delegation validity window")


def _verify_iiac_manifest(raw: dict, pins: dict[str, str], *, repo_root: Path) -> None:
    rel_text = str(raw.get("iiac_manifest_path", "")).strip()
    rel = Path(rel_text)
    if not rel_text or rel.is_absolute() or ".." in rel.parts:
        raise ValueError("delegate record: iiac_manifest_path missing or not repo-relative")
    cert_root = (repo_root / IIAC_CERTIFICATION_REL).resolve()
    manifest_path = (repo_root / rel).resolve()
    try:
        manifest_path.relative_to(cert_root)
    except ValueError as exc:
        raise ValueError(
            f"delegate record: IIAC manifest must live under {IIAC_CERTIFICATION_REL.as_posix()}/"
        ) from exc
    if not manifest_path.is_file():
        raise ValueError("delegate record: IIAC audit manifest missing")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != str(raw.get("iiac_manifest_digest", "")).lower():
        raise ValueError("delegate record: iiac_manifest_digest does not match the manifest file")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage_id") != IIAC_STAGE_ID:
        raise ValueError(f"delegate record: IIAC manifest stage_id must be {IIAC_STAGE_ID}")
    if str(manifest.get("verdict", "")) not in _ACCEPTING_VERDICTS:
        raise ValueError("delegate record: IIAC manifest verdict must be PASS or PASS_SCOPED")
    for key in ("test_execution_sha", "evidence_publication_sha"):
        if manifest.get(key) != pins[key]:
            raise ValueError(f"delegate record: IIAC manifest {key} does not match the record pin")
    sessions = [
        str(manifest.get(key, "")).strip()
        for key in ("implementation_session_id", "auditor_session_id", "certifier_session_id")
    ]
    if not all(sessions) or len(set(sessions)) != len(sessions):
        raise ValueError("delegate record: IIAC manifest session ids must be present and pairwise distinct")
    designated = str(raw.get("designated_session_id", "")).strip()
    if designated in {sessions[0], sessions[1]}:
        raise ValueError("delegate record: designated session equals the implementation or auditor session")


def verify_operator_decision_record(
    path: Path,
    *,
    for_promotion: bool = False,
    repo_root: Path | None = None,
    blob_reader: BlobReader | None = None,
) -> dict:
    """Verify an OD-ENT-01 record.

    Two origins are accepted: a record written by the operator (``operator_origin: true``), and a record
    signed by an operator-designated signing session under IIAC Amendment 01 (``record_origin:
    operator-delegate``), which additionally needs a published, operator-signed delegation, an in-force
    IIAC policy and an accepting independent-audit manifest. Agent origins never authorize promotion.
    """
    if not path.is_file():
        raise ValueError("OD-ENT-01 operator decision record missing")
    raw = json.loads(path.read_text(encoding="utf-8"))
    decision_id = str(raw.get("decision_id", ""))
    if decision_id != OD_ENT_01_DECISION_ID:
        raise ValueError(f"expected decision_id {OD_ENT_01_DECISION_ID}")
    origin = str(raw.get("record_origin", "")).lower()
    if origin in _AGENT_ORIGINS:
        raise ValueError("agent-origin OD-ENT-01 records cannot authorize promotion")
    delegated = origin == DELEGATE_RECORD_ORIGIN
    if delegated:
        _delegate_signature_parts(raw)
    elif raw.get("operator_origin") is not True:
        raise ValueError("operator_origin must be true for OD-ENT-01 promotion")
    pins = {
        key: _require_sha40(raw.get(key), field=key)
        for key in (
            "b6_acceptance_sha",
            "evidence_publication_sha",
            "test_execution_sha",
            "accepted_baseline_sha",
        )
    }
    if not for_promotion:
        return {**raw, **pins}

    if repo_root is None:
        raise ValueError("repo_root required for promotion verification")

    resolved = path.resolve()
    records_rel = IIAC_DECISIONS_REL if delegated else _OPERATOR_RECORDS_REL
    try:
        resolved.relative_to((repo_root / records_rel).resolve())
    except ValueError as exc:
        label = "delegated OD-ENT-01 record" if delegated else "OD-ENT-01 record"
        raise ValueError(f"{label} must live under {records_rel.name}/") from exc

    ladder = parse_sha_ladder_table(repo_root / _LADDER_REL)
    tes = pins["test_execution_sha"]
    evid = pins["evidence_publication_sha"]
    b6 = pins["b6_acceptance_sha"]
    accepted = pins["accepted_baseline_sha"]

    ladder_b6 = ladder.get("B6_ACCEPTANCE_PUBLICATION_SHA")
    ladder_evid = ladder.get("EVIDENCE_PUBLICATION_SHA")
    ladder_tes = ladder.get("TEST_EXECUTION_SHA")
    ladder_accepted = ladder.get("ACCEPTED_BASELINE_SHA")

    if ladder_b6 is None:
        raise ValueError("B6_ACCEPTANCE_PUBLICATION_SHA unset in ladder; promotion blocked")
    if b6 != ladder_b6:
        raise ValueError("b6_acceptance_sha does not match SHA ladder B6_ACCEPTANCE_PUBLICATION_SHA")
    if ladder_evid is not None and evid != ladder_evid:
        raise ValueError("evidence_publication_sha does not match SHA ladder EVIDENCE_PUBLICATION_SHA")
    if ladder_tes is not None and tes != ladder_tes:
        raise ValueError("test_execution_sha does not match SHA ladder TEST_EXECUTION_SHA")
    if ladder_accepted is not None and accepted != ladder_accepted:
        raise ValueError("accepted_baseline_sha does not match SHA ladder ACCEPTED_BASELINE_SHA")

    from intergrax.platform.enterprise_audit.repo_identity import assert_git_ancestor

    assert_git_ancestor(repo_root, ancestor_sha=tes)

    dossier = repo_root / _DOSSIER_REL
    if not _dossier_lists_shas(dossier, tes, evid, b6):
        raise ValueError("Protocol 03 dossier must list test_execution, evidence, and b6 SHAs")

    if delegated:
        _verify_delegation(raw, record_path=path, repo_root=repo_root, blob_reader=blob_reader)
        _verify_iiac_manifest(raw, pins, repo_root=repo_root)

    return {**raw, **pins}


def promote_accepted_baseline(
    baseline_dir: Path,
    *,
    od_record_path: Path,
    verify_closed_bundle_dir: Path | None = None,
    repo_root: Path | None = None,
    blob_reader: BlobReader | None = None,
) -> Path:
    """Promote candidate findings fingerprint to accepted baseline (operator boundary)."""
    from intergrax.platform.enterprise_audit.validators import verify_closed_evidence_bundle

    root = repo_root or Path.cwd()
    record = verify_operator_decision_record(
        od_record_path, for_promotion=True, repo_root=root, blob_reader=blob_reader
    )
    candidate = load_candidate_baseline(candidate_baseline_path(baseline_dir))
    if candidate is None:
        raise ValueError("candidate baseline missing; run certifying FREEZE bootstrap first")
    if verify_closed_bundle_dir is None:
        raise ValueError("verify_closed_bundle_dir required for OD-ENT-01 promotion (closed evidence bundle)")
    errors = verify_closed_evidence_bundle(verify_closed_bundle_dir)
    if errors:
        raise ValueError("; ".join(errors))
    manifest_path = verify_closed_bundle_dir / "evidence_manifest.v1.json"
    if manifest_path.is_file():
        bundle_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundle_sha = str(bundle_manifest.get("qualification_sha", ""))
        if bundle_sha != record["test_execution_sha"]:
            raise ValueError("test_execution_sha must equal bundle manifest qualification_sha")
    accepted_sha = str(record["accepted_baseline_sha"])
    if candidate.accepted_baseline_sha != accepted_sha:
        raise ValueError("candidate_baseline_sha does not match OD-ENT-01 accepted_baseline_sha pin")
    out = baseline_path(baseline_dir)
    save_baseline(
        out,
        EnterpriseAuditBaseline(
            accepted_baseline_sha=accepted_sha,
            findings=dict(candidate.findings),
            schema_version=candidate.schema_version,
        ),
    )
    return out


def _message_fingerprint(message: str) -> str:
    return hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]


def _regressed(baseline_row: dict, finding: AuditFinding, *, current_check_schema: int) -> bool:
    old_sev = AuditSeverity(str(baseline_row.get("severity", AuditSeverity.INFO)))
    if _SEVERITY_RANK.get(finding.severity, 0) > _SEVERITY_RANK.get(old_sev, 0):
        return True
    old_fp = str(baseline_row.get("message_fingerprint", ""))
    if old_fp and old_fp != _message_fingerprint(finding.message):
        return True
    old_schema = int(baseline_row.get("check_schema_version", baseline_row.get("schema_version", 1)))
    if current_check_schema > old_schema:
        return True
    return False


def apply_drift(
    findings: tuple[AuditFinding, ...],
    baseline: EnterpriseAuditBaseline | None,
    *,
    checks_by_id: dict[str, ManifestCheck] | None = None,
) -> tuple[AuditFinding, ...]:
    if baseline is None:
        return findings
    checks_by_id = checks_by_id or {}
    updated: list[AuditFinding] = []
    baseline_ids = set(baseline.findings)
    for finding in findings:
        if finding.finding_id in baseline.findings:
            row = baseline.findings[finding.finding_id]
            check = checks_by_id.get(finding.check_id)
            current_schema = check.schema_version if check else 1
            drift = (
                DriftStatus.REGRESSED
                if _regressed(row, finding, current_check_schema=current_schema)
                else DriftStatus.UNCHANGED
            )
        else:
            drift = DriftStatus.NEW
        updated.append(
            AuditFinding(
                finding_id=finding.finding_id,
                check_id=finding.check_id,
                severity=finding.severity,
                category=finding.category,
                message=finding.message,
                source_path=finding.source_path,
                frz_mapping=finding.frz_mapping,
                drift=drift,
            )
        )
    for resolved_id in baseline_ids - {f.finding_id for f in findings}:
        old = baseline.findings[resolved_id]
        updated.append(
            AuditFinding(
                finding_id=resolved_id,
                check_id=str(old.get("check_id", "")),
                severity=AuditSeverity(old.get("severity", AuditSeverity.INFO)),
                category=AuditCategory(old.get("category", AuditCategory.ARCHITECTURE)),
                message="Resolved since baseline",
                drift=DriftStatus.RESOLVED,
            )
        )
    return tuple(updated)


def findings_to_baseline_rows(
    findings: tuple[AuditFinding, ...],
    checks_by_id: dict[str, ManifestCheck] | None = None,
) -> dict[str, dict]:
    checks_by_id = checks_by_id or {}
    rows: dict[str, dict] = {}
    for finding in findings:
        check = checks_by_id.get(finding.check_id)
        rows[finding.finding_id] = {
            "check_id": finding.check_id,
            "severity": finding.severity.value,
            "message_fingerprint": _message_fingerprint(finding.message),
            "schema_version": 1,
            "check_schema_version": check.schema_version if check else 1,
        }
    return rows


def assert_baseline_manifest_digest_current(baseline: EnterpriseAuditBaseline, current_digest: str) -> None:
    if not baseline.check_manifest_digest:
        raise ValueError("accepted baseline missing check_manifest_digest pin")
    if baseline.check_manifest_digest != current_digest:
        raise ValueError("check_manifest_digest changed since accepted baseline (fail-closed)")
