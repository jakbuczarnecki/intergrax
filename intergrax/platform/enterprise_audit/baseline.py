# © Artur Czarnecki. All rights reserved.

"""Baseline load/store for enterprise audit drift (candidate vs accepted FSM)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
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


def verify_operator_decision_record(
    path: Path,
    *,
    for_promotion: bool = False,
    repo_root: Path | None = None,
) -> dict:
    if not path.is_file():
        raise ValueError("OD-ENT-01 operator decision record missing")
    raw = json.loads(path.read_text(encoding="utf-8"))
    decision_id = str(raw.get("decision_id", ""))
    if decision_id != OD_ENT_01_DECISION_ID:
        raise ValueError(f"expected decision_id {OD_ENT_01_DECISION_ID}")
    origin = str(raw.get("record_origin", "")).lower()
    if origin in {"agent", "cursor", "automation"}:
        raise ValueError("agent-origin OD-ENT-01 records cannot authorize promotion")
    if raw.get("operator_origin") is not True:
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
    operator_root = (repo_root / _OPERATOR_RECORDS_REL).resolve()
    try:
        resolved.relative_to(operator_root)
    except ValueError as exc:
        raise ValueError("OD-ENT-01 record must live under operator_records/") from exc

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

    return {**raw, **pins}


def promote_accepted_baseline(
    baseline_dir: Path,
    *,
    od_record_path: Path,
    verify_closed_bundle_dir: Path | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Promote candidate findings fingerprint to accepted baseline (operator boundary)."""
    from intergrax.platform.enterprise_audit.validators import verify_closed_evidence_bundle

    root = repo_root or Path.cwd()
    record = verify_operator_decision_record(od_record_path, for_promotion=True, repo_root=root)
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
