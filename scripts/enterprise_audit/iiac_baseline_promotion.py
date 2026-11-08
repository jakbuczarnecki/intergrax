"""OD-ENT-01 baseline promotion for records signed by an operator-designated signing session (IIAC Amendment 01).

``intergrax.platform.enterprise_audit.baseline`` accepts only a record that claims ``operator_origin: true``, so a
delegated signing session (an agent) can never promote the accepted baseline. This module adds the delegated
path **outside** ``intergrax/``: the IIAC activation gate (ACT-N11) invalidates the policy activation when any
``intergrax/``, ``agents/`` or ``applications/`` path changes after the audited SHA, so the platform package is
reused through its public API and left untouched. Records that are not delegated take the unchanged human path.

A record with ``record_origin: operator-delegate`` is accepted only when ALL of the following hold:

* the IIAC policy is IN_FORCE and the record names it;
* ``signature_line`` is ``Approved OD-ENT-01: operator-delegate @ <ISO-8601 with timezone>`` and
  ``delegation_line`` is ``Delegation: <repo-relative path>@<40-hex commit sha>``;
* the delegation is read from that published commit (an ancestor of HEAD, under
  ``iiac_evidence/ENT-AUDIT-X/operator_decisions/``) and carries a real operator signature line, the same
  decision id, policy, record path and designated session, an implementer list that excludes the signer and a
  validity window containing the signature time;
* the delegation's SHA-256 equals the digest of the record minus its two signature lines, so the record cannot
  change after it was delegated;
* a digest-pinned ENT-AUDIT-X IIAC audit manifest shows PASS or PASS_SCOPED, the same TES and evidence pins and
  pairwise-distinct sessions, and the signer is neither the implementation nor the auditor session;
* the ladder B6 pin, git ancestry and dossier checks of the human path also hold.

Agent origins never authorize promotion.

Command line (works for human and delegated records)::

    python scripts/enterprise_audit/iiac_baseline_promotion.py --od-record <record.json> --evidence-dir <closed bundle>

``run_enterprise_audit.py promote-baseline`` is deliberately left unchanged: it is a qualification-sensitive path
(``mechanical_readiness_check`` requires a new TES when it changes) and it keeps serving human records only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from intergrax.platform.enterprise_audit.baseline import (
    OD_ENT_01_DECISION_ID,
    EnterpriseAuditBaseline,
    assert_baseline_dir_jailed,
    baseline_path,
    candidate_baseline_path,
    load_candidate_baseline,
    parse_sha_ladder_table,
    save_baseline,
)
from intergrax.platform.enterprise_audit.baseline import promote_accepted_baseline as promote_human_record
from intergrax.platform.enterprise_audit.baseline import verify_operator_decision_record as verify_human_record
from intergrax.platform.enterprise_audit.repo_identity import assert_git_ancestor

BlobReader = Callable[[str, str], str | None]

DELEGATE_RECORD_ORIGIN = "operator-delegate"
IIAC_STAGE_ID = "ENT-AUDIT-X"
_QUAL_REL = Path("docs/project/maintainers/qualification")
IIAC_POLICY_ACTIVATION_REL = _QUAL_REL / "IIAC_POLICY_ACTIVATION_RECORD.json"
IIAC_DECISIONS_REL = _QUAL_REL / "iiac_evidence" / IIAC_STAGE_ID / "operator_decisions"
IIAC_CERTIFICATION_REL = _QUAL_REL / "iiac_evidence" / IIAC_STAGE_ID / "certification"
_LADDER_REL = _QUAL_REL / "ent_audit_x" / "ENT_AUDIT_X_SHA_LADDER.md"
_DOSSIER_REL = _QUAL_REL / "ent_audit_x" / "ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md"

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_AGENT_ORIGINS = frozenset({"agent", "cursor", "automation"})
_SIGNATURE_KEYS = frozenset({"signature_line", "delegation_line"})
_ACCEPTING_VERDICTS = frozenset({"PASS", "PASS_SCOPED"})
_PIN_KEYS = ("b6_acceptance_sha", "evidence_publication_sha", "test_execution_sha", "accepted_baseline_sha")
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


def _require_sha40(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA40.match(value):
        raise ValueError(f"missing or invalid ladder pin: {field}")
    return value


def _signature_parts(raw: dict) -> tuple[re.Match[str], re.Match[str]]:
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


def _read_policy_version(repo_root: Path) -> str:
    path = repo_root / IIAC_POLICY_ACTIVATION_REL
    if not path.is_file():
        raise ValueError("IIAC policy activation record missing; delegated promotion blocked")
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    if str(record.get("status", "")).upper() != "IN_FORCE":
        raise ValueError("IIAC policy is not IN_FORCE; delegated promotion blocked")
    version = str(record.get("policy_version", "")).strip()
    if not version:
        raise ValueError("IIAC policy activation record has no policy_version")
    return version


def _verify_publication(pins: dict[str, str], repo_root: Path) -> None:
    ladder = parse_sha_ladder_table(repo_root / _LADDER_REL)
    tes, evid, b6, accepted = (
        pins[key]
        for key in ("test_execution_sha", "evidence_publication_sha", "b6_acceptance_sha", "accepted_baseline_sha")
    )
    ladder_b6 = ladder.get("B6_ACCEPTANCE_PUBLICATION_SHA")
    if ladder_b6 is None:
        raise ValueError("B6_ACCEPTANCE_PUBLICATION_SHA unset in ladder; promotion blocked")
    if b6 != ladder_b6:
        raise ValueError("b6_acceptance_sha does not match SHA ladder B6_ACCEPTANCE_PUBLICATION_SHA")
    for pinned, role, label in (
        (ladder.get("EVIDENCE_PUBLICATION_SHA"), "EVIDENCE_PUBLICATION_SHA", evid),
        (ladder.get("TEST_EXECUTION_SHA"), "TEST_EXECUTION_SHA", tes),
        (ladder.get("ACCEPTED_BASELINE_SHA"), "ACCEPTED_BASELINE_SHA", accepted),
    ):
        if pinned is not None and label != pinned:
            raise ValueError(f"{role.lower()} does not match SHA ladder {role}")
    assert_git_ancestor(repo_root, ancestor_sha=tes)
    dossier = repo_root / _DOSSIER_REL
    text = dossier.read_text(encoding="utf-8-sig") if dossier.is_file() else ""
    if not all(sha in text for sha in (tes, evid, b6)):
        raise ValueError("Protocol 03 dossier must list test_execution, evidence, and b6 SHAs")


def _verify_delegation(raw: dict, *, record_path: Path, repo_root: Path, blob_reader: BlobReader | None) -> None:
    signature, delegation = _signature_parts(raw)
    policy = _read_policy_version(repo_root)
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
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != str(raw.get("iiac_manifest_digest", "")).lower():
        raise ValueError("delegate record: iiac_manifest_digest does not match the manifest file")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
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
    if str(raw.get("designated_session_id", "")).strip() in {sessions[0], sessions[1]}:
        raise ValueError("delegate record: designated session equals the implementation or auditor session")


def verify_delegated_record(
    path: Path,
    *,
    for_promotion: bool = True,
    repo_root: Path | None = None,
    blob_reader: BlobReader | None = None,
) -> dict:
    """Verify an OD-ENT-01 record signed by an operator-designated signing session."""
    if not path.is_file():
        raise ValueError("OD-ENT-01 operator decision record missing")
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if str(raw.get("decision_id", "")) != OD_ENT_01_DECISION_ID:
        raise ValueError(f"expected decision_id {OD_ENT_01_DECISION_ID}")
    origin = str(raw.get("record_origin", "")).lower()
    if origin in _AGENT_ORIGINS:
        raise ValueError("agent-origin OD-ENT-01 records cannot authorize promotion")
    if origin != DELEGATE_RECORD_ORIGIN:
        raise ValueError(f"record_origin must be {DELEGATE_RECORD_ORIGIN!r} for the delegated path")
    _signature_parts(raw)
    pins = {key: _require_sha40(raw.get(key), field=key) for key in _PIN_KEYS}
    if not for_promotion:
        return {**raw, **pins}
    if repo_root is None:
        raise ValueError("repo_root required for promotion verification")

    try:
        path.resolve().relative_to((repo_root / IIAC_DECISIONS_REL).resolve())
    except ValueError as exc:
        raise ValueError(f"delegated OD-ENT-01 record must live under {IIAC_DECISIONS_REL.name}/") from exc
    _verify_publication(pins, repo_root)
    _verify_delegation(raw, record_path=path, repo_root=repo_root, blob_reader=blob_reader)
    _verify_iiac_manifest(raw, pins, repo_root=repo_root)
    return {**raw, **pins}


def _is_delegated(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return False
    return isinstance(raw, dict) and str(raw.get("record_origin", "")).lower() == DELEGATE_RECORD_ORIGIN


def verify_record(
    path: Path,
    *,
    for_promotion: bool = False,
    repo_root: Path | None = None,
    blob_reader: BlobReader | None = None,
) -> dict:
    """Verify an OD-ENT-01 record of either origin (human records take the unchanged platform path)."""
    if _is_delegated(path):
        return verify_delegated_record(path, for_promotion=for_promotion, repo_root=repo_root, blob_reader=blob_reader)
    return verify_human_record(path, for_promotion=for_promotion, repo_root=repo_root)


def promote_accepted_baseline(
    baseline_dir: Path,
    *,
    od_record_path: Path,
    verify_closed_bundle_dir: Path | None = None,
    repo_root: Path | None = None,
    blob_reader: BlobReader | None = None,
) -> Path:
    """Promote the candidate baseline for a human or a delegated OD-ENT-01 record."""
    root = repo_root or Path.cwd()
    if not _is_delegated(od_record_path):
        return promote_human_record(
            baseline_dir,
            od_record_path=od_record_path,
            verify_closed_bundle_dir=verify_closed_bundle_dir,
            repo_root=root,
        )

    from intergrax.platform.enterprise_audit.validators import verify_closed_evidence_bundle

    record = verify_delegated_record(od_record_path, for_promotion=True, repo_root=root, blob_reader=blob_reader)
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
        bundle_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if str(bundle_manifest.get("qualification_sha", "")) != record["test_execution_sha"]:
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote the accepted ENT-AUDIT-X baseline (OD-ENT-01).")
    parser.add_argument("--od-record", type=Path, required=True, help="OD-ENT-01 JSON record (human or delegated)")
    parser.add_argument("--evidence-dir", type=Path, required=True, help="closed evidence bundle directory")
    parser.add_argument("--baseline-dir", type=Path, help="qualification baseline directory")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(None if argv is None else list(argv))

    root = args.repo_root.resolve()
    try:
        baseline_dir = assert_baseline_dir_jailed(args.baseline_dir or (root / "tests/qualification/ent_audit_x"), root)
        promote_accepted_baseline(
            baseline_dir,
            od_record_path=args.od_record,
            verify_closed_bundle_dir=args.evidence_dir,
            repo_root=root,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"accepted baseline promoted under {baseline_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
