#!/usr/bin/env python3
"""Shared helpers for IIAC governance validators (Session A implementation)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
QUAL = REPO / "docs/project/maintainers/qualification"
ACTIVATION_RECORD = QUAL / "IIAC_POLICY_ACTIVATION_RECORD.json"
IIAC_PROTOCOL = QUAL / "INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md"
MANIFEST_SCHEMA = QUAL / "schemas/iiac_audit_session_manifest.v1.schema.json"
HANDOFF_TEMPLATE = QUAL / "IIAC_SESSION_HANDOFF_RECORD.template.md"
OD_ENT_02_THREAT = QUAL / "ent_audit_x/OD_ENT_02_THREAT_MODEL.md"
OD_ENT_02_PACKAGE = QUAL / "ent_audit_x/OD_ENT_02_COMPENSATING_CONTROL_PACKAGE.md"
NEGATIVE_TEST_MODULE = REPO / "tests/unit/governance/test_iiac_negative_guards.py"

PRODUCT_PATH_PREFIXES = ("intergrax/", "agents/", "applications/")
# Extra write surface an auditor may not touch between the audited and the audit-execution SHA (v2 manifests only).
GOVERNANCE_WRITE_SURFACE = (
    "scripts/governance/",
    "tests/unit/governance/",
    "docs/project/maintainers/qualification/",
)

# Manifest v2 (ticket T-IGX-044-F): the extra strictness applies only to manifests that carry the marker.
MANIFEST_SCHEMA_V2 = "v2"
MANIFEST_STATES = ("draft", "final")
V2_ONLY_FIELDS = ("manifest_state", "identity_provenance", "handoff_record_ref")
FINAL_VERDICTS = frozenset({"PASS", "PASS_SCOPED"})
ROLE_FIELDS = {
    "implementer": ("implementation_session_id", "implementation_agent_id"),
    "auditor": ("auditor_session_id", "auditor_agent_id"),
    "certifier": ("certifier_session_id", "certifier_agent_id"),
}
PROVENANCE_FIELDS = ("issuer", "session_uuid", "started_at_utc", "host_receipt_ref")

MANIFEST_REQUIRED_STRINGS = (
    "stage_id",
    "certification_profile",
    "policy_version",
    "implementation_session_id",
    "implementation_agent_id",
    "auditor_session_id",
    "auditor_agent_id",
    "certifier_session_id",
    "certifier_agent_id",
    "audited_implementation_sha",
    "test_execution_sha",
    "evidence_publication_sha",
    "audit_execution_sha",
    "audit_report_digest",
    "audit_scope_manifest_digest",
    "test_matrix_digest",
    "findings_digest",
    "verdict",
    "limitations",
    "certification_publication_sha",
)

SESSION_ID_FIELDS = (
    "implementation_session_id",
    "auditor_session_id",
    "certifier_session_id",
)

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_PLACEHOLDER = re.compile(r"(?i)(pending|todo|tbd|placeholder|unrecorded|unknown|change_?me)")
_ID_SPLIT = re.compile(r"[+,;|\s]+")
_DELEGATE_LINE = re.compile(r"(?im)^\s*(?:[-*]\s*)?Delegate session:\s*(\S.*?)\s*$")


def load_activation_record() -> dict[str, Any]:
    if not ACTIVATION_RECORD.is_file():
        return {"status": "NOT_IN_FORCE", "policy_version": None}
    return json.loads(ACTIVATION_RECORD.read_text(encoding="utf-8"))


def policy_status() -> str:
    rec = load_activation_record()
    return str(rec.get("status") or "NOT_IN_FORCE").upper()


def is_policy_in_force() -> bool:
    return policy_status() == "IN_FORCE"


def git_rev_parse(ref: str = "HEAD", repo: Path | None = None) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=repo or REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git rev-parse {ref} failed")
    return proc.stdout.strip()


def git_merge_base_is_ancestor(ancestor: str, descendant: str, repo: Path | None = None) -> bool:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo or REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def product_paths_changed_between(base_sha: str, head_sha: str, repo: Path | None = None) -> list[str]:
    proc = subprocess.run(
        ["git", "diff", "--name-only", base_sha, head_sha],
        cwd=repo or REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git diff failed")
    changed: list[str] = []
    for line in proc.stdout.splitlines():
        rel = line.strip().replace("\\", "/")
        if any(rel.startswith(p) for p in PRODUCT_PATH_PREFIXES):
            changed.append(rel)
    return changed


def paths_changed_between(
    base_sha: str, head_sha: str, prefixes: tuple[str, ...], repo: Path | None = None
) -> list[str]:
    proc = subprocess.run(
        ["git", "diff", "--name-only", base_sha, head_sha],
        cwd=repo or REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git diff failed")
    changed: list[str] = []
    for line in proc.stdout.splitlines():
        rel = line.strip().replace("\\", "/")
        if any(rel.startswith(p) for p in prefixes):
            changed.append(rel)
    return changed


def normalize_identity(value: str) -> str:
    """Strip, collapse whitespace and case-fold: the form in which identities are compared."""
    return re.sub(r"\s+", " ", value).strip().casefold()


def _norm(value: object) -> str:
    return normalize_identity(value) if isinstance(value, str) else ""


def manifest_schema_version(manifest: dict[str, Any]) -> str | None:
    """``v1`` without a marker, ``v2`` with the v2 marker, ``None`` for any other marker value."""
    if "manifest_schema_version" not in manifest:
        return "v1"
    return MANIFEST_SCHEMA_V2 if manifest["manifest_schema_version"] == MANIFEST_SCHEMA_V2 else None


def is_v2(manifest: dict[str, Any]) -> bool:
    return manifest_schema_version(manifest) == MANIFEST_SCHEMA_V2


def is_v2_final(manifest: dict[str, Any]) -> bool:
    return is_v2(manifest) and manifest.get("manifest_state") == "final"


def split_ref(ref: object) -> tuple[str, str] | None:
    """Parse ``<repo path>@<40-hex sha>``; absolute, traversing and ``.tmp/`` paths are refused."""
    if not isinstance(ref, str):
        return None
    path, sep, sha = ref.strip().rpartition("@")
    path = path.strip().replace("\\", "/")
    if not sep or not _SHA40.match(sha) or not path:
        return None
    if path.startswith(("/", ".tmp/", "./.tmp/")) or re.match(r"^[A-Za-z]:", path) or ".." in Path(path).parts:
        return None
    return path, sha


def manifest_countersign_digest(manifest: dict[str, Any]) -> str:
    """Digest an operator countersignature binds to.

    It is the sha256 of the canonical JSON of the manifest without ``handoff_record_ref``. That field names the
    commit holding the countersignature, so hashing it would make the digest depend on itself.
    """
    body = {key: value for key, value in manifest.items() if key != "handoff_record_ref"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _utc_timestamp_ok(value: object) -> bool:
    if not isinstance(value, str) or not re.search(r"(Z|\+00:00)$", value):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def provenance_errors(manifest: dict[str, Any]) -> list[str]:
    """Format and consistency of ``identity_provenance`` (no git access)."""
    block = manifest.get("identity_provenance")
    if not isinstance(block, dict):
        return ["identity_provenance is required for a final v2 manifest and must be an object"]
    errors: list[str] = []
    seen_uuid: dict[str, str] = {}
    seen_receipt: dict[str, str] = {}
    for role, (session_field, _agent_field) in ROLE_FIELDS.items():
        entry = block.get(role)
        if not isinstance(entry, dict):
            errors.append(f"identity_provenance.{role} is required and must be an object")
            continue
        for key in PROVENANCE_FIELDS:
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"identity_provenance.{role}.{key} must be a non-empty string")
            elif _PLACEHOLDER.search(value):
                errors.append(f"identity_provenance.{role}.{key} carries a placeholder marker")
        if not _utc_timestamp_ok(entry.get("started_at_utc")):
            errors.append(f"identity_provenance.{role}.started_at_utc must be an ISO-8601 UTC timestamp")
        uuid = _norm(entry.get("session_uuid"))
        if uuid and uuid != _norm(manifest.get(session_field)):
            errors.append(f"identity_provenance.{role}.session_uuid differs from {session_field}")
        if uuid and uuid in seen_uuid:
            errors.append(f"identity_provenance session_uuid of {role} repeats the one of {seen_uuid[uuid]}")
        elif uuid:
            seen_uuid[uuid] = role
        receipt = split_ref(entry.get("host_receipt_ref"))
        if receipt is None:
            errors.append(f"identity_provenance.{role}.host_receipt_ref must be <repo path>@<40-hex sha>")
            continue
        key = receipt[0].casefold()
        if key in seen_receipt:
            errors.append(f"host receipt of {role} repeats the receipt of {seen_receipt[key]}")
        else:
            seen_receipt[key] = role
    return errors


def _validate_manifest_v2(manifest: dict[str, Any], errors: list[str]) -> None:
    version = manifest_schema_version(manifest)
    if version is None:
        errors.append(f"unsupported manifest_schema_version {manifest.get('manifest_schema_version')!r}")
        return
    if version != MANIFEST_SCHEMA_V2:
        for key in V2_ONLY_FIELDS:
            if key in manifest:
                errors.append(f"{key} is a v2 field and needs manifest_schema_version {MANIFEST_SCHEMA_V2}")
        return
    state = manifest.get("manifest_state")
    if state not in MANIFEST_STATES:
        errors.append(f"manifest_state must be one of {list(MANIFEST_STATES)} for a v2 manifest, got {state!r}")
        return
    agents: dict[str, str] = {}
    for role, (_session_field, agent_field) in ROLE_FIELDS.items():
        agent = _norm(manifest.get(agent_field))
        if agent and agent in agents:
            errors.append(f"{agent_field} equals the agent label of the {agents[agent]} role")
        elif agent:
            agents[agent] = role
    if state != "final":
        return
    verdict = str(manifest.get("verdict") or "").upper()
    if verdict not in FINAL_VERDICTS:
        errors.append(f"a final v2 manifest needs verdict PASS or PASS_SCOPED, got {verdict or 'none'}")
    for field in (
        *(name for pair in ROLE_FIELDS.values() for name in pair),
        "audited_implementation_sha",
        "test_execution_sha",
        "evidence_publication_sha",
        "audit_execution_sha",
        "certification_publication_sha",
        "audit_report_digest",
        "audit_scope_manifest_digest",
        "test_matrix_digest",
        "findings_digest",
    ):
        value = manifest.get(field)
        if isinstance(value, str) and (_PLACEHOLDER.search(value) or (value and set(value) == {"0"})):
            errors.append(f"{field} carries a placeholder value in a final v2 manifest")
    if split_ref(manifest.get("handoff_record_ref")) is None:
        errors.append("handoff_record_ref (<repo path>@<40-hex sha>) is required for a final v2 manifest")
    errors.extend(provenance_errors(manifest))


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=repo, capture_output=True, check=False)


def resolve_delegation_ref(
    repo: Path,
    path: str,
    sha: str,
    at: str,
    *,
    stage: str | None = None,
    implementer: str | None = None,
) -> list[str]:
    """Read-only check of a delegation record; the returned list is empty when it resolves.

    The commit must exist, hold the path and be an ancestor of ``at``; the record must name the stage and a
    ``Delegate session:`` that is not the implementer. Who wrote the record is not proven (process_enforced).
    """
    parsed = split_ref(f"{path}@{sha}")
    if parsed is None:
        return [f"delegation reference {path!r}@{sha!r} is not a tracked <repo path>@<40-hex sha>"]
    if not at or at.startswith("-"):
        return [f"refusing empty or option-like revision {at!r}"]
    rel, commit = parsed
    if _git(repo, "cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        return [f"delegation commit {commit[:12]} is not a commit in this repository"]
    shown = _git(repo, "show", f"{commit}:{rel}")
    if shown.returncode != 0:
        return [f"delegation record {rel} does not exist at {commit[:12]}"]
    errors: list[str] = []
    if _git(repo, "merge-base", "--is-ancestor", commit, at).returncode != 0:
        errors.append(f"delegation commit {commit[:12]} is not an ancestor of {at[:12]}")
    text = shown.stdout.decode("utf-8", errors="replace")
    if stage and not re.search(rf"(?<![A-Za-z0-9._-]){re.escape(stage)}(?![A-Za-z0-9_-])", text, re.IGNORECASE):
        errors.append(f"delegation record {rel} does not name the stage {stage}")
    delegates = _DELEGATE_LINE.findall(text)
    if not delegates:
        errors.append(f"delegation record {rel} names no 'Delegate session:'")
    elif implementer:
        wanted = normalize_identity(implementer)
        for line in delegates:
            parts = {normalize_identity(part) for part in _ID_SPLIT.split(line) if part}
            if wanted in parts or normalize_identity(line) == wanted:
                errors.append("delegation record names the implementer as the delegate")
                break
    return errors


def validate_manifest_dict(manifest: dict[str, Any], errors: list[str]) -> None:
    for key in MANIFEST_REQUIRED_STRINGS:
        if key not in manifest:
            errors.append(f"manifest missing required field: {key}")
            continue
        val = manifest[key]
        if key == "limitations":
            if not isinstance(val, list):
                errors.append("limitations must be an array")
            continue
        if not isinstance(val, str) or not val.strip():
            errors.append(f"manifest field {key} must be non-empty string")

    impl = _norm(manifest.get("implementation_session_id"))
    aud = _norm(manifest.get("auditor_session_id"))
    cert = _norm(manifest.get("certifier_session_id"))
    if impl and aud and impl == aud:
        errors.append("implementation_session_id must differ from auditor_session_id (N-01)")
    if impl and cert and impl == cert:
        errors.append("implementation_session_id must differ from certifier_session_id")
    if aud and cert and aud == cert:
        errors.append("auditor_session_id must differ from certifier_session_id")
    for role, (session_field, _agent_field) in ROLE_FIELDS.items():
        session = _norm(manifest.get(session_field))
        if not session:
            continue
        for other, (_other_session, other_agent_field) in ROLE_FIELDS.items():
            if other != role and session == _norm(manifest.get(other_agent_field)):
                errors.append(
                    f"{session_field} equals {other_agent_field} of the {other} role (session id used as a role label)"
                )

    for sha_field in (
        "audited_implementation_sha",
        "test_execution_sha",
        "evidence_publication_sha",
        "audit_execution_sha",
        "certification_publication_sha",
    ):
        sha = manifest.get(sha_field)
        if isinstance(sha, str) and sha.strip() and not _SHA40.match(sha.strip()):
            errors.append(f"{sha_field} must be 40-char lowercase hex SHA")

    profile = manifest.get("certification_profile")
    if profile and profile != "iiac-v1":
        errors.append("certification_profile must be iiac-v1 when present")

    verdict = str(manifest.get("verdict") or "").upper()
    if verdict and verdict not in {"PASS", "FAIL", "PASS_SCOPED", "CONDITIONAL"}:
        errors.append(f"unsupported verdict: {verdict}")

    digest_fields = (
        "audit_report_digest",
        "audit_scope_manifest_digest",
        "test_matrix_digest",
        "findings_digest",
    )
    for field in digest_fields:
        val = manifest.get(field)
        if isinstance(val, str) and val.strip():
            if len(val.strip()) < 16:
                errors.append(f"{field} too short for digest (N-11)")

    _validate_manifest_v2(manifest, errors)
