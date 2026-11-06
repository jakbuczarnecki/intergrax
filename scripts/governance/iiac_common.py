#!/usr/bin/env python3
"""Shared helpers for IIAC governance validators (Session A implementation)."""

from __future__ import annotations

import json
import re
import subprocess
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

    impl = manifest.get("implementation_session_id")
    aud = manifest.get("auditor_session_id")
    cert = manifest.get("certifier_session_id")
    if impl and aud and impl == aud:
        errors.append("implementation_session_id must differ from auditor_session_id (N-01)")
    if impl and cert and impl == cert:
        errors.append("implementation_session_id must differ from certifier_session_id")
    if aud and cert and aud == cert:
        errors.append("auditor_session_id must differ from certifier_session_id")

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
