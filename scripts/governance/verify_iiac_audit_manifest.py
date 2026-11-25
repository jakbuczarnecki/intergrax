#!/usr/bin/env python3
"""Validate IIAC audit session manifest (schema + session rules).

v1 manifests (no ``manifest_schema_version``) keep today's verdict. A v2 manifest (``manifest_schema_version: v2``)
is held to more:

* an unknown schema marker or ``manifest_state`` fails;
* a final v2 manifest carries ``identity_provenance`` (checked through ``iiac_common``), names its handoff record in
  ``handoff_record_ref`` (``path@sha``), and that record must carry the line
  ``Countersigned: operator @ <ts> manifest-digest <sha256>`` whose digest equals the countersign digest of the
  manifest (``iiac_common.manifest_countersign_digest``; canonical JSON without ``handoff_record_ref``). Presence and
  digest are checked, authorship is not;
* a final v2 manifest is read as strict JSON: a duplicate object key, a float or non-finite number is an error, and
  the root must be a JSON object (a UTF-8 byte order mark is ignored when reading).

The five-SHA strict chain and the binding of the four digests to their file bytes are NOT checked here: the
activation gate runs the PT-E04 artifact verifier in profile ``final`` for a v2 final manifest. This script imports
and names no PT-E04 module and has no ``--profile`` argument.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import jsonschema

from scripts.governance.iiac_common import (
    HANDOFF_TEMPLATE,
    MANIFEST_SCHEMA,
    REPO,
    git_commit_exists,
    git_merge_base_is_ancestor,
    git_show_blob,
    is_v2_final,
    manifest_countersign_digest,
    split_ref,
    validate_manifest_dict,
)

_COUNTERSIGN = re.compile(r"(?im)^\s*Countersigned:\s*operator\s*@\s*(\S+)\s+manifest-digest\s+([0-9a-f]{64})\s*$")


def parse_manifest(text: str) -> tuple[Any, list[str]]:
    """Parse the manifest and report, besides the value, the findings that only matter for a v2 final manifest."""
    findings: list[str] = []

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        for key, _value in items:
            if key in seen:
                findings.append(f"duplicate key {key!r} (a v2 final manifest must not repeat a key)")
            seen.add(key)
        return dict(items)

    def number(token: str) -> float:
        findings.append(f"float number {token} (a v2 final manifest allows integers only)")
        return float(token)

    def constant(token: str) -> float:
        findings.append(f"non-finite float {token} (a v2 final manifest allows integers only)")
        return float("nan")

    value = json.loads(text, object_pairs_hook=pairs, parse_float=number, parse_constant=constant)
    return value, findings


def countersignature_errors(manifest: dict[str, Any], repo: Path) -> list[str]:
    ref = split_ref(manifest.get("handoff_record_ref"))
    if ref is None:
        return []  # the format error comes from validate_manifest_dict
    path, sha = ref
    if not git_commit_exists(repo, sha):
        return [f"handoff record commit {sha[:12]} is not a commit in the repository"]
    blob = git_show_blob(repo, sha, path)
    if blob is None:
        return [f"handoff record {path} does not exist at {sha[:12]}"]
    found = _COUNTERSIGN.findall(blob.decode("utf-8", errors="replace"))
    if not found:
        return [f"handoff record {path} has no 'Countersigned: operator @ <ts> manifest-digest <sha256>' line"]
    try:
        expected = manifest_countersign_digest(manifest)
    except ValueError:
        return []  # reported by the strict-JSON findings
    if expected not in {digest for _ts, digest in found}:
        return [f"countersignature digest in {path} does not match the manifest (expected {expected[:16]}...)"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="Path to iiac_audit_session_manifest JSON")
    parser.add_argument(
        "--require-handoff-template",
        action="store_true",
        help="Fail if IIAC handoff template missing (N-02 organizational gate)",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Git repository for ancestry checks (defaults to the Intergrax checkout)",
    )
    args = parser.parse_args()

    errors: list[str] = []
    if not args.manifest.is_file():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    if not MANIFEST_SCHEMA.is_file():
        errors.append(f"missing schema: {MANIFEST_SCHEMA.relative_to(REPO)}")

    try:
        manifest, findings = parse_manifest(args.manifest.read_text(encoding="utf-8-sig"))
    except ValueError as exc:
        print("verify_iiac_audit_manifest: FAIL", file=sys.stderr)
        print(f"  - manifest is not valid JSON: {exc}", file=sys.stderr)
        return 1
    if not isinstance(manifest, dict):
        print("verify_iiac_audit_manifest: FAIL", file=sys.stderr)
        print("  - root must be a JSON object", file=sys.stderr)
        return 1

    if MANIFEST_SCHEMA.is_file():
        schema = json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8"))
        try:
            jsonschema.validate(manifest, schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"jsonschema: {exc.message}")

    validate_manifest_dict(manifest, errors)

    audited = manifest.get("audited_implementation_sha")
    audit_exec = manifest.get("audit_execution_sha")
    if isinstance(audited, str) and isinstance(audit_exec, str):
        if audited.strip() and audit_exec.strip():
            if audit_exec.strip() != audited.strip():
                try:
                    if not git_merge_base_is_ancestor(audited.strip(), audit_exec.strip(), repo=args.repo):
                        errors.append("audited_implementation_sha not ancestor of audit_execution_sha (N-04)")
                except RuntimeError as exc:
                    errors.append(str(exc))

    if is_v2_final(manifest):
        errors.extend(findings)
        errors.extend(countersignature_errors(manifest, args.repo or REPO))
    elif manifest.get("manifest_schema_version") == "v2" and manifest.get("manifest_state") == "draft":
        print("NOTE: v2 draft manifest, not certifying", file=sys.stderr)

    if args.require_handoff_template and not HANDOFF_TEMPLATE.is_file():
        errors.append("missing auditor handoff template (N-02)")

    if errors:
        print("verify_iiac_audit_manifest: FAIL", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("verify_iiac_audit_manifest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
