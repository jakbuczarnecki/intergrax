#!/usr/bin/env python3
"""Validate IIAC audit session manifest (schema + session rules).

v1 manifests (no ``manifest_schema_version``) keep today's verdict. A v2 manifest (``manifest_schema_version: v2``)
is held to more:

* an unknown schema marker or ``manifest_state`` fails;
* ``--profile final|legacy`` (default ``final`` for a final v2 manifest, ignored for v1): a final v2 manifest must
  pass the strict artifact rules of the PT-E04 verifiers (SHAs exist and follow the strict chain, digests equal the
  bytes of the files they name, no placeholders, verdict PASS or PASS_SCOPED) and ``legacy`` is refused for it;
  a v2 draft is not certifying: ``--profile final`` is refused for it;
* a final v2 manifest names its handoff record in ``handoff_record_ref`` (``path@sha``); that record must carry the
  line ``Countersigned: operator @ <ts> manifest-digest <sha256>`` whose digest equals the countersign digest of the
  manifest (``iiac_common.manifest_countersign_digest``). Presence and digest are checked, authorship is not.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import jsonschema

from scripts.governance.iiac_common import (
    HANDOFF_TEMPLATE,
    MANIFEST_SCHEMA,
    REPO,
    git_merge_base_is_ancestor,
    is_v2,
    is_v2_final,
    manifest_countersign_digest,
    split_ref,
    validate_manifest_dict,
)
from scripts.governance.iiac_verifier_common import check_manifest_artifacts, commit_exists, errors_of, show_blob

_COUNTERSIGN = re.compile(r"(?im)^\s*Countersigned:\s*operator\s*@\s*(\S+)\s+manifest-digest\s+([0-9a-f]{64})\s*$")


def countersignature_errors(manifest: dict, repo: Path) -> list[str]:
    ref = split_ref(manifest.get("handoff_record_ref"))
    if ref is None:
        return []  # the format error comes from validate_manifest_dict
    path, sha = ref
    if not commit_exists(repo, sha):
        return [f"handoff record commit {sha[:12]} is not a commit in the repository"]
    blob = show_blob(repo, sha, path)
    if blob is None:
        return [f"handoff record {path} does not exist at {sha[:12]}"]
    found = _COUNTERSIGN.findall(blob.decode("utf-8", errors="replace"))
    if not found:
        return [f"handoff record {path} has no 'Countersigned: operator @ <ts> manifest-digest <sha256>' line"]
    expected = manifest_countersign_digest(manifest)
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
    parser.add_argument(
        "--profile",
        choices=("final", "legacy"),
        default=None,
        help="v2 only: strictness of the artifact checks (default final for a final manifest; ignored for v1)",
    )
    args = parser.parse_args()

    errors: list[str] = []
    if not args.manifest.is_file():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    if not MANIFEST_SCHEMA.is_file():
        errors.append(f"missing schema: {MANIFEST_SCHEMA.relative_to(REPO)}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

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

    if is_v2(manifest):
        repo = args.repo or REPO
        if is_v2_final(manifest):
            if args.profile == "legacy":
                errors.append("a final v2 manifest is verified with --profile final only")
            else:
                for finding in errors_of(check_manifest_artifacts(manifest, repo, profile="final")):
                    if finding.code != "MANIFEST_FORMAT":  # already reported by validate_manifest_dict
                        errors.append(f"{finding.code}: {finding.message}")
                errors.extend(countersignature_errors(manifest, repo))
        elif manifest.get("manifest_state") == "draft":
            if args.profile == "final":
                errors.append("a v2 draft manifest is not certifying and cannot be verified with --profile final")
            else:
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
