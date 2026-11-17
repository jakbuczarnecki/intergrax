#!/usr/bin/env python3
"""Verify IIAC role separation (session inequality + auditor write-surface + identity provenance).

v1 manifests (no ``manifest_schema_version``) keep today's behaviour: normalised session comparison, and an
auditor product write-surface that is a warning unless ``--strict-product-diff`` is given.

v2 manifests (``manifest_schema_version: v2``) are stricter, by default:
  * the write surface is the product paths plus ``GOVERNANCE_WRITE_SURFACE`` between the audited and the
    audit-execution SHA, and a change there fails; ``--allow-product-diff`` is the explicit opt-out
    (``--strict-product-diff`` is accepted and is a no-op);
  * a final manifest's ``identity_provenance`` host receipts must exist at their commit, that commit must be an
    ancestor of the audit-execution SHA, the receipt must contain the session uuid, and neither the uuid nor the
    receipt may repeat across the three roles or in another v2 manifest of the corpus (``--corpus-root``).

This proves the consistency of recorded identifiers, not who typed them: independence stays process_enforced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.governance.iiac_common import (
    GOVERNANCE_WRITE_SURFACE,
    PRODUCT_PATH_PREFIXES,
    REPO,
    ROLE_FIELDS,
    is_v2,
    is_v2_final,
    normalize_identity,
    paths_changed_between,
    product_paths_changed_between,
    split_ref,
    validate_manifest_dict,
)
from scripts.governance.iiac_verifier_common import commit_exists, is_ancestor_or_equal, show_blob

DEFAULT_CORPUS_ROOT = "docs/project/maintainers/qualification"


def receipt_errors(manifest: dict, repo: Path, audit_exec: str) -> list[str]:
    """Git-level checks of the host receipts of a final v2 manifest."""
    block = manifest.get("identity_provenance")
    if not isinstance(block, dict):
        return []  # reported by validate_manifest_dict
    errors: list[str] = []
    for role in ROLE_FIELDS:
        entry = block.get(role)
        if not isinstance(entry, dict):
            continue
        ref = split_ref(entry.get("host_receipt_ref"))
        if ref is None:
            continue
        path, sha = ref
        if not commit_exists(repo, sha):
            errors.append(f"{role} host receipt commit {sha[:12]} is not a commit in the repository")
            continue
        blob = show_blob(repo, sha, path)
        if blob is None:
            errors.append(f"{role} host receipt {path} does not exist at {sha[:12]}")
            continue
        if audit_exec and commit_exists(repo, audit_exec) and not is_ancestor_or_equal(repo, sha, audit_exec):
            errors.append(f"{role} host receipt commit {sha[:12]} is not an ancestor of audit_execution_sha")
        uuid = entry.get("session_uuid")
        text = blob.decode("utf-8", errors="replace")
        if isinstance(uuid, str) and uuid.strip() and normalize_identity(uuid) not in normalize_identity(text):
            errors.append(f"{role} host receipt {path} does not contain the session uuid")
    return errors


def corpus_errors(manifest: dict, manifest_path: Path, corpus_root: Path) -> list[str]:
    """A session uuid or receipt path of a v2 manifest may not repeat in another v2 manifest of the corpus."""
    block = manifest.get("identity_provenance")
    if not isinstance(block, dict) or not corpus_root.is_dir():
        return []
    mine_uuid: dict[str, str] = {}
    mine_receipt: dict[str, str] = {}
    for role in ROLE_FIELDS:
        entry = block.get(role)
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("session_uuid"), str):
            mine_uuid[normalize_identity(entry["session_uuid"])] = role
        ref = split_ref(entry.get("host_receipt_ref"))
        if ref is not None:
            mine_receipt[ref[0].casefold()] = role
    errors: list[str] = []
    own = manifest_path.resolve()
    for other in sorted(corpus_root.rglob("iiac_audit_session_manifest*.json")):
        if other.name.endswith(".schema.json") or other.resolve() == own:
            continue
        try:
            data = json.loads(other.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or not is_v2(data) or not isinstance(data.get("identity_provenance"), dict):
            continue
        for role, entry in data["identity_provenance"].items():
            if not isinstance(entry, dict):
                continue
            uuid = entry.get("session_uuid")
            if isinstance(uuid, str) and normalize_identity(uuid) in mine_uuid:
                mine = mine_uuid[normalize_identity(uuid)]
                errors.append(f"session uuid of the {mine} role repeats {other.name} ({role})")
            ref = split_ref(entry.get("host_receipt_ref"))
            if ref is not None and ref[0].casefold() in mine_receipt:
                errors.append(f"host receipt {ref[0]} is already used by {other.name} ({role})")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="IIAC manifest JSON")
    parser.add_argument(
        "--strict-product-diff",
        action="store_true",
        help="Fail when product paths changed between audited and audit execution SHAs (v1; no-op for v2)",
    )
    parser.add_argument(
        "--allow-product-diff",
        action="store_true",
        help="v2 only: explicit opt-out of the default failure on a changed write surface",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Git repository for product-path diffs (defaults to the Intergrax checkout)",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="v2 only: directory scanned for other v2 manifests (default: the qualification root of the repository)",
    )
    args = parser.parse_args()

    errors: list[str] = []
    if not args.manifest.is_file():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest_dict(manifest, errors)
    v2 = is_v2(manifest)

    audited = str(manifest.get("audited_implementation_sha") or "").strip()
    audit_exec = str(manifest.get("audit_execution_sha") or "").strip()
    repo = args.repo or REPO
    if audited and audit_exec and audited != audit_exec:
        try:
            if v2:
                changed = paths_changed_between(
                    audited, audit_exec, PRODUCT_PATH_PREFIXES + GOVERNANCE_WRITE_SURFACE, repo=args.repo
                )
            else:
                changed = product_paths_changed_between(audited, audit_exec, repo=args.repo)
            if changed:
                msg = f"auditor product write-surface detected ({len(changed)} paths) (N-03): {changed[0]}"
                if v2 and not args.allow_product_diff:
                    errors.append(msg)
                elif not v2 and args.strict_product_diff:
                    errors.append(msg)
                else:
                    print(f"WARN: {msg}", file=sys.stderr)
        except RuntimeError as exc:
            errors.append(str(exc))

    if is_v2_final(manifest):
        errors.extend(receipt_errors(manifest, repo, audit_exec))
        root = args.corpus_root if args.corpus_root is not None else repo / DEFAULT_CORPUS_ROOT
        errors.extend(corpus_errors(manifest, args.manifest, root))

    if errors:
        print("verify_iiac_role_separation: FAIL", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("verify_iiac_role_separation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
