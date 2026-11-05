#!/usr/bin/env python3
"""Validate IIAC audit session manifest (schema + session rules)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema

from scripts.governance.iiac_common import (
    HANDOFF_TEMPLATE,
    MANIFEST_SCHEMA,
    REPO,
    git_merge_base_is_ancestor,
    validate_manifest_dict,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="Path to iiac_audit_session_manifest JSON")
    parser.add_argument(
        "--require-handoff-template",
        action="store_true",
        help="Fail if IIAC handoff template missing (N-02 organizational gate)",
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
                    if not git_merge_base_is_ancestor(audited.strip(), audit_exec.strip()):
                        errors.append("audited_implementation_sha not ancestor of audit_execution_sha (N-04)")
                except RuntimeError as exc:
                    errors.append(str(exc))

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
