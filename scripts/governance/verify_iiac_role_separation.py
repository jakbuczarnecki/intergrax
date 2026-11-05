#!/usr/bin/env python3
"""Verify IIAC role separation (session inequality + auditor product write-surface)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.governance.iiac_common import (
    product_paths_changed_between,
    validate_manifest_dict,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="IIAC manifest JSON")
    parser.add_argument(
        "--strict-product-diff",
        action="store_true",
        help="Fail when product paths changed between audited and audit execution SHAs",
    )
    args = parser.parse_args()

    errors: list[str] = []
    if not args.manifest.is_file():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest_dict(manifest, errors)

    audited = str(manifest.get("audited_implementation_sha") or "").strip()
    audit_exec = str(manifest.get("audit_execution_sha") or "").strip()
    if audited and audit_exec and audited != audit_exec:
        try:
            changed = product_paths_changed_between(audited, audit_exec)
            if changed:
                msg = f"auditor product write-surface detected ({len(changed)} paths) (N-03): {changed[0]}"
                if args.strict_product_diff:
                    errors.append(msg)
                else:
                    print(f"WARN: {msg}", file=sys.stderr)
        except RuntimeError as exc:
            errors.append(str(exc))

    if errors:
        print("verify_iiac_role_separation: FAIL", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("verify_iiac_role_separation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
