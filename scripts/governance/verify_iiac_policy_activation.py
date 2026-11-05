#!/usr/bin/env python3
"""Verify IIAC policy activation record (fail-closed for premature IN_FORCE)."""

from __future__ import annotations

import argparse
import sys

from scripts.governance.iiac_common import (
    ACTIVATION_RECORD,
    git_merge_base_is_ancestor,
    git_rev_parse,
    load_activation_record,
    policy_status,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expect-not-in-force",
        action="store_true",
        help="Exit 0 when status is NOT_IN_FORCE (Session A default)",
    )
    args = parser.parse_args()

    if not ACTIVATION_RECORD.is_file():
        print("IIAC policy activation: NOT_IN_FORCE (missing activation record)")
        return 0 if args.expect_not_in_force else 1

    rec = load_activation_record()
    status = policy_status()

    if status != "IN_FORCE":
        print(f"IIAC policy activation: {status}")
        if args.expect_not_in_force or status in {"NOT_IN_FORCE", "DRAFT", "PENDING"}:
            return 0
        return 1

    errors: list[str] = []
    session_b_digest = rec.get("session_b_governance_audit_report_digest")
    if not session_b_digest or not str(session_b_digest).strip():
        errors.append("IN_FORCE without session_b_governance_audit_report_digest (N-08)")

    effective_sha = rec.get("effective_from_sha")
    if not effective_sha or len(str(effective_sha)) != 40:
        errors.append("IN_FORCE requires effective_from_sha (40-char git SHA)")
    else:
        try:
            head = git_rev_parse("HEAD")
            if not git_merge_base_is_ancestor(str(effective_sha), head):
                errors.append("effective_from_sha is not ancestor of HEAD")
        except RuntimeError as exc:
            errors.append(str(exc))

    gov_manifest = rec.get("session_b_manifest_path")
    if not gov_manifest:
        errors.append("IN_FORCE requires session_b_manifest_path")

    policy_version = rec.get("policy_version")
    if not policy_version:
        errors.append("IN_FORCE requires policy_version")

    if errors:
        print("IIAC policy activation: FAIL", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"IIAC policy activation: IN_FORCE @ {policy_version} from {effective_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
