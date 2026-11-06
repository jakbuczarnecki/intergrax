#!/usr/bin/env python3
"""Verify IIAC policy activation (fail-closed).

Modes:
  default and --require-in-force
      exit 0 only when status is IN_FORCE and every activation proof passes.
  --expect-not-in-force
      exit 0 only when the policy is inactive and states no active certification claim.

The canonical record is expected to stay NOT_IN_FORCE until a separate
certification session. This script does not edit that record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.governance.iiac_activation_gate import (
    CANONICAL_RECORD,
    evaluate_activation,
)
from scripts.governance.iiac_common import REPO


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--require-in-force",
        action="store_true",
        help="Exit 0 only when the policy is IN_FORCE and activation evidence is valid",
    )
    mode.add_argument(
        "--expect-not-in-force",
        action="store_true",
        help="Exit 0 only when the policy is inactive and makes no active certification claim",
    )
    parser.add_argument(
        "--activation-record",
        type=Path,
        default=CANONICAL_RECORD,
        help="Activation record JSON (defaults to the canonical policy record)",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=REPO,
        help="Git repository used for exact-SHA checks",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help="Fixture-only evidence root. Rejected for the canonical activation record",
    )
    args = parser.parse_args(argv)

    record_path = args.activation_record if args.activation_record.is_absolute() else args.repo / args.activation_record
    repo = args.repo.resolve()
    if args.evidence_root is not None and record_path.resolve() == CANONICAL_RECORD.resolve():
        print(
            "IIAC policy activation: FAIL\n  - evidence-root override is forbidden for the canonical record",
            file=sys.stderr,
        )
        return 1
    if not record_path.is_file():
        print("IIAC policy activation: FAIL\n  - activation record is missing", file=sys.stderr)
        return 1
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"IIAC policy activation: FAIL\n  - malformed activation record: {exc}", file=sys.stderr)
        return 1
    if not isinstance(record, dict):
        print("IIAC policy activation: FAIL\n  - activation record must be a JSON object", file=sys.stderr)
        return 1

    require = not args.expect_not_in_force
    errors = evaluate_activation(
        record,
        repo=repo,
        record_path=record_path,
        evidence_root=args.evidence_root,
        require_in_force=require,
    )
    if errors:
        print("IIAC policy activation: FAIL", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    status = str(record.get("status") or "").upper()
    if require:
        version = record.get("policy_version")
        print(f"IIAC policy activation: IN_FORCE @ {version} from {record.get('effective_from_sha')}")
    else:
        print(f"IIAC policy activation: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
