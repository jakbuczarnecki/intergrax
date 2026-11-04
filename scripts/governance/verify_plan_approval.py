#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Verify Plan digest / Approved fields in a ticket file."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from scripts.governance.plan_digest import digest_file

_REPO = Path(__file__).resolve().parents[2]
_DIGEST_RE = re.compile(r"^Plan digest:\s*([0-9a-f]{64})\s*$", re.MULTILINE | re.IGNORECASE)
_APPROVED_RE = re.compile(r"^Approved:\s*.+$", re.MULTILINE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify ticket plan approval fields.")
    parser.add_argument("ticket", type=Path, help="Ticket markdown under governance/tickets/")
    parser.add_argument("--plan", type=Path, required=True, help="Plan file to compare digest against")
    args = parser.parse_args(argv)

    ticket_path = args.ticket if args.ticket.is_absolute() else _REPO / args.ticket
    plan_path = args.plan if args.plan.is_absolute() else _REPO / args.plan
    if not ticket_path.is_file():
        print(f"[ERROR] ticket not found: {ticket_path}", file=sys.stderr)
        return 1
    if not plan_path.is_file():
        print(f"[ERROR] plan not found: {plan_path}", file=sys.stderr)
        return 1

    body = ticket_path.read_text(encoding="utf-8")
    match = _DIGEST_RE.search(body)
    if not match:
        print("[ERROR] ticket missing Plan digest: line", file=sys.stderr)
        return 1
    recorded = match.group(1).lower()
    current = digest_file(plan_path)
    if recorded != current:
        print("[ERROR] plan digest stale — re-run plan_digest.py and update ticket", file=sys.stderr)
        print(f"  recorded={recorded}", file=sys.stderr)
        print(f"  current ={current}", file=sys.stderr)
        return 1
    if not _APPROVED_RE.search(body):
        print("[ERROR] ticket missing Approved: line", file=sys.stderr)
        return 1
    print("[INFO] verify_plan_approval: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
