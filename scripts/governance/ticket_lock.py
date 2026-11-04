#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Advisory ticket lock files under .tmp/session/ (no PostgreSQL)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_LOCK_ROOT = _REPO / ".tmp" / "session" / "ticket-locks"


def _lock_path(ticket_id: str) -> Path:
    safe = ticket_id.replace("/", "_").replace("\\", "_")
    return _LOCK_ROOT / f"{safe}.json"


def cmd_acquire(args: argparse.Namespace) -> int:
    _LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    path = _lock_path(args.ticket_id)
    if path.is_file() and not args.force:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"[ERROR] lock held by {data.get('holder')} since {data.get('since')}", file=sys.stderr)
        return 1
    payload = {"ticket_id": args.ticket_id, "holder": args.holder, "since": time.time()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[INFO] advisory lock acquired: {path.relative_to(_REPO)}")
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    path = _lock_path(args.ticket_id)
    if path.is_file():
        path.unlink()
    print("[INFO] advisory lock released")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    path = _lock_path(args.ticket_id)
    if not path.is_file():
        print("[INFO] no lock")
        return 0
    print(path.read_text(encoding="utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Advisory ticket locks (.tmp/session only).")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("acquire", cmd_acquire), ("release", cmd_release), ("status", cmd_status)):
        p = sub.add_parser(name)
        p.add_argument("ticket_id", help="e.g. T-IGX-001-A")
        p.add_argument("--holder", default="session", help="acquire: holder id")
        p.add_argument("--force", action="store_true", help="acquire: overwrite")
        p.set_defaults(func=func)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
