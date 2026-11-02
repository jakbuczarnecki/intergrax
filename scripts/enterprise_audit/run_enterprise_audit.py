#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CLI entrypoint for ENT-AUDIT-X enterprise audit runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import AuditMode
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_mode(raw: str) -> AuditMode:
    normalized = raw.upper().replace("_", "-")
    return AuditMode(normalized)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Intergrax enterprise audit (ENT-AUDIT-X)")
    parser.add_argument(
        "--mode",
        default="FAST",
        choices=[m.value for m in AuditMode],
        help="Audit mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / ".tmp" / "proof" / "ent-audit-x" / "cli-run",
        help="Report output directory",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan checks without executing gates")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--check-id",
        action="append",
        dest="check_ids",
        help="Restrict to specific check id (repeatable)",
    )
    parser.add_argument(
        "--since-sha",
        help="For CHANGED-SINCE: baseline git SHA (uses all EVOL-mapped checks when set)",
    )
    args = parser.parse_args(argv)
    mode = _parse_mode(args.mode)
    changed: frozenset[str] | None = None
    if mode is AuditMode.CHANGED_SINCE:
        changed = frozenset(
            cid
            for cid in (
                "ENT-A-EVOL-01",
                "ENT-A-EVOL-02",
                "ENT-A-EVOL-03",
                "ENT-A-INV-01",
                "ENT-A-INV-02",
                "ENT-A-INV-03",
                "ENT-A-REG-02",
            )
        )
    check_filter = frozenset(args.check_ids) if args.check_ids else None
    runner = EnterpriseAuditRunner(
        _repo_root(),
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
    _, _, exit_code = runner.run(
        mode,
        output_dir=args.output_dir,
        changed_check_ids=changed,
        check_ids_filter=check_filter,
    )
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
