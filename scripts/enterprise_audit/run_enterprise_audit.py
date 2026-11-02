#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CLI entrypoint for ENT-AUDIT-X enterprise audit runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from intergrax.platform.enterprise_audit.changed_since import select_checks_for_changed_since
from intergrax.platform.enterprise_audit.contracts import AuditMode
from intergrax.platform.enterprise_audit.manifest import load_default_manifest
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
    parser.add_argument(
        "--check-id",
        action="append",
        dest="check_ids",
        help="Restrict to specific check id (repeatable)",
    )
    parser.add_argument(
        "--since-sha",
        help="For CHANGED-SINCE: baseline git SHA (required)",
    )
    parser.add_argument(
        "--bootstrap-baseline",
        action="store_true",
        help="FREEZE: write candidate ENTERPRISE_AUDIT_BASELINE.json after clean run",
    )
    parser.add_argument(
        "--evidence-bundle",
        action="store_true",
        help="Write evidence_manifest.v1.json under output-dir",
    )
    args = parser.parse_args(argv)
    mode = _parse_mode(args.mode)
    root = _repo_root()
    checks = load_default_manifest()
    changed: frozenset[str] | None = None
    if mode is AuditMode.CHANGED_SINCE:
        if not args.since_sha:
            print("CHANGED-SINCE requires --since-sha", file=sys.stderr)
            return int(EnterpriseAuditExitCode_from_internal())
        try:
            changed = select_checks_for_changed_since(checks, repo_root=root, since_sha=args.since_sha)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return int(EnterpriseAuditExitCode_from_internal())
    check_filter = frozenset(args.check_ids) if args.check_ids else None
    runner = EnterpriseAuditRunner(
        root,
        dry_run=args.dry_run,
        require_clean_repo=mode is AuditMode.FREEZE,
    )
    _, _, exit_code = runner.run(
        mode,
        output_dir=args.output_dir,
        changed_check_ids=changed,
        check_ids_filter=check_filter,
        bootstrap_baseline=args.bootstrap_baseline,
        write_evidence_bundle=args.evidence_bundle,
    )
    return int(exit_code)


def EnterpriseAuditExitCode_from_internal() -> int:
    from intergrax.platform.enterprise_audit.contracts import EnterpriseAuditExitCode

    return EnterpriseAuditExitCode.INTERNAL


if __name__ == "__main__":
    sys.exit(main())
