#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""CLI entrypoint for ENT-AUDIT-X enterprise audit runner."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from intergrax.platform.enterprise_audit.baseline import (
    assert_baseline_dir_jailed,
    promote_accepted_baseline,
)
from intergrax.platform.enterprise_audit.changed_since import select_checks_for_changed_since
from intergrax.platform.enterprise_audit.contracts import AuditMode, EnterpriseAuditExitCode
from intergrax.platform.enterprise_audit.manifest import load_default_manifest
from intergrax.platform.enterprise_audit.report_summary import print_run_summary
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_mode(raw: str) -> AuditMode:
    normalized = raw.upper().replace("_", "-")
    return AuditMode(normalized)


def _cmd_run(args: argparse.Namespace) -> int:
    mode = _parse_mode(args.mode)
    root = _repo_root()
    checks = load_default_manifest()
    certifying_run = not args.non_certifying
    if args.check_ids and certifying_run:
        print("certifying runs require --non-certifying when using --check-id", file=sys.stderr)
        return int(EnterpriseAuditExitCode.VIOLATION)
    changed: frozenset[str] | None = None
    if mode is AuditMode.CHANGED_SINCE:
        if not args.since_sha:
            print("CHANGED-SINCE requires --since-sha", file=sys.stderr)
            return int(EnterpriseAuditExitCode.INTERNAL)
        try:
            changed = select_checks_for_changed_since(checks, repo_root=root, since_sha=args.since_sha)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return int(EnterpriseAuditExitCode.INTERNAL)
    check_filter = frozenset(args.check_ids) if args.check_ids else None
    bootstrap = args.bootstrap_candidate or args.bootstrap_baseline
    runner = EnterpriseAuditRunner(
        root,
        dry_run=args.dry_run,
        require_clean_repo=mode is AuditMode.FREEZE and certifying_run,
    )
    started = time.monotonic()
    report, manifest, exit_code = runner.run(
        mode,
        output_dir=args.output_dir,
        changed_check_ids=changed,
        check_ids_filter=check_filter,
        bootstrap_baseline=bootstrap,
        write_evidence_bundle=args.evidence_bundle,
        certifying_run=certifying_run,
    )
    duration_ms = int((time.monotonic() - started) * 1000)
    if args.print_summary:
        print_run_summary(
            report,
            manifest,
            output_dir=args.output_dir,
            duration_ms=duration_ms,
            certifying_run=certifying_run,
        )
    if args.open_report:
        md_path = args.output_dir / "enterprise_audit_report.md"
        if md_path.is_file() and sys.platform == "win32":
            import os

            os.startfile(md_path)  # noqa: S606
    return int(exit_code)


def _cmd_promote(args: argparse.Namespace) -> int:
    root = _repo_root()
    baseline_dir = assert_baseline_dir_jailed(args.baseline_dir or (root / "tests/qualification/ent_audit_x"), root)
    try:
        promote_accepted_baseline(
            baseline_dir,
            od_record_path=args.od_record,
            verify_closed_bundle_dir=args.evidence_dir,
            repo_root=root,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return int(EnterpriseAuditExitCode.VIOLATION)
    print(f"accepted baseline promoted under {baseline_dir}", file=sys.stdout)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Intergrax enterprise audit (ENT-AUDIT-X)")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Execute enterprise audit (default)")
    run_parser.set_defaults(command="run")
    run_parser.add_argument(
        "--mode",
        default="FAST",
        choices=[m.value for m in AuditMode],
        help="Audit mode",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / ".tmp" / "proof" / "ent-audit-x" / "cli-run",
        help="Report output directory",
    )
    run_parser.add_argument("--dry-run", action="store_true", help="Plan checks without executing gates")
    run_parser.add_argument(
        "--check-id",
        action="append",
        dest="check_ids",
        help="Restrict to specific check id (requires --non-certifying)",
    )
    run_parser.add_argument("--non-certifying", action="store_true", help="Diagnostic run (allows --check-id)")
    run_parser.add_argument(
        "--since-sha",
        help="For CHANGED-SINCE: baseline git SHA (required)",
    )
    run_parser.add_argument(
        "--bootstrap-candidate",
        action="store_true",
        help="FREEZE: write candidate baseline only (never accepted)",
    )
    run_parser.add_argument(
        "--bootstrap-baseline",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    run_parser.add_argument(
        "--evidence-bundle",
        action="store_true",
        help="Write evidence_manifest.v1.json under output-dir (certifying only)",
    )
    run_parser.add_argument("--print-summary", action="store_true", default=True)
    run_parser.add_argument("--no-print-summary", action="store_false", dest="print_summary")
    run_parser.add_argument("--open", dest="open_report", action="store_true", help="Open Markdown report locally")

    promote_parser = sub.add_parser("promote-baseline", help="Operator OD-ENT-01 baseline promotion")
    promote_parser.set_defaults(command="promote-baseline")
    promote_parser.add_argument("--od-record", type=Path, required=True, help="Operator OD-ENT-01 JSON record")
    promote_parser.add_argument("--baseline-dir", type=Path, help="Qualification baseline directory")
    promote_parser.add_argument(
        "--evidence-dir",
        type=Path,
        required=True,
        help="Closed evidence bundle directory (required for promotion)",
    )

    # Back-compat: no subcommand → run
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] not in {"run", "promote-baseline"}:
        argv = ["run", *argv]
    args = parser.parse_args(argv)
    if args.command == "promote-baseline":
        return _cmd_promote(args)
    return _cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
