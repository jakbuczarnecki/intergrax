# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""``intergrax evidence`` — harness evidence posture CLI (HEP Band 2ae · EVID-POSTURE-03/04)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from intergrax.runtime.evidence.evidence_posture_collector import (
    collect_evidence_posture,
    resolve_core_report_path,
    resolve_trace_timeline_path,
)
from intergrax.runtime.evidence.evidence_posture_export import (
    DEFAULT_POSTURE_OUTPUT_DIR,
    POSTURE_OPERATOR_NOTE,
    format_evidence_posture_cli,
    write_evidence_posture,
)

_POSTURE_ERROR_HINT = (
    "Run first: uv run intergrax certify core --level L2\n"
    "Then: uv run intergrax trace export"
)


def _add_posture_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root when resolving default artifact paths.",
    )
    parser.add_argument(
        "--core-report",
        type=Path,
        default=None,
        help="Path to core certification report.json "
        "(default: build/evidence/core_certification/report.json).",
    )
    parser.add_argument(
        "--trace-timeline",
        type=Path,
        default=None,
        help="Path to trace timeline.json (default: build/evidence/trace/timeline.json).",
    )
    parser.add_argument(
        "--root-label",
        default="local",
        help="Label for evidence posture ID generation (default: local).",
    )
    parser.add_argument(
        "--no-operational-unknowns",
        action="store_true",
        help="Do not include REPO_HEALTH/PYTEST_GATE as UNKNOWN signals.",
    )


def register_parser(sub: argparse._SubParsersAction) -> None:
    evidence = sub.add_parser("evidence", help="Harness evidence posture.")
    evidence_sub = evidence.add_subparsers(dest="evidence_command", required=True)

    posture = evidence_sub.add_parser(
        "posture",
        help="Render read-only evidence posture from existing artifacts.",
    )
    _add_posture_common_args(posture)
    posture_sub = posture.add_subparsers(dest="posture_command")

    export = posture_sub.add_parser(
        "export",
        help="Write posture.json and posture.md under build/evidence/posture/.",
    )
    _add_posture_common_args(export)
    export.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Posture artifact output directory (default: build/evidence/posture).",
    )


def _collect_from_args(args: argparse.Namespace):
    try:
        return collect_evidence_posture(
            root=args.root,
            core_report_path=args.core_report,
            trace_timeline_path=args.trace_timeline,
            include_unknown_operational_signals=not args.no_operational_unknowns,
            root_label=args.root_label,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(_POSTURE_ERROR_HINT, file=sys.stderr)
        raise SystemExit(1) from None


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "output_dir", None) is not None:
        return args.output_dir.resolve()
    return (args.root.resolve() / DEFAULT_POSTURE_OUTPUT_DIR).resolve()


def run_evidence_posture(args: argparse.Namespace) -> int:
    summary = _collect_from_args(args)
    print(format_evidence_posture_cli(summary))
    return 0


def run_evidence_posture_export(args: argparse.Namespace) -> int:
    summary = _collect_from_args(args)
    output_dir = _resolve_output_dir(args)
    core_path = resolve_core_report_path(
        root=args.root,
        core_report_path=args.core_report,
    )
    timeline_path = resolve_trace_timeline_path(
        root=args.root,
        trace_timeline_path=args.trace_timeline,
    )
    json_path, md_path = write_evidence_posture(summary, output_dir)
    print(f"core report: {core_path}")
    print(f"trace timeline: {timeline_path}")
    print(f"note: {POSTURE_OPERATOR_NOTE}")
    print(f"posture: {json_path}")
    print(f"posture: {md_path}")
    return 0


def run_evidence(args: argparse.Namespace) -> int:
    if args.evidence_command != "posture":
        return 2
    if args.posture_command == "export":
        return run_evidence_posture_export(args)
    if args.posture_command is None:
        return run_evidence_posture(args)
    return 2
