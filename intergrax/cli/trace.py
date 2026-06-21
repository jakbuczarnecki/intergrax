# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""``intergrax trace`` — harness evidence timeline CLI (HEP Band 2ae · EVID-TRACE-03/04)."""

from __future__ import annotations

import argparse
from pathlib import Path

from intergrax.runtime.evidence.certification_report import DEFAULT_CORE_CERTIFICATION_OUTPUT_DIR
from intergrax.runtime.evidence.trace_timeline_adapter import (
    build_timeline_from_certification_report,
    load_core_certification_report,
)
from intergrax.runtime.evidence.trace_timeline_export import (
    DEFAULT_TRACE_EVIDENCE_OUTPUT_DIR,
    TRACE_TIMELINE_OPERATOR_NOTE,
    format_trace_timeline_cli,
    write_trace_timeline,
)


def _default_report_path(root: Path) -> Path:
    return root.resolve() / DEFAULT_CORE_CERTIFICATION_OUTPUT_DIR / "report.json"


def _default_trace_output_dir(root: Path) -> Path:
    return root.resolve() / DEFAULT_TRACE_EVIDENCE_OUTPUT_DIR


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to core certification report.json "
        "(default: build/evidence/core_certification/report.json).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root when resolving default report/output paths.",
    )


def register_parser(sub: argparse._SubParsersAction) -> None:
    trace = sub.add_parser("trace", help="Harness evidence trace timeline.")
    trace_sub = trace.add_subparsers(dest="trace_command", required=True)

    show = trace_sub.add_parser(
        "show",
        help="Render report-derived evidence timeline to stdout (no artifact write).",
    )
    _add_common_args(show)

    export = trace_sub.add_parser(
        "export",
        help="Write report-derived timeline.json and timeline.md under build/evidence/trace/.",
    )
    _add_common_args(export)
    export.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Trace artifact output directory (default: build/evidence/trace).",
    )


def _resolve_report_path(args: argparse.Namespace) -> Path:
    if args.report is not None:
        return args.report.resolve()
    return _default_report_path(args.root)


def _resolve_trace_output_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "output_dir", None) is not None:
        return args.output_dir.resolve()
    return _default_trace_output_dir(args.root)


def _build_timeline_from_args(args: argparse.Namespace):
    report_path = _resolve_report_path(args)
    try:
        report = load_core_certification_report(report_path)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n"
            "Run certification first: uv run intergrax certify core --level L2"
        ) from None
    return build_timeline_from_certification_report(
        report,
        source_report_path=str(report_path),
    )


def run_trace_show(args: argparse.Namespace) -> int:
    timeline = _build_timeline_from_args(args)
    print(format_trace_timeline_cli(timeline))
    return 0


def run_trace_export(args: argparse.Namespace) -> int:
    timeline = _build_timeline_from_args(args)
    output_dir = _resolve_trace_output_dir(args)
    report_path = _resolve_report_path(args)
    json_path, md_path = write_trace_timeline(timeline, output_dir)
    print(f"report: {report_path}")
    print(f"note: {TRACE_TIMELINE_OPERATOR_NOTE}")
    print(f"timeline: {json_path}")
    print(f"timeline: {md_path}")
    return 0


def run_trace(args: argparse.Namespace) -> int:
    if args.trace_command == "show":
        return run_trace_show(args)
    if args.trace_command == "export":
        return run_trace_export(args)
    return 2
