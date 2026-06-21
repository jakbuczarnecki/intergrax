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
from intergrax.runtime.evidence.live_core_probe_contracts import LiveCoreProbeStatus
from intergrax.runtime.evidence.live_core_probe_export import (
    DEFAULT_LIVE_CORE_PROBE_OUTPUT_DIR,
    format_live_core_probe_cli,
    write_live_core_probe_report,
)
from intergrax.runtime.evidence.eval_evidence_contracts import EvalEvidenceStatus
from intergrax.runtime.evidence.eval_evidence_export import (
    DEFAULT_EVAL_EVIDENCE_OUTPUT_DIR,
    format_eval_evidence_cli,
    write_eval_evidence_report,
)
from intergrax.runtime.evidence.eval_evidence_runner import run_eval_evidence_checks
from intergrax.runtime.evidence.live_core_probe_runner import (
    LIVE_CORE_PROBE_OPERATOR_NOTE,
    run_live_core_probes,
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
        "--live-core-report",
        type=Path,
        default=None,
        help="Path to live core probe report.json "
        "(default: build/evidence/live_core_probes/live_core_report.json).",
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

    live_core = evidence_sub.add_parser(
        "live-core",
        help="Run selected live Tier-0 probes and write live core probe report.",
    )
    live_core.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root for default output directory.",
    )
    live_core.add_argument(
        "--root-label",
        default="local",
        help="Label for deterministic report id/run id (default: local).",
    )
    live_core.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Live core probe output directory "
            "(default: build/evidence/live_core_probes)."
        ),
    )
    live_core.add_argument(
        "--no-write",
        action="store_true",
        help="Render report to stdout only; do not write files.",
    )

    eval_cmd = evidence_sub.add_parser(
        "eval",
        help="Run eval regression evidence checks and write eval report.",
    )
    eval_cmd.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root for default output dir and source check path.",
    )
    eval_cmd.add_argument(
        "--root-label",
        default="local",
        help="Label for deterministic report id/run id (default: local).",
    )
    eval_cmd.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Eval evidence output directory (default: build/evidence/eval).",
    )
    eval_cmd.add_argument(
        "--no-write",
        action="store_true",
        help="Render report to stdout only; do not write files.",
    )

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
            live_core_probe_report_path=getattr(args, "live_core_report", None),
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


def _resolve_live_core_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (args.root.resolve() / DEFAULT_LIVE_CORE_PROBE_OUTPUT_DIR).resolve()


def _live_core_exit_code(status: LiveCoreProbeStatus) -> int:
    if status is LiveCoreProbeStatus.PASSED:
        return 0
    return 1


def _eval_exit_code(status: EvalEvidenceStatus) -> int:
    if status is EvalEvidenceStatus.PASSED:
        return 0
    return 1


def _resolve_eval_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (args.root.resolve() / DEFAULT_EVAL_EVIDENCE_OUTPUT_DIR).resolve()


def run_evidence_eval(args: argparse.Namespace) -> int:
    report = run_eval_evidence_checks(root=args.root, root_label=args.root_label)
    print(format_eval_evidence_cli(report))

    if args.no_write:
        return _eval_exit_code(report.status)

    output_dir = _resolve_eval_output_dir(args)
    json_path, md_path = write_eval_evidence_report(report, output_dir)
    print(f"eval evidence report: {json_path}")
    print(f"eval evidence report: {md_path}")
    return _eval_exit_code(report.status)


def run_evidence_live_core(args: argparse.Namespace) -> int:
    report = run_live_core_probes(root_label=args.root_label)
    print(format_live_core_probe_cli(report))

    if args.no_write:
        return _live_core_exit_code(report.status)

    output_dir = _resolve_live_core_output_dir(args)
    json_path, md_path = write_live_core_probe_report(report, output_dir)
    print(f"note: {LIVE_CORE_PROBE_OPERATOR_NOTE}")
    print(f"live core report: {json_path}")
    print(f"live core report: {md_path}")
    return _live_core_exit_code(report.status)


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
    if args.evidence_command == "live-core":
        return run_evidence_live_core(args)
    if args.evidence_command == "eval":
        return run_evidence_eval(args)
    if args.evidence_command != "posture":
        return 2
    if args.posture_command == "export":
        return run_evidence_posture_export(args)
    if args.posture_command is None:
        return run_evidence_posture(args)
    return 2
