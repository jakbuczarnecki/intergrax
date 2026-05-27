# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Intergrax debug CLI (Phase D.1, architecture §19).

Usage::

    python -m intergrax.debug tasks list [--tenant T] [--limit N] [--db PATH]
    python -m intergrax.debug tasks show RUN_ID [--tenant T] [--db PATH]
    python -m intergrax.debug tasks trace RUN_ID [--tenant T] [--format text|json] [--runtime] [--db PATH]

Environment:

    INTERGRAX_TRACE_DB — path to SQLite trace database (default: build/intergrax_trace.db)
"""

from __future__ import annotations

import argparse
import json
import sys

from intergrax.debug.formatters import (
    build_trace_payload,
    format_run_list,
    format_run_show,
    format_trace_timeline,
)
from intergrax.debug.store import ENV_TRACE_DB, open_trace_reader, resolve_trace_db_path


def _cmd_tasks_list(args: argparse.Namespace) -> int:
    store = open_trace_reader(resolve_trace_db_path(args.db))
    runs = store.list_runs(args.tenant, limit=args.limit)
    print(format_run_list(runs))
    return 0


def _cmd_tasks_show(args: argparse.Namespace) -> int:
    store = open_trace_reader(resolve_trace_db_path(args.db))
    persisted = store.read_run(args.run_id, args.tenant)
    print(format_run_show(persisted))
    return 0


def _cmd_tasks_trace(args: argparse.Namespace) -> int:
    store = open_trace_reader(resolve_trace_db_path(args.db))
    persisted = store.read_run(args.run_id, args.tenant)
    if args.format == "json":
        payload = build_trace_payload(persisted, include_runtime=args.runtime)
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(format_trace_timeline(persisted))
        if args.runtime:
            payload = build_trace_payload(persisted, include_runtime=True)
            print("\n--- runtime events (from trace_bridge) ---\n")
            print(json.dumps(payload.get("runtime_events", []), indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="intergrax.debug",
        description="Inspect Nexus task runs and traces (Phase D.1).",
    )
    parser.add_argument(
        "--db",
        dest="db",
        default=None,
        help=f"SQLite trace DB path (default: ${ENV_TRACE_DB} or build/intergrax_trace.db)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    tasks = sub.add_parser("tasks", help="Task/run inspection commands")
    tasks_sub = tasks.add_subparsers(dest="tasks_command", required=True)

    list_parser = tasks_sub.add_parser("list", help="List recent finalized runs")
    list_parser.add_argument("--tenant", default="default", help="Tenant id filter")
    list_parser.add_argument("--limit", type=int, default=20, help="Max rows")
    list_parser.set_defaults(handler=_cmd_tasks_list)

    show_parser = tasks_sub.add_parser("show", help="Show run metadata")
    show_parser.add_argument("run_id", help="Run / task id")
    show_parser.add_argument("--tenant", default="default", help="Tenant id")
    show_parser.set_defaults(handler=_cmd_tasks_show)

    trace_parser = tasks_sub.add_parser("trace", help="Show trace event timeline")
    trace_parser.add_argument("run_id", help="Run / task id")
    trace_parser.add_argument("--tenant", default="default", help="Tenant id")
    trace_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format",
    )
    trace_parser.add_argument(
        "--runtime",
        action="store_true",
        help="Include RuntimeEvent view via trace_bridge (JSON or appendix)",
    )
    trace_parser.set_defaults(handler=_cmd_tasks_trace)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    try:
        return handler(args)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
