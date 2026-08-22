# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Intergrax debug CLI (Phase D.1–D.3, architecture §19, §35).

Usage::

    python -m intergrax.debug tasks list [--tenant T] [--limit N] [--db PATH]
    python -m intergrax.debug tasks show RUN_ID [--tenant T] [--db PATH]
    python -m intergrax.debug tasks trace RUN_ID [--tenant T] [--format text|json] [--runtime] [--db PATH]
    python -m intergrax.debug experiments register --hypothesis "..." --capability echo.basic
    python -m intergrax.debug experiments list [--decision pending]
    python -m intergrax.debug experiments show EXPERIMENT_ID
    python -m intergrax.debug experiments decide EXPERIMENT_ID --decision keep
    python -m intergrax.debug experiments link-run EXPERIMENT_ID RUN_ID

Environment:

    INTERGRAX_TRACE_DB — trace SQLite database (default: build/intergrax_trace.db)
    INTERGRAX_EXPERIMENTS_DB — experiment registry database (default: build/intergrax_experiments.db)
"""

from __future__ import annotations
from intergrax.utils import attribute_access

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
from intergrax.experiments.composition import resolve_experiment_persistence
from intergrax.experiments.formatters import format_experiment_list, format_experiment_show
from intergrax.experiments.models import ExperimentDecision, RegisterExperimentRequest
from intergrax.experiments.store import (
    ENV_EXPERIMENTS_DB,
    resolve_experiments_db_path,
)


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


def _open_experiments(args: argparse.Namespace):
    return resolve_experiment_persistence(
        experiments_db=resolve_experiments_db_path(args.experiments_db)
    )


def _cmd_experiments_register(args: argparse.Namespace) -> int:
    store = _open_experiments(args)
    record = store.register(
        RegisterExperimentRequest(
            hypothesis=args.hypothesis,
            capability=args.capability,
            agent_id=args.agent_id,
            expected_output=args.expected_output or "",
            validation_criteria=args.validation_criteria or "",
            notes=args.notes or "",
        )
    )
    print(record.experiment_id)
    return 0


def _cmd_experiments_list(args: argparse.Namespace) -> int:
    store = _open_experiments(args)
    decision = ExperimentDecision(args.decision) if args.decision else None
    records = store.list_experiments(limit=args.limit, decision=decision)
    print(format_experiment_list(records))
    return 0


def _cmd_experiments_show(args: argparse.Namespace) -> int:
    store = _open_experiments(args)
    record = store.get(args.experiment_id)
    if args.format == "json":
        print(json.dumps(record.model_dump(mode="json"), indent=2))
    else:
        print(format_experiment_show(record))
    return 0


def _cmd_experiments_decide(args: argparse.Namespace) -> int:
    store = _open_experiments(args)
    decision = ExperimentDecision(args.decision)
    record = store.set_decision(
        args.experiment_id,
        decision,
        notes=args.notes,
    )
    if decision == ExperimentDecision.DELETE:
        print(f"deleted {args.experiment_id}")
    elif args.format == "json":
        print(json.dumps(record.model_dump(mode="json"), indent=2))
    else:
        print(format_experiment_show(record))
    return 0


def _cmd_experiments_link_run(args: argparse.Namespace) -> int:
    store = _open_experiments(args)
    record = store.link_run(args.experiment_id, args.run_id)
    if args.format == "json":
        print(json.dumps(record.model_dump(mode="json"), indent=2))
    else:
        print(format_experiment_show(record))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="intergrax.debug",
        description="Inspect Nexus runs/traces and manage experiment registry.",
    )
    parser.add_argument(
        "--db",
        dest="db",
        default=None,
        help=f"SQLite trace DB path (default: ${ENV_TRACE_DB} or build/intergrax_trace.db)",
    )
    parser.add_argument(
        "--experiments-db",
        dest="experiments_db",
        default=None,
        help=(
            f"Experiment registry DB path "
            f"(default: ${ENV_EXPERIMENTS_DB} or build/intergrax_experiments.db)"
        ),
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

    experiments = sub.add_parser("experiments", help="Experiment registry (§35)")
    exp_sub = experiments.add_subparsers(dest="experiments_command", required=True)

    register_parser = exp_sub.add_parser("register", help="Register a new experiment")
    register_parser.add_argument("--hypothesis", required=True)
    register_parser.add_argument("--capability", required=True)
    register_parser.add_argument("--agent-id", dest="agent_id", default=None)
    register_parser.add_argument("--expected-output", dest="expected_output", default="")
    register_parser.add_argument("--validation-criteria", dest="validation_criteria", default="")
    register_parser.add_argument("--notes", default="")
    register_parser.set_defaults(handler=_cmd_experiments_register)

    exp_list = exp_sub.add_parser("list", help="List experiments")
    exp_list.add_argument("--limit", type=int, default=20)
    exp_list.add_argument(
        "--decision",
        choices=[d.value for d in ExperimentDecision],
        default=None,
    )
    exp_list.set_defaults(handler=_cmd_experiments_list)

    exp_show = exp_sub.add_parser("show", help="Show experiment details")
    exp_show.add_argument("experiment_id")
    exp_show.add_argument("--format", choices=("text", "json"), default="text")
    exp_show.set_defaults(handler=_cmd_experiments_show)

    decide_parser = exp_sub.add_parser("decide", help="Set keep/improve/pause/delete verdict")
    decide_parser.add_argument("experiment_id")
    decide_parser.add_argument(
        "--decision",
        required=True,
        choices=[d.value for d in ExperimentDecision if d != ExperimentDecision.PENDING],
    )
    decide_parser.add_argument("--notes", default=None)
    decide_parser.add_argument("--format", choices=("text", "json"), default="text")
    decide_parser.set_defaults(handler=_cmd_experiments_decide)

    link_parser = exp_sub.add_parser("link-run", help="Attach a Nexus run_id to an experiment")
    link_parser.add_argument("experiment_id")
    link_parser.add_argument("run_id")
    link_parser.add_argument("--format", choices=("text", "json"), default="text")
    link_parser.set_defaults(handler=_cmd_experiments_link_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = attribute_access.optional(args, "handler", None)
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
