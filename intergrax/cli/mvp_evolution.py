# © Artur Czarnecki. All rights reserved.

"""MVP evolution CLI — simulator and trace replay (MVP-EVOL.2, MVP-EVOL.3)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from intergrax.runtime.replay.persisted_trace_event_store import PersistedRunTraceEventStore
from intergrax.runtime.replay.trace_replay_bridge import serialized_trace_events_to_replay_dtos


def register_parser(sub: argparse._SubParsersAction) -> None:
    mvp = sub.add_parser("mvp", help="MVP evolution tools (simulate, replay).")
    mvp_sub = mvp.add_subparsers(dest="mvp_command", required=True)

    sim = mvp_sub.add_parser("simulate", help="Run orchestration CFG harness simulation.")
    sim.add_argument(
        "-k",
        "--pytest-expression",
        default="test_orchestration_cfg_simulation",
        help="Pytest -k expression for simulation tests.",
    )

    replay = mvp_sub.add_parser("replay", help="Reconstruct a run from persisted trace store.")
    replay.add_argument("--tenant-id", required=True)
    replay.add_argument("--run-id", required=True)
    replay.add_argument("--trace-db", type=Path, help="SQLite trace DB path (optional).")


def run_mvp_simulate(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/integration/runtime/",
        "-k",
        args.pytest_expression,
        "-q",
    ]
    return subprocess.call(cmd)


def run_mvp_replay(args: argparse.Namespace) -> int:
    if args.trace_db is None:
        print("trace replay requires --trace-db")
        return 1
    from intergrax.runtime.nexus.tracing.sqlite_trace_store import SQLiteRunTraceStore

    reader = SQLiteRunTraceStore(str(args.trace_db))
    trace_store = PersistedRunTraceEventStore(reader)
    events = list(trace_store.get_events(args.tenant_id, args.run_id))
    persisted = reader.read_run(args.run_id, args.tenant_id)
    replay_events = serialized_trace_events_to_replay_dtos(persisted.events)
    print(
        {
            "tenant_id": args.tenant_id,
            "run_id": args.run_id,
            "trace_event_count": len(events),
            "replay_dto_count": len(replay_events),
        }
    )
    return 0


def run_mvp_command(args: argparse.Namespace) -> int:
    if args.mvp_command == "simulate":
        return run_mvp_simulate(args)
    if args.mvp_command == "replay":
        return run_mvp_replay(args)
    return 2
