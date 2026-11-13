# © Artur Czarnecki. All rights reserved.

"""Child-process entry for the PDC-D7-R012 two-process fencing proof.

Each child is an independent OS process running the REAL ``LongRunningScheduler``
against a shared on-disk SQLite ledger. Every fire is appended to a shared log with a
single ``O_APPEND`` write so the parent can count fires per occurrence.

Everything is bounded: the go-barrier, the tick loop and the whole process all carry
explicit deadlines, so a child can never hang the suite.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--sync-dir", required=True)
    parser.add_argument("--fire-log", required=True)
    parser.add_argument("--now", required=True)
    parser.add_argument("--mode", choices=("tick", "claim_and_die"), default="tick")
    parser.add_argument("--deadline-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sys.path.insert(0, args.repo_root)
    deadline = time.monotonic() + args.deadline_seconds

    from intergrax.runtime.long_running.recurrence import RecurrenceMaterializer
    from intergrax.runtime.long_running.scheduler import LongRunningScheduler
    from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
    from tests.qualification.pdc_x.d7._support import ok_task_result

    now = datetime.fromisoformat(args.now)
    sync_dir = Path(args.sync_dir)
    store = SQLiteTaskCheckpointStore(db_path=Path(args.db))

    # Barrier: announce readiness, then wait for the parent's go signal.
    (sync_dir / f"ready.{args.owner}").write_text("ready", encoding="utf-8")
    go_file = sync_dir / "go"
    while not go_file.exists():
        if time.monotonic() > deadline:
            print("barrier timeout", file=sys.stderr)
            return 4
        time.sleep(0.005)

    if args.mode == "claim_and_die":
        claims = store.claim_due(
            before_utc_iso=now.isoformat(),
            owner_id=args.owner,
            lease_seconds=0,
        )
        print(f"claimed={len(claims)}", flush=True)
        os._exit(3)  # crash after claiming, before completing or firing

    class _FireLogExecutor:
        async def resume_task(self, task, *, checkpoint):
            line = f"{task.metadata.get('schedule_id')}\t{args.owner}\n".encode()
            fd = os.open(args.fire_log, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
            return ok_task_result(checkpoint.task_id)

    scheduler = LongRunningScheduler(
        store,
        _FireLogExecutor(),
        schedule_store=store,
        ledger=store,
        owner_id=args.owner,
        recurrence=RecurrenceMaterializer(store),
    )

    async def run() -> int:
        fired = 0
        idle_rounds = 0
        while idle_rounds < 20:
            if time.monotonic() > deadline:
                print("tick deadline", file=sys.stderr)
                return 5
            processed = await scheduler.tick(now=now)
            fired += processed
            idle_rounds = idle_rounds + 1 if processed == 0 else 0
            await asyncio.sleep(0.01)
        print(f"fired={fired}", flush=True)
        return 0

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
