# © Artur Czarnecki. All rights reserved.

"""Child process for F-D7-02: a scheduler that dies (``os._exit``) while an occurrence runs."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--now", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.repo_root)

    from intergrax.runtime.long_running.recurrence import RecurrenceMaterializer
    from intergrax.runtime.long_running.scheduler import LongRunningScheduler
    from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore

    store = SQLiteTaskCheckpointStore(db_path=Path(args.db))

    class _DyingExecutor:
        async def resume_task(self, task, *, checkpoint):
            os._exit(3)  # killed while the occurrence is RUNNING

    scheduler = LongRunningScheduler(
        store,
        _DyingExecutor(),
        schedule_store=store,
        ledger=store,  # type: ignore[arg-type]
        owner_id="proc-dying",
        recurrence=RecurrenceMaterializer(store),
    )
    asyncio.run(scheduler.tick(now=datetime.fromisoformat(args.now)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
