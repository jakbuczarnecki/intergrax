# © Artur Czarnecki. All rights reserved.

"""Self-bounded OS process that holds the hosting instance lease (PDC-D4-R006).

Run as ``python _lock_holder_process.py <run_directory> <application_id> <max_seconds>``.
It acquires the real file instance lease, prints ``acquired`` and holds it until it is
terminated or ``max_seconds`` elapse.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from intergrax.hosting import HostedApplicationProcessIdentity, InstancePolicy
from intergrax.hosting.instance.contracts import HostedApplicationInstanceIdentity
from intergrax.hosting.instance.file_guard import FileHostedApplicationInstanceGuard


class _Clock:
    def now(self) -> datetime:
        return datetime.now(UTC)


async def _acquire(run_directory: Path, application_id: str):
    clock = _Clock()
    process_identity = HostedApplicationProcessIdentity(
        process_id=os.getpid(), started_at=clock.now(), host_id="pdc-d4-holder"
    )
    guard = FileHostedApplicationInstanceGuard(
        run_directory=run_directory,
        instance_policy=InstancePolicy(),
        process_identity=process_identity,
        clock=clock,
    )
    identity = HostedApplicationInstanceIdentity(
        application_id=application_id,
        instance_id="holder-001",
        profile_digest="sha256:" + "b" * 64,
        process_identity=process_identity,
    )
    return await guard.acquire(identity)


def main() -> int:
    run_directory = Path(sys.argv[1])
    application_id = sys.argv[2]
    max_seconds = float(sys.argv[3])
    acquisition = asyncio.run(_acquire(run_directory, application_id))
    print("acquired", flush=True)
    deadline = time.monotonic() + max_seconds
    while time.monotonic() < deadline:
        time.sleep(0.05)
    asyncio.run(acquisition.lease.release())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
