# © Artur Czarnecki. All rights reserved.

"""Durable async task index resolver (AUDIT-IDEAL-28.1)."""

from __future__ import annotations

import os
from pathlib import Path

from intergrax.applications._shared.async_task_dispatch import InMemoryAsyncTaskIndex
from intergrax.applications._shared.async_task_index_protocol import AsyncTaskIndexProtocol
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def resolve_async_task_index(
    env: ApplicationEnvironmentProfile,
    *,
    db_path: Path | None = None,
) -> AsyncTaskIndexProtocol:
    """
    Return durable SQLite index for product/strict hosts; in-memory for lab.

    Override with ``INTERGRAX_DURABLE_QUEUE=memory`` to force in-process index.
    """
    override = os.getenv("INTERGRAX_DURABLE_QUEUE", "").strip().lower()
    if override in ("memory", "inmemory", "off", "false", "0"):
        return InMemoryAsyncTaskIndex()

    use_durable = (
        env.application_profile is ApplicationProfile.PRODUCT
        or env.execution_mode.value == "strict"
        or env.features.durable_async_index_default
    )
    if not use_durable:
        return InMemoryAsyncTaskIndex()

    from intergrax.applications._shared.sqlite_async_task_index import SqliteAsyncTaskIndex

    resolved_path = db_path or Path("build/async_task_index.db")
    return SqliteAsyncTaskIndex(resolved_path)
