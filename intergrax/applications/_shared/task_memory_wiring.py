# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""TaskMemory persistence wiring for Tier-3 applications (Phase Q-M.2, H-APP.4.4)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.providers.relational_store.sqlite.paths import (
    ENV_TASK_MEMORY_DB,
    resolve_task_memory_db_path,
)
from intergrax.runtime.task_memory.persistence_contract import TaskMemoryPersistence
from intergrax.runtime.task_memory.store import resolve_task_memory_persistence


@dataclass(frozen=True)
class TaskMemoryWiring:
    store: Optional[TaskMemoryPersistence]
    db_path: Optional[Path]


def wire_task_memory(
    *,
    db_path: Path | None = None,
    enabled: bool | None = None,
    warn_if_disabled: bool = False,
) -> TaskMemoryWiring:
    """
    Resolve TaskMemory for lab/product hosts.

    Enabled when ``db_path`` is set, ``INTERGRAX_TASK_MEMORY_DB`` is set, or
    ``enabled=True``. Pass ``warn_if_disabled=True`` in harness lab to surface hints.
    """
    if enabled is False:
        if warn_if_disabled:
            resolve_task_memory_persistence(warn_if_disabled=True)
        return TaskMemoryWiring(store=None, db_path=None)
    resolved_path = resolve_task_memory_db_path(db_path) if (
        db_path is not None or os.environ.get(ENV_TASK_MEMORY_DB, "").strip()
    ) else None
    if enabled is True and resolved_path is None:
        resolved_path = resolve_task_memory_db_path(None)
    store = resolve_task_memory_persistence(
        db_path=resolved_path,
        warn_if_disabled=warn_if_disabled,
    )
    return TaskMemoryWiring(store=store, db_path=resolved_path)


def wire_task_memory_from_profile(
    env: ApplicationEnvironmentProfile,
    *,
    db_path: Path | None = None,
    warn_if_disabled: bool = False,
) -> TaskMemoryWiring:
    """Unify task memory wiring under environment memory profile (H-APP.4.4)."""
    memory = env.memory_profile
    enabled = (
        memory.enable_task_memory
        or memory.enable_user_memory
        or memory.enable_org_memory
        or memory.enable_long_term_memory
    )
    return wire_task_memory(
        db_path=db_path,
        enabled=enabled if enabled else None,
        warn_if_disabled=warn_if_disabled,
    )
