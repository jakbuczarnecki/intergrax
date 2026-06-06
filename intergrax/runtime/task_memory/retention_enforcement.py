# © Artur Czarnecki. All rights reserved.

"""STM retention enforcement helpers (FAUDIT-MEM.1)."""

from __future__ import annotations

from intergrax.runtime.task_memory.retention import is_record_expired


def should_forget_stm_record(
    *,
    updated_at_utc: str,
    retention_days: int | None,
    namespace: str,
) -> bool:
    """Return True when a short-term memory record should be purged."""
    if not namespace.startswith("stm:"):
        return False
    return is_record_expired(updated_at_utc, retention_days=retention_days)
