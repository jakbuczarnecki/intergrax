# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Low-level SQLite store openers — internal to the sqlite integration package.

Only this module and tests of implementation classes may construct SQLite backends
directly. All composition roots use ``bundle.create_sqlite_*`` factories.
"""

from __future__ import annotations

from pathlib import Path

from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.nexus.session.sqlite_session_storage import SQLiteSessionStorage
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore
from intergrax.runtime.organization.stores.sqlite_organization_profile_store import (
    SQLiteOrganizationProfileStore,
)
from intergrax.runtime.task_memory.stores.sqlite_task_memory_store import SQLiteTaskMemoryStore
from intergrax.runtime.tools.sqlite_idempotency_store import SQLiteIdempotencyStore


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def open_trace_store_at(path: Path) -> SQLiteRunTraceStore:
    return SQLiteRunTraceStore(db_path=_ensure_parent(path))


def open_runtime_event_store_at(path: Path) -> SQLiteRuntimeEventStore:
    return SQLiteRuntimeEventStore(db_path=_ensure_parent(path))


def open_task_checkpoint_store_at(path: Path):
    from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore

    return SQLiteTaskCheckpointStore(db_path=_ensure_parent(path))


def open_human_decision_store_at(path: Path):
    from intergrax.runtime.human.store import SQLiteHumanDecisionStore

    return SQLiteHumanDecisionStore(db_path=_ensure_parent(path))


def open_task_memory_store_at(path: Path) -> SQLiteTaskMemoryStore:
    return SQLiteTaskMemoryStore(db_path=_ensure_parent(path))


def open_experiment_store_at(path: Path):
    from intergrax.experiments.store import SQLiteExperimentStore

    return SQLiteExperimentStore(db_path=_ensure_parent(path))


def open_idempotency_store_at(path: Path) -> SQLiteIdempotencyStore:
    return SQLiteIdempotencyStore(str(_ensure_parent(path)))


def open_session_storage_at(path: Path) -> SQLiteSessionStorage:
    return SQLiteSessionStorage(str(_ensure_parent(path)))


def open_organization_profile_store_at(path: Path) -> SQLiteOrganizationProfileStore:
    return SQLiteOrganizationProfileStore(db_path=str(_ensure_parent(path)))


def open_delivery_ledger_at(path: Path):
    from intergrax.runtime.notifications.deliveries.sqlite_delivery_ledger import SQLiteDeliveryLedger

    return SQLiteDeliveryLedger(db_path=_ensure_parent(path))
