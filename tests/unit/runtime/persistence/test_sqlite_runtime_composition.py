# © Artur Czarnecki. All rights reserved.

"""Runtime SQLite persistence composition (R6-AUDIT-MAJOR-01 remediation)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.experiments.store import SQLiteExperimentStore
from intergrax.integrations.providers.relational_store.sqlite.integration import (
    SqliteRelationalStoreIntegration,
)
from intergrax.integrations.providers.relational_store.sqlite.paths import (
    EXPERIMENTS_DB_NAME,
    RELATIONAL_DB_NAME,
    TRACE_DB_NAME,
)
from intergrax.runtime.events.stores.sqlite_runtime_event_store import (
    SQLiteRuntimeEventStore,
)
from intergrax.runtime.human.store import SQLiteHumanDecisionStore
from intergrax.memory.stores.sqlite_user_profile_store import SQLiteUserProfileStore
from intergrax.runtime.nexus.session.sqlite_session_storage import SQLiteSessionStorage
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore
from intergrax.runtime.organization.stores.sqlite_organization_profile_store import (
    SQLiteOrganizationProfileStore,
)
from intergrax.runtime.persistence.sqlite_composition import (
    SQLiteRuntimePersistenceBundle,
    create_sqlite_runtime_persistence,
    create_sqlite_trace_store,
)
from intergrax.runtime.task_memory.stores.sqlite_task_memory_store import (
    SQLiteTaskMemoryStore,
)
from intergrax.runtime.tools.sqlite_idempotency_store import SQLiteIdempotencyStore

pytestmark = pytest.mark.unit


def test_create_sqlite_runtime_persistence_uses_shared_data_dir(tmp_path: Path) -> None:
    bundle = create_sqlite_runtime_persistence(data_dir=tmp_path)

    assert isinstance(bundle, SQLiteRuntimePersistenceBundle)
    assert bundle.paths.data_dir == tmp_path
    assert bundle.paths.relational == tmp_path / RELATIONAL_DB_NAME
    assert bundle.paths.trace == tmp_path / TRACE_DB_NAME
    assert bundle.paths.experiments == tmp_path / EXPERIMENTS_DB_NAME

    assert isinstance(bundle.relational_store, SqliteRelationalStoreIntegration)
    assert isinstance(bundle.trace_store, SQLiteRunTraceStore)
    assert isinstance(bundle.runtime_event_store, SQLiteRuntimeEventStore)
    assert type(bundle.task_checkpoint_store).__name__ == "SQLiteTaskCheckpointStore"
    assert isinstance(bundle.human_decision_store, SQLiteHumanDecisionStore)
    assert isinstance(bundle.task_memory_store, SQLiteTaskMemoryStore)
    assert isinstance(bundle.experiment_store, SQLiteExperimentStore)
    assert isinstance(bundle.idempotency_store, SQLiteIdempotencyStore)
    assert isinstance(bundle.session_storage, SQLiteSessionStorage)
    assert isinstance(bundle.organization_profile_store, SQLiteOrganizationProfileStore)
    assert isinstance(bundle.user_profile_store, SQLiteUserProfileStore)

    assert bundle.relational_store.db_path.exists()
    assert bundle.paths.trace.exists()


def test_create_sqlite_trace_store_factory(tmp_path: Path) -> None:
    store = create_sqlite_trace_store(data_dir=tmp_path)
    assert isinstance(store, SQLiteRunTraceStore)
