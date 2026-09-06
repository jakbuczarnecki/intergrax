# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite reference durable adapters for Agent Distribution store protocols (EA-01/EA-02).

Single-host / multi-process persistence with optimistic concurrency preserved by
reload-mutate-save transactions. Stores remain protocol implementations only —
no lifecycle authority.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from intergrax.agent_distribution.application_environment_identity import (
    ApplicationEnvironmentIdentity,
)
from intergrax.agent_distribution.binding import (
    SCHEMA_APPLICATION_AGENT_BINDING_V1,
    ApplicationAgentBinding,
)
from intergrax.agent_distribution.dependency import (
    SCHEMA_MATERIALIZED_RUNTIME_LOCK_V1,
    MaterializedRuntimeLock,
)
from intergrax.agent_distribution.deployment import (
    DeploymentInstanceRecord,
    DeploymentInstanceState,
)
from intergrax.agent_distribution.in_memory_stores import (
    AgentDistributionStoreState,
    InMemoryAgentArtifactMetadataStore,
    InMemoryAgentInstallationStore,
    InMemoryApplicationAgentBindingStore,
    InMemoryApplicationEnvironmentActivationStore,
    InMemoryApplicationEnvironmentServingStore,
    InMemoryDeploymentInstanceStore,
    InMemoryEffectiveRosterSnapshotStore,
    InMemoryMaterializedRuntimeLockStore,
    InMemoryRuntimeMaterializationStore,
    InMemoryRuntimeRevisionStore,
)
from intergrax.agent_distribution.installation import (
    SCHEMA_AGENT_INSTALLATION_RECORD_V1,
    AgentInstallationRecord,
)
from intergrax.agent_distribution.installation_slot_scope import InstallationSlotScope
from intergrax.agent_distribution.roster import (
    SCHEMA_EFFECTIVE_ROSTER_V1,
    EffectiveRoster,
)
from intergrax.agent_distribution.runtime_materialization_record import (
    RuntimeMaterializationRecord,
)
from intergrax.agent_distribution.runtime_revision import (
    SCHEMA_RUNTIME_REVISION_V1,
    RuntimeRevision,
    RuntimeRevisionState,
)
from intergrax.agent_distribution.stores import (
    ActivationAtomicCommitResult,
    AgentArtifactMetadata,
    ApplicationEnvironmentServingRecord,
    RollbackAtomicCommitResult,
)
from intergrax.applications._shared.registry_projection_descriptor import (
    SCHEMA_RUNTIME_REGISTRY_PROJECTION_DESCRIPTOR_V1,
    RuntimeRegistryProjectionDescriptor,
)

SCHEMA_AGENT_DISTRIBUTION_SQLITE_V1 = "agent_distribution_sqlite.v1"
_MODEL = TypeVar("_MODEL", bound=BaseModel)


def _encode(model: BaseModel) -> str:
    return model.model_dump_json()


def _decode(model_type: type[_MODEL], payload: str, expected_schema: str) -> _MODEL:
    data = json.loads(payload)
    schema_version = data.get("schema_version")
    if schema_version != expected_schema:
        raise ValueError(
            f"unsupported schema version {schema_version!r} for {model_type.__name__}; "
            f"expected {expected_schema!r}"
        )
    return model_type.model_validate(data)


def _scope_key(scope: ApplicationEnvironmentIdentity) -> str:
    return f"{scope.application_id}\0{scope.application_environment_id}"


def _slot_key(scope: InstallationSlotScope) -> str:
    return f"{scope.environment_id}\0{scope.installation_slot_id}"


def _deployment_key(
    application_id: str,
    application_environment_id: str,
    runtime_revision_id: str,
) -> str:
    return f"{application_id}\0{application_environment_id}\0{runtime_revision_id}"


class SqliteAgentDistributionDatabase:
    """Shared SQLite backing for one durable agent distribution universe."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._activation_lock = threading.RLock()
        self._init_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=FULL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_distribution_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS installations (
                    installation_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS active_installation_slot (
                    slot_key TEXT PRIMARY KEY,
                    installation_id TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bindings (
                    application_binding_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS revisions (
                    runtime_revision_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS active_revision_scope (
                    scope_key TEXT PRIMARY KEY,
                    runtime_revision_id TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifact_metadata (
                    package_digest TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_locks (
                    lock_id TEXT PRIMARY KEY,
                    lock_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS materializations (
                    runtime_revision_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS effective_roster_snapshots (
                    effective_roster_revision_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS deployment_instances (
                    instance_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS serving_records (
                    scope_key TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS projection_descriptors (
                    runtime_revision_id TEXT PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_distribution_meta(key, value)
                VALUES ('schema_version', ?)
                """,
                (SCHEMA_AGENT_DISTRIBUTION_SQLITE_V1,),
            )

    @contextmanager
    def mutation_transaction(self):
        with self._activation_lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                state = self._load_state(conn)
                yield state
                self._save_state(conn, state)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def load_state(self) -> AgentDistributionStoreState:
        with self._connect() as conn:
            return self._load_state(conn)

    def _load_state(self, conn: sqlite3.Connection) -> AgentDistributionStoreState:
        state = AgentDistributionStoreState()
        for row in conn.execute("SELECT installation_id, payload_json FROM installations"):
            record = _decode(
                AgentInstallationRecord,
                row["payload_json"],
                SCHEMA_AGENT_INSTALLATION_RECORD_V1,
            )
            state.installations[record.installation_id] = record
        for row in conn.execute(
            "SELECT slot_key, installation_id FROM active_installation_slot"
        ):
            environment_id, installation_slot_id = row["slot_key"].split("\0", 1)
            state.active_installation_by_scope[
                InstallationSlotScope(
                    environment_id=environment_id,
                    installation_slot_id=installation_slot_id,
                )
            ] = row["installation_id"]
        for row in conn.execute(
            "SELECT application_binding_id, payload_json FROM bindings"
        ):
            binding = _decode(
                ApplicationAgentBinding,
                row["payload_json"],
                SCHEMA_APPLICATION_AGENT_BINDING_V1,
            )
            state.bindings[binding.application_binding_id] = binding
        for row in conn.execute(
            "SELECT runtime_revision_id, payload_json FROM revisions"
        ):
            revision = _decode(
                RuntimeRevision,
                row["payload_json"],
                SCHEMA_RUNTIME_REVISION_V1,
            )
            state.revisions[revision.runtime_revision_id] = revision
        for row in conn.execute(
            "SELECT scope_key, runtime_revision_id FROM active_revision_scope"
        ):
            application_id, application_environment_id = row["scope_key"].split("\0", 1)
            state.active_revision_by_scope[
                ApplicationEnvironmentIdentity(
                    application_id=application_id,
                    application_environment_id=application_environment_id,
                )
            ] = row["runtime_revision_id"]
        for row in conn.execute(
            "SELECT package_digest, payload_json FROM artifact_metadata"
        ):
            metadata = AgentArtifactMetadata.model_validate_json(row["payload_json"])
            state.artifact_metadata[metadata.package_digest] = metadata
        for row in conn.execute("SELECT lock_id, payload_json FROM runtime_locks"):
            lock = _decode(
                MaterializedRuntimeLock,
                row["payload_json"],
                SCHEMA_MATERIALIZED_RUNTIME_LOCK_V1,
            )
            state.locks[lock.lock_id] = lock
        for row in conn.execute(
            "SELECT runtime_revision_id, payload_json FROM materializations"
        ):
            record = RuntimeMaterializationRecord.model_validate_json(row["payload_json"])
            state.materializations[record.runtime_revision_id] = record
        for row in conn.execute(
            "SELECT effective_roster_revision_id, payload_json FROM effective_roster_snapshots"
        ):
            roster = _decode(
                EffectiveRoster,
                row["payload_json"],
                SCHEMA_EFFECTIVE_ROSTER_V1,
            )
            state.effective_roster_snapshots[roster.effective_roster_revision_id] = roster
        for row in conn.execute("SELECT instance_key, payload_json FROM deployment_instances"):
            instance = DeploymentInstanceRecord.model_validate_json(row["payload_json"])
            key = (
                ApplicationEnvironmentIdentity(
                    application_id=instance.application_id,
                    application_environment_id=instance.application_environment_id,
                ),
                instance.runtime_revision_id,
            )
            state.deployment_instances[key] = instance
        for row in conn.execute("SELECT scope_key, payload_json FROM serving_records"):
            application_id, application_environment_id = row["scope_key"].split("\0", 1)
            serving = ApplicationEnvironmentServingRecord.model_validate_json(
                row["payload_json"]
            )
            state.serving_records[
                ApplicationEnvironmentIdentity(
                    application_id=application_id,
                    application_environment_id=application_environment_id,
                )
            ] = serving
        return state

    def _save_state(self, conn: sqlite3.Connection, state: AgentDistributionStoreState) -> None:
        conn.execute("DELETE FROM installations")
        conn.execute("DELETE FROM active_installation_slot")
        conn.execute("DELETE FROM bindings")
        conn.execute("DELETE FROM revisions")
        conn.execute("DELETE FROM active_revision_scope")
        conn.execute("DELETE FROM artifact_metadata")
        conn.execute("DELETE FROM runtime_locks")
        conn.execute("DELETE FROM materializations")
        conn.execute("DELETE FROM effective_roster_snapshots")
        conn.execute("DELETE FROM deployment_instances")
        conn.execute("DELETE FROM serving_records")
        for record in state.installations.values():
            conn.execute(
                "INSERT INTO installations(installation_id, payload_json) VALUES (?, ?)",
                (record.installation_id, _encode(record)),
            )
        for scope, installation_id in state.active_installation_by_scope.items():
            conn.execute(
                "INSERT INTO active_installation_slot(slot_key, installation_id) VALUES (?, ?)",
                (_slot_key(scope), installation_id),
            )
        for binding in state.bindings.values():
            conn.execute(
                "INSERT INTO bindings(application_binding_id, payload_json) VALUES (?, ?)",
                (binding.application_binding_id, _encode(binding)),
            )
        for revision in state.revisions.values():
            conn.execute(
                "INSERT INTO revisions(runtime_revision_id, payload_json) VALUES (?, ?)",
                (revision.runtime_revision_id, _encode(revision)),
            )
        for scope, revision_id in state.active_revision_by_scope.items():
            conn.execute(
                "INSERT INTO active_revision_scope(scope_key, runtime_revision_id) VALUES (?, ?)",
                (_scope_key(scope), revision_id),
            )
        for metadata in state.artifact_metadata.values():
            conn.execute(
                "INSERT INTO artifact_metadata(package_digest, payload_json) VALUES (?, ?)",
                (metadata.package_digest, _encode(metadata)),
            )
        for lock in state.locks.values():
            conn.execute(
                "INSERT INTO runtime_locks(lock_id, lock_digest, payload_json) VALUES (?, ?, ?)",
                (lock.lock_id, lock.lock_digest, _encode(lock)),
            )
        for record in state.materializations.values():
            conn.execute(
                "INSERT INTO materializations(runtime_revision_id, payload_json) VALUES (?, ?)",
                (record.runtime_revision_id, record.model_dump_json()),
            )
        for roster in state.effective_roster_snapshots.values():
            conn.execute(
                """
                INSERT INTO effective_roster_snapshots(
                    effective_roster_revision_id, payload_json
                ) VALUES (?, ?)
                """,
                (roster.effective_roster_revision_id, _encode(roster)),
            )
        for key, instance in state.deployment_instances.items():
            scope, revision_id = key
            conn.execute(
                "INSERT INTO deployment_instances(instance_key, payload_json) VALUES (?, ?)",
                (
                    _deployment_key(
                        scope.application_id,
                        scope.application_environment_id,
                        revision_id,
                    ),
                    instance.model_dump_json(),
                ),
            )
        for scope, serving in state.serving_records.items():
            conn.execute(
                "INSERT INTO serving_records(scope_key, payload_json) VALUES (?, ?)",
                (_scope_key(scope), serving.model_dump_json()),
            )


class _SqliteMutationMixin:
    _db: SqliteAgentDistributionDatabase

    def _refresh(self) -> None:
        self._state = self._db.load_state()

    def _mutate(self, operation):
        with self._db.mutation_transaction() as state:
            self._state = state
            return operation()


class SqliteAgentInstallationStore(_SqliteMutationMixin, InMemoryAgentInstallationStore):
    def __init__(self, db: SqliteAgentDistributionDatabase) -> None:
        self._db = db
        super().__init__(db.load_state())

    def get_installation(self, installation_id: str) -> AgentInstallationRecord | None:
        self._refresh()
        return super().get_installation(installation_id)

    def get_active_installation_for_slot(
        self,
        environment_id: str,
        installation_slot_id: str,
    ) -> AgentInstallationRecord | None:
        self._refresh()
        return super().get_active_installation_for_slot(
            environment_id,
            installation_slot_id,
        )

    def list_installations_for_slot(
        self,
        environment_id: str,
        installation_slot_id: str,
    ) -> list[AgentInstallationRecord]:
        self._refresh()
        return super().list_installations_for_slot(environment_id, installation_slot_id)

    def list_installations_for_environment(
        self,
        environment_id: str,
    ) -> list[AgentInstallationRecord]:
        self._refresh()
        return super().list_installations_for_environment(environment_id)

    def persist_installation(
        self,
        record: AgentInstallationRecord,
        *,
        expected_active_installation_id: str | None = None,
    ) -> AgentInstallationRecord:
        return self._mutate(
            lambda: super(SqliteAgentInstallationStore, self).persist_installation(
                record,
                expected_active_installation_id=expected_active_installation_id,
            )
        )

    def atomic_promote_active_installation(
        self,
        *,
        demoted_prior: AgentInstallationRecord | None,
        promoted: AgentInstallationRecord,
        expected_active_installation_id: str | None,
    ) -> tuple[AgentInstallationRecord, AgentInstallationRecord | None]:
        return self._mutate(
            lambda: super(
                SqliteAgentInstallationStore, self
            ).atomic_promote_active_installation(
                demoted_prior=demoted_prior,
                promoted=promoted,
                expected_active_installation_id=expected_active_installation_id,
            )
        )


class SqliteApplicationAgentBindingStore(
    _SqliteMutationMixin, InMemoryApplicationAgentBindingStore
):
    def __init__(self, db: SqliteAgentDistributionDatabase) -> None:
        self._db = db
        super().__init__(db.load_state())

    def get_binding(
        self, application_binding_id: str
    ) -> ApplicationAgentBinding | None:
        self._refresh()
        return super().get_binding(application_binding_id)

    def list_bindings_for_environment(
        self,
        application_id: str,
        application_environment_id: str,
    ) -> list[ApplicationAgentBinding]:
        self._refresh()
        return super().list_bindings_for_environment(
            application_id,
            application_environment_id,
        )

    def list_bindings_for_slot(
        self, installation_slot_id: str
    ) -> list[ApplicationAgentBinding]:
        self._refresh()
        return super().list_bindings_for_slot(installation_slot_id)

    def persist_binding(
        self,
        binding: ApplicationAgentBinding,
        *,
        expected_revision: int | None = None,
    ) -> ApplicationAgentBinding:
        return self._mutate(
            lambda: super(SqliteApplicationAgentBindingStore, self).persist_binding(
                binding,
                expected_revision=expected_revision,
            )
        )


class SqliteRuntimeRevisionStore(_SqliteMutationMixin, InMemoryRuntimeRevisionStore):
    def __init__(self, db: SqliteAgentDistributionDatabase) -> None:
        self._db = db
        super().__init__(db.load_state())

    def get_revision(self, runtime_revision_id: str) -> RuntimeRevision | None:
        self._refresh()
        return super().get_revision(runtime_revision_id)

    def get_active_revision(
        self,
        application_id: str,
        application_environment_id: str,
    ) -> RuntimeRevision | None:
        self._refresh()
        return super().get_active_revision(application_id, application_environment_id)

    def list_revisions_for_environment(
        self,
        application_id: str,
        application_environment_id: str,
    ) -> list[RuntimeRevision]:
        self._refresh()
        return super().list_revisions_for_environment(
            application_id,
            application_environment_id,
        )

    def persist_candidate_revision(
        self,
        revision: RuntimeRevision,
        *,
        expected_revision_state: RuntimeRevisionState | None = None,
    ) -> RuntimeRevision:
        return self._mutate(
            lambda: super(SqliteRuntimeRevisionStore, self).persist_candidate_revision(
                revision,
                expected_revision_state=expected_revision_state,
            )
        )

    def swap_active_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        new_active_revision_id: str,
        prior_active_revision_id: str | None = None,
    ) -> RuntimeRevision:
        return self._mutate(
            lambda: super(SqliteRuntimeRevisionStore, self).swap_active_revision(
                application_id=application_id,
                application_environment_id=application_environment_id,
                new_active_revision_id=new_active_revision_id,
                prior_active_revision_id=prior_active_revision_id,
            )
        )

    def atomic_activate_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        promoted: RuntimeRevision,
        demoted_prior: RuntimeRevision | None,
        expected_prior_active_revision_id: str | None,
    ) -> tuple[RuntimeRevision, RuntimeRevision | None]:
        return self._mutate(
            lambda: super(SqliteRuntimeRevisionStore, self).atomic_activate_revision(
                application_id=application_id,
                application_environment_id=application_environment_id,
                promoted=promoted,
                demoted_prior=demoted_prior,
                expected_prior_active_revision_id=expected_prior_active_revision_id,
            )
        )


class SqliteAgentArtifactMetadataStore(
    _SqliteMutationMixin, InMemoryAgentArtifactMetadataStore
):
    def __init__(self, db: SqliteAgentDistributionDatabase) -> None:
        self._db = db
        super().__init__(db.load_state())

    def get_by_digest(self, package_digest: str) -> AgentArtifactMetadata | None:
        self._refresh()
        return super().get_by_digest(package_digest)

    def persist_metadata(
        self, metadata: AgentArtifactMetadata
    ) -> AgentArtifactMetadata:
        return self._mutate(
            lambda: super(SqliteAgentArtifactMetadataStore, self).persist_metadata(
                metadata
            )
        )


class SqliteMaterializedRuntimeLockStore(
    _SqliteMutationMixin, InMemoryMaterializedRuntimeLockStore
):
    def __init__(self, db: SqliteAgentDistributionDatabase) -> None:
        self._db = db
        super().__init__(db.load_state())

    def get_lock(self, lock_id: str) -> MaterializedRuntimeLock | None:
        self._refresh()
        return super().get_lock(lock_id)

    def get_lock_by_digest(self, lock_digest: str) -> MaterializedRuntimeLock | None:
        self._refresh()
        return super().get_lock_by_digest(lock_digest)

    def persist_lock(self, lock: MaterializedRuntimeLock) -> MaterializedRuntimeLock:
        return self._mutate(
            lambda: super(SqliteMaterializedRuntimeLockStore, self).persist_lock(lock)
        )


class SqliteRuntimeMaterializationStore(
    _SqliteMutationMixin, InMemoryRuntimeMaterializationStore
):
    def __init__(self, db: SqliteAgentDistributionDatabase) -> None:
        self._db = db
        super().__init__(db.load_state())

    def get_by_revision(
        self, runtime_revision_id: str
    ) -> RuntimeMaterializationRecord | None:
        self._refresh()
        return super().get_by_revision(runtime_revision_id)

    def persist(
        self, record: RuntimeMaterializationRecord
    ) -> RuntimeMaterializationRecord:
        return self._mutate(
            lambda: super(SqliteRuntimeMaterializationStore, self).persist(record)
        )


class SqliteEffectiveRosterSnapshotStore(
    _SqliteMutationMixin, InMemoryEffectiveRosterSnapshotStore
):
    def __init__(self, db: SqliteAgentDistributionDatabase) -> None:
        self._db = db
        super().__init__(db.load_state())

    def get_by_revision(
        self,
        effective_roster_revision_id: str,
    ) -> EffectiveRoster | None:
        self._refresh()
        return super().get_by_revision(effective_roster_revision_id)

    def persist(self, roster: EffectiveRoster) -> EffectiveRoster:
        return self._mutate(
            lambda: super(SqliteEffectiveRosterSnapshotStore, self).persist(roster)
        )


class SqliteDeploymentInstanceStore(_SqliteMutationMixin, InMemoryDeploymentInstanceStore):
    def __init__(self, db: SqliteAgentDistributionDatabase) -> None:
        self._db = db
        super().__init__(db.load_state())

    def get_instance(
        self,
        application_id: str,
        application_environment_id: str,
        runtime_revision_id: str,
    ) -> DeploymentInstanceRecord | None:
        self._refresh()
        return super().get_instance(
            application_id,
            application_environment_id,
            runtime_revision_id,
        )

    def persist_instance(
        self, instance: DeploymentInstanceRecord
    ) -> DeploymentInstanceRecord:
        return self._mutate(
            lambda: super(SqliteDeploymentInstanceStore, self).persist_instance(instance)
        )

    def update_instance(
        self,
        instance: DeploymentInstanceRecord,
        *,
        expected_state: DeploymentInstanceState | None = None,
        expected_record_revision: int | None = None,
    ) -> DeploymentInstanceRecord:
        return self._mutate(
            lambda: super(SqliteDeploymentInstanceStore, self).update_instance(
                instance,
                expected_state=expected_state,
                expected_record_revision=expected_record_revision,
            )
        )


class SqliteApplicationEnvironmentServingStore(
    _SqliteMutationMixin, InMemoryApplicationEnvironmentServingStore
):
    def __init__(self, db: SqliteAgentDistributionDatabase) -> None:
        self._db = db
        super().__init__(db.load_state())

    def get_serving_record(
        self,
        application_id: str,
        application_environment_id: str,
    ) -> ApplicationEnvironmentServingRecord | None:
        self._refresh()
        return super().get_serving_record(application_id, application_environment_id)

    def atomic_swap_serving_revision(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        expected_current_revision_id: str | None,
        expected_pointer_revision: int,
        new_revision_id: str,
        prior_revision_id: str | None,
        committed_at: datetime,
    ) -> ApplicationEnvironmentServingRecord:
        return self._mutate(
            lambda: super(
                SqliteApplicationEnvironmentServingStore, self
            ).atomic_swap_serving_revision(
                application_id=application_id,
                application_environment_id=application_environment_id,
                expected_current_revision_id=expected_current_revision_id,
                expected_pointer_revision=expected_pointer_revision,
                new_revision_id=new_revision_id,
                prior_revision_id=prior_revision_id,
                committed_at=committed_at,
            )
        )


class SqliteApplicationEnvironmentActivationStore(
    _SqliteMutationMixin, InMemoryApplicationEnvironmentActivationStore
):
    def __init__(self, db: SqliteAgentDistributionDatabase) -> None:
        self._db = db
        super().__init__(db.load_state())

    def atomic_commit_activation(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        expected_current_revision_id: str | None,
        expected_pointer_revision: int,
        candidate_revision_id: str,
        expected_artifact_digest: str,
        committed_at: datetime,
    ) -> ActivationAtomicCommitResult:
        return self._mutate(
            lambda: super(
                SqliteApplicationEnvironmentActivationStore, self
            ).atomic_commit_activation(
                application_id=application_id,
                application_environment_id=application_environment_id,
                expected_current_revision_id=expected_current_revision_id,
                expected_pointer_revision=expected_pointer_revision,
                candidate_revision_id=candidate_revision_id,
                expected_artifact_digest=expected_artifact_digest,
                committed_at=committed_at,
            )
        )

    def atomic_commit_rollback(
        self,
        *,
        application_id: str,
        application_environment_id: str,
        expected_current_revision_id: str,
        expected_pointer_revision: int,
        target_revision_id: str,
        committed_at: datetime,
    ) -> RollbackAtomicCommitResult:
        return self._mutate(
            lambda: super(
                SqliteApplicationEnvironmentActivationStore, self
            ).atomic_commit_rollback(
                application_id=application_id,
                application_environment_id=application_environment_id,
                expected_current_revision_id=expected_current_revision_id,
                expected_pointer_revision=expected_pointer_revision,
                target_revision_id=target_revision_id,
                committed_at=committed_at,
            )
        )


class SqliteRuntimeRegistryProjectionDescriptorStore:
    """SQLite-backed durable projection descriptor store."""

    def __init__(self, db: SqliteAgentDistributionDatabase) -> None:
        self._db = db
        self._lock = threading.RLock()

    def put(self, descriptor: RuntimeRegistryProjectionDescriptor) -> None:
        scope_key = _scope_key(
            ApplicationEnvironmentIdentity(
                application_id=descriptor.application_id,
                application_environment_id=descriptor.application_environment_id,
            )
        )
        payload = descriptor.model_dump_json()
        with self._lock:
            with self._db._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                existing = conn.execute(
                    """
                    SELECT payload_json FROM projection_descriptors
                    WHERE runtime_revision_id = ?
                    """,
                    (descriptor.runtime_revision_id,),
                ).fetchone()
                if existing is not None:
                    if existing["payload_json"] != payload:
                        conn.rollback()
                        raise ValueError(
                            f"conflicting projection descriptor for "
                            f"{descriptor.runtime_revision_id!r}"
                        )
                    conn.commit()
                    return
                conn.execute(
                    """
                    INSERT INTO projection_descriptors(
                        runtime_revision_id, scope_key, payload_json
                    ) VALUES (?, ?, ?)
                    """,
                    (descriptor.runtime_revision_id, scope_key, payload),
                )
                conn.commit()

    def get_for_revision(
        self,
        application_id: str,
        application_environment_id: str,
        runtime_revision_id: str,
    ) -> RuntimeRegistryProjectionDescriptor | None:
        with self._lock:
            with self._db._connect() as conn:
                row = conn.execute(
                    """
                    SELECT payload_json FROM projection_descriptors
                    WHERE runtime_revision_id = ?
                    """,
                    (runtime_revision_id,),
                ).fetchone()
        if row is None:
            return None
        descriptor = _decode(
            RuntimeRegistryProjectionDescriptor,
            row["payload_json"],
            SCHEMA_RUNTIME_REGISTRY_PROJECTION_DESCRIPTOR_V1,
        )
        if descriptor.application_id != application_id:
            raise ValueError(
                "projection descriptor application_id mismatch with lookup scope"
            )
        if descriptor.application_environment_id != application_environment_id:
            raise ValueError(
                "projection descriptor application_environment_id mismatch with lookup scope"
            )
        return descriptor


@dataclass(frozen=True, slots=True)
class SqliteAgentDistributionStoreBundle:
    database: SqliteAgentDistributionDatabase
    installation_store: SqliteAgentInstallationStore
    binding_store: SqliteApplicationAgentBindingStore
    revision_store: SqliteRuntimeRevisionStore
    artifact_metadata_store: SqliteAgentArtifactMetadataStore
    lock_store: SqliteMaterializedRuntimeLockStore
    materialization_store: SqliteRuntimeMaterializationStore
    effective_roster_snapshot_store: SqliteEffectiveRosterSnapshotStore
    deployment_instance_store: SqliteDeploymentInstanceStore
    serving_store: SqliteApplicationEnvironmentServingStore
    activation_store: SqliteApplicationEnvironmentActivationStore
    projection_descriptor_store: SqliteRuntimeRegistryProjectionDescriptorStore


def build_sqlite_agent_distribution_store_bundle(
    db_path: Path,
) -> SqliteAgentDistributionStoreBundle:
    database = SqliteAgentDistributionDatabase(db_path)
    return SqliteAgentDistributionStoreBundle(
        database=database,
        installation_store=SqliteAgentInstallationStore(database),
        binding_store=SqliteApplicationAgentBindingStore(database),
        revision_store=SqliteRuntimeRevisionStore(database),
        artifact_metadata_store=SqliteAgentArtifactMetadataStore(database),
        lock_store=SqliteMaterializedRuntimeLockStore(database),
        materialization_store=SqliteRuntimeMaterializationStore(database),
        effective_roster_snapshot_store=SqliteEffectiveRosterSnapshotStore(database),
        deployment_instance_store=SqliteDeploymentInstanceStore(database),
        serving_store=SqliteApplicationEnvironmentServingStore(database),
        activation_store=SqliteApplicationEnvironmentActivationStore(database),
        projection_descriptor_store=SqliteRuntimeRegistryProjectionDescriptorStore(
            database
        ),
    )


__all__ = [
    "SCHEMA_AGENT_DISTRIBUTION_SQLITE_V1",
    "SqliteAgentDistributionDatabase",
    "SqliteAgentDistributionStoreBundle",
    "SqliteRuntimeRegistryProjectionDescriptorStore",
    "build_sqlite_agent_distribution_store_bundle",
]
