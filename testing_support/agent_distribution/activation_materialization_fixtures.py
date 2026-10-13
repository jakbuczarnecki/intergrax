# © Artur Czarnecki. All rights reserved.

"""Canonical materialization authority helpers for activation orchestration tests."""

from __future__ import annotations

from intergrax.agent_distribution.in_memory_stores import InMemoryRuntimeMaterializationStore
from intergrax.agent_distribution.runtime_materialization_record import RuntimeMaterializationRecord
from intergrax.agent_distribution.runtime_revision import RuntimeRevision


def persist_materialization_for_revision(
    store: InMemoryRuntimeMaterializationStore,
    revision: RuntimeRevision,
    *,
    artifact_locator: str,
) -> RuntimeMaterializationRecord:
    """Persist immutable materialization authority aligned with a runtime revision."""
    topology = revision.materialization_topology
    digest = revision.materialization_artifact_digest
    lock_id = revision.materialized_runtime_lock_id
    lock_digest = revision.materialized_runtime_lock_digest
    if topology is None or digest is None or lock_id is None or lock_digest is None:
        raise ValueError("revision lacks materialization identity for test fixture")
    record = RuntimeMaterializationRecord(
        runtime_revision_id=revision.runtime_revision_id,
        application_id=revision.application_id,
        application_environment_id=revision.application_environment_id,
        materialization_topology=topology,
        artifact_locator=artifact_locator,
        materialization_artifact_digest=digest,
        materialized_runtime_lock_id=lock_id,
        materialized_runtime_lock_digest=lock_digest,
    )
    return store.persist(record)
