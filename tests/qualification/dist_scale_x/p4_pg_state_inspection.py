# © Artur Czarnecki. All rights reserved.

"""Independent-process PostgreSQL durable state inspection (P4 qual)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from intergrax.integrations.providers.document_store.postgresql.bundle import (
    create_postgresql_document_store,
)
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)
from intergrax.runtime.execution.decision_finalization_conformance import (
    conformance_artifact_payload_codec_registry,
)
from intergrax.runtime.execution.budget.persistence import DocumentStoreRunBudgetPersistence
from intergrax.runtime.long_running.postgresql_task_checkpoint_store import (
    PostgreSQLTaskCheckpointStore,
)
from intergrax.runtime.persistence.postgresql_opens import open_postgresql_decision_checkpoint_store

from tests.qualification.dist_scale_x.p4_decision_checkpoint_sample import (
    build_p4_qualification_decision_checkpoint,
)

BoundaryId = Literal["G-01", "G-02", "G-03", "G-04", "G-05", "G-06"]


@dataclass(slots=True)
class P4DurableSnapshot:
    f01_revision: int | None
    f03_present: bool
    f07_event_present: bool
    f09_present: bool
    checkpoint_count: int


def inspect_p4_durable_state(
    *,
    dsn: str,
    schema: str,
    tenant_id: str,
    task_id: str,
    run_id: str,
    decision_id: str,
    decision_subject: str,
    event_id: str,
) -> P4DurableSnapshot:
    config = PostgreSQLIntegrationConfig.from_env().model_copy(
        update={"dsn": dsn, "tenant_schema": schema},
    )
    f01_store = PostgreSQLTaskCheckpointStore(config=config)
    latest = f01_store.get_latest(task_id, tenant_id) if task_id else None
    checkpoints = f01_store.list_for_task(task_id, tenant_id) if task_id else []
    key = build_p4_qualification_decision_checkpoint(
        tenant_id=tenant_id,
        subject=decision_subject,
        decision_id=decision_id,
    ).finalization.key
    f03 = open_postgresql_decision_checkpoint_store(
        config=config,
        payload_codecs=conformance_artifact_payload_codec_registry(),
    ).load_materialized(key=key)
    doc = create_postgresql_document_store(tenant_schema=schema, dsn=dsn)
    budget = DocumentStoreRunBudgetPersistence(doc).load_snapshot(
        tenant_id=tenant_id,
        run_id=run_id,
    ) if run_id else None
    event_present = False
    if event_id:
        record = doc.get(f"{tenant_id}|event", event_id)
        if record is None:
            record = doc.get("runtime_event|event_id", event_id)
        event_present = record is not None
    return P4DurableSnapshot(
        f01_revision=latest.revision if latest else None,
        f03_present=f03 is not None,
        f07_event_present=event_present,
        f09_present=budget is not None,
        checkpoint_count=len(checkpoints),
    )


def assert_boundary_snapshot(boundary: BoundaryId, snap: P4DurableSnapshot) -> None:
    if boundary == "G-01":
        assert snap.checkpoint_count == 0
        assert not snap.f03_present and not snap.f07_event_present and not snap.f09_present
        return
    if boundary == "G-02":
        assert snap.f01_revision == 1
        assert not snap.f03_present and not snap.f07_event_present and not snap.f09_present
        return
    if boundary == "G-03":
        assert snap.f01_revision == 1
        assert snap.f03_present
        assert not snap.f07_event_present and not snap.f09_present
        return
    if boundary == "G-04":
        assert snap.f01_revision == 1
        assert snap.f03_present and snap.f07_event_present
        assert not snap.f09_present
        return
    if boundary in {"G-05", "G-06"}:
        assert snap.f01_revision == 1
        assert snap.f03_present and snap.f07_event_present and snap.f09_present
        return
    raise AssertionError(f"unknown boundary {boundary}")
