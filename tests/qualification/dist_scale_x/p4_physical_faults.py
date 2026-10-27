# © Artur Czarnecki. All rights reserved.

"""P4 dimension F — bounded storage and event-path failures."""

from __future__ import annotations

import subprocess
import threading
import time

import pytest

from tests.qualification.dist_scale_x._p4_support import (
    P4_PG_CONTAINER,
    docker_pause_pg,
    docker_unpause_pg,
    new_run_id,
    prime_p4_qualification_stores,
    qual_pg_schema,
    require_p4_docker_infra,
    require_p4_pg_infra,
    resolve_pg_dsn,
)
from tests.qualification.dist_scale_x.p4_persistence_op import execute_p4_persistence_op

pytestmark = [pytest.mark.qualification, pytest.mark.integration]


def test_p4_f_storage_pg_pause_surfaces_write_error() -> None:
    require_p4_pg_infra()
    require_p4_docker_infra()
    dsn = resolve_pg_dsn()
    schema = qual_pg_schema(new_run_id("f-storage"))
    prime_p4_qualification_stores(dsn, schema)
    errors: list[str] = []
    attempts = 0

    def _writer() -> None:
        nonlocal attempts
        for i in range(40):
            attempts += 1
            result = execute_p4_persistence_op(
                dsn=dsn,
                schema=schema,
                tenant_id="tenant-a",
                op_index=i + 1,
                payload_pad="f" * 1024,
            )
            if result.error:
                errors.append(result.error)
            time.sleep(0.15)

    docker_pause_pg()
    try:
        thread = threading.Thread(target=_writer, daemon=True)
        thread.start()
        thread.join(timeout=90)
    finally:
        docker_unpause_pg()
    assert attempts > 0, "no write attempts during F-STORAGE window"
    assert errors, f"expected persistence errors during pause on {P4_PG_CONTAINER}"


def test_p4_f_event_injected_append_outage_fail_closed() -> None:
    """Injected F07 append failure (not PostgreSQL-wide outage)."""
    from unittest.mock import patch

    from intergrax.runtime.events.stores.document_backed_runtime_event_store import (
        DocumentBackedRuntimeEventStore,
    )
    from tests.qualification.dist_scale_x.p4_pg_state_inspection import inspect_p4_durable_state

    require_p4_pg_infra()
    dsn = resolve_pg_dsn()
    schema = qual_pg_schema(new_run_id("f-event"))
    prime_p4_qualification_stores(dsn, schema)

    def _boom(self, event, *, tenant_id):  # noqa: ANN001
        raise RuntimeError("qualification_event_append_outage")

    with patch.object(DocumentBackedRuntimeEventStore, "append", _boom):
        result = execute_p4_persistence_op(
            dsn=dsn,
            schema=schema,
            tenant_id="tenant-a",
            op_index=1,
            payload_pad="e" * 1024,
        )
    assert result.error is not None
    assert "qualification_event_append_outage" in result.error
    if result.task_id:
        snap = inspect_p4_durable_state(
            dsn=dsn,
            schema=schema,
            tenant_id="tenant-a",
            task_id=result.task_id,
            run_id="",
            decision_id="decision_0123456789abcdef0123456789abcdef",
            decision_subject="wl5-1",
            event_id="",
        )
        assert snap.f01_revision == 1
        assert snap.f03_present
        assert not snap.f07_event_present
        assert not snap.f09_present
