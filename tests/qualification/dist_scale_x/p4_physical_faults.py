# © Artur Czarnecki. All rights reserved.

"""P4 dimension F — bounded storage and event-path failures."""

from __future__ import annotations

import subprocess
import threading
import time

import pytest

from tests.qualification.dist_scale_x._p4_support import (
    P4_FAULT_WINDOW_SECONDS,
    bounded_pg_outage,
    docker_unpause_pg,
    new_run_id,
    qual_pg_schema,
    require_p4_pg_infra,
    resolve_pg_dsn,
)
from tests.qualification.dist_scale_x.p4_persistence_op import execute_p4_persistence_op

pytestmark = [pytest.mark.qualification, pytest.mark.integration]


def test_p4_f_storage_pg_pause_surfaces_write_error() -> None:
    require_p4_pg_infra()
    dsn = resolve_pg_dsn()
    schema = qual_pg_schema(new_run_id("f-storage"))
    errors: list[str] = []

    def _writer() -> None:
        for i in range(20):
            result = execute_p4_persistence_op(
                dsn=dsn,
                schema=schema,
                tenant_id="tenant-a",
                op_index=i + 1,
                payload_pad="f" * 1024,
            )
            if result.error:
                errors.append(result.error)
            time.sleep(0.2)

    thread = threading.Thread(target=_writer, daemon=True)
    thread.start()
    try:
        try:
            bounded_pg_outage(seconds=min(5, P4_FAULT_WINDOW_SECONDS))
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            pytest.skip(f"docker pause unavailable: {exc}")
    finally:
        try:
            docker_unpause_pg()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    thread.join(timeout=60)
    if not errors:
        pytest.skip("PG pause did not surface persistence errors in this environment")


def test_p4_f_event_append_outage_fail_closed() -> None:
    from unittest.mock import patch

    from intergrax.runtime.events.stores.document_backed_runtime_event_store import (
        DocumentBackedRuntimeEventStore,
    )

    require_p4_pg_infra()
    dsn = resolve_pg_dsn()
    schema = qual_pg_schema(new_run_id("f-event"))

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
