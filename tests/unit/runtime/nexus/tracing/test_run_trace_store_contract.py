# © Artur Czarnecki. All rights reserved.

"""RunTraceStore contract — canonical observability trace backends."""

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.observability_wiring import wire_nexus_observability
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.persistence_models import (
    RunTraceReader,
    RunTraceStore,
    RunTraceWriter,
)
from intergrax.runtime.nexus.tracing.sqlite_run_trace_store import SQLiteRunTraceStore
from intergrax.runtime.nexus.tracing.store import open_run_trace_store

pytestmark = pytest.mark.unit


def test_canonical_trace_stores_satisfy_run_trace_store_contract(tmp_path) -> None:
    stores = (
        InMemoryRunTraceStore(),
        SQLiteRunTraceStore(db_path=tmp_path / "trace.db"),
    )
    for store in stores:
        assert isinstance(store, RunTraceStore)
        assert isinstance(store, RunTraceWriter)
        assert isinstance(store, RunTraceReader)


def test_wire_nexus_observability_exposes_run_trace_store() -> None:
    in_memory = wire_nexus_observability(use_in_memory_trace=True)
    assert isinstance(in_memory.trace_store, RunTraceStore)

    sqlite = wire_nexus_observability(trace_store=open_run_trace_store())
    assert isinstance(sqlite.trace_store, RunTraceStore)
