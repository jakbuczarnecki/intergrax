# © Artur Czarnecki. All rights reserved.

"""SESSION-01 — production-shaped durable continuation composition proof."""

from __future__ import annotations

import pytest

from intergrax.runtime.execution.continuation.composition import (
    wire_execution_engine_continuation_dependencies,
)
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    BackingExecutionContinuationStateStore,
    backing_execution_continuation_state_store,
    export_durable_continuation_state,
    execution_continuation_state_store_from_durable_export,
    wire_execution_continuation_state_store,
)
from testing_support.mp4r7_enterprise_integration.composition import (
    open_mp4r7_enterprise_integration_composition,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_session_01_default_composition_store_is_explicit_ephemeral() -> None:
    deps = wire_execution_engine_continuation_dependencies()
    store = wire_execution_continuation_state_store(state_store=None)
    assert store.is_durable is False
    assert deps.continuation is not None


def test_session_01_durable_backing_composition_is_explicit_and_replaceable() -> None:
    backing = ExecutionContinuationDurableBacking()
    store = backing_execution_continuation_state_store(backing)
    assert isinstance(store, BackingExecutionContinuationStateStore)
    assert store.is_durable is False  # reconnect client; restart qualification uses export path
    export = export_durable_continuation_state(backing)
    restart_store = execution_continuation_state_store_from_durable_export(export)
    assert restart_store.is_durable is True
    deps = wire_execution_engine_continuation_dependencies(state_store=restart_store)
    assert deps.continuation is not None
    assert deps.lifecycle_driver is not None


def test_session_01_mp4r7_enterprise_integration_durable_wiring() -> None:
    composition = open_mp4r7_enterprise_integration_composition(durable_continuation=True)
    assert composition.continuation_state_store is not None
    assert composition.continuation_backing is not None
    assert composition.continuation_port is not None
    export = export_durable_continuation_state(composition.continuation_backing)
    assert export["schema_version"] == "execution_continuation_durable_state.v1"
    restart_store = execution_continuation_state_store_from_durable_export(export)
    assert restart_store.is_durable is True
