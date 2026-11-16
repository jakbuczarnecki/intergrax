# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-H (F-D8-07): ``RuntimeContext.create_artifact`` writes a durable, attributable artifact."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from intergrax.runtime.execution.agent_runtime_context import RuntimeContext
from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
from tests.qualification.pdc_x.d8._d8_support import TENANT_A, TENANT_B, sqlite_store

pytestmark = pytest.mark.unit


class _State:
    def __init__(self, *, tenant_id: str = TENANT_A, run_id: str = "run-7", agent_id: str | None = "agent-7") -> None:
        self.tenant_id = tenant_id
        self.run_id = run_id
        self.request = SimpleNamespace(agent_id=agent_id)
        self.added: list[Any] = []
        self.events: list[dict[str, Any]] = []

    def add_artifact(self, ref: Any) -> None:
        self.added.append(ref)

    def trace_event(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


def _create(store: Any, state: _State, data: bytes = b"evidence-bytes"):
    context = SimpleNamespace(artifact_store=store)
    create: Any = RuntimeContext.create_artifact
    return create(context, state=state, kind="report", mime_type="text/plain", data=data, step_id="step-1")


def test_gateway_stores_a_tenant_scoped_attributable_artifact(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    state = _State()
    ref = _create(store, state)
    stored = store.get(TENANT_A, ref.artifact_id)
    assert stored.tenant_id == TENANT_A
    assert stored.run_id == "run-7" and stored.step_id == "step-1"
    assert stored.data == b"evidence-bytes"
    assert stored.producer is not None and stored.producer.created_by_agent_id == "agent-7"
    assert stored.lineage is not None and stored.lineage.run_id == "run-7"
    durable = stored.to_durable_ref()
    assert durable.scope.tenant_id == TENANT_A
    assert state.added == [ref] and state.events


def test_gateway_artifact_is_invisible_to_another_tenant(tmp_path: Path) -> None:
    store = sqlite_store(tmp_path)
    ref = _create(store, _State())
    with pytest.raises(KeyError):
        store.get(TENANT_B, ref.artifact_id)


def test_gateway_without_a_request_agent_still_records_a_producer() -> None:
    store = InMemoryArtifactStore()
    ref = _create(store, _State(agent_id=None))
    stored = store.get(TENANT_A, ref.artifact_id)
    assert stored.producer is not None and stored.lineage is not None


def test_gateway_without_a_store_still_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="ArtifactStore not configured"):
        _create(None, _State())
