# © Artur Czarnecki. All rights reserved.

"""HARNESS-01-R5-W2-R4 — typed shared-context ACP seam (no ``Any`` masking)."""

from __future__ import annotations

import inspect

import pytest

from intergrax.agents.authoring.acp_runtime_session_ports import (
    SharedContextLoadPort,
    SharedContextPersistPort,
    SharedContextProjectionPort,
)
from intergrax.agents.authoring.shared_context_access import (
    load_view,
    persist_view,
    view_from_task_metadata,
)
from intergrax.contracts.shared_context import SharedContextView

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _callable_param_types(protocol: type) -> list[str]:
    call = inspect.signature(protocol.__call__).parameters["metadata"]
    return [str(call.annotation)]


def test_shared_context_ports_use_typed_metadata_not_any() -> None:
    for protocol in (
        SharedContextLoadPort,
        SharedContextPersistPort,
        SharedContextProjectionPort,
    ):
        for annotation in _callable_param_types(protocol):
            assert "Any" not in annotation
            assert annotation != "object"


def test_shared_context_access_round_trip() -> None:
    metadata: dict[str, object] = {}
    view = SharedContextView(task_id="task-1")
    persist_view(metadata, view)
    loaded = load_view(metadata)
    assert loaded is not None
    assert loaded.task_id == "task-1"
    projected = view_from_task_metadata(metadata, task_id="task-2")
    assert projected.task_id == "task-2"


def test_shared_context_access_rejects_non_mapping_carrier() -> None:
    with pytest.raises(TypeError, match="mutable mapping"):
        load_view([])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mutable mapping"):
        persist_view(42, SharedContextView(task_id="x"))  # type: ignore[arg-type]
