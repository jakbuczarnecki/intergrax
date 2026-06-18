# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from pydantic import Field

from intergrax.runtime.events.event_kind_registry import (
    EventKindRegistryError,
    clear_event_kind_registry,
    get_event_kind_entry,
    list_registered_event_kinds,
    register_event_kind,
    require_registered_event_kind,
)
from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.observability.extension_sdk import register_extension_runtime_payload

pytestmark = pytest.mark.gate


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    clear_event_kind_registry()
    yield
    clear_event_kind_registry()


class _AgentSignalV1(RuntimeEventPayload):
    schema_id = "agents.echo.risk_flagged.v1"
    score: float = Field(ge=0.0, le=1.0)


def test_register_event_kind_binds_payload_schema() -> None:
    register_extension_runtime_payload(_AgentSignalV1)
    entry = register_event_kind("agents.echo.risk_flagged", _AgentSignalV1.schema_id)
    assert entry.payload_schema_id == _AgentSignalV1.schema_id
    assert get_event_kind_entry("agents.echo.risk_flagged") is entry


def test_register_extension_runtime_payload_with_event_kind() -> None:
    register_extension_runtime_payload(
        _AgentSignalV1,
        event_kind="agents.echo.risk_flagged",
    )
    assert "agents.echo.risk_flagged" in list_registered_event_kinds()


def test_require_registered_event_kind_rejects_unknown() -> None:
    with pytest.raises(EventKindRegistryError, match="unregistered"):
        require_registered_event_kind("agents.echo.missing")


def test_register_rejects_unregistered_payload_schema() -> None:
    with pytest.raises(EventKindRegistryError, match="not registered"):
        register_event_kind("agents.echo.orphan", "agents.echo.orphan.v1")


def test_register_rejects_llm_stream_namespace_on_bus() -> None:
    register_extension_runtime_payload(_AgentSignalV1)
    with pytest.raises(EventKindRegistryError, match="reserved"):
        register_event_kind("intergrax.llm.stream.delta", _AgentSignalV1.schema_id)


def test_llm_stream_event_uses_separate_kind_field() -> None:
    """SAR-08 — LLM stream chunks must not use HOS bus ``event_kind`` namespace."""
    from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent, LLMStreamEventKind

    event = LLMStreamEvent(kind=LLMStreamEventKind.PARTIAL, delta_content="hi")
    assert event.kind == LLMStreamEventKind.PARTIAL
    assert not hasattr(event, "event_kind")
