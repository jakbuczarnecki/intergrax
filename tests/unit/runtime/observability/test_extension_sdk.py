# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from pydantic import Field

from intergrax.runtime.events.payloads.base import RuntimeEventPayload
from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload
from intergrax.runtime.observability.extension_sdk import (
    ExtensionSchemaError,
    PayloadSchemaRegistry,
    agent_diagnostic_schema_id,
    application_diagnostic_schema_id,
    get_registered_diagnostic_payload,
    register_extension_runtime_payload,
)

pytestmark = pytest.mark.gate


def test_agent_diagnostic_schema_id_helper() -> None:
    assert agent_diagnostic_schema_id("echo", "custom_check") == "agents.echo.diag.custom_check"


def test_application_diagnostic_schema_id_helper() -> None:
    assert (
        application_diagnostic_schema_id("my_lab", "host_lifecycle")
        == "applications.my_lab.diag.host_lifecycle"
    )


def test_payload_schema_registry_registers_agent_diagnostic() -> None:
    from dataclasses import dataclass
    from typing import Any, Dict

    @dataclass(frozen=True)
    class _EchoDiag(DiagnosticPayload):
        ok: bool

        @classmethod
        def schema_id(cls) -> str:
            return agent_diagnostic_schema_id("echo_sdk", "probe")

        def to_dict(self) -> Dict[str, Any]:
            return {"ok": self.ok}

        def redact(self) -> _EchoDiag:
            return self

    PayloadSchemaRegistry.register_agent_diagnostic(_EchoDiag, agent_slug="echo_sdk")
    assert get_registered_diagnostic_payload(_EchoDiag.schema_id()) is _EchoDiag


def test_payload_schema_registry_rejects_bad_agent_namespace() -> None:
    from dataclasses import dataclass
    from typing import Any, Dict

    @dataclass(frozen=True)
    class _BadDiag(DiagnosticPayload):
        @classmethod
        def schema_id(cls) -> str:
            return "intergrax.diag.bad"

        def to_dict(self) -> Dict[str, Any]:
            return {}

        def redact(self) -> _BadDiag:
            return self

    with pytest.raises(ExtensionSchemaError):
        PayloadSchemaRegistry.register_agent_diagnostic(_BadDiag, agent_slug="echo_sdk")


def test_register_extension_runtime_payload() -> None:
    class _RuntimeExt(RuntimeEventPayload):
        schema_id = "agents.echo_sdk.event.custom"

        detail: str = Field(default="")

    register_extension_runtime_payload(_RuntimeExt)
    from intergrax.runtime.events.payload_registry import get_payload_schema

    assert get_payload_schema("agents.echo_sdk.event.custom") is _RuntimeExt
