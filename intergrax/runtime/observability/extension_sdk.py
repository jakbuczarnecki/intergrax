# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Observability extension SDK for Tier-2 agents and Tier-3 applications (OBS-BUS-4)."""

from __future__ import annotations

import re
from typing import TypeVar

from intergrax.runtime.events.payload_registry import register_payload_schema
from intergrax.runtime.events.payloads import RuntimeEventPayload
from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload

TDiag = TypeVar("TDiag", bound=DiagnosticPayload)
TRuntime = TypeVar("TRuntime", bound=RuntimeEventPayload)

HARNESS_DIAG_PREFIX = "intergrax.diag."
AGENT_DIAG_PREFIX = "agents."
APPLICATION_DIAG_PREFIX = "applications."

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class ExtensionSchemaError(ValueError):
    """Raised when an extension schema_id violates namespace rules."""


_DIAGNOSTIC_REGISTRY: dict[str, type[DiagnosticPayload]] = {}


def _validate_slug(slug: str, *, label: str) -> None:
    if not _SLUG_RE.match(slug):
        raise ExtensionSchemaError(f"{label} must match { _SLUG_RE.pattern!r}, got {slug!r}")


def agent_diagnostic_schema_id(agent_slug: str, name: str) -> str:
    _validate_slug(agent_slug, label="agent_slug")
    _validate_slug(name, label="diagnostic name")
    return f"agents.{agent_slug}.diag.{name}"


def application_diagnostic_schema_id(app_slug: str, name: str) -> str:
    _validate_slug(app_slug, label="application_slug")
    _validate_slug(name, label="diagnostic name")
    return f"applications.{app_slug}.diag.{name}"


def assert_agent_diagnostic_schema_id(schema_id: str, *, agent_slug: str) -> None:
    _validate_slug(agent_slug, label="agent_slug")
    expected = f"agents.{agent_slug}.diag."
    if not schema_id.startswith(expected):
        raise ExtensionSchemaError(
            f"agent diagnostic schema_id must start with {expected!r}, got {schema_id!r}"
        )


def assert_application_diagnostic_schema_id(schema_id: str, *, app_slug: str) -> None:
    _validate_slug(app_slug, label="application_slug")
    expected = f"applications.{app_slug}.diag."
    if not schema_id.startswith(expected):
        raise ExtensionSchemaError(
            f"application diagnostic schema_id must start with {expected!r}, got {schema_id!r}"
        )


def assert_extension_runtime_schema_id(schema_id: str) -> None:
    if schema_id.startswith(AGENT_DIAG_PREFIX) or schema_id.startswith(APPLICATION_DIAG_PREFIX):
        return
    raise ExtensionSchemaError(
        "runtime extension payload schema_id must start with "
        f"{AGENT_DIAG_PREFIX!r} or {APPLICATION_DIAG_PREFIX!r}, got {schema_id!r}"
    )


def register_agent_diagnostic_payload(
    schema_cls: type[TDiag],
    *,
    agent_slug: str,
) -> type[TDiag]:
    schema_id = schema_cls.schema_id()
    assert_agent_diagnostic_schema_id(schema_id, agent_slug=agent_slug)
    if schema_id in _DIAGNOSTIC_REGISTRY and _DIAGNOSTIC_REGISTRY[schema_id] is not schema_cls:
        raise ExtensionSchemaError(f"duplicate diagnostic schema_id: {schema_id!r}")
    _DIAGNOSTIC_REGISTRY[schema_id] = schema_cls
    return schema_cls


def register_application_diagnostic_payload(
    schema_cls: type[TDiag],
    *,
    app_slug: str,
) -> type[TDiag]:
    schema_id = schema_cls.schema_id()
    assert_application_diagnostic_schema_id(schema_id, app_slug=app_slug)
    if schema_id in _DIAGNOSTIC_REGISTRY and _DIAGNOSTIC_REGISTRY[schema_id] is not schema_cls:
        raise ExtensionSchemaError(f"duplicate diagnostic schema_id: {schema_id!r}")
    _DIAGNOSTIC_REGISTRY[schema_id] = schema_cls
    return schema_cls


def register_extension_runtime_payload(
    schema_cls: type[TRuntime],
    *,
    event_kind: str | None = None,
) -> type[TRuntime]:
    assert_extension_runtime_schema_id(schema_cls.schema_id)
    registered = register_payload_schema(schema_cls, extension=True)
    if event_kind is not None:
        from intergrax.runtime.events.event_kind_registry import register_event_kind

        register_event_kind(event_kind, schema_cls.schema_id)
    return registered


def get_registered_diagnostic_payload(schema_id: str) -> type[DiagnosticPayload] | None:
    return _DIAGNOSTIC_REGISTRY.get(schema_id)


def list_registered_diagnostic_schema_ids() -> list[str]:
    return sorted(_DIAGNOSTIC_REGISTRY.keys())


class PayloadSchemaRegistry:
    """Developer-facing registry facade for observability extension schemas."""

    @staticmethod
    def register_agent_diagnostic(schema_cls: type[TDiag], *, agent_slug: str) -> type[TDiag]:
        return register_agent_diagnostic_payload(schema_cls, agent_slug=agent_slug)

    @staticmethod
    def register_application_diagnostic(schema_cls: type[TDiag], *, app_slug: str) -> type[TDiag]:
        return register_application_diagnostic_payload(schema_cls, app_slug=app_slug)

    @staticmethod
    def register_runtime_extension(
        schema_cls: type[TRuntime],
        *,
        event_kind: str | None = None,
    ) -> type[TRuntime]:
        return register_extension_runtime_payload(schema_cls, event_kind=event_kind)
