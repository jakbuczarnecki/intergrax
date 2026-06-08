# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Harness observability spine helpers (OBS-BUS-2+)."""

from intergrax.runtime.observability.emitter import EmittedDiagnostic, ObservabilityEmitter
from intergrax.runtime.observability.persistence_conformance import (
    assert_runtime_event_persistence_conformance,
    sample_runtime_event,
)
from intergrax.runtime.observability.extension_sdk import (
    ExtensionSchemaError,
    PayloadSchemaRegistry,
    agent_diagnostic_schema_id,
    application_diagnostic_schema_id,
    get_registered_diagnostic_payload,
    list_registered_diagnostic_schema_ids,
    register_agent_diagnostic_payload,
    register_application_diagnostic_payload,
    register_extension_runtime_payload,
)
from intergrax.runtime.observability.trace_scope import (
    TraceScope,
    TraceScopeState,
    bind_parent_event_id,
    current_parent_event_id,
    current_trace_scope,
)

__all__ = [
    "assert_runtime_event_persistence_conformance",
    "EmittedDiagnostic",
    "ExtensionSchemaError",
    "ObservabilityEmitter",
    "PayloadSchemaRegistry",
    "TraceScope",
    "agent_diagnostic_schema_id",
    "application_diagnostic_schema_id",
    "get_registered_diagnostic_payload",
    "list_registered_diagnostic_schema_ids",
    "register_agent_diagnostic_payload",
    "register_application_diagnostic_payload",
    "register_extension_runtime_payload",
    "sample_runtime_event",
    "TraceScopeState",
    "bind_parent_event_id",
    "current_parent_event_id",
    "current_trace_scope",
]
