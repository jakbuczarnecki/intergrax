# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Harness observability spine helpers (OBS-BUS-2+)."""

from intergrax.runtime.observability.export_bridge import (
    is_journal_export_enabled,
    make_journal_export_runtime_plugin,
    register_journal_export_plugin,
)
from intergrax.runtime.observability.emitter import EmittedDiagnostic, ObservabilityEmitter
from intergrax.runtime.observability.journal_export import (
    JournalExportSnapshot,
    JournalRef,
    build_journal_export_snapshot,
    build_journal_ref,
    build_journal_ref_payload,
    render_journal_otlp_json,
    serialize_runtime_event,
)
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
    "build_journal_export_snapshot",
    "build_journal_ref",
    "build_journal_ref_payload",
    "EmittedDiagnostic",
    "is_journal_export_enabled",
    "JournalExportSnapshot",
    "JournalRef",
    "make_journal_export_runtime_plugin",
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
    "register_journal_export_plugin",
    "render_journal_otlp_json",
    "sample_runtime_event",
    "serialize_runtime_event",
    "TraceScopeState",
    "bind_parent_event_id",
    "current_parent_event_id",
    "current_trace_scope",
]
