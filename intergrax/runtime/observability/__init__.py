# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Harness observability spine helpers (OBS-BUS-2+)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.runtime.observability.export_attributes import (
    APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA,
    ApplicationObservabilityAttributePolicyResult,
    ApplicationObservabilityAttributes,
    OBSERVABILITY_ARTIFACT_REFERENCE_SCHEMA,
    ObservabilityArtifactReference,
    SANITIZED_APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA,
    SanitizedApplicationObservabilityAttributes,
    ObservabilityAttributeValue,
    observability_attribute_key,
    sanitize_application_observability_attributes,
    sanitized_application_attributes_are_content_safe,
)
from intergrax.runtime.observability.trace_scope import (
    TraceScope,
    TraceScopeState,
    bind_parent_event_id,
    current_parent_event_id,
    current_trace_scope,
)

if TYPE_CHECKING:
    from intergrax.runtime.observability.causal_evidence_persistence import CausalEvidencePersistence
    from intergrax.runtime.observability.emitter import EmittedDiagnostic, ObservabilityEmitter
    from intergrax.runtime.observability.export_boundary import (
        ExportRecordKind,
        ExportStatus,
        GatewayCallExportSource,
        InMemoryObservabilityExporter,
        NoOpObservabilityExporter,
        ObservabilityExportEnvelope,
        ObservabilityExporter,
        RuntimeEventExportSource,
        TestObservabilityExporter,
    )
    from intergrax.runtime.observability.jsonl_exporter import JsonlObservabilityExporter
    from intergrax.runtime.observability.otlp_exporter import (
        OtlpObservabilityExporter,
        OtlpObservabilityExporterConfig,
        OtlpTransport,
    )
    from intergrax.runtime.observability.otlp_http_transport import OtlpHttpTransport
    from intergrax.runtime.observability.problem_reporter import ProblemReporter

__all__ = [
    "CausalEvidencePersistence",
    "ApplicationObservabilityAttributePolicyResult",
    "ApplicationObservabilityAttributes",
    "APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA",
    "ExportRecordKind",
    "ExportStatus",
    "FORBIDDEN_EXPORT_CONTENT_FIELDS",
    "GatewayCallExportSource",
    "InMemoryObservabilityExporter",
    "JsonlObservabilityExporter",
    "NoOpObservabilityExporter",
    "OtlpObservabilityExporter",
    "OtlpObservabilityExporterConfig",
    "OtlpHttpTransport",
    "OtlpTransport",
    "OBSERVABILITY_ARTIFACT_REFERENCE_SCHEMA",
    "OBSERVABILITY_EXPORT_ENVELOPE_SCHEMA",
    "ObservabilityArtifactReference",
    "ObservabilityAttributeValue",
    "ObservabilityExportBackendRegistry",
    "ObservabilityExportBackendRegistryError",
    "ObservabilityExportEnvelope",
    "ObservabilityExportRoute",
    "ObservabilityExportOperatorConfig",
    "ObservabilityExportOperatorConfigError",
    "ObservabilityExporter",
    "ObservabilityFanoutResult",
    "ObservabilityRouteDeliveryResult",
    "OtlpExportOperatorConfig",
    "DEFAULT_OBSERVABILITY_EXPORT_BACKEND_REGISTRY",
    "build_observability_export_integration",
    "build_observability_export_runtime_plugin",
    "build_otlp_observability_export_runtime_plugin",
    "build_otlp_observability_exporter",
    "build_otlp_observability_integration",
    "parse_observability_export_backend_id",
    "problem_signal_export_status",
    "problem_signal_is_content_safe",
    "RuntimeEventExportSource",
    "SANITIZED_APPLICATION_OBSERVABILITY_ATTRIBUTES_SCHEMA",
    "SanitizedApplicationObservabilityAttributes",
    "assert_causal_evidence_persistence_conformance",
    "assert_runtime_event_persistence_conformance",
    "build_journal_export_snapshot",
    "build_journal_ref",
    "build_journal_ref_payload",
    "EmittedDiagnostic",
    "ExportPolicyResult",
    "FanoutObservabilityExporter",
    "envelope_from_gateway_call_source",
    "envelope_from_problem_signal",
    "envelope_from_journal_ref",
    "envelope_from_rag_call",
    "envelope_from_runtime_event",
    "envelope_from_runtime_event_source",
    "envelope_from_tool_call",
    "envelope_is_content_safe",
    "envelope_with_observability_extensions",
    "gateway_call_export_source_from_rag_call",
    "gateway_call_export_source_from_tool_call",
    "runtime_event_export_source_from_event",
    "route_matches_envelope",
    "is_journal_export_enabled",
    "apply_observability_export_policy",
    "JournalExportSnapshot",
    "JournalRef",
    "make_journal_export_runtime_plugin",
    "make_observability_export_runtime_plugin",
    "ObservabilityExportMode",
    "ObservabilityExportPolicy",
    "ObservabilityFieldAction",
    "PLATFORM_PROBLEM_SIGNAL_SCHEMA",
    "PROBLEM_KIND_PLATFORM_ARTIFACT_FAILURE",
    "PROBLEM_KIND_PLATFORM_CONFIGURATION_ERROR",
    "PROBLEM_KIND_PLATFORM_EXCEPTION",
    "PROBLEM_KIND_PLATFORM_INTEGRATION_FAILURE",
    "PROBLEM_KIND_PLATFORM_OBSERVABILITY_EXPORT_FAILURE",
    "PROBLEM_KIND_PLATFORM_POLICY_VIOLATION",
    "PROBLEM_KIND_PLATFORM_RAG_FAILURE",
    "PROBLEM_KIND_PLATFORM_TOOL_FAILURE",
    "PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE",
    "PROBLEM_SEVERITY_CRITICAL",
    "PROBLEM_SEVERITY_ERROR",
    "PROBLEM_SEVERITY_INFO",
    "PROBLEM_SEVERITY_WARNING",
    "PROBLEM_SOURCE_LAYER_AGENT",
    "PROBLEM_SOURCE_LAYER_APPLICATION",
    "PROBLEM_SOURCE_LAYER_INTEGRATION",
    "PROBLEM_SOURCE_LAYER_OBSERVABILITY",
    "PROBLEM_SOURCE_LAYER_POLICY",
    "PROBLEM_SOURCE_LAYER_RAG",
    "PROBLEM_SOURCE_LAYER_RUNTIME",
    "PROBLEM_SOURCE_LAYER_TOOL",
    "PROBLEM_STATUS_DETECTED",
    "PROBLEM_STATUS_IGNORED",
    "PROBLEM_STATUS_RESOLVED",
    "PlatformProblemSignal",
    "ProblemReportContext",
    "ProblemReporter",
    "build_problem_export_envelope",
    "build_problem_signal",
    "report_problem",
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
    "try_export_observability_envelope",
    "TestObservabilityExporter",
    "observability_attribute_key",
    "sanitize_application_observability_attributes",
    "sanitized_application_attributes_are_content_safe",
    "sample_causal_evidence",
    "sample_runtime_event",
    "serialize_runtime_event",
    "TraceScopeState",
    "bind_parent_event_id",
    "current_parent_event_id",
    "current_trace_scope",
]


def __getattr__(name: str):
    if name == "CausalEvidencePersistence":
        from intergrax.runtime.observability.causal_evidence_persistence import CausalEvidencePersistence

        return CausalEvidencePersistence
    if name in {
        "FORBIDDEN_EXPORT_CONTENT_FIELDS",
        "ExportRecordKind",
        "ExportStatus",
        "GatewayCallExportSource",
        "InMemoryObservabilityExporter",
        "NoOpObservabilityExporter",
        "OBSERVABILITY_EXPORT_ENVELOPE_SCHEMA",
        "ObservabilityExportEnvelope",
        "ObservabilityExporter",
        "RuntimeEventExportSource",
        "TestObservabilityExporter",
        "envelope_from_gateway_call_source",
        "envelope_from_journal_ref",
        "envelope_from_rag_call",
        "envelope_from_runtime_event",
        "envelope_from_runtime_event_source",
        "envelope_from_tool_call",
        "envelope_is_content_safe",
        "envelope_with_observability_extensions",
        "gateway_call_export_source_from_rag_call",
        "gateway_call_export_source_from_tool_call",
        "runtime_event_export_source_from_event",
    }:
        from intergrax.runtime.observability.export_boundary import (
            FORBIDDEN_EXPORT_CONTENT_FIELDS,
            ExportRecordKind,
            ExportStatus,
            GatewayCallExportSource,
            InMemoryObservabilityExporter,
            NoOpObservabilityExporter,
            OBSERVABILITY_EXPORT_ENVELOPE_SCHEMA,
            ObservabilityExportEnvelope,
            ObservabilityExporter,
            RuntimeEventExportSource,
            TestObservabilityExporter,
            envelope_from_gateway_call_source,
            envelope_from_journal_ref,
            envelope_from_rag_call,
            envelope_from_runtime_event,
            envelope_from_runtime_event_source,
            envelope_from_tool_call,
            envelope_is_content_safe,
            envelope_with_observability_extensions,
            gateway_call_export_source_from_rag_call,
            gateway_call_export_source_from_tool_call,
            runtime_event_export_source_from_event,
        )

        return locals()[name]
    if name in {
        "is_journal_export_enabled",
        "make_journal_export_runtime_plugin",
        "register_journal_export_plugin",
    }:
        from intergrax.runtime.observability.export_bridge import (
            is_journal_export_enabled,
            make_journal_export_runtime_plugin,
            register_journal_export_plugin,
        )

        return locals()[name]
    if name in {
        "ExportPolicyResult",
        "ObservabilityExportMode",
        "ObservabilityExportPolicy",
        "ObservabilityFieldAction",
        "apply_observability_export_policy",
        "try_export_observability_envelope",
    }:
        from intergrax.runtime.observability.export_policy import (
            ExportPolicyResult,
            ObservabilityExportMode,
            ObservabilityExportPolicy,
            ObservabilityFieldAction,
            apply_observability_export_policy,
            try_export_observability_envelope,
        )

        return locals()[name]
    if name in {
        "FanoutObservabilityExporter",
        "ObservabilityExportRoute",
        "ObservabilityFanoutResult",
        "ObservabilityRouteDeliveryResult",
        "route_matches_envelope",
    }:
        from intergrax.runtime.observability.export_routing import (
            FanoutObservabilityExporter,
            ObservabilityExportRoute,
            ObservabilityFanoutResult,
            ObservabilityRouteDeliveryResult,
            route_matches_envelope,
        )

        return locals()[name]
    if name == "make_observability_export_runtime_plugin":
        from intergrax.runtime.observability.export_wiring import make_observability_export_runtime_plugin

        return make_observability_export_runtime_plugin
    if name == "JsonlObservabilityExporter":
        from intergrax.runtime.observability.jsonl_exporter import JsonlObservabilityExporter

        return JsonlObservabilityExporter
    if name in {"OtlpObservabilityExporter", "OtlpObservabilityExporterConfig", "OtlpTransport"}:
        from intergrax.runtime.observability.otlp_exporter import (
            OtlpObservabilityExporter,
            OtlpObservabilityExporterConfig,
            OtlpTransport,
        )

        return locals()[name]
    if name in {"envelope_from_problem_signal", "problem_signal_export_status"}:
        from intergrax.runtime.observability.problem_export import (
            envelope_from_problem_signal,
            problem_signal_export_status,
        )

        return locals()[name]
    if name in {
        "ProblemReportContext",
        "ProblemReporter",
        "build_problem_export_envelope",
        "build_problem_signal",
        "report_problem",
    }:
        from intergrax.runtime.observability.problem_reporter import (
            ProblemReportContext,
            ProblemReporter,
            build_problem_export_envelope,
            build_problem_signal,
            report_problem,
        )

        return locals()[name]
    if name in {
        "PLATFORM_PROBLEM_SIGNAL_SCHEMA",
        "PROBLEM_KIND_PLATFORM_ARTIFACT_FAILURE",
        "PROBLEM_KIND_PLATFORM_CONFIGURATION_ERROR",
        "PROBLEM_KIND_PLATFORM_EXCEPTION",
        "PROBLEM_KIND_PLATFORM_INTEGRATION_FAILURE",
        "PROBLEM_KIND_PLATFORM_OBSERVABILITY_EXPORT_FAILURE",
        "PROBLEM_KIND_PLATFORM_POLICY_VIOLATION",
        "PROBLEM_KIND_PLATFORM_RAG_FAILURE",
        "PROBLEM_KIND_PLATFORM_TOOL_FAILURE",
        "PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE",
        "PROBLEM_SEVERITY_CRITICAL",
        "PROBLEM_SEVERITY_ERROR",
        "PROBLEM_SEVERITY_INFO",
        "PROBLEM_SEVERITY_WARNING",
        "PROBLEM_SOURCE_LAYER_AGENT",
        "PROBLEM_SOURCE_LAYER_APPLICATION",
        "PROBLEM_SOURCE_LAYER_INTEGRATION",
        "PROBLEM_SOURCE_LAYER_OBSERVABILITY",
        "PROBLEM_SOURCE_LAYER_POLICY",
        "PROBLEM_SOURCE_LAYER_RAG",
        "PROBLEM_SOURCE_LAYER_RUNTIME",
        "PROBLEM_SOURCE_LAYER_TOOL",
        "PROBLEM_STATUS_DETECTED",
        "PROBLEM_STATUS_IGNORED",
        "PROBLEM_STATUS_RESOLVED",
        "PlatformProblemSignal",
        "problem_signal_is_content_safe",
    }:
        from intergrax.runtime.observability.problem_signal import (
            PLATFORM_PROBLEM_SIGNAL_SCHEMA,
            PROBLEM_KIND_PLATFORM_ARTIFACT_FAILURE,
            PROBLEM_KIND_PLATFORM_CONFIGURATION_ERROR,
            PROBLEM_KIND_PLATFORM_EXCEPTION,
            PROBLEM_KIND_PLATFORM_INTEGRATION_FAILURE,
            PROBLEM_KIND_PLATFORM_OBSERVABILITY_EXPORT_FAILURE,
            PROBLEM_KIND_PLATFORM_POLICY_VIOLATION,
            PROBLEM_KIND_PLATFORM_RAG_FAILURE,
            PROBLEM_KIND_PLATFORM_TOOL_FAILURE,
            PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE,
            PROBLEM_SEVERITY_CRITICAL,
            PROBLEM_SEVERITY_ERROR,
            PROBLEM_SEVERITY_INFO,
            PROBLEM_SEVERITY_WARNING,
            PROBLEM_SOURCE_LAYER_AGENT,
            PROBLEM_SOURCE_LAYER_APPLICATION,
            PROBLEM_SOURCE_LAYER_INTEGRATION,
            PROBLEM_SOURCE_LAYER_OBSERVABILITY,
            PROBLEM_SOURCE_LAYER_POLICY,
            PROBLEM_SOURCE_LAYER_RAG,
            PROBLEM_SOURCE_LAYER_RUNTIME,
            PROBLEM_SOURCE_LAYER_TOOL,
            PROBLEM_STATUS_DETECTED,
            PROBLEM_STATUS_IGNORED,
            PROBLEM_STATUS_RESOLVED,
            PlatformProblemSignal,
            problem_signal_is_content_safe,
        )

        return locals()[name]
    if name in {
        "DEFAULT_OBSERVABILITY_EXPORT_BACKEND_REGISTRY",
        "ObservabilityExportBackendRegistry",
        "ObservabilityExportBackendRegistryError",
        "ObservabilityExportOperatorConfig",
        "ObservabilityExportOperatorConfigError",
        "OtlpExportOperatorConfig",
        "build_observability_export_integration",
        "build_observability_export_runtime_plugin",
        "build_otlp_observability_export_runtime_plugin",
        "build_otlp_observability_exporter",
        "build_otlp_observability_integration",
        "parse_observability_export_backend_id",
    }:
        from intergrax.runtime.observability.operator_wiring import (
            DEFAULT_OBSERVABILITY_EXPORT_BACKEND_REGISTRY,
            ObservabilityExportBackendRegistry,
            ObservabilityExportBackendRegistryError,
            ObservabilityExportOperatorConfig,
            ObservabilityExportOperatorConfigError,
            OtlpExportOperatorConfig,
            build_observability_export_integration,
            build_observability_export_runtime_plugin,
            build_otlp_observability_export_runtime_plugin,
            build_otlp_observability_exporter,
            build_otlp_observability_integration,
            parse_observability_export_backend_id,
        )

        return locals()[name]
    if name == "OtlpHttpTransport":
        from intergrax.runtime.observability.otlp_http_transport import OtlpHttpTransport

        return OtlpHttpTransport
    if name in {"EmittedDiagnostic", "ObservabilityEmitter"}:
        from intergrax.runtime.observability.emitter import EmittedDiagnostic, ObservabilityEmitter

        if name == "EmittedDiagnostic":
            return EmittedDiagnostic
        return ObservabilityEmitter
    if name in {
        "JournalExportSnapshot",
        "JournalRef",
        "build_journal_export_snapshot",
        "build_journal_ref",
        "build_journal_ref_payload",
        "render_journal_otlp_json",
        "serialize_runtime_event",
    }:
        from intergrax.runtime.observability.journal_export import (
            JournalExportSnapshot,
            JournalRef,
            build_journal_export_snapshot,
            build_journal_ref,
            build_journal_ref_payload,
            render_journal_otlp_json,
            serialize_runtime_event,
        )

        return locals()[name]
    if name in {
        "assert_causal_evidence_persistence_conformance",
        "assert_runtime_event_persistence_conformance",
        "sample_causal_evidence",
        "sample_runtime_event",
    }:
        from intergrax.runtime.observability.persistence_conformance import (
            assert_causal_evidence_persistence_conformance,
            assert_runtime_event_persistence_conformance,
            sample_causal_evidence,
            sample_runtime_event,
        )

        return locals()[name]
    if name in {
        "ExtensionSchemaError",
        "PayloadSchemaRegistry",
        "agent_diagnostic_schema_id",
        "application_diagnostic_schema_id",
        "get_registered_diagnostic_payload",
        "list_registered_diagnostic_schema_ids",
        "register_agent_diagnostic_payload",
        "register_application_diagnostic_payload",
        "register_extension_runtime_payload",
    }:
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

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
