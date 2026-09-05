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
    if name == "FORBIDDEN_EXPORT_CONTENT_FIELDS":
        from intergrax.runtime.observability.export_boundary import FORBIDDEN_EXPORT_CONTENT_FIELDS

        return FORBIDDEN_EXPORT_CONTENT_FIELDS
    if name == "ExportRecordKind":
        from intergrax.runtime.observability.export_boundary import ExportRecordKind

        return ExportRecordKind
    if name == "ExportStatus":
        from intergrax.runtime.observability.export_boundary import ExportStatus

        return ExportStatus
    if name == "GatewayCallExportSource":
        from intergrax.runtime.observability.export_boundary import GatewayCallExportSource

        return GatewayCallExportSource
    if name == "InMemoryObservabilityExporter":
        from intergrax.runtime.observability.export_boundary import InMemoryObservabilityExporter

        return InMemoryObservabilityExporter
    if name == "NoOpObservabilityExporter":
        from intergrax.runtime.observability.export_boundary import NoOpObservabilityExporter

        return NoOpObservabilityExporter
    if name == "OBSERVABILITY_EXPORT_ENVELOPE_SCHEMA":
        from intergrax.runtime.observability.export_boundary import OBSERVABILITY_EXPORT_ENVELOPE_SCHEMA

        return OBSERVABILITY_EXPORT_ENVELOPE_SCHEMA
    if name == "ObservabilityExportEnvelope":
        from intergrax.runtime.observability.export_boundary import ObservabilityExportEnvelope

        return ObservabilityExportEnvelope
    if name == "ObservabilityExporter":
        from intergrax.runtime.observability.export_boundary import ObservabilityExporter

        return ObservabilityExporter
    if name == "RuntimeEventExportSource":
        from intergrax.runtime.observability.export_boundary import RuntimeEventExportSource

        return RuntimeEventExportSource
    if name == "TestObservabilityExporter":
        from intergrax.runtime.observability.export_boundary import TestObservabilityExporter

        return TestObservabilityExporter
    if name == "envelope_from_gateway_call_source":
        from intergrax.runtime.observability.export_boundary import envelope_from_gateway_call_source

        return envelope_from_gateway_call_source
    if name == "envelope_from_journal_ref":
        from intergrax.runtime.observability.export_boundary import envelope_from_journal_ref

        return envelope_from_journal_ref
    if name == "envelope_from_rag_call":
        from intergrax.runtime.observability.export_boundary import envelope_from_rag_call

        return envelope_from_rag_call
    if name == "envelope_from_runtime_event":
        from intergrax.runtime.observability.export_boundary import envelope_from_runtime_event

        return envelope_from_runtime_event
    if name == "envelope_from_runtime_event_source":
        from intergrax.runtime.observability.export_boundary import envelope_from_runtime_event_source

        return envelope_from_runtime_event_source
    if name == "envelope_from_tool_call":
        from intergrax.runtime.observability.export_boundary import envelope_from_tool_call

        return envelope_from_tool_call
    if name == "envelope_is_content_safe":
        from intergrax.runtime.observability.export_boundary import envelope_is_content_safe

        return envelope_is_content_safe
    if name == "envelope_with_observability_extensions":
        from intergrax.runtime.observability.export_boundary import envelope_with_observability_extensions

        return envelope_with_observability_extensions
    if name == "gateway_call_export_source_from_rag_call":
        from intergrax.runtime.observability.export_boundary import gateway_call_export_source_from_rag_call

        return gateway_call_export_source_from_rag_call
    if name == "gateway_call_export_source_from_tool_call":
        from intergrax.runtime.observability.export_boundary import gateway_call_export_source_from_tool_call

        return gateway_call_export_source_from_tool_call
    if name == "runtime_event_export_source_from_event":
        from intergrax.runtime.observability.export_boundary import runtime_event_export_source_from_event

        return runtime_event_export_source_from_event
    if name == "is_journal_export_enabled":
        from intergrax.runtime.observability.export_bridge import is_journal_export_enabled

        return is_journal_export_enabled
    if name == "make_journal_export_runtime_plugin":
        from intergrax.runtime.observability.export_bridge import make_journal_export_runtime_plugin

        return make_journal_export_runtime_plugin
    if name == "register_journal_export_plugin":
        from intergrax.runtime.observability.export_bridge import register_journal_export_plugin

        return register_journal_export_plugin
    if name == "ExportPolicyResult":
        from intergrax.runtime.observability.export_policy import ExportPolicyResult

        return ExportPolicyResult
    if name == "ObservabilityExportMode":
        from intergrax.runtime.observability.export_policy import ObservabilityExportMode

        return ObservabilityExportMode
    if name == "ObservabilityExportPolicy":
        from intergrax.runtime.observability.export_policy import ObservabilityExportPolicy

        return ObservabilityExportPolicy
    if name == "ObservabilityFieldAction":
        from intergrax.runtime.observability.export_policy import ObservabilityFieldAction

        return ObservabilityFieldAction
    if name == "apply_observability_export_policy":
        from intergrax.runtime.observability.export_policy import apply_observability_export_policy

        return apply_observability_export_policy
    if name == "try_export_observability_envelope":
        from intergrax.runtime.observability.export_policy import try_export_observability_envelope

        return try_export_observability_envelope
    if name == "FanoutObservabilityExporter":
        from intergrax.runtime.observability.export_routing import FanoutObservabilityExporter

        return FanoutObservabilityExporter
    if name == "ObservabilityExportRoute":
        from intergrax.runtime.observability.export_routing import ObservabilityExportRoute

        return ObservabilityExportRoute
    if name == "ObservabilityFanoutResult":
        from intergrax.runtime.observability.export_routing import ObservabilityFanoutResult

        return ObservabilityFanoutResult
    if name == "ObservabilityRouteDeliveryResult":
        from intergrax.runtime.observability.export_routing import ObservabilityRouteDeliveryResult

        return ObservabilityRouteDeliveryResult
    if name == "route_matches_envelope":
        from intergrax.runtime.observability.export_routing import route_matches_envelope

        return route_matches_envelope
    if name == "OtlpObservabilityExporter":
        from intergrax.runtime.observability.otlp_exporter import OtlpObservabilityExporter

        return OtlpObservabilityExporter
    if name == "OtlpObservabilityExporterConfig":
        from intergrax.runtime.observability.otlp_exporter import OtlpObservabilityExporterConfig

        return OtlpObservabilityExporterConfig
    if name == "OtlpTransport":
        from intergrax.runtime.observability.otlp_exporter import OtlpTransport

        return OtlpTransport
    if name == "envelope_from_problem_signal":
        from intergrax.runtime.observability.problem_export import envelope_from_problem_signal

        return envelope_from_problem_signal
    if name == "problem_signal_export_status":
        from intergrax.runtime.observability.problem_export import problem_signal_export_status

        return problem_signal_export_status
    if name == "ProblemReportContext":
        from intergrax.runtime.observability.problem_reporter import ProblemReportContext

        return ProblemReportContext
    if name == "ProblemReporter":
        from intergrax.runtime.observability.problem_reporter import ProblemReporter

        return ProblemReporter
    if name == "build_problem_export_envelope":
        from intergrax.runtime.observability.problem_reporter import build_problem_export_envelope

        return build_problem_export_envelope
    if name == "build_problem_signal":
        from intergrax.runtime.observability.problem_reporter import build_problem_signal

        return build_problem_signal
    if name == "report_problem":
        from intergrax.runtime.observability.problem_reporter import report_problem

        return report_problem
    if name == "PLATFORM_PROBLEM_SIGNAL_SCHEMA":
        from intergrax.runtime.observability.problem_signal import PLATFORM_PROBLEM_SIGNAL_SCHEMA

        return PLATFORM_PROBLEM_SIGNAL_SCHEMA
    if name == "PROBLEM_KIND_PLATFORM_ARTIFACT_FAILURE":
        from intergrax.runtime.observability.problem_signal import PROBLEM_KIND_PLATFORM_ARTIFACT_FAILURE

        return PROBLEM_KIND_PLATFORM_ARTIFACT_FAILURE
    if name == "PROBLEM_KIND_PLATFORM_CONFIGURATION_ERROR":
        from intergrax.runtime.observability.problem_signal import PROBLEM_KIND_PLATFORM_CONFIGURATION_ERROR

        return PROBLEM_KIND_PLATFORM_CONFIGURATION_ERROR
    if name == "PROBLEM_KIND_PLATFORM_EXCEPTION":
        from intergrax.runtime.observability.problem_signal import PROBLEM_KIND_PLATFORM_EXCEPTION

        return PROBLEM_KIND_PLATFORM_EXCEPTION
    if name == "PROBLEM_KIND_PLATFORM_INTEGRATION_FAILURE":
        from intergrax.runtime.observability.problem_signal import PROBLEM_KIND_PLATFORM_INTEGRATION_FAILURE

        return PROBLEM_KIND_PLATFORM_INTEGRATION_FAILURE
    if name == "PROBLEM_KIND_PLATFORM_OBSERVABILITY_EXPORT_FAILURE":
        from intergrax.runtime.observability.problem_signal import PROBLEM_KIND_PLATFORM_OBSERVABILITY_EXPORT_FAILURE

        return PROBLEM_KIND_PLATFORM_OBSERVABILITY_EXPORT_FAILURE
    if name == "PROBLEM_KIND_PLATFORM_POLICY_VIOLATION":
        from intergrax.runtime.observability.problem_signal import PROBLEM_KIND_PLATFORM_POLICY_VIOLATION

        return PROBLEM_KIND_PLATFORM_POLICY_VIOLATION
    if name == "PROBLEM_KIND_PLATFORM_RAG_FAILURE":
        from intergrax.runtime.observability.problem_signal import PROBLEM_KIND_PLATFORM_RAG_FAILURE

        return PROBLEM_KIND_PLATFORM_RAG_FAILURE
    if name == "PROBLEM_KIND_PLATFORM_TOOL_FAILURE":
        from intergrax.runtime.observability.problem_signal import PROBLEM_KIND_PLATFORM_TOOL_FAILURE

        return PROBLEM_KIND_PLATFORM_TOOL_FAILURE
    if name == "PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE":
        from intergrax.runtime.observability.problem_signal import PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE

        return PROBLEM_KIND_PLATFORM_UNEXPECTED_STATE
    if name == "PROBLEM_SEVERITY_CRITICAL":
        from intergrax.runtime.observability.problem_signal import PROBLEM_SEVERITY_CRITICAL

        return PROBLEM_SEVERITY_CRITICAL
    if name == "PROBLEM_SEVERITY_ERROR":
        from intergrax.runtime.observability.problem_signal import PROBLEM_SEVERITY_ERROR

        return PROBLEM_SEVERITY_ERROR
    if name == "PROBLEM_SEVERITY_INFO":
        from intergrax.runtime.observability.problem_signal import PROBLEM_SEVERITY_INFO

        return PROBLEM_SEVERITY_INFO
    if name == "PROBLEM_SEVERITY_WARNING":
        from intergrax.runtime.observability.problem_signal import PROBLEM_SEVERITY_WARNING

        return PROBLEM_SEVERITY_WARNING
    if name == "PROBLEM_SOURCE_LAYER_AGENT":
        from intergrax.runtime.observability.problem_signal import PROBLEM_SOURCE_LAYER_AGENT

        return PROBLEM_SOURCE_LAYER_AGENT
    if name == "PROBLEM_SOURCE_LAYER_APPLICATION":
        from intergrax.runtime.observability.problem_signal import PROBLEM_SOURCE_LAYER_APPLICATION

        return PROBLEM_SOURCE_LAYER_APPLICATION
    if name == "PROBLEM_SOURCE_LAYER_INTEGRATION":
        from intergrax.runtime.observability.problem_signal import PROBLEM_SOURCE_LAYER_INTEGRATION

        return PROBLEM_SOURCE_LAYER_INTEGRATION
    if name == "PROBLEM_SOURCE_LAYER_OBSERVABILITY":
        from intergrax.runtime.observability.problem_signal import PROBLEM_SOURCE_LAYER_OBSERVABILITY

        return PROBLEM_SOURCE_LAYER_OBSERVABILITY
    if name == "PROBLEM_SOURCE_LAYER_POLICY":
        from intergrax.runtime.observability.problem_signal import PROBLEM_SOURCE_LAYER_POLICY

        return PROBLEM_SOURCE_LAYER_POLICY
    if name == "PROBLEM_SOURCE_LAYER_RAG":
        from intergrax.runtime.observability.problem_signal import PROBLEM_SOURCE_LAYER_RAG

        return PROBLEM_SOURCE_LAYER_RAG
    if name == "PROBLEM_SOURCE_LAYER_RUNTIME":
        from intergrax.runtime.observability.problem_signal import PROBLEM_SOURCE_LAYER_RUNTIME

        return PROBLEM_SOURCE_LAYER_RUNTIME
    if name == "PROBLEM_SOURCE_LAYER_TOOL":
        from intergrax.runtime.observability.problem_signal import PROBLEM_SOURCE_LAYER_TOOL

        return PROBLEM_SOURCE_LAYER_TOOL
    if name == "PROBLEM_STATUS_DETECTED":
        from intergrax.runtime.observability.problem_signal import PROBLEM_STATUS_DETECTED

        return PROBLEM_STATUS_DETECTED
    if name == "PROBLEM_STATUS_IGNORED":
        from intergrax.runtime.observability.problem_signal import PROBLEM_STATUS_IGNORED

        return PROBLEM_STATUS_IGNORED
    if name == "PROBLEM_STATUS_RESOLVED":
        from intergrax.runtime.observability.problem_signal import PROBLEM_STATUS_RESOLVED

        return PROBLEM_STATUS_RESOLVED
    if name == "PlatformProblemSignal":
        from intergrax.runtime.observability.problem_signal import PlatformProblemSignal

        return PlatformProblemSignal
    if name == "problem_signal_is_content_safe":
        from intergrax.runtime.observability.problem_signal import problem_signal_is_content_safe

        return problem_signal_is_content_safe
    if name == "DEFAULT_OBSERVABILITY_EXPORT_BACKEND_REGISTRY":
        from intergrax.runtime.observability.operator_wiring import DEFAULT_OBSERVABILITY_EXPORT_BACKEND_REGISTRY

        return DEFAULT_OBSERVABILITY_EXPORT_BACKEND_REGISTRY
    if name == "ObservabilityExportBackendRegistry":
        from intergrax.runtime.observability.operator_wiring import ObservabilityExportBackendRegistry

        return ObservabilityExportBackendRegistry
    if name == "ObservabilityExportBackendRegistryError":
        from intergrax.runtime.observability.operator_wiring import ObservabilityExportBackendRegistryError

        return ObservabilityExportBackendRegistryError
    if name == "ObservabilityExportOperatorConfig":
        from intergrax.runtime.observability.operator_wiring import ObservabilityExportOperatorConfig

        return ObservabilityExportOperatorConfig
    if name == "ObservabilityExportOperatorConfigError":
        from intergrax.runtime.observability.operator_wiring import ObservabilityExportOperatorConfigError

        return ObservabilityExportOperatorConfigError
    if name == "OtlpExportOperatorConfig":
        from intergrax.runtime.observability.operator_wiring import OtlpExportOperatorConfig

        return OtlpExportOperatorConfig
    if name == "build_observability_export_integration":
        from intergrax.runtime.observability.operator_wiring import build_observability_export_integration

        return build_observability_export_integration
    if name == "build_observability_export_runtime_plugin":
        from intergrax.runtime.observability.operator_wiring import build_observability_export_runtime_plugin

        return build_observability_export_runtime_plugin
    if name == "build_otlp_observability_export_runtime_plugin":
        from intergrax.runtime.observability.operator_wiring import build_otlp_observability_export_runtime_plugin

        return build_otlp_observability_export_runtime_plugin
    if name == "build_otlp_observability_exporter":
        from intergrax.runtime.observability.operator_wiring import build_otlp_observability_exporter

        return build_otlp_observability_exporter
    if name == "build_otlp_observability_integration":
        from intergrax.runtime.observability.operator_wiring import build_otlp_observability_integration

        return build_otlp_observability_integration
    if name == "parse_observability_export_backend_id":
        from intergrax.runtime.observability.operator_wiring import parse_observability_export_backend_id

        return parse_observability_export_backend_id
    if name == "JournalExportSnapshot":
        from intergrax.runtime.observability.journal_export import JournalExportSnapshot

        return JournalExportSnapshot
    if name == "JournalRef":
        from intergrax.runtime.observability.journal_export import JournalRef

        return JournalRef
    if name == "build_journal_export_snapshot":
        from intergrax.runtime.observability.journal_export import build_journal_export_snapshot

        return build_journal_export_snapshot
    if name == "build_journal_ref":
        from intergrax.runtime.observability.journal_export import build_journal_ref

        return build_journal_ref
    if name == "build_journal_ref_payload":
        from intergrax.runtime.observability.journal_export import build_journal_ref_payload

        return build_journal_ref_payload
    if name == "render_journal_otlp_json":
        from intergrax.runtime.observability.journal_export import render_journal_otlp_json

        return render_journal_otlp_json
    if name == "serialize_runtime_event":
        from intergrax.runtime.observability.journal_export import serialize_runtime_event

        return serialize_runtime_event
    if name == "assert_causal_evidence_persistence_conformance":
        from intergrax.runtime.observability.persistence_conformance import assert_causal_evidence_persistence_conformance

        return assert_causal_evidence_persistence_conformance
    if name == "assert_runtime_event_persistence_conformance":
        from intergrax.runtime.observability.persistence_conformance import assert_runtime_event_persistence_conformance

        return assert_runtime_event_persistence_conformance
    if name == "sample_causal_evidence":
        from intergrax.runtime.observability.persistence_conformance import sample_causal_evidence

        return sample_causal_evidence
    if name == "sample_runtime_event":
        from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

        return sample_runtime_event
    if name == "ExtensionSchemaError":
        from intergrax.runtime.observability.extension_sdk import ExtensionSchemaError

        return ExtensionSchemaError
    if name == "PayloadSchemaRegistry":
        from intergrax.runtime.observability.extension_sdk import PayloadSchemaRegistry

        return PayloadSchemaRegistry
    if name == "agent_diagnostic_schema_id":
        from intergrax.runtime.observability.extension_sdk import agent_diagnostic_schema_id

        return agent_diagnostic_schema_id
    if name == "application_diagnostic_schema_id":
        from intergrax.runtime.observability.extension_sdk import application_diagnostic_schema_id

        return application_diagnostic_schema_id
    if name == "get_registered_diagnostic_payload":
        from intergrax.runtime.observability.extension_sdk import get_registered_diagnostic_payload

        return get_registered_diagnostic_payload
    if name == "list_registered_diagnostic_schema_ids":
        from intergrax.runtime.observability.extension_sdk import list_registered_diagnostic_schema_ids

        return list_registered_diagnostic_schema_ids
    if name == "register_agent_diagnostic_payload":
        from intergrax.runtime.observability.extension_sdk import register_agent_diagnostic_payload

        return register_agent_diagnostic_payload
    if name == "register_application_diagnostic_payload":
        from intergrax.runtime.observability.extension_sdk import register_application_diagnostic_payload

        return register_application_diagnostic_payload
    if name == "register_extension_runtime_payload":
        from intergrax.runtime.observability.extension_sdk import register_extension_runtime_payload

        return register_extension_runtime_payload
    if name == "make_observability_export_runtime_plugin":
        from intergrax.runtime.observability.export_wiring import make_observability_export_runtime_plugin

        return make_observability_export_runtime_plugin
    if name == "JsonlObservabilityExporter":
        from intergrax.runtime.observability.jsonl_exporter import JsonlObservabilityExporter

        return JsonlObservabilityExporter
    if name == "OtlpHttpTransport":
        from intergrax.runtime.observability.otlp_http_transport import OtlpHttpTransport

        return OtlpHttpTransport
    if name == "EmittedDiagnostic":
        from intergrax.runtime.observability.emitter import EmittedDiagnostic

        return EmittedDiagnostic
    if name == "ObservabilityEmitter":
        from intergrax.runtime.observability.emitter import ObservabilityEmitter

        return ObservabilityEmitter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
