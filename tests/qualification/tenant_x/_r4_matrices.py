# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R4 required qualification matrices."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._r4_mechanisms import build_r4_families
from tests.qualification.tenant_x._r4_types import (
    R4AttributionTenantOptionalClass,
    R4ChainVerdict,
    R4ContextAssemblyPathClass,
    R4ContextTenantSourceClass,
    R4CrossTenantClass,
    R4EventTenantSourceClass,
    R4ExtensionTenantClass,
    R4LlmCallTenantParamClass,
    R4MatrixRow,
    R4ObservabilityRole,
    R4SubstitutionClass,
)

CONTEXT_MODEL_CONTINUITY_MATRIX: Final[tuple[R4MatrixRow, ...]] = (
    R4MatrixRow(
        row_id="context_assembled_event",
        surface_hint="context_skill_recording.py::record_context_assembled_from_engine",
        verdict=R4ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("RuntimeEventType.CONTEXT_ASSEMBLED + tenant_id from active scope",),
        tenant_expression="peek_active_runtime_event_tenant_id",
    ),
    R4MatrixRow(
        row_id="llm_call_event",
        surface_hint="record_llm_call_runtime_event",
        verdict=R4ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("LLM_CALL carries tenant_id + context_assembly_event_id",),
        tenant_expression="llm_event.tenant_id",
    ),
    R4MatrixRow(
        row_id="model_context_attribution",
        surface_hint="model_context_attribution.py::try_attribute_model_call_to_context",
        verdict=R4ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("tenant_mismatch → not attributable", "evidence.tenant_id = llm_event.tenant_id"),
        tenant_expression="context_event.tenant_id == llm_event.tenant_id",
    ),
)

CONTEXT_VIEW_ISOLATION_MATRIX: Final[tuple[R4MatrixRow, ...]] = (
    R4MatrixRow(
        row_id="request.scope.tenant_id",
        surface_hint="ContextViewRequest._validate_scope_and_authority",
        verdict=R4ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("authority_request.tenant_id must match scope",),
    ),
    R4MatrixRow(
        row_id="membership.tenant_id",
        surface_hint="ContextViewRequest membership alignment",
        verdict=R4ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("membership tenant_id must match scope tenant_id",),
    ),
    R4MatrixRow(
        row_id="delegation.tenant_id",
        surface_hint="ContextViewRequest delegation alignment",
        verdict=R4ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("delegation tenant_id must match scope tenant_id",),
    ),
    R4MatrixRow(
        row_id="entry.entry_scope.tenant_id",
        surface_hint="validate_context_view_matches_composition_request",
        verdict=R4ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("entry tenant_id must match view scope tenant_id",),
    ),
    R4MatrixRow(
        row_id="source_ref.tenant_id",
        surface_hint="context_view_source_ref_identity_key",
        verdict=R4ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("typed source_ref includes tenant_id in identity key",),
    ),
)

EVIDENCE_TENANT_CONTINUITY_MATRIX: Final[tuple[R4MatrixRow, ...]] = (
    R4MatrixRow(
        row_id="functional_evidence_exec_ctx",
        surface_hint="require_tenant_id_from_exec_ctx",
        verdict=R4ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("canonical_request_identity.tenant_id required",),
        tenant_expression="exec_ctx.canonical_request_identity.tenant_id",
    ),
    R4MatrixRow(
        row_id="model_attribution_evidence",
        surface_hint="ModelContextAttributionEvidence",
        verdict=R4ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("emitted only after tenant equality gate",),
        tenant_expression="llm_event.tenant_id",
    ),
)

EVENT_TRACE_CONTINUITY_MATRIX: Final[tuple[R4MatrixRow, ...]] = (
    R4MatrixRow(
        row_id="event.context_assembled",
        surface_hint="RuntimeEventType.CONTEXT_ASSEMBLED",
        verdict=R4EventTenantSourceClass.VALIDATED_FROM_BOUND_CONTEXT,
        evidence=("tenant_id on RuntimeEvent envelope",),
    ),
    R4MatrixRow(
        row_id="event.llm_call",
        surface_hint="RuntimeEventType.LLM_CALL",
        verdict=R4EventTenantSourceClass.DIRECT_FROM_CANONICAL_EXECUTION,
        evidence=("tenant_id on RuntimeEvent envelope",),
    ),
    R4MatrixRow(
        row_id="background_task_lifecycle",
        surface_hint="intergrax/background_tasks/events.py",
        verdict=R4EventTenantSourceClass.RESTORED_FROM_TYPED_ASYNC_IDENTITY,
        evidence=("background lifecycle events carry tenant_id when tenant-bound",),
    ),
    R4MatrixRow(
        row_id="trace_attribution_consumer",
        surface_hint="try_attribute_model_call_to_context",
        verdict=R4ObservabilityRole.VALIDATOR,
        evidence=("compares tenants — does not mint",),
    ),
)

BACKGROUND_EXECUTION_IDENTITY_MATRIX: Final[tuple[R4MatrixRow, ...]] = (
    R4MatrixRow(
        row_id="blank_transport_tenant",
        surface_hint="bootstrap.py::_resolve_tenant_scope",
        verdict=R4SubstitutionClass.ILLEGAL_INPUT_REJECTED,
        evidence=("transport_tenant_id must be non-empty",),
    ),
    R4MatrixRow(
        row_id="blank_task_tenant",
        surface_hint="bootstrap.py::_resolve_tenant_scope",
        verdict=R4SubstitutionClass.ILLEGAL_INPUT_REJECTED,
        evidence=("task_tenant_id must be non-empty when provided",),
    ),
    R4MatrixRow(
        row_id="transport_task_mismatch",
        surface_hint="BackgroundExecutionTenantMismatchError",
        verdict=R4CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=("transport A + task B → reject",),
    ),
    R4MatrixRow(
        row_id="identity_materialization",
        surface_hint="BackgroundExecutionIdentity.tenant_id",
        verdict=R4ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("no run_id tenant inference",),
    ),
)

QUEUE_WORKER_CONTINUITY_MATRIX: Final[tuple[R4MatrixRow, ...]] = (
    R4MatrixRow(
        row_id="task_request_enqueue",
        surface_hint="TaskRequest.tenant_id",
        verdict=R4ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("tenant on enqueue contract",),
    ),
    R4MatrixRow(
        row_id="worker_handler_tenant",
        surface_hint="queueing/worker/execution.py",
        verdict=R4ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("handler receives execution_identity.tenant_id",),
    ),
    R4MatrixRow(
        row_id="document_store_worker_intake",
        surface_hint="DocumentStoreTaskWorker",
        verdict=R4ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("bootstrap_background_execution at worker boundary",),
    ),
)

ASYNC_DOMAIN_WORKER_MATRIX: Final[tuple[R4MatrixRow, ...]] = (
    R4MatrixRow(
        row_id="vendor_knowledge_queue_namespace",
        surface_hint="_TenantVendorKnowledgeTaskQueue",
        verdict=R4ChainVerdict.PROVEN_CURRENT_HEAD,
        evidence=("namespace vendor_knowledge.sync_queue.v1:{tenant_id}:",),
    ),
    R4MatrixRow(
        row_id="vendor_enqueue_validation",
        surface_hint="sync_runtime.py enqueue",
        verdict=R4CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=("queue bound A + request B → ValueError",),
    ),
    R4MatrixRow(
        row_id="vendor_worker_payload",
        surface_hint="sync_worker.py handler",
        verdict=R4CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=("job.tenant_id != handler tenant_id → fail",),
    ),
)

EXTENSION_SCOPE_MATRIX: Final[tuple[R4MatrixRow, ...]] = (
    R4MatrixRow(
        row_id="nexus_context_engine",
        surface_hint="context_engine.py → record_context_assembled_from_engine",
        verdict=R4ExtensionTenantClass.PRESERVED,
        evidence=("engine path uses active runtime event tenant scope",),
    ),
    R4MatrixRow(
        row_id="context_metadata_helper_raw",
        surface_hint="record_context_assembly(metadata tenant_id)",
        verdict=R4ContextAssemblyPathClass.RAW_HELPER_CALLER_CONTROLLED,
        evidence=("metadata.get('tenant_id') — non-authoritative helper surface",),
    ),
    R4MatrixRow(
        row_id="context_manager_metadata_path",
        surface_hint="context_manager.py production caller",
        verdict=R4ContextAssemblyPathClass.PRODUCTION_CALLSITE_CANONICALLY_BOUND,
        evidence=('metadata={**bundle_metadata, "tenant_id": task.tenant_id}',),
        tenant_expression="task.tenant_id",
    ),
)

MISSING_DEFAULT_GLOBAL_MATRIX: Final[tuple[R4MatrixRow, ...]] = (
    R4MatrixRow(
        row_id="bg_blank_transport",
        surface_hint="bootstrap background transport tenant",
        verdict=R4SubstitutionClass.ILLEGAL_INPUT_REJECTED,
        evidence=("empty transport tenant rejected",),
    ),
    R4MatrixRow(
        row_id="functional_evidence_no_identity",
        surface_hint="require_tenant_id_from_exec_ctx",
        verdict=R4SubstitutionClass.ILLEGAL_INPUT_REJECTED,
        evidence=("missing canonical_request_identity → error",),
    ),
    R4MatrixRow(
        row_id="vendor_queue_empty_tenant",
        surface_hint="_TenantVendorKnowledgeTaskQueue.__init__",
        verdict=R4SubstitutionClass.ILLEGAL_INPUT_REJECTED,
        evidence=("blank tenant_id → ValueError",),
    ),
    R4MatrixRow(
        row_id="event_schema_version",
        surface_hint="RuntimeEvent payload schema ids",
        verdict=R4SubstitutionClass.LEGAL_NON_TENANT,
        evidence=("schema versioning — not customer tenant substitution",),
    ),
)

CROSS_TENANT_R4_MATRIX: Final[tuple[R4MatrixRow, ...]] = (
    R4MatrixRow(
        row_id="context_a_llm_b",
        surface_hint="try_attribute_model_call_to_context",
        verdict=R4CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=("tenant_mismatch",),
    ),
    R4MatrixRow(
        row_id="context_view_request_a_authority_b",
        surface_hint="ContextViewRequest validator",
        verdict=R4CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=("ValidationError",),
    ),
    R4MatrixRow(
        row_id="context_view_request_a_membership_b",
        surface_hint="ContextViewRequest validator",
        verdict=R4CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=("ValidationError",),
    ),
    R4MatrixRow(
        row_id="transport_a_task_b",
        surface_hint="bootstrap_background_execution",
        verdict=R4CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=("BackgroundExecutionTenantMismatchError",),
    ),
    R4MatrixRow(
        row_id="queue_a_request_b",
        surface_hint="vendor knowledge queue enqueue",
        verdict=R4CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=("tenant mismatch",),
    ),
    R4MatrixRow(
        row_id="request_a_payload_job_b",
        surface_hint="vendor knowledge worker handler",
        verdict=R4CrossTenantClass.CROSS_TENANT_REJECTED,
        evidence=("vendor_knowledge_sync_tenant_mismatch",),
    ),
    R4MatrixRow(
        row_id="same_tenant_attribution",
        surface_hint="trace_x_p4 e2e attribution",
        verdict=R4CrossTenantClass.SAME_TENANT_PROVEN,
        evidence=("test_txp4_q12_primary_context_model_e2e_attribution",),
    ),
)

LLM_CALL_TENANT_SOURCE_MATRIX: Final[tuple[R4MatrixRow, ...]] = (
    R4MatrixRow(
        row_id="maybe_record_llm_from_usage",
        surface_hint="maybe_record_llm_call_from_usage_end",
        verdict=R4LlmCallTenantParamClass.CANONICAL_EXECUTION_TENANT,
        evidence=("tenant_id=evidence.tenant_id",),
        tenant_expression="evidence.tenant_id",
    ),
    R4MatrixRow(
        row_id="record_llm_ambient_fallback",
        surface_hint="record_llm_call_runtime_event resolved_tenant",
        verdict=R4LlmCallTenantParamClass.AMBIENT_RUNTIME_TENANT,
        evidence=("peek_active_runtime_event_tenant_id when tenant_id omitted",),
    ),
    R4MatrixRow(
        row_id="record_llm_direct_production",
        surface_hint="production direct record_llm_call_runtime_event",
        verdict=R4LlmCallTenantParamClass.TEST_ONLY,
        evidence=("no production module outside llm_call_recording.py",),
    ),
)

CONTEXT_ASSEMBLY_TENANT_SOURCE_MATRIX: Final[tuple[R4MatrixRow, ...]] = (
    R4MatrixRow(
        row_id="record_context_assembly",
        surface_hint="metadata.get('tenant_id')",
        verdict=R4ContextTenantSourceClass.UNTRUSTED_CALLER_METADATA,
        evidence=("tenant from caller metadata — not active execution scope",),
    ),
    R4MatrixRow(
        row_id="record_context_assembled_from_engine",
        surface_hint="peek_active_runtime_event_tenant_id",
        verdict=R4ContextTenantSourceClass.AMBIENT_RUNTIME_TENANT,
        evidence=("active runtime event recorder tenant scope",),
    ),
)

MODEL_ATTRIBUTION_OPTIONAL_TENANT_MATRIX: Final[tuple[R4MatrixRow, ...]] = (
    R4MatrixRow(
        row_id="ModelContextAttributionEvidence.tenant_id optional",
        surface_hint="model_context_attribution.py",
        verdict=R4AttributionTenantOptionalClass.LEGAL_NON_TENANT,
        evidence=(
            "optional field on evidence contract; successful attribution copies llm_event.tenant_id",
            "cross-tenant rejected before evidence emission",
        ),
    ),
)

R4_FAMILY_MATRIX: Final[tuple[R4MatrixRow, ...]] = tuple(
    R4MatrixRow(
        row_id=f.family_id,
        surface_hint=f.family_label,
        verdict=R4ChainVerdict.PROVEN_CURRENT_HEAD
        if f.binding.value == "TENANT_BOUND"
        else R4ChainVerdict.MISSING,
        evidence=f.evidence,
        tenant_expression=f.authoritative_tenant_source,
    )
    for f in build_r4_families()
)
