# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R4 bounded cross-boundary family inventory (A–J)."""

from __future__ import annotations

from tests.qualification.tenant_x._r1_structural_io import all_structural_patterns_present
from tests.qualification.tenant_x._r4_types import R4FamilyBinding, R4FamilyRow

_CONTEXT_MODEL_CHECKS: tuple[tuple[str, str], ...] = (
    ("intergrax/runtime/llm/model_context_attribution.py", "tenant_mismatch"),
    ("intergrax/runtime/llm/model_context_attribution.py", "left_tenant = context_event.tenant_id"),
    ("intergrax/runtime/events/context_skill_recording.py", "RuntimeEventType.CONTEXT_ASSEMBLED"),
)

_CONTEXT_VIEW_CHECKS: tuple[tuple[str, str], ...] = (
    ("intergrax/contracts/context_view.py", "authority_request tenant_id must match scope tenant_id"),
    ("intergrax/contracts/context_view_composition.py", "entry tenant_id must match view scope tenant_id"),
)

_BACKGROUND_IDENTITY_CHECKS: tuple[tuple[str, str], ...] = (
    ("intergrax/runtime/background_execution/bootstrap.py", "BackgroundExecutionTenantMismatchError"),
    ("intergrax/runtime/background_execution/bootstrap.py", "transport_tenant_id must be non-empty"),
)

_VENDOR_QUEUE_CHECKS: tuple[tuple[str, str], ...] = (
    ("intergrax/runtime/vendor_knowledge/sync_runtime.py", "vendor knowledge task queue tenant mismatch"),
    ("intergrax/runtime/vendor_knowledge/sync_worker.py", "vendor_knowledge_sync_tenant_mismatch"),
)


def _binding_from_checks(
    checks: tuple[tuple[str, str], ...],
) -> tuple[R4FamilyBinding, tuple[str, ...]]:
    matched = all_structural_patterns_present(checks)
    if len(matched) == len(checks):
        return R4FamilyBinding.TENANT_BOUND, matched
    return R4FamilyBinding.UNKNOWN, ("discovery: structural markers incomplete",)


def build_r4_families() -> tuple[R4FamilyRow, ...]:
    ctx_model_binding, ctx_model_ev = _binding_from_checks(_CONTEXT_MODEL_CHECKS)
    ctx_view_binding, ctx_view_ev = _binding_from_checks(_CONTEXT_VIEW_CHECKS)
    bg_binding, bg_ev = _binding_from_checks(_BACKGROUND_IDENTITY_CHECKS)
    vk_binding, vk_ev = _binding_from_checks(_VENDOR_QUEUE_CHECKS)
    return (
        R4FamilyRow(
            family_id="A.context_assembly",
            family_label="context assembly",
            binding=R4FamilyBinding.TENANT_BOUND,
            authoritative_tenant_source=(
                "record_context_assembled_from_engine: peek_active_runtime_event_tenant_id; "
                "record_context_assembly: metadata tenant_id (caller-supplied)"
            ),
            evidence=(
                "intergrax/runtime/events/context_skill_recording.py::record_context_assembly",
                "intergrax/runtime/events/context_skill_recording.py::record_context_assembled_from_engine",
            ),
        ),
        R4FamilyRow(
            family_id="B.model_call_attribution",
            family_label="model-call attribution",
            binding=ctx_model_binding,
            authoritative_tenant_source="try_attribute_model_call_to_context tenant equality gate",
            evidence=ctx_model_ev
            + (
                "intergrax/runtime/llm/model_context_attribution.py::ModelContextAttributionEvidence",
            ),
        ),
        R4FamilyRow(
            family_id="C.context_view_composition",
            family_label="ContextView composition",
            binding=ctx_view_binding,
            authoritative_tenant_source="ContextViewRequest + composition validators (scope alignment)",
            evidence=ctx_view_ev,
        ),
        R4FamilyRow(
            family_id="D.runtime_functional_evidence",
            family_label="runtime functional evidence",
            binding=R4FamilyBinding.TENANT_BOUND,
            authoritative_tenant_source="require_tenant_id_from_exec_ctx(exec_ctx)",
            evidence=(
                "intergrax/contracts/functional_evidence/tenant.py::require_tenant_id_from_exec_ctx",
                "intergrax/runtime/observability/functional_evidence_recorder.py",
            ),
        ),
        R4FamilyRow(
            family_id="E.runtime_events",
            family_label="runtime events (CONTEXT_ASSEMBLED / LLM_CALL / background lifecycle)",
            binding=R4FamilyBinding.TENANT_BOUND,
            authoritative_tenant_source="RuntimeEvent.tenant_id on typed lifecycle families",
            evidence=(
                "intergrax/contracts/runtime_event.py::RuntimeEvent",
                "intergrax/background_tasks/events.py",
            ),
        ),
        R4FamilyRow(
            family_id="F.trace_diagnostics",
            family_label="trace / diagnostics",
            binding=R4FamilyBinding.TENANT_BOUND,
            authoritative_tenant_source="observational carrier — no tenant mint on attribution path",
            evidence=(
                "intergrax/runtime/llm/model_context_attribution.py",
                "tests/qualification/trace_x/test_trace_x_p4_model_context_attribution.py",
            ),
            notes="ILLEGAL_AUTHORITY count must remain 0",
        ),
        R4FamilyRow(
            family_id="G.background_execution_identity",
            family_label="background execution identity",
            binding=bg_binding,
            authoritative_tenant_source="resolve_background_execution transport+task tenant reconcile",
            evidence=bg_ev,
        ),
        R4FamilyRow(
            family_id="H.queue_worker_transport",
            family_label="queue / worker transport",
            binding=R4FamilyBinding.TENANT_BOUND,
            authoritative_tenant_source="TaskRequest.tenant_id at enqueue + worker intake",
            evidence=(
                "intergrax/queueing/providers/document_store.py::DocumentStoreTaskWorker",
                "intergrax/runtime/background_execution/bootstrap.py",
            ),
        ),
        R4FamilyRow(
            family_id="I.async_domain_workers",
            family_label="async domain workers (vendor knowledge sync)",
            binding=vk_binding,
            authoritative_tenant_source="tenant-bound queue namespace + handler decode validation",
            evidence=vk_ev,
        ),
        R4FamilyRow(
            family_id="J.extension_context_async_seam",
            family_label="extension / context async seam",
            binding=R4FamilyBinding.TENANT_BOUND,
            authoritative_tenant_source="extensions receive bound scope — no tenant mint in R4 paths",
            evidence=(
                "intergrax/runtime/nexus/context/context_engine.py::record_context_assembled_from_engine",
            ),
            notes="architecture change required if extension becomes tenant authority",
        ),
    )


def r4_critical_unknown_families() -> tuple[str, ...]:
    return tuple(
        f.family_id for f in build_r4_families() if f.binding == R4FamilyBinding.UNKNOWN
    )
