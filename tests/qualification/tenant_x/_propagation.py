# © Artur Czarnecki. All rights reserved.

"""TENANT-X continuity edges (typed propagation inventory)."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._types import PropagationRelation, TenantPropagationEdge

TENANT_PROPAGATION_EDGES: Final[tuple[TenantPropagationEdge, ...]] = (
    TenantPropagationEdge(
        edge_id="edge.request_to_governance",
        source_surface_id="intergrax/runtime/execution/agent_runtime_io.py::RuntimeRequest::canonical.runtime_request",
        target_surface_id="intergrax/contracts/agent_runtime_governance.py::GovernanceAuditEvent::canonical.governance_event",
        expected_relation=PropagationRelation.SAME,
        frz_ten_criteria=("FRZ-TEN-02",),
    ),
    TenantPropagationEdge(
        edge_id="edge.governance_to_execution",
        source_surface_id="intergrax/contracts/agent_runtime_governance.py::GovernanceAuditEvent::canonical.governance_event",
        target_surface_id="intergrax/contracts/runtime_execution_admission.py::RootExecutionAuthorityAdmissionRequest::canonical.execution_admission",
        expected_relation=PropagationRelation.SAME,
        frz_ten_criteria=("FRZ-TEN-02",),
    ),
    TenantPropagationEdge(
        edge_id="edge.execution_to_tool",
        source_surface_id="intergrax/contracts/runtime_execution_admission.py::RootExecutionAuthorityAdmissionRequest::canonical.execution_admission",
        target_surface_id="intergrax/contracts/tool_runtime_read.py::ToolRuntimeExecutionScope::canonical.tool_runtime_scope",
        expected_relation=PropagationRelation.NARROWER,
        frz_ten_criteria=("FRZ-TEN-02", "FRZ-TEN-03"),
    ),
    TenantPropagationEdge(
        edge_id="edge.execution_to_state",
        source_surface_id="intergrax/contracts/runtime_execution_admission.py::RootExecutionAuthorityAdmissionRequest::canonical.execution_admission",
        target_surface_id="intergrax/memory/memory_scope_authority.py::assert_memory_scope_authorized::canonical.memory_scope_authority",
        expected_relation=PropagationRelation.SAME,
        frz_ten_criteria=("FRZ-TEN-02", "FRZ-TEN-04"),
    ),
    TenantPropagationEdge(
        edge_id="edge.execution_to_evidence",
        source_surface_id="intergrax/contracts/runtime_execution_admission.py::RootExecutionAuthorityAdmissionRequest::canonical.execution_admission",
        target_surface_id="intergrax/contracts/functional_evidence/tenant.py::require_tenant_id_from_exec_ctx::canonical.functional_evidence_tenant",
        expected_relation=PropagationRelation.SAME,
        frz_ten_criteria=("FRZ-TEN-02", "FRZ-TEN-07"),
    ),
    TenantPropagationEdge(
        edge_id="edge.parent_to_child",
        source_surface_id="intergrax/runtime/execution/agent_runtime_io.py::RuntimeRequest::canonical.runtime_request",
        target_surface_id="intergrax/runtime/execution/child.py::ChildExecutionRunner::canonical.child_execution_runner",
        expected_relation=PropagationRelation.NARROWER,
        frz_ten_criteria=("FRZ-TEN-03",),
    ),
    TenantPropagationEdge(
        edge_id="edge.config_to_provider_binding",
        source_surface_id="intergrax/integrations/_shared/p3/configs.py::require_tenant_id::validator.marker",
        target_surface_id="intergrax/integrations/credentials/broker.py::ScopedCredentialBroker::canonical.credential_broker",
        expected_relation=PropagationRelation.SAME,
        frz_ten_criteria=("FRZ-TEN-05",),
    ),
    TenantPropagationEdge(
        edge_id="edge.context_to_model_scope",
        source_surface_id="intergrax/contracts/runtime_intelligence/context.py::RuntimeIntelligenceContext.tenant_id::class.field.tenant",
        target_surface_id="intergrax/contracts/runtime_intelligence/context.py::RuntimeIntelligenceContext.tenant_id::class.field.tenant",
        expected_relation=PropagationRelation.SAME,
        frz_ten_criteria=("FRZ-TEN-02", "FRZ-TEN-11"),
    ),
    TenantPropagationEdge(
        edge_id="edge.checkpoint_to_recovery",
        source_surface_id="intergrax/runtime/execution/agent_runtime_io.py::RuntimeRequest::canonical.runtime_request",
        target_surface_id="intergrax/runtime/execution/child.py::ChildExecutionRunner::canonical.child_execution_runner",
        expected_relation=PropagationRelation.SAME,
        frz_ten_criteria=("FRZ-TEN-08",),
    ),
    TenantPropagationEdge(
        edge_id="edge.credential_to_provider",
        source_surface_id="intergrax/integrations/credentials/broker.py::ScopedCredentialBroker::canonical.credential_broker",
        target_surface_id="intergrax/integrations/credentials/broker.py::ScopedCredentialBroker::canonical.credential_broker",
        expected_relation=PropagationRelation.SAME,
        frz_ten_criteria=("FRZ-TEN-06",),
    ),
)


def propagation_edge_count() -> int:
    return len(TENANT_PROPAGATION_EDGES)
