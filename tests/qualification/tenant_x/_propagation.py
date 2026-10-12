# © Artur Czarnecki. All rights reserved.

"""TENANT-X continuity edges (typed propagation inventory)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tests.qualification.tenant_x._types import PropagationEdgeStatus, PropagationRelation, TenantPropagationEdge


@dataclass(frozen=True, slots=True)
class PropagationEdgeGap:
    gap_id: str
    description: str
    frz_ten_criteria: tuple[str, ...]
    status: PropagationEdgeStatus


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
        edge_id="edge.config_validator_to_credential_broker",
        source_surface_id="intergrax/integrations/_shared/p3/configs.py::require_tenant_id::validator.marker",
        target_surface_id="intergrax/integrations/credentials/broker.py::ScopedCredentialBroker::canonical.credential_broker",
        expected_relation=PropagationRelation.SAME,
        frz_ten_criteria=("FRZ-TEN-05",),
    ),
)

TENANT_PROPAGATION_GAPS: Final[tuple[PropagationEdgeGap, ...]] = (
    PropagationEdgeGap(
        gap_id="gap.checkpoint_to_recovery",
        description="Checkpoint/resume tenant continuity endpoint not identified — do not substitute RuntimeRequest→ChildExecutionRunner",
        frz_ten_criteria=("FRZ-TEN-08",),
        status=PropagationEdgeStatus.MISSING,
    ),
    PropagationEdgeGap(
        gap_id="gap.credential_to_provider_consumption",
        description="Provider credential consumption endpoint not identified — no self-edge proof",
        frz_ten_criteria=("FRZ-TEN-06",),
        status=PropagationEdgeStatus.MISSING,
    ),
    PropagationEdgeGap(
        gap_id="gap.context_model_scope",
        description="RuntimeIntelligenceContext self-edge removed — not a propagation mechanism",
        frz_ten_criteria=("FRZ-TEN-02", "FRZ-TEN-11"),
        status=PropagationEdgeStatus.INVALID,
    ),
)


def propagation_edge_count() -> int:
    return len(TENANT_PROPAGATION_EDGES)


def validate_propagation_edges(known_surface_ids: frozenset[str]) -> tuple[str, ...]:
    violations: list[str] = []
    for edge in TENANT_PROPAGATION_EDGES:
        if edge.source_surface_id not in known_surface_ids:
            violations.append(f"{edge.edge_id}:missing_source")
        if edge.target_surface_id not in known_surface_ids:
            violations.append(f"{edge.edge_id}:missing_target")
        if edge.source_surface_id == edge.target_surface_id:
            violations.append(f"{edge.edge_id}:self_edge")
    return tuple(violations)
