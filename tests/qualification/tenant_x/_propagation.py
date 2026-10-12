# © Artur Czarnecki. All rights reserved.

"""TENANT-X continuity edges (typed propagation inventory)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tests.qualification.tenant_x._types import (
    PropagationEdgeContinuityStatus,
    PropagationEdgeStatus,
    PropagationRelation,
    TenantPropagationEdge,
    TenantProvenanceEdge,
    TenantSurfaceRecord,
)


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
        description=(
            "CLOSED @ TENANT-X-R3 — checkpoint.tenant_id → validate_checkpoint_identity_binding / "
            "durable scoped load → resume materialization (current-HEAD structural + adversarial)"
        ),
        frz_ten_criteria=("FRZ-TEN-08",),
        status=PropagationEdgeStatus.PROVEN,
    ),
    PropagationEdgeGap(
        gap_id="gap.credential_to_provider_consumption",
        description=(
            "CLOSED @ TENANT-X-R2 — credential broker → provider operation consumption "
            "(exec.aw7c.reference_effect + R2 canonical mechanisms @ 9bc53dfa)"
        ),
        frz_ten_criteria=("FRZ-TEN-06",),
        status=PropagationEdgeStatus.PROVEN,
    ),
    PropagationEdgeGap(
        gap_id="gap.context_model_scope",
        description=(
            "context/model continuity = PROVEN_CURRENT_HEAD at R4 qualification scope "
            "(per-transition Chain 1 + production call-site inventory); "
            "pending independent R4 audit — not FRZ promotion"
        ),
        frz_ten_criteria=("FRZ-TEN-02", "FRZ-TEN-11"),
        status=PropagationEdgeStatus.PROVEN,
    ),
)


def propagation_edge_count() -> int:
    return len(TENANT_PROPAGATION_EDGES)


@dataclass(frozen=True, slots=True)
class PropagationEdgeContinuityAssessment:
    edge_id: str
    status: PropagationEdgeContinuityStatus
    note: str


def assess_propagation_edge_continuity(
    inventory: tuple[TenantSurfaceRecord, ...],
) -> tuple[PropagationEdgeContinuityAssessment, ...]:
    by_id = {r.surface_id: r for r in inventory}
    out: list[PropagationEdgeContinuityAssessment] = []
    for edge in TENANT_PROPAGATION_EDGES:
        src = by_id.get(edge.source_surface_id)
        tgt = by_id.get(edge.target_surface_id)
        if src is None or tgt is None:
            out.append(
                PropagationEdgeContinuityAssessment(
                    edge_id=edge.edge_id,
                    status=PropagationEdgeContinuityStatus.MISSING,
                    note="endpoint not in current inventory",
                )
            )
            continue
        src_refs = "".join(src.evidence_refs)
        tgt_refs = "".join(tgt.evidence_refs)
        if "provenance:current_head" in src_refs or "provenance:current_head" in tgt_refs:
            out.append(
                PropagationEdgeContinuityAssessment(
                    edge_id=edge.edge_id,
                    status=PropagationEdgeContinuityStatus.PROVEN_CURRENT_HEAD,
                    note="structural current-head provenance evidence on endpoint",
                )
            )
        elif "canonical_anchor_profile" in src_refs and "canonical_anchor_profile" in tgt_refs:
            out.append(
                PropagationEdgeContinuityAssessment(
                    edge_id=edge.edge_id,
                    status=PropagationEdgeContinuityStatus.HISTORICAL_EVIDENCE_REPLAY_REQUIRED,
                    note="canonical anchors only — replay required for current continuity",
                )
            )
        else:
            out.append(
                PropagationEdgeContinuityAssessment(
                    edge_id=edge.edge_id,
                    status=PropagationEdgeContinuityStatus.HISTORICAL_EVIDENCE_REPLAY_REQUIRED,
                    note="no current-head structural propagation proof",
                )
            )
    for gap in TENANT_PROPAGATION_GAPS:
        status = PropagationEdgeContinuityStatus.MISSING
        if gap.status == PropagationEdgeStatus.INVALID:
            status = PropagationEdgeContinuityStatus.INVALID
        elif gap.status == PropagationEdgeStatus.PROVEN:
            status = PropagationEdgeContinuityStatus.PROVEN_CURRENT_HEAD
        out.append(
            PropagationEdgeContinuityAssessment(
                edge_id=gap.gap_id,
                status=status,
                note=gap.description,
            )
        )
    return tuple(out)


def build_structural_provenance_edges(
    inventory: tuple[TenantSurfaceRecord, ...],
) -> tuple[TenantProvenanceEdge, ...]:
    edges: list[TenantProvenanceEdge] = []
    for record in inventory:
        joined = "".join(record.evidence_refs)
        if "provenance:current_head:derive_same" not in joined:
            continue
        edges.append(
            TenantProvenanceEdge(
                source_surface_id=record.module_path + "::parent",
                target_surface_id=record.surface_id,
                relation=PropagationRelation.SAME,
                evidence=joined,
            )
        )
    return tuple(edges)


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
