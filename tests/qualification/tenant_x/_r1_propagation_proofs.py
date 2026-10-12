# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R1 structural current-HEAD proofs for the seven P0 propagation edges."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from tests.qualification.tenant_x._propagation import TENANT_PROPAGATION_EDGES
from tests.qualification.tenant_x._r1_types import R1PropagationEdgeCertification, R1ProvenanceStatus
from tests.qualification.tenant_x._types import PropagationRelation

_REPO_ROOT = Path(__file__).resolve().parents[3]

_HISTORICAL_SUPPORT: Final[dict[str, tuple[str, ...]]] = {
    "edge.request_to_governance": ("GOV-X2 local tenant audit (supporting)",),
    "edge.governance_to_execution": ("GOV-X2 execution admission chain (supporting)",),
    "edge.execution_to_tool": ("GOV-X2 tool path (supporting)",),
    "edge.execution_to_state": ("STATE-X scoped memory evidence (supporting)",),
    "edge.execution_to_evidence": ("TRACE-X functional evidence (supporting)",),
    "edge.parent_to_child": ("GOV-X2 child execution negatives (supporting)",),
    "edge.config_validator_to_credential_broker": ("CONFIG-X credential broker partial (supporting)",),
}


def _read(rel: str) -> str:
    return (_REPO_ROOT / rel).read_text(encoding="utf-8")


def _markers_present(rel: str, markers: tuple[str, ...]) -> tuple[str, ...]:
    text = _read(rel)
    return tuple(m for m in markers if m in text)


def certify_r1_propagation_edges() -> tuple[R1PropagationEdgeCertification, ...]:
    out: list[R1PropagationEdgeCertification] = []
    for edge in TENANT_PROPAGATION_EDGES:
        current: tuple[str, ...] = ()
        status = R1ProvenanceStatus.MISSING
        if edge.edge_id == "edge.request_to_governance":
            current = _markers_present(
                "intergrax/runtime/execution/agent_runtime_io.py",
                (
                    "tenant_id is required for RuntimeRequest",
                    "metadata tenant_id cannot override canonical RuntimeRequest.tenant_id",
                ),
            ) + _markers_present(
                "intergrax/contracts/agent_runtime_governance.py",
                ('tenant_id: str = _NON_EMPTY',),
            )
        elif edge.edge_id == "edge.governance_to_execution":
            current = _markers_present(
                "intergrax/contracts/runtime_execution_admission.py",
                (
                    "tenant_id: str",
                    "tenant_id must be non-empty",
                ),
            )
        elif edge.edge_id == "edge.execution_to_tool":
            current = _markers_present(
                "intergrax/contracts/tool_runtime_read.py",
                ("tenant_id: str",),
            ) + _markers_present(
                "intergrax/runtime/execution/protected_work_admission.py",
                ("narrow_protected_work_admission_for_child",),
            )
        elif edge.edge_id == "edge.execution_to_state":
            current = _markers_present(
                "intergrax/memory/memory_scope_authority.py",
                ("scope tenant_id conflicts with canonical identity",),
            )
        elif edge.edge_id == "edge.execution_to_evidence":
            current = _markers_present(
                "intergrax/contracts/functional_evidence/tenant.py",
                (
                    "never guess or default",
                    "canonical_request_identity with tenant_id",
                ),
            )
        elif edge.edge_id == "edge.parent_to_child":
            current = _markers_present(
                "intergrax/runtime/execution/child.py",
                ("narrow_protected_work_admission_for_child", "ChildExecutionRunner"),
            ) + _markers_present(
                "intergrax/runtime/execution/agent_runtime_io.py",
                ("canonical_runtime_request_tenant_id",),
            )
        elif edge.edge_id == "edge.config_validator_to_credential_broker":
            current = _markers_present(
                "intergrax/integrations/credentials/broker.py",
                ("assert_tenant_consistency", "scope.tenant_id"),
            ) + _markers_present(
                "intergrax/integrations/_shared/p3/configs.py",
                ("require_tenant_id",),
            )

        if current:
            status = R1ProvenanceStatus.PROVEN_CURRENT_HEAD
        out.append(
            R1PropagationEdgeCertification(
                edge_id=edge.edge_id,
                source=edge.source_surface_id,
                target=edge.target_surface_id,
                relation=edge.expected_relation,
                current_head_evidence=current,
                historical_evidence=_HISTORICAL_SUPPORT.get(edge.edge_id, ()),
                final_status=status,
            )
        )
    return tuple(out)


def r1_critical_propagation_missing() -> tuple[str, ...]:
    return tuple(
        e.edge_id for e in certify_r1_propagation_edges() if e.final_status == R1ProvenanceStatus.MISSING
    )
