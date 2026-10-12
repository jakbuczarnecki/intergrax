# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R1 structural current-HEAD proofs for the seven P0 propagation edges."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._propagation import TENANT_PROPAGATION_EDGES
from tests.qualification.tenant_x._r1_structural_io import all_structural_patterns_present, discovery_markers
from tests.qualification.tenant_x._r1_types import R1PropagationEdgeCertification, R1ProvenanceStatus
from tests.qualification.tenant_x._types import PropagationRelation

_HISTORICAL_SUPPORT: Final[dict[str, tuple[str, ...]]] = {
    "edge.request_to_governance": ("GOV-X2 local tenant audit (supporting)",),
    "edge.governance_to_execution": ("GOV-X2 execution admission chain (supporting)",),
    "edge.execution_to_tool": ("GOV-X2 tool path (supporting)",),
    "edge.execution_to_state": ("STATE-X scoped memory evidence (supporting)",),
    "edge.execution_to_evidence": ("TRACE-X functional evidence (supporting)",),
    "edge.parent_to_child": ("GOV-X2 child execution negatives (supporting)",),
    "edge.config_validator_to_credential_broker": ("CONFIG-X credential broker partial (supporting)",),
}

# (relative_path, exact substring) chains — all must match for PROVEN_CURRENT_HEAD.
_STRUCTURAL_PROVEN: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "edge.request_to_governance": (
        ("intergrax/runtime/execution/agent_runtime_io.py", "tenant_id=str(self.tenant_id).strip()"),
        ("intergrax/runtime/agent_governance/request_builder.py", "tenant_id=state.tenant_id"),
        ("intergrax/runtime/agent_governance/audit.py", "tenant_id=request.agent.tenant_id"),
    ),
    "edge.governance_to_execution": (
        ("intergrax/runtime/execution/runtime.py", "tenant_id = governance_identity.tenant_id"),
        (
            "intergrax/runtime/governance/default_root_execution_launcher.py",
            "RootExecutionAuthorityAdmissionRequest(",
        ),
        ("intergrax/runtime/governance/default_root_execution_launcher.py", "tenant_id=request.tenant_id"),
    ),
    "edge.execution_to_tool": (
        (
            "intergrax/runtime/runtime_inspection/adapters/tool_runtime_read.py",
            "ToolRuntimeExecutionScope(",
        ),
        (
            "intergrax/runtime/runtime_inspection/adapters/tool_runtime_read.py",
            "tenant_id=scope.tenant_id",
        ),
        (
            "intergrax/runtime/nexus/tools/tool_invocation_inner_governance.py",
            "tenant_id=governance_identity.tenant_id",
        ),
    ),
    "edge.execution_to_state": (
        (
            "intergrax/memory/memory_scope_authority.py",
            "scope.tenant_id != identity.tenant_id",
        ),
        (
            "intergrax/memory/memory_scope_authority.py",
            "scope tenant_id conflicts with canonical identity",
        ),
    ),
    "edge.execution_to_evidence": (
        ("intergrax/contracts/functional_evidence/tenant.py", "never guess or default"),
        ("intergrax/contracts/functional_evidence/tenant.py", "tenant_id = identity.tenant_id"),
    ),
    "edge.parent_to_child": (
        ("intergrax/runtime/execution/child.py", "resolve_child_authority"),
        ("intergrax/runtime/execution/authority/policy.py", "child authority cannot exceed immediate parent authority"),
        ("intergrax/runtime/execution/child.py", "narrow_child_deadline_at_utc"),
        ("intergrax/runtime/execution/child.py", "narrow_protected_work_admission_for_child"),
        ("intergrax/runtime/execution/agent_runtime_io.py", "def canonical_runtime_request_tenant_id"),
    ),
    "edge.config_validator_to_credential_broker": (
        ("intergrax/integrations/credentials/broker.py", "assert_tenant_consistency(ref=grant.credential_ref, grant=grant, scope=scope)"),
        ("intergrax/integrations/_shared/p3/configs.py", "def require_tenant_id(self) -> str:"),
    ),
}

_DISCOVERY_ONLY: Final[dict[str, tuple[tuple[str, tuple[str, ...]], ...]]] = {
    "edge.config_validator_to_credential_broker": (
        (
            "intergrax/integrations/_shared/p3/factories.py",
            ("config.require_tenant_id()",),
        ),
    ),
}


def _structural_evidence(edge_id: str) -> tuple[str, ...]:
    checks = _STRUCTURAL_PROVEN.get(edge_id, ())
    if not checks:
        return ()
    return all_structural_patterns_present(checks)


def _certify_edge(edge_id: str) -> tuple[tuple[str, ...], R1ProvenanceStatus]:
    structural = _structural_evidence(edge_id)
    discovery: list[str] = []
    for rel, markers in _DISCOVERY_ONLY.get(edge_id, ()):
        for m in discovery_markers(rel, markers):
            discovery.append(f"discovery:{rel}:{m}")

    if structural:
        if edge_id == "edge.config_validator_to_credential_broker":
            return (*structural, *discovery), R1ProvenanceStatus.HISTORICAL_REPLAY_CONFIRMED
        return (*structural, *discovery), R1ProvenanceStatus.PROVEN_CURRENT_HEAD

    if discovery:
        return tuple(discovery), R1ProvenanceStatus.HISTORICAL_REPLAY_CONFIRMED

    return (), R1ProvenanceStatus.MISSING


def certify_r1_propagation_edges() -> tuple[R1PropagationEdgeCertification, ...]:
    out: list[R1PropagationEdgeCertification] = []
    for edge in TENANT_PROPAGATION_EDGES:
        current, status = _certify_edge(edge.edge_id)

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


def r1_owned_propagation_missing() -> tuple[str, ...]:
    """R1-owned edges still MISSING (excludes R2 handoff-only gaps)."""
    handoff_edges = {"edge.config_validator_to_credential_broker"}
    return tuple(
        e.edge_id
        for e in certify_r1_propagation_edges()
        if e.final_status == R1ProvenanceStatus.MISSING and e.edge_id not in handoff_edges
    )


def r1_critical_propagation_missing() -> tuple[str, ...]:
    return tuple(e.edge_id for e in certify_r1_propagation_edges() if e.final_status == R1ProvenanceStatus.MISSING)
