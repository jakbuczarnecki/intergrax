# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R1 mandatory boundary families A–J (derived from evidence collectors)."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._boundary_continuity import (
    qualify_credential_boundary,
    qualify_recovery_boundary,
)
from tests.qualification.tenant_x._r1_propagation_proofs import certify_r1_propagation_edges
from tests.qualification.tenant_x._r1_structural_io import all_structural_patterns_present
from tests.qualification.tenant_x._r1_types import (
    R1CriticalBoundaryFamily,
    R1ProvenanceStatus,
    SANCTIONED_ROOT_TENANT_CONTRACT,
)
from tests.qualification.tenant_x._types import BoundaryContinuityClass, PropagationRelation

_EDGE_BY_ID = {e.edge_id: e for e in certify_r1_propagation_edges()}


def _edge_status(*edge_ids: str) -> R1ProvenanceStatus:
    statuses = [_EDGE_BY_ID[e].final_status for e in edge_ids if e in _EDGE_BY_ID]
    if not statuses:
        return R1ProvenanceStatus.MISSING
    if any(s == R1ProvenanceStatus.INVALID for s in statuses):
        return R1ProvenanceStatus.INVALID
    if all(s == R1ProvenanceStatus.PROVEN_CURRENT_HEAD for s in statuses):
        return R1ProvenanceStatus.PROVEN_CURRENT_HEAD
    if any(s == R1ProvenanceStatus.HISTORICAL_REPLAY_CONFIRMED for s in statuses):
        return R1ProvenanceStatus.HISTORICAL_REPLAY_CONFIRMED
    return R1ProvenanceStatus.MISSING


def _family_a_status() -> R1ProvenanceStatus:
    checks = (
        ("intergrax/runtime/execution/agent_runtime_io.py", "metadata tenant_id cannot override canonical RuntimeRequest.tenant_id"),
        ("intergrax/contracts/runtime_execution_admission.py", "class RootExecutionAuthorityAdmissionRequest"),
    )
    if not all_structural_patterns_present(checks):
        return R1ProvenanceStatus.MISSING
    return _edge_status("edge.request_to_governance")


def _family_i_status() -> R1ProvenanceStatus:
    cred = qualify_credential_boundary()
    if cred.status == BoundaryContinuityClass.UNKNOWN:
        return R1ProvenanceStatus.MISSING
    if cred.status == BoundaryContinuityClass.PARTIAL_EVIDENCE:
        edge = _EDGE_BY_ID.get("edge.config_validator_to_credential_broker")
        if edge and edge.final_status == R1ProvenanceStatus.HISTORICAL_REPLAY_CONFIRMED:
            return R1ProvenanceStatus.HISTORICAL_REPLAY_CONFIRMED
        return R1ProvenanceStatus.MISSING
    if cred.status == BoundaryContinuityClass.GAP_QUALIFICATION_ONLY:
        return R1ProvenanceStatus.MISSING
    return _edge_status("edge.config_validator_to_credential_broker")


def build_r1_critical_boundary_families() -> tuple[R1CriticalBoundaryFamily, ...]:
    return (
        R1CriticalBoundaryFamily(
            family_id="A",
            label="ingress / RuntimeRequest / admission",
            frz_ten_criteria=("FRZ-TEN-01", "FRZ-TEN-09"),
            provenance_status=_family_a_status(),
            relation=PropagationRelation.SAME,
            structural_evidence=(
                "RuntimeRequest.tenant_id required; metadata cannot override",
                f"sanctioned root admission contract: {SANCTIONED_ROOT_TENANT_CONTRACT}",
                "RootExecutionAuthorityAdmissionRequest fail-closed on empty tenant_id",
            ),
        ),
        R1CriticalBoundaryFamily(
            family_id="B",
            label="Governance subject / authority",
            frz_ten_criteria=("FRZ-TEN-02",),
            provenance_status=_edge_status("edge.request_to_governance"),
            relation=PropagationRelation.SAME,
            structural_evidence=_EDGE_BY_ID["edge.request_to_governance"].current_head_evidence,
        ),
        R1CriticalBoundaryFamily(
            family_id="C",
            label="Execution authority / execution scope",
            frz_ten_criteria=("FRZ-TEN-02", "FRZ-TEN-03"),
            provenance_status=_edge_status("edge.governance_to_execution"),
            relation=PropagationRelation.SAME,
            structural_evidence=_EDGE_BY_ID["edge.governance_to_execution"].current_head_evidence,
        ),
        R1CriticalBoundaryFamily(
            family_id="D",
            label="parent → child execution",
            frz_ten_criteria=("FRZ-TEN-03",),
            provenance_status=_edge_status("edge.parent_to_child"),
            relation=PropagationRelation.NARROWER,
            structural_evidence=_EDGE_BY_ID["edge.parent_to_child"].current_head_evidence,
        ),
        R1CriticalBoundaryFamily(
            family_id="E",
            label="execution → tool invocation",
            frz_ten_criteria=("FRZ-TEN-02", "FRZ-TEN-03"),
            provenance_status=_edge_status("edge.execution_to_tool"),
            relation=PropagationRelation.NARROWER,
            structural_evidence=_EDGE_BY_ID["edge.execution_to_tool"].current_head_evidence,
        ),
        R1CriticalBoundaryFamily(
            family_id="F",
            label="execution → state access identity",
            frz_ten_criteria=("FRZ-TEN-02", "FRZ-TEN-04"),
            provenance_status=_edge_status("edge.execution_to_state"),
            relation=PropagationRelation.SAME,
            structural_evidence=_EDGE_BY_ID["edge.execution_to_state"].current_head_evidence,
        ),
        R1CriticalBoundaryFamily(
            family_id="G",
            label="execution → evidence/event identity",
            frz_ten_criteria=("FRZ-TEN-02", "FRZ-TEN-07"),
            provenance_status=_edge_status("edge.execution_to_evidence"),
            relation=PropagationRelation.SAME,
            structural_evidence=_EDGE_BY_ID["edge.execution_to_evidence"].current_head_evidence,
        ),
        R1CriticalBoundaryFamily(
            family_id="H",
            label="execution/context → model/context boundary",
            frz_ten_criteria=("FRZ-TEN-11",),
            provenance_status=R1ProvenanceStatus.NOT_APPLICABLE,
            relation=None,
            structural_evidence=("R4 owns full context/model + async continuity",),
            handoff_wave="TENANT-X-R4",
        ),
        R1CriticalBoundaryFamily(
            family_id="I",
            label="provider/config tenant input boundary",
            frz_ten_criteria=("FRZ-TEN-05", "FRZ-TEN-06"),
            provenance_status=_family_i_status(),
            relation=PropagationRelation.SAME,
            structural_evidence=(
                "configs.require_tenant_id validator → ScopedCredentialBroker assert_tenant_consistency",
                "credential → provider consumption: handoff TENANT-X-R2 (continuity classification only)",
            ),
            handoff_wave="TENANT-X-R2",
        ),
        R1CriticalBoundaryFamily(
            family_id="J",
            label="checkpoint/recovery identity boundary",
            frz_ten_criteria=("FRZ-TEN-08",),
            provenance_status=R1ProvenanceStatus.NOT_APPLICABLE,
            relation=None,
            structural_evidence=(
                qualify_recovery_boundary().note,
                "gap.checkpoint_to_recovery remains R3 mandatory closure",
            ),
            handoff_wave="TENANT-X-R3",
        ),
    )


R1_CRITICAL_BOUNDARY_FAMILIES: Final[tuple[R1CriticalBoundaryFamily, ...]] = build_r1_critical_boundary_families()


def r1_critical_boundary_missing_or_invalid() -> tuple[str, ...]:
    bad: list[str] = []
    for family in R1_CRITICAL_BOUNDARY_FAMILIES:
        if family.provenance_status in {R1ProvenanceStatus.MISSING, R1ProvenanceStatus.INVALID}:
            if family.handoff_wave and family.family_id in {"I"}:
                if family.provenance_status == R1ProvenanceStatus.MISSING:
                    bad.append(family.family_id)
                continue
            if family.handoff_wave:
                continue
            bad.append(family.family_id)
    return tuple(bad)


def r1_owned_critical_boundary_missing_or_invalid() -> tuple[str, ...]:
    bad: list[str] = []
    for family in R1_CRITICAL_BOUNDARY_FAMILIES:
        if family.handoff_wave and family.family_id in {"H", "J"}:
            continue
        if family.provenance_status in {R1ProvenanceStatus.MISSING, R1ProvenanceStatus.INVALID}:
            bad.append(family.family_id)
    return tuple(bad)
