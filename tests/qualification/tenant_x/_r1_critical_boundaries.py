# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R1 mandatory boundary families A–J (family certification, not 319 surfaces)."""

from __future__ import annotations

from typing import Final

from tests.qualification.tenant_x._boundary_continuity import (
    qualify_credential_boundary,
    qualify_recovery_boundary,
)
from tests.qualification.tenant_x._r1_types import (
    R1CriticalBoundaryFamily,
    R1ProvenanceStatus,
    SANCTIONED_ROOT_TENANT_CONTRACT,
)
from tests.qualification.tenant_x._types import BoundaryContinuityClass, PropagationRelation

R1_CRITICAL_BOUNDARY_FAMILIES: Final[tuple[R1CriticalBoundaryFamily, ...]] = (
    R1CriticalBoundaryFamily(
        family_id="A",
        label="ingress / RuntimeRequest / admission",
        frz_ten_criteria=("FRZ-TEN-01", "FRZ-TEN-09"),
        provenance_status=R1ProvenanceStatus.PROVEN_CURRENT_HEAD,
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
        provenance_status=R1ProvenanceStatus.PROVEN_CURRENT_HEAD,
        relation=PropagationRelation.SAME,
        structural_evidence=(
            "Governance audit/event contracts carry tenant_id with non-empty validators",
            "Governance does not mint alternate root tenant — derives from admitted execution context",
        ),
    ),
    R1CriticalBoundaryFamily(
        family_id="C",
        label="Execution authority / execution scope",
        frz_ten_criteria=("FRZ-TEN-02", "FRZ-TEN-03"),
        provenance_status=R1ProvenanceStatus.PROVEN_CURRENT_HEAD,
        relation=PropagationRelation.SAME,
        structural_evidence=(
            "RootExecutionAuthorityAdmissionPort trusted admission",
            "host_task admit_root_execution / admit_root_governance_identity bridge",
        ),
    ),
    R1CriticalBoundaryFamily(
        family_id="D",
        label="parent → child execution",
        frz_ten_criteria=("FRZ-TEN-03",),
        provenance_status=R1ProvenanceStatus.PROVEN_CURRENT_HEAD,
        relation=PropagationRelation.NARROWER,
        structural_evidence=(
            "ChildExecutionRunner + narrow_protected_work_admission_for_child",
            "child scope ⊆ parent protected-work admission (structural narrowing)",
        ),
    ),
    R1CriticalBoundaryFamily(
        family_id="E",
        label="execution → tool invocation",
        frz_ten_criteria=("FRZ-TEN-02", "FRZ-TEN-03"),
        provenance_status=R1ProvenanceStatus.PROVEN_CURRENT_HEAD,
        relation=PropagationRelation.NARROWER,
        structural_evidence=("ToolRuntimeExecutionScope.tenant_id contract field",),
    ),
    R1CriticalBoundaryFamily(
        family_id="F",
        label="execution → state access identity",
        frz_ten_criteria=("FRZ-TEN-02", "FRZ-TEN-04"),
        provenance_status=R1ProvenanceStatus.PROVEN_CURRENT_HEAD,
        relation=PropagationRelation.SAME,
        structural_evidence=("assert_memory_scope_authorized tenant mismatch fail-closed",),
    ),
    R1CriticalBoundaryFamily(
        family_id="G",
        label="execution → evidence/event identity",
        frz_ten_criteria=("FRZ-TEN-02", "FRZ-TEN-07"),
        provenance_status=R1ProvenanceStatus.PROVEN_CURRENT_HEAD,
        relation=PropagationRelation.SAME,
        structural_evidence=("require_tenant_id_from_exec_ctx — no guess/default",),
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
        provenance_status=(
            R1ProvenanceStatus.PROVEN_CURRENT_HEAD
            if qualify_credential_boundary().status != BoundaryContinuityClass.UNKNOWN
            else R1ProvenanceStatus.MISSING
        ),
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


def r1_critical_boundary_missing_or_invalid() -> tuple[str, ...]:
    bad: list[str] = []
    for family in R1_CRITICAL_BOUNDARY_FAMILIES:
        if family.provenance_status in {R1ProvenanceStatus.MISSING, R1ProvenanceStatus.INVALID}:
            bad.append(family.family_id)
    return tuple(bad)
