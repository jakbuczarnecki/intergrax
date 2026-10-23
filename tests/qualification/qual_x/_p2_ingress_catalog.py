# © Artur Czarnecki. All rights reserved.

"""QUAL-X-P2 closed-world ingress matrix (G1–G8) — regression linkage catalog only."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

QUAL_X_P2_START_HEAD: str = "9b9a85a98a3817744f542341c7a4fa68b15bbca3"

_P2 = "tests/qualification/qual_x/test_qual_x_p2_negative_ingress_matrix.py"
_G1 = "tests/unit/runtime/governance/test_runtime_execution_policy_admission.py"
_G2A = "tests/unit/runtime/governance/test_gr2_execution_admission_governance.py"
_G2B = "tests/unit/runtime/governance/test_multi_agent_coordination_collaborative_authority.py"
_G3A = _P2
_G3B = "tests/unit/runtime/governance/test_gr3_canonical_inner_enforcement.py"
_G4 = "tests/unit/tools/test_qualified_tool_invocation_resolver.py"
_G5A = (
    "tests/qualification/existing_capability_configuration/"
    "test_configuration_realization_certification.py"
)
_G5B = "tests/qualification/tenant_x/test_tenant_x_r2_fast.py"
_G6 = "tests/qualification/tenant_x/_r4_adversarial.py"
_G7A = "tests/unit/runtime/governance/test_control_plane_mutation_approval.py"
_G7B = "tests/unit/runtime/governance/test_control_plane_mutation_authorization.py"
_G8A = "tests/qualification/inspect_01/test_inspect_01b_c1_governance_integrity.py"
_G8B = "tests/qualification/control_planes/ctrl_x/test_ctrl_x_r3_r2_tenant_scope_fail_closed.py"
_CTRL = "tests/qualification/control_planes/ctrl_x/test_ctrl_x_r1_security_composition.py"


def _nid(module: str, test_name: str) -> str:
    return f"{module}::{test_name}"


class QualXP2EvidenceStrength(StrEnum):
    DIRECT = "DIRECT INVARIANT PROOF"
    HISTORICAL = "HISTORICAL EVIDENCE"
    SUPPORTING = "SUPPORTING STATIC EVIDENCE"


@dataclass(frozen=True, slots=True)
class QualXP2IngressEntry:
    ingress_id: str
    title: str
    canonical_owner: str
    public_contract: str
    sanctioned_caller: str
    authority_boundary: str
    effect_boundary: str
    primary_proof_node_ids: tuple[str, ...]
    strength: QualXP2EvidenceStrength
    coverage_gap: str
    frz_criteria: tuple[str, ...]


QUAL_X_P2_INGRESS_CATALOG: tuple[QualXP2IngressEntry, ...] = (
    QualXP2IngressEntry(
        ingress_id="G1",
        title="Request identity admission",
        canonical_owner="Runtime execution policy admission + CTRL-X tenant scope",
        public_contract="intergrax.contracts.runtime_execution_policy_admission",
        sanctioned_caller="Root execution authority composition; application security middleware",
        authority_boundary="RuntimeExecutionPolicyAdmissionRequest validation; tenant-scoped defense hooks",
        effect_boundary="No root launch / no protected hook continuation on invalid identity",
        primary_proof_node_ids=(
            _nid(_P2, "test_qual_x_p2_g1_runtime_policy_admission_rejects_blank_tenant_before_evaluator"),
            _nid(_G8B, "test_r3_r2_defense_plugin_tenant_fail_closed_before_inspect"),
            _nid(_CTRL, "test_r1_sec_05_tenant_scoped_fail_closed_defense_still_blocks"),
        ),
        strength=QualXP2EvidenceStrength.DIRECT,
        coverage_gap="Platform-wide HTTP ingress negatives remain in PROD-Q scoped suites (HISTORICAL).",
        frz_criteria=("FRZ-SEC-01", "FRZ-TEN-01", "FRZ-REG-03"),
    ),
    QualXP2IngressEntry(
        ingress_id="G2",
        title="Governance authorization",
        canonical_owner="GOV-X2 / GR-2 governance resolution",
        public_contract="intergrax.contracts.runtime_policy.PolicyDecision",
        sanctioned_caller="Runtime governance composition; collaborative authority coordinator",
        authority_boundary="Policy vs authority conjunction; kernel governance resolution",
        effect_boundary="DENY / REQUIRE_HUMAN execute zero protected effects",
        primary_proof_node_ids=(
            _nid(_G2A, "test_gr2_pluggable_policy_port_allow_vs_deny_without_consumer_change"),
            _nid(_G2B, "test_policy_allow_cannot_override_authority_deny"),
            _nid(_G2B, "test_deny_has_zero_downstream_effects_with_collaborative_workspace"),
        ),
        strength=QualXP2EvidenceStrength.DIRECT,
        coverage_gap="GR-10 strategy orchestration negatives scoped to strategy qual (not re-certified here).",
        frz_criteria=("FRZ-GOV-01", "FRZ-SEC-02", "FRZ-REG-03"),
    ),
    QualXP2IngressEntry(
        ingress_id="G3",
        title="Execution admission",
        canonical_owner="GR-2 / GR-3 execution guard composition",
        public_contract="intergrax.contracts.root_execution_launch",
        sanctioned_caller="DefaultRootExecutionLauncher + canonical intake port",
        authority_boundary="RootExecutionAuthorityAdmissionPort",
        effect_boundary="Canonical execution intake not dispatched after DENY",
        primary_proof_node_ids=(
            _nid(_P2, "test_qual_x_p2_q05_policy_deny_blocks_intake_without_physical_dispatch"),
            _nid(_G3B, "test_deny_and_require_human_execute_zero_effects"),
            _nid(
                "tests/unit/runtime/governance/test_gr2_r3_root_execution_launcher.py",
                "test_launcher_deny_skips_intake[root.execution.agent]",
            ),
        ),
        strength=QualXP2EvidenceStrength.DIRECT,
        coverage_gap="Full execution guard surface = GOV-X2 historical matrix (not replayed in P2 batch).",
        frz_criteria=("FRZ-EXE-01", "FRZ-REG-03"),
    ),
    QualXP2IngressEntry(
        ingress_id="G4",
        title="Tool and capability invocation",
        canonical_owner="Tools qualified invocation + execution-bound catalog gateway",
        public_contract="intergrax.contracts.tools.qualified_tool_invocation",
        sanctioned_caller="DefaultQualifiedToolInvocationResolver; marketplace qualified handler composition",
        authority_boundary="require_non_empty_text on resolver mapping; typed material request validation",
        effect_boundary="No ExecutionBoundCatalogToolInvokeRequest on invalid identity/material",
        primary_proof_node_ids=(
            _nid(_G4, "test_resolve_rejects_blank_tenant_before_catalog_request[]"),
            _nid(_G4, "test_qualified_material_request_rejects_blank_tenant_at_contract"),
            _nid(_G4, "test_resolve_rejects_non_basemodel_material_before_catalog_request"),
        ),
        strength=QualXP2EvidenceStrength.DIRECT,
        coverage_gap="Physical ToolRuntime dispatch negatives remain in marketplace unit + AW qual (scoped).",
        frz_criteria=("FRZ-EXE-02", "FRZ-SEC-03", "FRZ-REG-03"),
    ),
    QualXP2IngressEntry(
        ingress_id="G5",
        title="Provider and integration resolution",
        canonical_owner="Existing capability configuration realization",
        public_contract="integrations existing-capability configuration realization port",
        sanctioned_caller="ExistingCapabilityConfigurationRealizationFacade",
        authority_boundary="Control-plane authorization port + strategy tenant/provider guards",
        effect_boundary="Resolver/strategy realize_calls == 0 on DENY / mismatch",
        primary_proof_node_ids=(
            _nid(_G5A, "test_cert_a02_authorization_deny_zero_downstream"),
            _nid(_G5A, "test_cert_t04_strategy_output_tenant_widening_rejected"),
            _nid(_G5A, "test_cert_p03_strategy_output_provider_mismatch_rejected"),
            _nid(_G5B, "test_tx_r2_q05_adversarial_cross_tenant_credential_broker"),
        ),
        strength=QualXP2EvidenceStrength.DIRECT,
        coverage_gap="Live provider qualification = PROD-Q Dimension B (OPEN).",
        frz_criteria=("FRZ-TEN-02", "FRZ-CTR-01", "FRZ-REG-03"),
    ),
    QualXP2IngressEntry(
        ingress_id="G6",
        title="Asynchronous continuation",
        canonical_owner="Runtime background execution + TENANT-X-R4",
        public_contract="intergrax.runtime.background_execution.bootstrap",
        sanctioned_caller="bootstrap_background_execution; vendor knowledge sync worker",
        authority_boundary="BackgroundExecutionTenantMismatchError",
        effect_boundary="No cross-tenant transport ref persistence",
        primary_proof_node_ids=(
            _nid(
                "tests/qualification/tenant_x/test_tenant_x_r4_fast.py",
                "test_tx_r4_q07_adversarial_background_identity",
            ),
            _nid(
                "tests/unit/runtime/nexus/tools/test_harness_w4_r1_production_tool_admission_behavior.py",
                "test_w4_r1_reject_second_invocation_without_physical_executor_call",
            ),
        ),
        strength=QualXP2EvidenceStrength.DIRECT,
        coverage_gap="CTRL-X W4 node referenced via catalog row — collectability enforced in P2 batch.",
        frz_criteria=("FRZ-EXE-03", "FRZ-TEN-03", "FRZ-REG-03"),
    ),
    QualXP2IngressEntry(
        ingress_id="G7",
        title="Control-plane mutations",
        canonical_owner="Control plane mutation approval / authorization",
        public_contract="intergrax.contracts.control_plane_mutation",
        sanctioned_caller="Runtime governance CPMA ports",
        authority_boundary="Approver identity + evidence binding",
        effect_boundary="Mutation not applied on authorization DENY",
        primary_proof_node_ids=(
            _nid(_G7A, "test_cpma_2_non_user_approver_rejected"),
            _nid(_G7A, "test_cpma_4_wrong_tenant_approver_rejected"),
            _nid(_G7B, "test_cp2_deny_preserved"),
            _nid(_G7B, "test_cp4_missing_principal_fail_closed"),
        ),
        strength=QualXP2EvidenceStrength.DIRECT,
        coverage_gap="Full CP registry = CTRL-X catalog (historical closure @ CTRL-X evidence SHA).",
        frz_criteria=("FRZ-GOV-02", "FRZ-SEC-04", "FRZ-REG-03"),
    ),
    QualXP2IngressEntry(
        ingress_id="G8",
        title="Tenant-isolated state and evidence",
        canonical_owner="INSPECT-01 / TENANT-X",
        public_contract="intergrax.contracts.governance_audit_read; runtime inspection federation",
        sanctioned_caller="FederatedRuntimeInspectionReadService",
        authority_boundary="GovernanceAuditReadTenantBoundaryError",
        effect_boundary="Cross-tenant execution_id probe returns no foreign audit rows",
        primary_proof_node_ids=(
            _nid(_G8A, "test_c1_q2_cross_tenant_same_execution_id_fail_closed"),
            _nid(_G8A, "test_c1_q1_tenant_filtered_by_governance_read_port"),
            _nid(
                "tests/qualification/tenant_x/test_tenant_x_final_fast.py",
                "test_tx_final_q03_adversarial_matrix_f1_f10_closed_world",
            ),
            _nid(
                "tests/qualification/tenant_x/test_tenant_x_final_current_head.py",
                "test_tx_final_head_q01_final_ready_for_audit",
            ),
        ),
        strength=QualXP2EvidenceStrength.DIRECT,
        coverage_gap=(
            "TENANT-X FINAL historical production delta @ 5ccb72fc…; "
            "current-HEAD behavioral replay via test_tenant_x_final_current_head + FINAL fast gates."
        ),
        frz_criteria=("FRZ-TEN-04", "FRZ-TRC-01", "FRZ-REG-03"),
    ),
)


def qual_x_p2_all_proof_pytest_node_ids() -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in QUAL_X_P2_INGRESS_CATALOG:
        for node_id in entry.primary_proof_node_ids:
            if node_id not in seen:
                seen.add(node_id)
                ordered.append(node_id)
    return tuple(ordered)
