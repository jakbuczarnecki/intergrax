# © Artur Czarnecki. All rights reserved.

"""GR-12 control-plane mutation production surface inventory SSOT (A1 foundation)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class Gr12Applicability(StrEnum):
    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    REQUIRES_ARCHITECTURE_DECISION = "REQUIRES_ARCHITECTURE_DECISION"


class Gr12CoverageStatus(StrEnum):
    DISCOVERED = "DISCOVERED"
    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    GAP = "GAP"
    WIRED_NOT_QUALIFIED = "WIRED_NOT_QUALIFIED"
    QUALIFIED = "QUALIFIED"
    ARCHITECTURE_DECISION_REQUIRED = "ARCHITECTURE_DECISION_REQUIRED"


GR12_CONSEQUENTIAL_MUTATION_DEFINITION: Final[str] = (
    "Operacja poza standardowym strategy execution flow (GR-10), która zmienia "
    "aktywną konfigurację, authority, dostępne capability, routing, capacity, "
    "lifecycle środowiska lub operator-visible task control dla bieżących lub "
    "przyszłych wykonań — wykonywana przez domain executor po (docelowo) wspólnej "
    "ewaluacji CONTROL_PLANE_MUTATION."
)

GR12_CANONICAL_BOUNDARY_CLASS: Final[str] = (
    "intergrax.runtime.governance.control_plane_mutation_authorization."
    "ControlPlaneMutationAuthorizationBoundary"
)

GR12_CANONICAL_POLICY_PORT: Final[str] = (
    "intergrax.contracts.control_plane_mutation.ControlPlaneMutationPolicyEvaluator"
)

GR12_CONTRACT_DECISION: Final[str] = "VARIANT_A_EXISTING_CLA04_CONTRACT_REUSABLE"

GR12_ADR_REQUIRED: Final[bool] = False


@dataclass(frozen=True, slots=True)
class Gr12ControlPlaneSurface:
    path_id: str
    surface: str
    production_entrypoint: str
    mutation: str
    consequential: bool
    current_guard: str
    current_authority: str
    audit_evidence: str
    applicability: Gr12Applicability
    coverage: Gr12CoverageStatus
    recommended_owner: str
    future_remediation: str
    qualification_proof: str = ""


@dataclass(frozen=True, slots=True)
class Gr12ExecutionPlaneExclusion:
    path_id: str
    surface: str
    reason: str


@dataclass(frozen=True, slots=True)
class Gr12GovernanceMechanism:
    mechanism: str
    reusable: str
    layer: str
    limitation: str


@dataclass(frozen=True, slots=True)
class Gr12NextRemediation:
    task_name: str
    exact_blocker: str
    why_highest: str


GR12_A3_NEXT_REMEDIATION: Gr12NextRemediation = Gr12NextRemediation(
    task_name=(
        "GR-12-A4 — Residual Control-Plane Governance: Catalog Hot Reload, "
        "Vector Administration & Specialized Domain Decisions"
    ),
    exact_blocker=(
        "Core AD/AHI/ECP/Task Control surfaces qualified in GR-12-A3; residual "
        "catalog hot reload, vector admin, and memory specialized mutations remain."
    ),
    why_highest=(
        "Next bounded slice after core production control-plane qualification."
    ),
)

GR12_A4_RESIDUAL_PATH_IDS: Final[tuple[str, ...]] = (
    "CP-PLUGIN-CATALOG-HOT-RELOAD",
    "CP-VECTOR-INDEX-ADMIN",
    "CP-MEM-SPECIALIZED-MUTATION",
)

GR12_A4_NEXT_REMEDIATION: Gr12NextRemediation = Gr12NextRemediation(
    task_name=(
        "GR-12-A4-R2 — Vector Administration Governance Architecture & CLA-04 Mapping Decision"
    ),
    exact_blocker=(
        "Vector index administration port exists without CLA-04 resource mapping and "
        "governed operator admin API; destructive lifecycle semantics undecided."
    ),
    why_highest=(
        "Catalog hot reload qualification closed in GR-12-A4-R1-R1; Vector is the next "
        "residual consequential control-plane surface."
    ),
)

GR12_A4_R1_R1_QUALIFICATION_PROOF: Final[str] = (
    "tests/qualification/governance/gr12/test_gr12_a4_r1_r1_catalog_revision_and_identity_qualification.py"
)

GR12_A4_R1_R1_EXECUTION_PROOF_NODES: Final[tuple[str, ...]] = (
    "tests/unit/integrations/registry/test_catalog_revision_and_mutation.py::test_rev7_aba_generations_increase_when_digest_returns",
    "tests/unit/integrations/registry/test_catalog_atomic_registration_concurrency.py::test_reg_conc_1_concurrent_same_slug_exactly_one_success",
    "tests/unit/integrations/registry/test_catalog_atomic_registration_concurrency.py::test_reg_conc_3_generation_increases_exactly_once_from_initial",
    "tests/unit/applications/test_catalog_hot_reload_governance.py::test_chr_r1r1_4_explicit_principal_reaches_cla04_request",
    "tests/unit/applications/test_catalog_hot_reload_governance.py::test_chr_r1r1_5_missing_principal_fails_closed",
    "tests/unit/applications/test_catalog_hot_reload_bypass_inventory.py::test_chr16_live_operator_catalog_mutation_paths_only_governed_reload",
)

GR12_A4_R1_QUALIFICATION_PROOF: Final[str] = GR12_A4_R1_R1_QUALIFICATION_PROOF


GR12_A1_NEXT_REMEDIATION: Gr12NextRemediation = Gr12NextRemediation(
    task_name=(
        "GR-12-A2 — Mandatory CLA-04 Composition, Live Wiring & Evidence Correlation"
    ),
    exact_blocker=(
        "CLA-04 contracts and ControlPlaneMutationAuthorizationBoundary exist; "
        "production hosts may omit boundary, domain-local evaluators diverge, "
        "GR-8 GovernanceDecisionEvidenceFact not correlated for CONTROL_PLANE_MUTATION, "
        "scenario CP remains GAP."
    ),
    why_highest=(
        "Narrowest slice: mandatory composition + single evaluator wiring path before "
        "per-surface qualification proofs."
    ),
)


GR12_CONTROL_PLANE_SURFACES: tuple[Gr12ControlPlaneSurface, ...] = (
    Gr12ControlPlaneSurface(
        path_id="CP-AD-ACTIVATE",
        surface="Agent Distribution",
        production_entrypoint="intergrax.agent_distribution.admin_service.AgentPlatformAdminService.activate_revision",
        mutation="agent_distribution.activate_runtime_revision",
        consequential=True,
        current_guard="authorize_scoped_control_plane_mutation + required boundary",
        current_authority="RequestIdentity + ApplicationEnvironmentTenantResolver",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="agents/Tier-3 composition",
        future_remediation="",
        qualification_proof="tests/unit/agent_distribution/test_agent_distribution_control_plane_governance.py::test_ad1_activation_allow_commits_once",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-AD-ROLLBACK",
        surface="Agent Distribution",
        production_entrypoint="intergrax.agent_distribution.admin_service.AgentPlatformAdminService.rollback_revision",
        mutation="agent_distribution.rollback_runtime_revision",
        consequential=True,
        current_guard="authorize_scoped_control_plane_mutation + required boundary",
        current_authority="RequestIdentity + tenant resolver",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="agents/Tier-3 composition",
        future_remediation="",
        qualification_proof="tests/unit/agent_distribution/test_agent_distribution_control_plane_governance.py::test_ad4_rollback_allow_commits_once",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-AD-INSTALL",
        surface="Agent Distribution",
        production_entrypoint="intergrax.agent_distribution.admin_service.AgentPlatformAdminService.install_agent",
        mutation="agent_distribution.install_agent",
        consequential=True,
        current_guard="authorize_scoped_control_plane_mutation",
        current_authority="RequestIdentity + tenant resolver",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="",
        qualification_proof="tests/unit/agent_distribution/test_agent_distribution_desired_state_remediation.py::test_ads1_install_allow_one_mutation_sequence",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-AD-BIND",
        surface="Agent Distribution",
        production_entrypoint="intergrax.agent_distribution.admin_service.AgentPlatformAdminService.bind_agent",
        mutation="agent_distribution.bind_agent",
        consequential=True,
        current_guard="authorize_scoped_control_plane_mutation",
        current_authority="RequestIdentity + tenant resolver",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="",
        qualification_proof="tests/unit/agent_distribution/test_agent_distribution_desired_state_remediation.py::test_ads6_bind_allow_one_create",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-AD-BINDING-CONFIG",
        surface="Agent Distribution",
        production_entrypoint="intergrax.agent_distribution.admin_service.AgentPlatformAdminService.update_binding_config",
        mutation="agent_distribution.update_binding_config",
        consequential=True,
        current_guard="authorize_scoped_control_plane_mutation",
        current_authority="RequestIdentity + tenant resolver",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="",
        qualification_proof="tests/unit/agent_distribution/test_agent_distribution_desired_state_remediation.py::test_ads9_update_config_allow_one_update",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-AD-ENABLE-BINDING",
        surface="Agent Distribution",
        production_entrypoint="intergrax.agent_distribution.admin_service.AgentPlatformAdminService.enable_binding",
        mutation="agent_distribution.enable_binding",
        consequential=True,
        current_guard="authorize_scoped_control_plane_mutation",
        current_authority="RequestIdentity + tenant resolver",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="",
        qualification_proof="tests/unit/agent_distribution/test_agent_distribution_desired_state_remediation.py::test_ads14_enable_allow_once",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-AD-DISABLE-BINDING",
        surface="Agent Distribution",
        production_entrypoint="intergrax.agent_distribution.admin_service.AgentPlatformAdminService.disable_binding",
        mutation="agent_distribution.disable_binding",
        consequential=True,
        current_guard="authorize_scoped_control_plane_mutation",
        current_authority="RequestIdentity + tenant resolver",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="",
        qualification_proof="tests/unit/agent_distribution/test_agent_distribution_desired_state_remediation.py::test_ads16_disable_allow_once",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-AD-ADMIT",
        surface="Agent Distribution",
        production_entrypoint="intergrax.agent_distribution.admin_service.AgentPlatformAdminService.activate_revision",
        mutation="agent_distribution.admit_runtime_revision",
        consequential=True,
        current_guard="authorize_scoped_control_plane_mutation",
        current_authority="RequestIdentity + tenant resolver",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="",
        qualification_proof="tests/unit/agent_distribution/test_agent_distribution_build_remediation.py::test_r1_7_reference_admission_allow_persists_revision",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-AD-BUILD",
        surface="Agent Distribution",
        production_entrypoint="intergrax.agent_distribution.admin_service.AgentPlatformAdminService.build_application_revision",
        mutation="agent_distribution.build_runtime_revision",
        consequential=True,
        current_guard="authorize_scoped_control_plane_mutation",
        current_authority="RequestIdentity + tenant resolver",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="",
        qualification_proof="tests/unit/agent_distribution/test_agent_distribution_build_remediation.py::test_adb1_build_allow_persists_once",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-AD-DRAIN",
        surface="Agent Distribution",
        production_entrypoint="intergrax.agent_distribution.admin_service.AgentPlatformAdminService.complete_revision_drain",
        mutation="agent_distribution.complete_drain",
        consequential=True,
        current_guard="authorize_scoped_control_plane_mutation",
        current_authority="RequestIdentity + tenant resolver",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="",
        qualification_proof="tests/unit/agent_distribution/test_agent_distribution_drain_recovery_remediation.py::test_dr1_complete_drain_allow_stops_once",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-AD-POST-CUTOVER-FAIL",
        surface="Agent Distribution",
        production_entrypoint="intergrax.agent_distribution.admin_service.AgentPlatformAdminService.handle_post_cutover_failure",
        mutation="agent_distribution.mark_post_cutover_failure",
        consequential=True,
        current_guard="authorize_scoped_control_plane_mutation",
        current_authority="RequestIdentity + tenant resolver",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="",
        qualification_proof="tests/unit/agent_distribution/test_agent_distribution_drain_recovery_remediation.py::test_dr11_failure_mark_is_control_plane_mutation",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-AHI-APPLY",
        surface="AHI (Adaptive Harness Intelligence)",
        production_entrypoint="intergrax.runtime.architecture.runtime_governance_bridge.RuntimeArchitectureGovernanceBridge.apply_approved",
        mutation="ahi.apply_profile",
        consequential=True,
        current_guard="authorize_scoped_ahi_control_plane_mutation; boundary required",
        current_authority="RequestIdentity + AhiTenantScopeResolver",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="runtime/adaptive + host composition",
        future_remediation="",
        qualification_proof="tests/unit/runtime/adaptive/test_ahi_control_plane_governance.py::test_ahicpm1_apply_allow_executes_with_evidence",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-AHI-ROLLBACK",
        surface="AHI",
        production_entrypoint="intergrax.runtime.architecture.runtime_governance_bridge.RuntimeArchitectureGovernanceBridge.rollback_profile",
        mutation="ahi.rollback_profile",
        consequential=True,
        current_guard="authorize_scoped_ahi_control_plane_mutation",
        current_authority="RequestIdentity + AhiTenantScopeResolver",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="runtime/adaptive",
        future_remediation="",
        qualification_proof="tests/unit/runtime/adaptive/test_ahi_control_plane_governance.py::test_ahicpm10_rollback_allow_uses_canonical_previous",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-ECP-SCALE-K8S",
        surface="ECP (Elastic Capacity)",
        production_entrypoint="intergrax.runtime.capacity.governed_capacity_mutation.GovernedCapacityMutationExecutor.scale_k8s_deployment",
        mutation="ecp.scale_k8s_deployment",
        consequential=True,
        current_guard="authorize_scoped_ecp_control_plane_mutation; stale-state CAS in scheduler",
        current_authority="RequestIdentity + EcpResourceTenantResolver",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence + scheduler blocker codes",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="runtime/capacity",
        future_remediation="",
        qualification_proof="tests/unit/runtime/capacity/test_ecp_control_plane_governance.py::test_ecp_cpm1_allow_k8s_exact_target_and_evidence",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-ECP-SCALE-CELERY",
        surface="ECP",
        production_entrypoint="intergrax.runtime.capacity.governed_capacity_mutation.GovernedCapacityMutationExecutor.scale_celery_workers",
        mutation="ecp.scale_celery_workers",
        consequential=True,
        current_guard="authorize_scoped_ecp_control_plane_mutation",
        current_authority="RequestIdentity + EcpResourceTenantResolver",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="runtime/capacity",
        future_remediation="",
        qualification_proof="tests/unit/runtime/capacity/test_ecp_control_plane_governance.py::test_ecp_cpm9_allow_celery_exact_target_and_evidence",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-TASK-CANCEL",
        surface="Live task control",
        production_entrypoint="intergrax.applications._shared.task_control.cancel_task_execution",
        mutation="task_control.cancel_task_execution",
        consequential=True,
        current_guard="ControlPlaneMutationAuthorizationBoundary when wired",
        current_authority="RequestIdentity; tenant from task scope",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="applications harness wiring",
        future_remediation="",
        qualification_proof="tests/unit/applications/test_task_control_governed_cancel.py::test_taskcpm_c1_allow_matching_binding_requests_cancel_once",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-TASK-RESUME",
        surface="Live task control",
        production_entrypoint="intergrax.applications._shared.task_control.resume_task_execution",
        mutation="task_control.resume_task_execution",
        consequential=True,
        current_guard="boundary + approval coordinator optional",
        current_authority="RequestIdentity + checkpoint pause_record binding",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="applications harness",
        future_remediation="",
        qualification_proof="tests/unit/applications/test_task_control_governed_resume.py::test_taskcpm_r1_allow_exact_checkpoint_invokes_runner_once",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-TASK-AUTONOMY",
        surface="Live task control",
        production_entrypoint="intergrax.applications._shared.task_control.set_task_autonomy",
        mutation="task_control.set_task_autonomy",
        consequential=True,
        current_guard="ControlPlaneMutationAuthorizationBoundary when wired",
        current_authority="RequestIdentity",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="applications harness",
        future_remediation="",
        qualification_proof="tests/unit/applications/test_task_control_governed_autonomy.py::test_taskcpm_a1_allow_matching_binding_changes_autonomy_once",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-REF-ACTIVATE",
        surface="Reference production lifecycle",
        production_entrypoint="intergrax.applications._shared.reference_production_lifecycle.ReferenceProductionLifecycleLauncher.deploy_and_activate",
        mutation="agent_distribution.activate_runtime_revision (via AD helpers)",
        consequential=True,
        current_guard="required mutation boundary + authorize_scoped_control_plane_mutation",
        current_authority="composition-injected boundary",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="applications reference host",
        future_remediation="",
        qualification_proof="tests/unit/agent_distribution/test_agent_distribution_activation_remediation.py::test_adr7_reference_production_allow_commits",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-HOST-BOUNDARY-OPTIONAL",
        surface="Harness host composition",
        production_entrypoint="intergrax.applications._shared.harness_control_plane_governance_wiring.build_harness_control_plane_governance",
        mutation="(composition) omission allows ungoverned downstream mutations",
        consequential=True,
        current_guard="PRODUCT auto-builds boundary; wire_harness_task_control requires boundary when routes enabled",
        current_authority="build_harness_control_plane_governance + wire_harness_task_control",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="applications composition root",
        future_remediation="",
        qualification_proof="tests/qualification/governance/gr12/test_gr12_a2_mandatory_cla04_composition.py::test_gr12_a2_product_task_control_wiring_requires_canonical_boundary",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-ECP-BOUNDARY-OPTIONAL",
        surface="Production capacity wiring",
        production_entrypoint="intergrax.applications._shared.production_capacity_governance_wiring.build_production_capacity_governance",
        mutation="ECP mutations without injected boundary",
        consequential=True,
        current_guard="PRODUCT+adapters require explicit boundary; resolve_production_capacity_wiring fails closed without authority",
        current_authority="build_production_capacity_governance + GovernedCapacityMutationExecutor",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="applications capacity wiring",
        future_remediation="",
        qualification_proof="tests/unit/runtime/capacity/test_ecp_control_plane_governance.py::test_ecp_cpm16_product_without_authority_fails_at_wiring",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-PLUGIN-CATALOG-HOT-RELOAD",
        surface="Integration catalog",
        production_entrypoint=(
            "intergrax.applications._shared.catalog_hot_reload_service."
            "CatalogHotReloadService.reload"
        ),
        mutation="integration_catalog.hot_reload",
        consequential=True,
        current_guard=(
            "CatalogHotReloadService + ControlPlaneMutationAuthorizationBoundary; "
            "revision CAS in registry"
        ),
        current_authority="composition-injected CLA-04 boundary",
        audit_evidence="ControlPlaneMutationAuthorizationEvidence + CatalogHotReloadResult",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.QUALIFIED,
        recommended_owner="integrations registry + applications composition",
        future_remediation="",
        qualification_proof=GR12_A4_R1_QUALIFICATION_PROOF,
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-MEM-SPECIALIZED-MUTATION",
        surface="Memory LTM/entity mutations",
        production_entrypoint="intergrax.memory.memory_specialized_mutation_governance",
        mutation="memory domain writes/deletes/compactions",
        consequential=True,
        current_guard="MemoryGovernanceEvaluationRequest (separate contract)",
        current_authority="memory control scope + memory policy port",
        audit_evidence="MemoryGovernanceDecision (not GR-8 fact)",
        applicability=Gr12Applicability.REQUIRES_ARCHITECTURE_DECISION,
        coverage=Gr12CoverageStatus.ARCHITECTURE_DECISION_REQUIRED,
        recommended_owner="memory domain",
        future_remediation="GR-12-A4-R3 specialized memory governance architecture ADR",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-MARKETPLACE-ACQUIRE",
        surface="Marketplace acquisition",
        production_entrypoint="intergrax.marketplace.acquisition.service.MachineCapabilityAcquisitionService",
        mutation="recommendation/handoff only — no install authority in service",
        consequential=False,
        current_guard="CapabilityGovernanceDecision on candidates",
        current_authority="catalog governance evaluator",
        audit_evidence="GovernanceDecisionEvidence (capability catalog)",
        applicability=Gr12Applicability.NOT_APPLICABLE,
        coverage=Gr12CoverageStatus.NOT_APPLICABLE,
        recommended_owner="marketplace",
        future_remediation="Revisit if install becomes live mutation API",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-BOOT-PLUGIN-REGISTER",
        surface="Plugin bootstrap",
        production_entrypoint="intergrax.core.plugins.discovery.register_plugins_with_report",
        mutation="process-start registry population",
        consequential=False,
        current_guard="composition / profile validation (_validate_requested_plugin_kinds_activated)",
        current_authority="host startup only",
        audit_evidence="registration report",
        applicability=Gr12Applicability.NOT_APPLICABLE,
        coverage=Gr12CoverageStatus.NOT_APPLICABLE,
        recommended_owner="core/plugins",
        future_remediation="N/A unless hot plugin admission added",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-POLICY-BUNDLE-COMPOSE",
        surface="Runtime policy",
        production_entrypoint="intergrax.applications._shared.policy_wiring",
        mutation="immutable bundle at host compose (not live API)",
        consequential=False,
        current_guard="deployment-time composition",
        current_authority="environment profile",
        audit_evidence="bundle digest in execution evidence receipts",
        applicability=Gr12Applicability.NOT_APPLICABLE,
        coverage=Gr12CoverageStatus.NOT_APPLICABLE,
        recommended_owner="applications policy wiring",
        future_remediation="N/A until live policy activation API exists",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-VECTOR-INDEX-ADMIN",
        surface="Vector index administration",
        production_entrypoint=(
            "intergrax.integrations.contracts.vector_index_administration."
            "VectorIndexAdministration"
        ),
        mutation="prepare_index (idempotent create/align); destructive ops provider-internal only",
        consequential=True,
        current_guard="provider adapter + bootstrap callers; no CLA-04 bridge",
        current_authority="integration credentials",
        audit_evidence="provider logs only",
        applicability=Gr12Applicability.REQUIRES_ARCHITECTURE_DECISION,
        coverage=Gr12CoverageStatus.ARCHITECTURE_DECISION_REQUIRED,
        recommended_owner="integrations/RAG",
        future_remediation="GR-12-A4-R2 vector admin CLA-04 mapping / operator exposure ADR",
    ),
)


GR12_EXECUTION_PLANE_EXCLUSIONS: tuple[Gr12ExecutionPlaneExclusion, ...] = (
    Gr12ExecutionPlaneExclusion(
        path_id="EP-UAEP-TOOL",
        surface="UAEP / tool invoke",
        reason="GR-10 inner governance + MSE; not control-plane mutation",
    ),
    Gr12ExecutionPlaneExclusion(
        path_id="EP-MSE-ORCH",
        surface="Orchestration meaningful side effect",
        reason="GR-10 ORCHESTRATION GEP coverage; distinct evaluation class",
    ),
    Gr12ExecutionPlaneExclusion(
        path_id="EP-TASK-PAUSE-HITL",
        surface="Execution pause for HITL",
        reason="Execution lifecycle + checkpoint; resume is CP (CP-TASK-RESUME)",
    ),
    Gr12ExecutionPlaneExclusion(
        path_id="EP-INFERENCE-CALL",
        surface="PRE_MODEL / inference executor",
        reason="GR-10 INFERENCE strategy scope",
    ),
)


GR12_EXISTING_MECHANISMS: tuple[Gr12GovernanceMechanism, ...] = (
    Gr12GovernanceMechanism(
        mechanism="ControlPlaneMutationAuthorizationBoundary",
        reusable="YES — canonical pre-mutation evaluation",
        layer="intergrax/runtime/governance",
        limitation="Not mandatory on all hosts; no unified GR-8 fact emission",
    ),
    Gr12GovernanceMechanism(
        mechanism="ControlPlaneMutationPolicyEvaluator + BundleBackedControlPlaneMutationEvaluator",
        reusable="YES — pluggable policy via bundle match_action=mutation_type",
        layer="contracts + runtime/governance",
        limitation="Requires task_id/run_id for bundle mapping",
    ),
    Gr12GovernanceMechanism(
        mechanism="ControlPlaneMutationApprovalCoordinator",
        reusable="YES — HITL continuation for REQUIRE_HUMAN/ESCALATE",
        layer="runtime/governance",
        limitation="In-process only; not durable",
    ),
    Gr12GovernanceMechanism(
        mechanism="Domain authorize_scoped_* helpers (AD/AHI/ECP)",
        reusable="PARTIAL — shared request builder + boundary call",
        layer="domain modules",
        limitation="Duplicated tenant-scope patterns",
    ),
    Gr12GovernanceMechanism(
        mechanism="MemoryGovernanceEvaluationRequest",
        reusable="NO for CLA-04 without ADR",
        layer="memory",
        limitation="Different semantics and evidence model",
    ),
    Gr12GovernanceMechanism(
        mechanism="AgentRuntimeGovernancePort",
        reusable="NO — execution-time agent governance",
        layer="runtime/agent_governance",
        limitation="GR-10 domain",
    ),
)


def gr12_path_ids() -> tuple[str, ...]:
    return tuple(row.path_id for row in GR12_CONTROL_PLANE_SURFACES)


def gr12_applicable_surfaces() -> tuple[Gr12ControlPlaneSurface, ...]:
    return tuple(
        row
        for row in GR12_CONTROL_PLANE_SURFACES
        if row.applicability is Gr12Applicability.APPLICABLE
    )
