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
        coverage=Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
        recommended_owner="agents/Tier-3 composition",
        future_remediation="GR-12-A3 AD activation qualification slice",
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
        coverage=Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
        recommended_owner="agents/Tier-3 composition",
        future_remediation="GR-12-A3 AD rollback slice",
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
        coverage=Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="GR-12-A3",
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
        coverage=Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="GR-12-A3",
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
        coverage=Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="GR-12-A3",
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
        coverage=Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="GR-12-A3",
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
        coverage=Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="GR-12-A3",
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
        coverage=Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="GR-12-A3",
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
        coverage=Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="GR-12-A3",
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
        coverage=Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="GR-12-A3",
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
        coverage=Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
        recommended_owner="agent_distribution",
        future_remediation="GR-12-A3",
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
        coverage=Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
        recommended_owner="runtime/adaptive + host composition",
        future_remediation="GR-12-A3 AHI apply qualification",
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
        coverage=Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
        recommended_owner="runtime/adaptive",
        future_remediation="GR-12-A3",
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
        coverage=Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
        recommended_owner="runtime/capacity",
        future_remediation="GR-12-A3 ECP qualification (extends ECP-CPM proofs)",
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
        coverage=Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
        recommended_owner="runtime/capacity",
        future_remediation="GR-12-A3",
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
        coverage=Gr12CoverageStatus.GAP,
        recommended_owner="applications harness wiring",
        future_remediation="Mandatory boundary on PRODUCT hosts (nullable today)",
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
        coverage=Gr12CoverageStatus.GAP,
        recommended_owner="applications harness",
        future_remediation="GR-12-A2 composition",
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
        coverage=Gr12CoverageStatus.GAP,
        recommended_owner="applications harness",
        future_remediation="GR-12-A2",
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
        coverage=Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
        recommended_owner="applications reference host",
        future_remediation="GR-12-A3",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-HOST-BOUNDARY-OPTIONAL",
        surface="Harness host composition",
        production_entrypoint="intergrax.applications._shared.harness_control_plane_governance_wiring.build_harness_control_plane_governance",
        mutation="(composition) omission allows ungoverned downstream mutations",
        consequential=True,
        current_guard="PRODUCT profile auto-builds boundary; else None",
        current_authority="ApplicationEnvironmentProfile.application_profile",
        audit_evidence="none when boundary None",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.GAP,
        recommended_owner="applications composition root",
        future_remediation="GR-12-A2 fail-closed when consequential routes enabled",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-ECP-BOUNDARY-OPTIONAL",
        surface="Production capacity wiring",
        production_entrypoint="intergrax.applications._shared.production_capacity_governance_wiring.build_production_capacity_governance",
        mutation="ECP mutations without injected boundary",
        consequential=True,
        current_guard="GovernedCapacityMutationExecutor blocks ECP_BLOCKED_MISSING_BOUNDARY",
        current_authority="service principal + tenant resolver",
        audit_evidence="blocker codes only",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
        recommended_owner="applications capacity wiring",
        future_remediation="GR-12-A2 mandatory evaluator injection",
    ),
    Gr12ControlPlaneSurface(
        path_id="CP-PLUGIN-CATALOG-HOT-RELOAD",
        surface="Integration catalog",
        production_entrypoint="intergrax.integrations.registry.catalog_hot_reload.reload_integration_catalog",
        mutation="in-process integration registry replace (override=True)",
        consequential=True,
        current_guard="ApplicationProfile.PRODUCT + feature flag only",
        current_authority="environment profile flag",
        audit_evidence="CatalogHotReloadReport only",
        applicability=Gr12Applicability.APPLICABLE,
        coverage=Gr12CoverageStatus.GAP,
        recommended_owner="integrations registry",
        future_remediation="GR-12-A4 catalog mutation governance slice",
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
        future_remediation="ADR: bridge vs parallel port under CLA-04",
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
        production_entrypoint="intergrax.integrations.providers.vector_store.qdrant.index_administration.QdrantVectorIndexAdministration",
        mutation="collection create/delete/reindex",
        consequential=True,
        current_guard="provider adapter only",
        current_authority="integration credentials",
        audit_evidence="provider logs only",
        applicability=Gr12Applicability.REQUIRES_ARCHITECTURE_DECISION,
        coverage=Gr12CoverageStatus.GAP,
        recommended_owner="integrations/RAG",
        future_remediation="Scope decision: operator API vs internal only",
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
