# © Artur Czarnecki. All rights reserved.

"""GR-11 closed-world Governance extensibility inventory SSOT (GOV-X1 child)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class Gr11QualificationStatus(StrEnum):
    QUALIFIED = "QUALIFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Gr11DynamicRegistrationApplicability(StrEnum):
    COMPOSITION_TIME_ONLY = "COMPOSITION_TIME_ONLY"
    NOT_SUPPORTED_BY_SEAM = "NOT_SUPPORTED_BY_SEAM"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Gr11AuthorityClass(StrEnum):
    GOVERNANCE_PERMISSION = "GOVERNANCE_PERMISSION"
    GOVERNANCE_PROPOSAL_MATERIAL = "GOVERNANCE_PROPOSAL_MATERIAL"
    EXECUTION_LIFECYCLE = "EXECUTION_LIFECYCLE"
    RELIABILITY_FACT = "RELIABILITY_FACT"
    RELIABILITY_EVIDENCE_SINK = "RELIABILITY_EVIDENCE_SINK"
    PROVIDER_INTEGRATION = "PROVIDER_INTEGRATION"


GR11_QUALIFICATION_STATUS: Final[str] = "READY FOR AUDIT"

GR11_QUALIFICATION_START_HEAD: Final[str] = "a7f575a7fe3eea37df28eef73336db1eb276d7e3"

GR11_SEMANTIC_BASELINE_NOTE: Final[str] = (
    "Mechanical certification runs on current development HEAD; GR-12 independent "
    "audit acceptance SHA 03dde6c68a37ac0a8fe19cc5bcf683da8a3afc06 (semantic baseline "
    "b706c2c72a900575ec360b7f217c97a3656c71b9) is reconciled separately in GR-12 closure."
)

GR11_GR12_CONTROL_PLANE_AUDIT_SHA: Final[str] = (
    "03dde6c68a37ac0a8fe19cc5bcf683da8a3afc06"
)


@dataclass(frozen=True, slots=True)
class Gr11CanonicalCompositionBoundary:
    """Per-row canonical composition shapes registered for G08 positive proof."""

    entrypoint_function_names: tuple[str, ...] = ()
    boundary_class_names: tuple[str, ...] = ()
    emitter_function_names: tuple[str, ...] = ()
    delegate_boundary_class_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Gr11ExtensionSurface:
    capability_id: str
    capability: str
    contract: str
    semantic_owner: str
    composition_owner: str
    default_implementation: str
    semantic_owner_module: str
    composition_owner_module: str
    canonical_composition_boundary: Gr11CanonicalCompositionBoundary
    consumer_scan_modules: tuple[str, ...]
    structural_replaceability_proof_nodes: tuple[str, ...]
    authority_proof_nodes: tuple[str, ...]
    composition_proof_nodes: tuple[str, ...]
    negative_bypass_proof_nodes: tuple[str, ...]
    authority: Gr11AuthorityClass
    may_widen_authority: bool
    dynamic_registration: Gr11DynamicRegistrationApplicability
    status: Gr11QualificationStatus
    not_applicable_evidence: str = ""


@dataclass(frozen=True, slots=True)
class Gr11HistoricalPluginabilityRow:
    capability: str
    historical_result: str
    current_result: Gr11QualificationStatus
    reconciliation_notes: str


_GR11_E2E_PLUGIN = "tests/qualification/governance/test_governance_e2e_pluginability.py"
_GR11_GR2_LAUNCHER = (
    "tests/unit/runtime/governance/test_gr2_r3_root_execution_launcher.py"
)
_GR11_GR2_ADMISSION = (
    "tests/unit/runtime/governance/test_runtime_execution_policy_admission.py"
)
_GR11_GR3 = "tests/unit/runtime/governance/test_gr3_canonical_inner_enforcement.py"
_GR11_GR3_R3 = (
    "tests/unit/runtime/governance/test_gr3_r3_explicit_plugin_selection_semantics.py"
)
_GR11_GR6 = "tests/unit/runtime/policy/test_gr6_r1_decision_requirement_enforcement.py"
_GR11_MSE = "tests/unit/runtime/policy/test_meaningful_side_effect_policy.py"
_GR11_MP7C = (
    "tests/qualification/multiplayer/mp7c/test_runtime_policy_evaluator_injection.py"
)
_GR11_GR10_R8 = (
    "tests/unit/runtime/nexus/tools/"
    "test_gr10_r8_r1_orchestration_inner_guard_production_pluginability.py"
)
_GR11_GR12_A2 = (
    "tests/qualification/governance/gr12/test_gr12_a2_mandatory_cla04_composition.py"
)
_GR11_ECP = "tests/unit/runtime/capacity/test_ecp_control_plane_governance.py"
_GR11_CONT = "tests/unit/contracts/test_execution_continuation.py"
_GR11_MP4R3 = "tests/unit/runtime/architecture/test_mp4r3_execution_continuation_integration_gates.py"
_GR11_GR7_A3 = (
    "applications/governed_contractor_application/tests/host/"
    "test_gr7_a3_durable_provider_invocation.py"
)
_GR11_GR7_A2 = (
    "applications/governed_contractor_application/tests/host/"
    "test_gr7_a2_external_work_erl_bridge.py"
)
_GR11_GR7_A6 = (
    "applications/governed_contractor_application/tests/host/"
    "test_gr7_a6_provider_reconciliation.py"
)
_GR11_GR7_A8 = (
    "tests/unit/runtime/enterprise_reliability/"
    "test_gr7_a8_provider_invocation_reliability_evidence.py"
)
_GR11_GR7_A8_R3 = (
    "tests/unit/agents/external_contractor_adapter/"
    "test_gr7_a8_r3_explicit_dispatch_composition.py"
)


def _nid(path: str, name: str) -> str:
    return f"{path}::{name}"


GR11_EXTENSION_SURFACES: Final[tuple[Gr11ExtensionSurface, ...]] = (
    Gr11ExtensionSurface(
        "GR11-ROOT-ADMISSION",
        "Root execution policy admission",
        "intergrax.contracts.runtime_execution_policy_admission.RuntimeExecutionPolicyAdmissionPort",
        "Governance / root admission",
        "build_root_execution_authority_admission / DefaultRootExecutionLauncher composition",
        "AllowingRuntimeExecutionPolicyAdmission / DenyingRuntimeExecutionPolicyAdmission",
        "intergrax/contracts/runtime_execution_policy_admission.py",
        "intergrax/runtime/governance/execution_admission_composition.py",
        Gr11CanonicalCompositionBoundary(
            entrypoint_function_names=(
                "build_runtime_execution_policy_admission",
                "build_fail_closed_runtime_execution_policy_admission",
                "build_root_execution_authority_admission",
                "build_root_execution_authority_admission_from_rules",
                "build_default_root_execution_launcher",
            ),
        ),
        (
            "intergrax/runtime/governance/default_root_execution_launcher.py",
            "intergrax/runtime/governance/root_execution_authority_admission.py",
        ),
        (
            _nid(
                _GR11_E2E_PLUGIN,
                "test_scenario_plugin_custom_runtime_execution_policy_admission_via_composition_launcher",
            ),
        ),
        (
            _nid(
                _GR11_GR2_LAUNCHER,
                "test_launcher_deny_skips_intake[root.execution.agent]",
            ),
        ),
        (
            _nid(
                _GR11_E2E_PLUGIN,
                "test_scenario_plugin_custom_runtime_execution_policy_admission_via_composition_launcher",
            ),
        ),
        (
            _nid(_GR11_GR2_ADMISSION, "test_evaluator_unconfigured_fail_closed"),
            _nid(_GR11_GR2_ADMISSION, "test_unavailable_adapter_fail_closed"),
        ),
        Gr11AuthorityClass.GOVERNANCE_PERMISSION,
        False,
        Gr11DynamicRegistrationApplicability.COMPOSITION_TIME_ONLY,
        Gr11QualificationStatus.QUALIFIED,
    ),
    Gr11ExtensionSurface(
        "GR11-INNER-GUARD",
        "Canonical inner execution guard",
        "intergrax.contracts.canonical_inner_governance.CanonicalInnerExecutionGuardPort",
        "Governance / inner enforcement",
        "meaningful_side_effect_authorization_composition / orchestration runtime context",
        "DefaultCanonicalInnerExecutionGuard",
        "intergrax/contracts/canonical_inner_governance.py",
        "intergrax/runtime/governance/meaningful_side_effect_authorization_composition.py",
        Gr11CanonicalCompositionBoundary(
            entrypoint_function_names=(
                "build_canonical_inner_execution_guard",
                "build_default_canonical_inner_execution_guard",
                "build_meaningful_side_effect_authorization_boundary",
                "build_default_wired_meaningful_side_effect_authorization_boundary",
                "build_decision_governed_meaningful_side_effect_authorization_boundary",
            ),
        ),
        (
            "intergrax/runtime/governance/orchestration_meaningful_side_effect_composition.py",
            "intergrax/runtime/nexus/tools/runtime_tool_invoker_composition.py",
        ),
        (
            _nid(
                _GR11_GR10_R8, "test_gr10_r8_r1_runtime_context_custom_guard_end_to_end"
            ),
        ),
        (_nid(_GR11_GR3, "test_exact_four_id_match_allow_executes_once"),),
        (
            _nid(
                _GR11_GR10_R8, "test_gr10_r8_r1_runtime_context_custom_guard_end_to_end"
            ),
        ),
        (
            _nid(
                _GR11_GR3_R3,
                "test_falsey_custom_task_scope_resolve_is_invoked_through_wired_boundary",
            ),
        ),
        Gr11AuthorityClass.GOVERNANCE_PERMISSION,
        False,
        Gr11DynamicRegistrationApplicability.COMPOSITION_TIME_ONLY,
        Gr11QualificationStatus.QUALIFIED,
    ),
    Gr11ExtensionSurface(
        "GR11-DECISION-REQUIREMENT",
        "Decision material requirement policy",
        "intergrax.contracts.decision_requirement_policy.DecisionRequirementPolicy",
        "Governance / Decision-bound MSE",
        "meaningful_side_effect_authorization_composition",
        "PermissiveDecisionRequirementPolicy / ConfiguredDecisionRequirementPolicy",
        "intergrax/contracts/decision_requirement_policy.py",
        "intergrax/runtime/governance/meaningful_side_effect_authorization_composition.py",
        Gr11CanonicalCompositionBoundary(
            entrypoint_function_names=(
                "build_meaningful_side_effect_authorization_boundary",
                "build_default_wired_meaningful_side_effect_authorization_boundary",
                "build_decision_governed_meaningful_side_effect_authorization_boundary",
            ),
        ),
        ("intergrax/runtime/policy/meaningful_side_effect_authorization.py",),
        (_nid(_GR11_GR6, "test_custom_policy_without_subclassing_default"),),
        (
            _nid(
                _GR11_GR6,
                "test_direct_boundary_required_without_material_denies_no_effect",
            ),
        ),
        (_nid(_GR11_GR6, "test_custom_policy_without_subclassing_default"),),
        (
            _nid(
                _GR11_GR6,
                "test_direct_boundary_required_without_material_denies_no_effect",
            ),
        ),
        Gr11AuthorityClass.GOVERNANCE_PROPOSAL_MATERIAL,
        False,
        Gr11DynamicRegistrationApplicability.COMPOSITION_TIME_ONLY,
        Gr11QualificationStatus.QUALIFIED,
    ),
    Gr11ExtensionSurface(
        "GR11-RUNTIME-POLICY-EVALUATOR",
        "Runtime meaningful-side-effect policy evaluation",
        "intergrax.runtime.policy.runtime_policy_engine.RuntimePolicyEngine (MSE evaluator seam)",
        "Governance / runtime policy",
        "host / MSE authorization composition",
        "RuntimePolicyEngine default rule bundles",
        "intergrax/runtime/policy/runtime_policy_engine.py",
        "intergrax/runtime/governance/execution_admission_composition.py",
        Gr11CanonicalCompositionBoundary(
            entrypoint_function_names=(
                "build_runtime_execution_policy_admission",
                "build_root_execution_authority_admission_from_rules",
            ),
        ),
        (
            "intergrax/runtime/policy/meaningful_side_effect_authorization.py",
            "intergrax/runtime/governance/runtime_execution_policy_admission.py",
        ),
        (
            _nid(
                _GR11_MP7C,
                "test_custom_conforming_evaluator_is_accepted_without_concrete_branching",
            ),
        ),
        (_nid(_GR11_MSE, "test_action_filtering"),),
        (
            _nid(
                _GR11_MP7C,
                "test_custom_conforming_evaluator_is_accepted_without_concrete_branching",
            ),
        ),
        (_nid(_GR11_MSE, "test_action_filtering"),),
        Gr11AuthorityClass.GOVERNANCE_PERMISSION,
        False,
        Gr11DynamicRegistrationApplicability.COMPOSITION_TIME_ONLY,
        Gr11QualificationStatus.QUALIFIED,
    ),
    Gr11ExtensionSurface(
        "GR11-CONTROL-PLANE-EVALUATOR",
        "Control-plane mutation policy evaluation (CLA-04)",
        "intergrax.contracts.control_plane_mutation.ControlPlaneMutationPolicyEvaluator",
        "Governance / control-plane authorization boundary",
        "ControlPlaneMutationAuthorizationBoundary composition (application + runtime hosts)",
        "BundleBackedControlPlaneMutationEvaluator (default bundle-backed)",
        "intergrax/runtime/governance/control_plane_mutation_authorization.py",
        "intergrax/runtime/governance/control_plane_mutation_authorization.py",
        Gr11CanonicalCompositionBoundary(
            boundary_class_names=("ControlPlaneMutationAuthorizationBoundary",),
        ),
        (
            "intergrax/applications/_shared/uca6c_codecraft_qualified_execution_composition.py",
        ),
        (
            _nid(
                _GR11_GR12_A2,
                "test_gr12_a2_r2_external_evaluator_wired_through_product_host_composition",
            ),
            _nid(
                _GR11_ECP,
                "test_ecp_gr12_r2_external_evaluator_receives_composition_probe_mutations",
            ),
        ),
        (
            _nid(
                _GR11_GR12_A2,
                "test_gr12_a2_r2_product_task_control_fail_closed_when_canonical_boundary_unresolved",
            ),
        ),
        (
            _nid(
                _GR11_GR12_A2,
                "test_gr12_a2_r2_product_host_composition_exposes_cla04_boundary",
            ),
        ),
        (
            _nid(
                _GR11_GR12_A2,
                "test_gr12_a2_ecp_product_without_authority_fails_at_wiring",
            ),
        ),
        Gr11AuthorityClass.GOVERNANCE_PERMISSION,
        False,
        Gr11DynamicRegistrationApplicability.COMPOSITION_TIME_ONLY,
        Gr11QualificationStatus.QUALIFIED,
    ),
    Gr11ExtensionSurface(
        "GR11-CONTINUATION",
        "Execution pause/resume lifecycle",
        "intergrax.contracts.execution_continuation.ExecutionContinuationPort",
        "Execution Runtime",
        "execution continuation composition / MP-4R7 enterprise integration composition",
        "Governed execution continuation store implementations (Execution-owned)",
        "intergrax/contracts/execution_continuation.py",
        "intergrax/runtime/execution/suspended_operation/composition.py",
        Gr11CanonicalCompositionBoundary(
            entrypoint_function_names=(
                "wire_execution_suspended_work_reentry_coordinator",
            ),
        ),
        (
            "intergrax/runtime/nexus/nexus_loop.py",
            "intergrax/runtime/policy/mse_hitl_effect_gate.py",
        ),
        (
            _nid(
                _GR11_CONT,
                "test_pluginability_two_implementations[_DictExecutionContinuationPort]",
            ),
            _nid(
                _GR11_CONT,
                "test_pluginability_two_implementations[_ListExecutionContinuationPort]",
            ),
        ),
        (
            _nid(
                _GR11_MP4R3, "test_mp4r3_no_duplicate_continuation_lifecycle_authority"
            ),
        ),
        (_nid(_GR11_MP4R3, "test_mp4r3_contract_only_continuation_dependency"),),
        (_nid(_GR11_CONT, "test_state_machine_valid_spine"),),
        Gr11AuthorityClass.EXECUTION_LIFECYCLE,
        False,
        Gr11DynamicRegistrationApplicability.COMPOSITION_TIME_ONLY,
        Gr11QualificationStatus.QUALIFIED,
    ),
    Gr11ExtensionSurface(
        "GR11-PROVIDER-INVOCATION-STORE",
        "Durable provider invocation facts",
        "intergrax.contracts.provider_invocation_store.ProviderInvocationStore",
        "Enterprise Reliability / provider invocation lifecycle",
        "build_governed_external_work_production_runtime / reliability composition",
        "InMemoryProviderInvocationStore (host default)",
        "intergrax/contracts/provider_invocation_store.py",
        "applications/governed_contractor_application/host/production_external_work_composition.py",
        Gr11CanonicalCompositionBoundary(
            entrypoint_function_names=(
                "build_governed_external_work_production_runtime",
            ),
        ),
        (
            "intergrax/runtime/governance/orchestration_consequential_effect_reliability_composition.py",
        ),
        (_nid(_GR11_GR7_A3, "test_production_accepts_custom_durable_store"),),
        (_nid(_GR11_GR7_A3, "test_success_persists_intent_before_outcome_and_ger"),),
        (_nid(_GR11_GR7_A3, "test_production_accepts_custom_durable_store"),),
        (_nid(_GR11_GR7_A3, "test_success_persists_intent_before_outcome_and_ger"),),
        Gr11AuthorityClass.RELIABILITY_FACT,
        False,
        Gr11DynamicRegistrationApplicability.COMPOSITION_TIME_ONLY,
        Gr11QualificationStatus.QUALIFIED,
    ),
    Gr11ExtensionSurface(
        "GR11-PROVIDER-RELIABILITY-COLLABORATOR",
        "Governance→Reliability external-work handoff collaborators",
        "intergrax.contracts.enterprise_reliability.admission_boundary.ExternalEffectAdmissionRequest; "
        "intergrax.contracts.enterprise_reliability.plugin_spi.ReconciliationProbeExecutor",
        "Governance host + Reliability reconciliation",
        "governed_contractor_application production external-work composition",
        "Production bridge defaults with injectable admission + reconciliation registry",
        "intergrax/contracts/enterprise_reliability/admission_boundary.py",
        "applications/governed_contractor_application/host/production_external_work_composition.py",
        Gr11CanonicalCompositionBoundary(
            entrypoint_function_names=(
                "build_governed_external_work_production_runtime",
            ),
            delegate_boundary_class_names=(
                "GovernedExternalWorkEnterpriseReliabilityBridge",
            ),
        ),
        (
            "applications/governed_contractor_application/host/external_work_enterprise_reliability_bridge.py",
            "intergrax/runtime/enterprise_reliability/plugin_registry.py",
            "intergrax/runtime/governance/orchestration_consequential_effect_reliability_composition.py",
        ),
        (_nid(_GR11_GR7_A6, "test_pluginability_custom_probe_without_core_change"),),
        (_nid(_GR11_GR7_A2, "test_governance_deny_zero_provider_calls_no_admission"),),
        (
            _nid(
                _GR11_GR7_A8_R3,
                "test_custom_aware_port_observer_on_receives_reliability_context",
            ),
        ),
        (_nid(_GR11_GR7_A2, "test_governance_deny_zero_provider_calls_no_admission"),),
        Gr11AuthorityClass.PROVIDER_INTEGRATION,
        False,
        Gr11DynamicRegistrationApplicability.COMPOSITION_TIME_ONLY,
        Gr11QualificationStatus.QUALIFIED,
    ),
    Gr11ExtensionSurface(
        "GR11-RELIABILITY-OBSERVATION-EVIDENCE",
        "Provider invocation reliability evidence observation",
        "intergrax.contracts.enterprise_reliability.provider_invocation_reliability_evidence."
        "ProviderInvocationReliabilityEvidenceObserver",
        "Enterprise Reliability evidence projection",
        "provider_invocation_reliability_evidence.emit_* (optional observer parameter)",
        "NullProviderInvocationReliabilityEvidenceObserver",
        "intergrax/runtime/enterprise_reliability/provider_invocation_reliability_evidence.py",
        "intergrax/runtime/enterprise_reliability/provider_invocation_reliability_evidence.py",
        Gr11CanonicalCompositionBoundary(
            emitter_function_names=("emit_provider_invocation_reliability_fact",),
        ),
        (
            "applications/governed_contractor_application/host/production_external_work_composition.py",
        ),
        (_nid(_GR11_GR7_A8, "test_intent_and_outcome_phases_distinguish_unknown"),),
        (_nid(_GR11_GR7_A8, "test_succeeded_path_terminal_no_execution_mutation"),),
        (
            _nid(
                _GR11_GR7_A8_R3,
                "test_custom_aware_port_observer_on_receives_reliability_context",
            ),
        ),
        (_nid(_GR11_GR7_A8, "test_observer_failure_does_not_raise"),),
        Gr11AuthorityClass.RELIABILITY_EVIDENCE_SINK,
        False,
        Gr11DynamicRegistrationApplicability.NOT_APPLICABLE,
        Gr11QualificationStatus.QUALIFIED,
    ),
)

GR11_HISTORICAL_PLUGINABILITY_RECONCILIATION: Final[
    tuple[Gr11HistoricalPluginabilityRow, ...]
] = (
    Gr11HistoricalPluginabilityRow(
        "Continuation",
        "PARTIAL",
        Gr11QualificationStatus.QUALIFIED,
        "ExecutionContinuationPort two-implementation pluginability + MP-4R3 no-duplicate-authority gates.",
    ),
    Gr11HistoricalPluginabilityRow(
        "Provider integration",
        "PARTIAL",
        Gr11QualificationStatus.QUALIFIED,
        "Decomposed into typed host admission + reconciliation probe collaborators (GR-7-A2/A6).",
    ),
    Gr11HistoricalPluginabilityRow(
        "ProviderInvocationStore",
        "PARTIAL",
        Gr11QualificationStatus.QUALIFIED,
        "Custom durable store injectable via production composition (GR-7-A3).",
    ),
    Gr11HistoricalPluginabilityRow(
        "Reliability observer",
        "PARTIAL",
        Gr11QualificationStatus.QUALIFIED,
        "ProviderInvocationReliabilityEvidenceObserver optional sink; cannot mint permission.",
    ),
)

GR11_WEAK_BOUNDARY_SCAN_MODULES: Final[tuple[str, ...]] = (
    "intergrax/contracts/runtime_execution_policy_admission.py",
    "intergrax/contracts/canonical_inner_governance.py",
    "intergrax/contracts/decision_requirement_policy.py",
    "intergrax/contracts/meaningful_side_effect_authorization.py",
    "intergrax/contracts/execution_continuation.py",
    "intergrax/contracts/provider_invocation_store.py",
    "intergrax/contracts/control_plane_mutation.py",
    "intergrax/contracts/enterprise_reliability/provider_invocation_reliability_evidence.py",
    "intergrax/runtime/governance/runtime_execution_policy_admission.py",
    "intergrax/runtime/governance/canonical_inner_execution_guard.py",
    "intergrax/runtime/governance/control_plane_mutation_authorization.py",
    "intergrax/runtime/governance/meaningful_side_effect_authorization_composition.py",
    "intergrax/runtime/governance/decision_requirement_policy.py",
    "intergrax/runtime/policy/runtime_policy_engine.py",
    "intergrax/runtime/execution/suspended_operation/composition.py",
    "intergrax/runtime/enterprise_reliability/provider_invocation_lifecycle.py",
    "intergrax/runtime/enterprise_reliability/provider_invocation_reliability_evidence.py",
    "applications/governed_contractor_application/host/production_external_work_composition.py",
)

GR11_IMPLEMENTATION_BRANCH_SCAN_MODULES: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(
        module
        for row in GR11_EXTENSION_SURFACES
        for module in (
            row.semantic_owner_module,
            row.composition_owner_module,
            *row.consumer_scan_modules,
        )
    )
)

GR11_IMPLEMENTATION_BRANCH_FORBIDDEN_PATTERNS: Final[tuple[str, ...]] = (
    r"isinstance\s*\(\s*\w+\s*,\s*Default",
    r'if\s+\w+\.name\s*==\s*["\']',
    r"if\s+plugin_name\s*==",
    r'if\s+provider\s*==\s*["\']',
)

GR11_COMPOSITION_SELF_REGISTRATION_MARKERS: Final[tuple[str, ...]] = (
    "register_plugin",
    "auto_discover",
    "importlib.import_module",
    "__import__(",
)

GR11_FORBIDDEN_PARENT_STATUSES: Final[frozenset[str]] = frozenset(
    {"CLOSED", "FINAL CLOSED"}
)


def gr11_capability_ids() -> tuple[str, ...]:
    return tuple(row.capability_id for row in GR11_EXTENSION_SURFACES)


def gr11_qualified_or_na_rows() -> tuple[Gr11ExtensionSurface, ...]:
    return GR11_EXTENSION_SURFACES


def gr11_contract_symbol(contract: str) -> str:
    return gr11_contract_symbols(contract)[0]


def gr11_contract_symbols(contract: str) -> tuple[str, ...]:
    parts = [segment.strip() for segment in contract.split(";") if segment.strip()]
    symbols: list[str] = []
    for part in parts:
        tail = part.rsplit(".", 1)[-1]
        symbols.append(tail.split(" ")[0])
    return tuple(symbols)


def gr11_row_all_proof_nodes(row: Gr11ExtensionSurface) -> tuple[str, ...]:
    return (
        *row.structural_replaceability_proof_nodes,
        *row.authority_proof_nodes,
        *row.composition_proof_nodes,
        *row.negative_bypass_proof_nodes,
    )


def gr11_all_structural_replaceability_nodes() -> tuple[str, ...]:
    return tuple(
        node
        for row in GR11_EXTENSION_SURFACES
        for node in row.structural_replaceability_proof_nodes
    )


def gr11_all_registered_proof_nodes() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            node
            for row in GR11_EXTENSION_SURFACES
            for node in gr11_row_all_proof_nodes(row)
        )
    )
