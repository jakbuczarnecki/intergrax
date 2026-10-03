# © Artur Czarnecki. All rights reserved.

"""CTRL-X plane catalog (CX-01..CX-12) — mechanical evidence SSOT for FRZ-CTL-01..12."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

CTRL_X_START_HEAD: Final[str] = "3e574ea35ee2e3f9e9adac4088cd51ae89ed469c"


class CtrlXPlaneResult(StrEnum):
    PASS = "PASS"
    NOT_APPLICABLE_WITH_EVIDENCE = "N/A — WITH EVIDENCE"


class CtrlXTenantVerdict(StrEnum):
    PASS = "PASS"
    NOT_APPLICABLE_WITH_EVIDENCE = "N/A — WITH EVIDENCE"


@dataclass(frozen=True, slots=True)
class CtrlXPlaneEvidence:
    plane_id: str
    frz_criterion: str
    semantic_owner: str
    canonical_contract_boundary: str
    tenant_verdict: CtrlXTenantVerdict
    result: CtrlXPlaneResult
    primary_pytest_node_ids: tuple[str, ...]
    historical_evidence: tuple[str, ...] = ()
    notes: str = ""


def _nid(path: str, test_name: str) -> str:
    return f"{path}::{test_name}"


_PLUG03 = "tests/qualification/plug_03/test_plug_03_gates.py"
_H02 = "tests/qualification/harness_02/test_harness_02_gates.py"
_BUDGET = "tests/unit/runtime/execution/budget/test_execution_budget_ledger.py"
_ADV_EVAL = "tests/unit/runtime/token_optimization/test_advisory_evaluation.py"
_DS_MIG = "tests/unit/runtime/architecture/test_ds_mig_04_critic_runtime_deleted.py"
_DV = "tests/unit/runtime/test_decision_verification_composition.py"
_W5H = (
    "tests/unit/runtime/observability/test_enterprise_scale_resilience_w5_h_final_qualification.py"
)
_DIAG = "tests/unit/runtime/diagnostics/test_diagnostic_subsystem_failure_evidence.py"
_GOV_TOOL = "tests/qualification/governance/test_governance_e2e_pluginability.py"
_W4_TOOL = (
    "tests/unit/runtime/nexus/tools/test_harness_w4_r1_production_tool_admission_behavior.py"
)
_REG = "tests/unit/runtime/registry/test_agent_registry_read_boundary.py"
_CAP = "tests/unit/applications/test_capability_graph_deploy_gate.py"
_CE01 = "tests/qualification/ce_01/test_ce_01_gates.py"
_MEM_SEC = "tests/qualification/memory_behavior/test_mem_final_audit_6_security_behavior.py"
_W4_ARCH = "tests/unit/runtime/architecture/test_harness_w4_r1_production_tool_boundedness_gate.py"
_GR12 = (
    "tests/qualification/governance/gr12/test_gr12_final_control_plane_qualification.py"
)
_CP_MUT = "tests/unit/runtime/governance/test_control_plane_mutation_approval.py"
_R1_SEC = "tests/qualification/control_planes/ctrl_x/test_ctrl_x_r1_security_composition.py"
_R1_EVAL = "tests/qualification/control_planes/ctrl_x/test_ctrl_x_r1_evaluation_plane.py"
_R2_MW = "tests/qualification/control_planes/ctrl_x/test_ctrl_x_r2_middleware_composition.py"
_R2_EVAL = "tests/qualification/control_planes/ctrl_x/test_ctrl_x_r2_evaluation_plane.py"
_EVAL_WIRING = "tests/unit/applications/test_harness_evaluation_wiring.py"
_ONLINE_REG = "tests/unit/runtime/architecture/test_online_evaluation_registry.py"


CTRL_X_PLANE_CATALOG: tuple[CtrlXPlaneEvidence, ...] = (
    CtrlXPlaneEvidence(
        "CX-01",
        "FRZ-CTL-01",
        "Tier-3 SecurityEnvelope + Tier-1 PluginSecurityDefenseMiddleware / UAEP hook timeline",
        "SecurityDefensePlugin (defense-only); ADR-SEC-001 composition boundary",
        CtrlXTenantVerdict.PASS,
        CtrlXPlaneResult.PASS,
        (
            _nid(_PLUG03, "test_plug03_security_defense_canonical_hook_invokes_custom_plugin"),
            _nid(_MEM_SEC, "test_cross_tenant_isolation"),
            _nid(_R1_SEC, "test_r1_sec_01_canonical_composition_rejects_external_fail_open"),
            _nid(_R1_SEC, "test_r1_sec_02_fail_closed_external_defense_accepted"),
            _nid(_R1_SEC, "test_r1_sec_06_security_emitters_use_host_orchestration_event_port"),
            _nid(_R2_MW, "test_r2_mw_02_alternate_pipeline_port_accepts_security_middleware"),
            _nid(_R2_MW, "test_r2_mw_05_fail_open_rejected_before_attach"),
        ),
        ("AW-7C-P0-3B scoped substrate evidence", "PLUG-03 security plugin adoption"),
    ),
    CtrlXPlaneEvidence(
        "CX-02",
        "FRZ-CTL-02",
        "ResiliencePolicy + DependencyAttemptExecutionBoundary + ErrorClassifier",
        "intergrax.runtime.resilience.*; bounded admission at execution/tool boundaries",
        CtrlXTenantVerdict.PASS,
        CtrlXPlaneResult.PASS,
        (
            _nid(_H02, "test_harness_02_child_inherits_parent_global_deadline"),
            _nid(_W4_ARCH, "test_w4_r1_production_roots_pass_dependency_attempt_boundary"),
        ),
        ("HARNESS-W4", "HARNESS-02 cancellation/deadline propagation"),
    ),
    CtrlXPlaneEvidence(
        "CX-03",
        "FRZ-CTL-03",
        "Execution budget ledger (RunBudget / BudgetEnvelope materialization)",
        "intergrax.contracts.run_budget.RunBudget; execution budget ledger reservation semantics",
        CtrlXTenantVerdict.PASS,
        CtrlXPlaneResult.PASS,
        (
            _nid(_BUDGET, "test_reservation_greater_than_available_fails_closed"),
            _nid(_BUDGET, "test_two_children_reserve_without_oversubscribing"),
        ),
        ("HARNESS-FINAL budget wiring evidence",),
    ),
    CtrlXPlaneEvidence(
        "CX-04",
        "FRZ-CTL-04",
        "EvaluationProfile wiring + OnlineEvaluationRegistry + advisory/offline/shadow producers",
        "EvaluationProfile / evaluation_wiring; OnlineEvaluationRegistry; advisory evaluation contracts",
        CtrlXTenantVerdict.NOT_APPLICABLE_WITH_EVIDENCE,
        CtrlXPlaneResult.PASS,
        (
            _nid(_EVAL_WIRING, "test_wire_application_evaluation_builds_registry_and_bridge"),
            _nid(_ONLINE_REG, "test_in_memory_registry_append"),
            _nid(_ADV_EVAL, "test_evaluation_result_rejects_auto_apply"),
            _nid(_R1_EVAL, "test_r1_eval_03_shadow_observation_is_observation_not_permission"),
            _nid(_DV, "test_pipeline_factory_registers_selected_stages_only"),
            _nid(_R1_EVAL, "test_r1_eval_07_evaluation_observation_host_scoped_not_cross_run"),
            _nid(_R2_EVAL, "test_r2_eval_02_promotion_consumer_requires_separate_gate_evidence"),
            _nid(_R2_EVAL, "test_r2_eval_04_decision_verification_distinct_from_evaluation_registry"),
            _nid(_R2_EVAL, "test_r2_eval_05_tenant_verdict_not_tenant_aware_at_this_stage"),
        ),
        ("Offline RAG evaluation harness — product scope; not runtime permission",),
        notes="OnlineEvaluationObservation is run/host-scoped; TENANT-X debt on tenant_id surface.",
    ),
    CtrlXPlaneEvidence(
        "CX-05",
        "FRZ-CTL-05",
        "Decision Verification Pipeline (typed stages)",
        "intergrax.contracts.decision_verification_stage; Decision lifecycle",
        CtrlXTenantVerdict.PASS,
        CtrlXPlaneResult.PASS,
        (
            _nid(_DS_MIG, "test_production_code_has_no_live_critic_symbols"),
            _nid(_DV, "test_pipeline_factory_registers_selected_stages_only"),
        ),
        ("DS-MIG-04 legacy Critic retirement", "DECISION_VERIFICATION.md"),
    ),
    CtrlXPlaneEvidence(
        "CX-06",
        "FRZ-CTL-06",
        "RuntimeEventBus + BoundedEventSink + export sink pipeline",
        "RuntimeEvent → ObservabilityExportPayload → EventExportSinkPort",
        CtrlXTenantVerdict.PASS,
        CtrlXPlaneResult.PASS,
        (
            _nid(_W5H, "test_otlp_transport_port_export_is_typed_observability_export_payload"),
            _nid(
                "tests/unit/runtime/events/test_enterprise_scale_resilience_w5_c_event_export.py",
                "test_instance_isolation_between_environments",
            ),
        ),
        ("HARNESS-W5",),
    ),
    CtrlXPlaneEvidence(
        "CX-07",
        "FRZ-CTL-07",
        "DiagnosticOrchestrator + ProblemLifecycleEngine (interpretive)",
        "Diagnostics consumes observability facts; ProblemId minting ≠ ExecutionId minting",
        CtrlXTenantVerdict.PASS,
        CtrlXPlaneResult.PASS,
        (
            _nid(_DIAG, "test_record_without_active_execution_id_fails_closed"),
            _nid(_DIAG, "test_bridge_records_failure_evidence_without_changing_outcome"),
        ),
        ("HARNESS-W6 advisory intelligence boundaries (scoped)",),
    ),
    CtrlXPlaneEvidence(
        "CX-08",
        "FRZ-CTL-08",
        "ToolRuntime → gateway → RuntimeToolInvoker → ToolExecutor",
        "intent != execution; ToolRuntime admission",
        CtrlXTenantVerdict.PASS,
        CtrlXPlaneResult.PASS,
        (
            _nid(_GOV_TOOL, "test_scenario_plugin_allowing_vs_denying_admission_via_composition"),
            _nid(_W4_TOOL, "test_w4_r1_reject_second_invocation_without_physical_executor_call"),
        ),
        ("GOV-X2 tool governance E2E", "HARNESS-W4-R1 tool admission"),
    ),
    CtrlXPlaneEvidence(
        "CX-09",
        "FRZ-CTL-09",
        "SkillManifest → SkillRegistry → SkillResolver → AgentContract",
        "Skill declarative composition; ToolRuntime still enforces effects",
        CtrlXTenantVerdict.PASS,
        CtrlXPlaneResult.PASS,
        (
            _nid(_PLUG03, "test_plug03_without_custom_skill_tool_not_allowed"),
            _nid(_PLUG03, "test_plug03_custom_skill_enables_canonical_tool_execution"),
        ),
        ("PLUG-03 skill gateway proofs",),
    ),
    CtrlXPlaneEvidence(
        "CX-10",
        "FRZ-CTL-10",
        "Agent distribution revision → derived AgentRegistry projection",
        "build_registry_projection / MaterializedRegistryProjection read boundary",
        CtrlXTenantVerdict.PASS,
        CtrlXPlaneResult.PASS,
        (
            _nid(_REG, "test_materialized_projection_runtime_surface_has_no_register"),
            _nid(_REG, "test_revision_bound_host_registry_resolution_uses_read_only_projection_surface"),
        ),
        ("HOST-01 host roster", "GOV-X2 assembly paths"),
    ),
    CtrlXPlaneEvidence(
        "CX-11",
        "FRZ-CTL-11",
        "CapabilityGraph (declarative dependency/impact)",
        "intergrax.runtime.architecture.capability_graph.CapabilityGraph",
        CtrlXTenantVerdict.PASS,
        CtrlXPlaneResult.PASS,
        (
            _nid(_CAP, "test_validate_strict_capability_graph_deploy_blocks_experimental_agent"),
            _nid(
                "tests/unit/applications/architecture/test_binding_contract_identity_authority.py",
                "test_capability_graph_catalog_does_not_own_binding_identity_resolution",
            ),
        ),
        ("EBH-3 capability graph compatibility",),
    ),
    CtrlXPlaneEvidence(
        "CX-12",
        "FRZ-CTL-12",
        "Context Engineering (exactly-one model-call assembly owner)",
        "intergrax.context / CE entry surfaces — select/rank/filter/fit/assemble",
        CtrlXTenantVerdict.PASS,
        CtrlXPlaneResult.PASS,
        (
            _nid(_CE01, "test_ce_q1_canonical_context_engine_entry_surfaces"),
            _nid(_CE01, "test_ce_q3_foreign_tenant_fragment_rejected"),
            _nid(_CE01, "test_ce_q15_nexus_canonical_paths_forbid_direct_prompt_injection"),
        ),
        ("CE-01 CE-Q1..Q15", "HARNESS-RESIDUAL / TOKEN-CE → classified non-blocking or superseded"),
    ),
)

CTRL_X_CROSS_CUTTING_PROOF_NODE_IDS: Final[tuple[str, ...]] = (
    _nid(_GR12, "test_gr12_final_f02_applicable_rows_are_qualified"),
    _nid(_CP_MUT, "test_cpma_2_non_user_approver_rejected"),
)


def ctrl_x_all_proof_pytest_node_ids() -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for entry in CTRL_X_PLANE_CATALOG:
        for node_id in entry.primary_pytest_node_ids:
            if node_id in seen:
                continue
            seen.add(node_id)
            ordered.append(node_id)
    for node_id in CTRL_X_CROSS_CUTTING_PROOF_NODE_IDS:
        if node_id in seen:
            continue
        seen.add(node_id)
        ordered.append(node_id)
    return tuple(ordered)
