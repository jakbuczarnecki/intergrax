# © Artur Czarnecki. All rights reserved.

"""L3 production reachability evidence — per-mechanism records (Protocol 03 remediation R1)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from tests.qualification.ebh_6._closed_world_discovery import EBH6_AUDIT_LAYERS
from tests.qualification.ebh_6._l3_aj_group_reconciliation import (
    AjControlOutcome,
    aj_group_by_layer_class,
    format_group_evidence_ref,
)
from tests.qualification.ebh_6._layer_evidence import layer_aj_applicability


@dataclass(frozen=True, slots=True)
class AjControlResult:
    detection_class: str
    outcome: AjControlOutcome
    evidence_ref: str


def _aj_controls_for_mechanism(
    layer: str,
    mechanism_id: str,
    primary_gate: str,
    entrypoint: str,
    finding_id: str | None,
) -> tuple[AjControlResult, ...]:
    groups = aj_group_by_layer_class()
    rows: list[AjControlResult] = []
    for aj in layer_aj_applicability(layer):
        if not aj.applicable:
            rows.append(
                AjControlResult(
                    aj.detection_class,
                    AjControlOutcome.NA_WITH_EVIDENCE,
                    f"N/A — WITH EVIDENCE: {aj.rationale}",
                ),
            )
            continue

        if finding_id == "EBH6-DEBT-Q2-D2" and aj.detection_class == "F":
            rows.append(
                AjControlResult(
                    "F",
                    AjControlOutcome.FINDING,
                    f"{finding_id} — FRZ-TYP-03 tracked seam (RATIFIED register row)",
                ),
            )
            continue

        if finding_id == "EBH6-DEBT-BND-COMPAT-R3" and aj.detection_class == "H":
            rows.append(
                AjControlResult(
                    "H",
                    AjControlOutcome.FINDING,
                    f"{finding_id} — lab-only reachability; production_effective=no",
                ),
            )
            continue

        group = groups[(layer, aj.detection_class)]
        if group.outcome == AjControlOutcome.EVIDENCE_REQUIRED:
            rows.append(
                AjControlResult(
                    aj.detection_class,
                    AjControlOutcome.EVIDENCE_REQUIRED,
                    (f"{group.mandatory_blocker or group.group_id}: mechanism={mechanism_id}; entrypoint={entrypoint}"),
                ),
            )
        else:
            rows.append(
                AjControlResult(
                    aj.detection_class,
                    AjControlOutcome.PASS_CANDIDATE,
                    format_group_evidence_ref(group, mechanism_id, primary_gate),
                ),
            )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class L3ReachabilityRecord:
    mechanism_id: str
    layer: str
    entrypoint: str
    call_chain: tuple[str, ...]
    semantic_owner: str
    canonical_contract: str
    composition_owner: str
    aj_controls: tuple[AjControlResult, ...]
    production_effective: bool
    production_effective_rationale: str
    existing_finding_id: str | None = None


def _m(
    mid: str,
    layer: str,
    entry: str,
    chain: tuple[str, ...],
    owner: str,
    contract: str,
    comp_owner: str,
    gate: str,
    *,
    prod: bool = True,
    prod_rationale: str = "Sanctioned production composition path @ TRACE-X registry",
    finding_id: str | None = None,
) -> L3ReachabilityRecord:
    return L3ReachabilityRecord(
        mechanism_id=mid,
        layer=layer,
        entrypoint=entry,
        call_chain=chain,
        semantic_owner=owner,
        canonical_contract=contract,
        composition_owner=comp_owner,
        aj_controls=_aj_controls_for_mechanism(layer, mid, gate, entry, finding_id),
        production_effective=prod,
        production_effective_rationale=prod_rationale,
        existing_finding_id=finding_id,
    )


# ≥3 mechanisms per roadmap §3.0.8 layer — concrete entrypoints and call chains @ HEAD.
_L3_MECHANISMS: Final[tuple[L3ReachabilityRecord, ...]] = (
    _m(
        "L3-CTR-01",
        "contracts",
        "intergrax/applications/_shared/runtime_config_bridge.py::materialize_runtime_config",
        (
            "intergrax/applications/_shared/runtime_config_bridge.py",
            "intergrax/contracts/runtime/hosting_profile.py",
            "intergrax/contracts/integrations/integration_profile.py",
        ),
        "Platform contracts tier",
        "HostProfile / IntegrationProfile contracts",
        "Application composition (_shared)",
        "tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py::test_tier0_must_not_import_agents_or_applications",
    ),
    _m(
        "L3-CTR-02",
        "contracts",
        "intergrax/contracts/decision/integration/composition.py",
        (
            "intergrax/contracts/decision/integration/composition.py",
            "intergrax/contracts/decision/authorization.py",
        ),
        "Governance decision contracts",
        "DecisionAuthorization contracts",
        "Governance composition",
        "tests/unit/runtime/architecture/test_npsc42_h1_governance_boundary_freeze.py::test_npsc42_h1_governance_planes_do_not_mint_execution_identity",
    ),
    _m(
        "L3-CTR-03",
        "contracts",
        "intergrax/integrations/contracts/graph_store.py",
        (
            "intergrax/integrations/contracts/graph_store.py",
            "intergrax/rag/contracts/graph_store.py",
        ),
        "Integrations vs RAG contract split",
        "GraphStore port (Integration-owned materialization)",
        "IntegrationProfile resolver",
        "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_tracked_debt_findings_ratified",
    ),
    _m(
        "L3-SMC-01",
        "semantic_core",
        "intergrax/runtime/nexus/execution/graph_executor.py",
        (
            "intergrax/runtime/nexus/execution/graph_executor.py",
            "intergrax/runtime/nexus/execution/nexus_loop.py",
            "intergrax/runtime/execution/host_task.py",
        ),
        "Nexus orchestration",
        "ExecutionRequest / graph lifecycle",
        "Nexus host composition",
        "tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py",
    ),
    _m(
        "L3-SMC-02",
        "semantic_core",
        "intergrax/runtime/execution/orchestration.py",
        (
            "intergrax/runtime/execution/orchestration.py",
            "intergrax/runtime/execution/orchestration_topology_submission.py",
        ),
        "Execution orchestration",
        "StrategyExecutionRouter",
        "Runtime execution composition root",
        "test_trace_x_p4_r5_closed_world.py::test_txp4r5_q06_inference_executor_unreachable_from_sanctioned_routers",
    ),
    _m(
        "L3-SMC-03",
        "semantic_core",
        "intergrax/runtime/nexus/nexus_loop.py",
        ("intergrax/runtime/nexus/nexus_loop.py",),
        "Nexus loop",
        "AuthoritativeDecisionExposure seam",
        "Nexus host",
        "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_tracked_debt_findings_ratified",
        finding_id="EBH6-DEBT-Q2-D2",
    ),
    _m(
        "L3-APP-01",
        "application_services",
        "intergrax/applications/_shared/production_delegated_subtask_child_execution_wiring.py",
        (
            "intergrax/applications/_shared/production_delegated_subtask_child_execution_wiring.py",
            "intergrax/contracts/execution/child_execution.py",
        ),
        "Applications tier",
        "ChildExecutionPort",
        "Application _shared wiring",
        "tests/qualification/trace_x/test_trace_x_p4_r5_closed_world.py::test_txp4r5_q10_sanctioned_runtime_config_llm_adapter_wrap_propagation",
    ),
    _m(
        "L3-APP-02",
        "application_services",
        "intergrax/applications/_shared/runtime_event_delivery_wiring.py",
        (
            "intergrax/applications/_shared/runtime_event_delivery_wiring.py",
            "intergrax/contracts/observability/event_sink.py",
        ),
        "Observability wiring (app tier)",
        "EventSinkPort / RuntimeEventBus",
        "Application composition",
        "tests/unit/runtime/events/test_enterprise_scale_resilience_w5_b2_composition_wiring.py::test_production_composition_creates_bounded_sink",
    ),
    _m(
        "L3-APP-03",
        "application_services",
        "intergrax/applications/_shared/acp_runtime_session_hooks.py",
        (
            "intergrax/applications/_shared/acp_runtime_session_hooks.py",
            "intergrax/applications/_shared/runtime_config_bridge.py",
        ),
        "ACP session hooks",
        "RuntimeConfig bridge",
        "Application _shared",
        "tests/qualification/trace_x/test_trace_x_p4_r5_closed_world.py::test_txp4r5_q04_production_composition_registry_parity",
        prod_rationale="Registered site — DEVELOPMENT_LAB classification in TRACE-X; not primary production router",
    ),
    _m(
        "L3-CMP-01",
        "composition_roots",
        "intergrax/applications/_shared/runtime_config_bridge.py::materialize_runtime_config",
        (
            "runtime_config_bridge.materialize_runtime_config",
            "intergrax/runtime/hosting/bootstrap.py",
        ),
        "Sanctioned production root",
        "RuntimeConfig materialization",
        "Application composition root",
        "tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py::test_ebh6_exit_sanctioned_composition_roots_non_empty",
    ),
    _m(
        "L3-CMP-02",
        "composition_roots",
        "intergrax/runtime/execution/orchestration_topology_submission.py",
        (
            "orchestration_topology_submission.py",
            "intergrax/runtime/nexus/execution/graph_executor.py",
        ),
        "Topology submission factory",
        "MSE policy mandatory factory",
        "Execution composition root",
        "tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py::test_ebh6_exit_sanctioned_composition_roots_non_empty",
    ),
    _m(
        "L3-CMP-03",
        "composition_roots",
        "intergrax/runtime/transport/bootstrap.py",
        (
            "intergrax/runtime/transport/bootstrap.py",
            "intergrax/integrations/providers/message_bus/",
        ),
        "Queue transport composition",
        "Transport bootstrap single root",
        "Runtime transport composition",
        "tests/qualification/ebh_5/test_ebh_5_reachability_gates.py",
        finding_id="EBH6-DEBT-BND-COMPAT-R3",
        prod=False,
        prod_rationale="Lab debug path documented — production uses sanctioned roots only",
    ),
    _m(
        "L3-GOV-01",
        "governance",
        "intergrax/runtime/governance/plugin_security_defense_middleware.py",
        (
            "plugin_security_defense_middleware.py",
            "intergrax/contracts/governance/security_defense.py",
        ),
        "Governance plane",
        "SecurityDefensePlugin contract",
        "Host application_security_wiring",
        "tests/unit/runtime/architecture/test_npsc42_h1_governance_boundary_freeze.py",
    ),
    _m(
        "L3-GOV-02",
        "governance",
        "intergrax/runtime/governance/decision_verification_composition.py",
        (
            "decision_verification_composition.py",
            "intergrax/contracts/decision/verification.py",
        ),
        "Decision verification",
        "Verification stage contracts",
        "Governance composition",
        "tests/unit/runtime/architecture/test_npsc42_h1_governance_boundary_freeze.py::test_npsc42_h1_governance_planes_do_not_mint_execution_identity",
    ),
    _m(
        "L3-GOV-03",
        "governance",
        "intergrax/runtime/nexus/context/uaep_assemble.py",
        (
            "uaep_assemble.py",
            "intergrax/runtime/governance/uaep_hook_timeline.py",
        ),
        "UAEP governance attach",
        "UAEP hook context",
        "Nexus context bootstrap",
        "tests/unit/runtime/architecture/test_npsc42_h1_governance_boundary_freeze.py",
    ),
    _m(
        "L3-EXE-01",
        "execution",
        "intergrax/runtime/execution/host_task.py::build_host_task_strategy_router",
        (
            "host_task.py",
            "intergrax/runtime/execution/strategy_execution_router.py",
        ),
        "ExecutionRuntime",
        "ExecutionRequest / StrategyExecutionRouter",
        "Execution host composition",
        "tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py",
    ),
    _m(
        "L3-EXE-02",
        "execution",
        "intergrax/runtime/nexus/execution/graph_executor.py",
        (
            "graph_executor.py",
            "intergrax/runtime/execution/execution_runtime.py",
        ),
        "Nexus graph execution",
        "GraphExecutor lifecycle",
        "Nexus composition",
        "tests/unit/runtime/architecture/test_execution_identity_single_authority_gate.py",
    ),
    _m(
        "L3-EXE-03",
        "execution",
        "intergrax/runtime/execution/inference_composition.py",
        (
            "inference_composition.py",
            "intergrax/runtime/execution/inference_executor.py",
        ),
        "Inference executor factory",
        "Governed inference factory (internal — no production caller per TRACE-X Q07)",
        "Execution composition",
        "test_trace_x_p4_r5_closed_world.py::test_txp4r5_q07_build_governed_inference_executor_no_production_caller",
        prod=False,
        prod_rationale="TRACE-X Q07 — factory defined but no production caller",
    ),
    _m(
        "L3-INT-01",
        "integrations",
        "intergrax/integrations/providers/key_value_cache/redis/bundle.py",
        (
            "redis/bundle.py",
            "intergrax/integrations/factory.py",
            "IntegrationProfile",
        ),
        "Integrations",
        "KeyValueCache provider bundle",
        "Integration factory + profile",
        "tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py::test_ebh5_exit_inventory_complete",
    ),
    _m(
        "L3-INT-02",
        "integrations",
        "intergrax/integrations/providers/message_bus/celery/bundle.py",
        (
            "celery/bundle.py",
            "intergrax/integrations/factory.py",
        ),
        "Integrations",
        "MessageBus bundle",
        "Integration factory",
        "tests/qualification/ebh_5/test_ebh_5_reachability_gates.py",
    ),
    _m(
        "L3-INT-03",
        "integrations",
        "intergrax/integrations/providers/relational_persistence/sqlite/bundle.py",
        (
            "sqlite/bundle.py",
            "intergrax/contracts/persistence/relational.py",
        ),
        "Integrations persistence edge",
        "Relational persistence contract",
        "IntegrationProfile resolver",
        "tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py::test_ebh6_exit_sqlite_annex_accountability",
    ),
    _m(
        "L3-STA-01",
        "state_persistence",
        "intergrax/applications/_shared/runtime_config_bridge.py",
        (
            "IntegrationProfile relational provider slug",
            "intergrax/integrations/providers/relational_persistence/sqlite/bundle.py",
        ),
        "Integrations materialization",
        "RelationalPersistencePort",
        "Profile-driven composition",
        "tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py::test_ebh6_exit_sqlite_annex_accountability",
    ),
    _m(
        "L3-STA-02",
        "state_persistence",
        "intergrax/integrations/providers/relational_persistence/sqlite/bundle.py::connect",
        (
            "bundle.py",
            "sqlite3 via sanctioned adapter only",
        ),
        "SQLite adapter boundary",
        "Provider adapter — not generic core authority",
        "Integration bundle composition",
        "tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py",
        finding_id="EBH6-SQLITE-01",
    ),
    _m(
        "L3-STA-03",
        "state_persistence",
        "tests/qualification/ebh_5/_sqlite_touchpoint_registry.py",
        (
            "38 touchpoints @ HEAD",
            "EBH-5 annex replay",
        ),
        "Qualification SSOT",
        "SqliteTouchpointRegistry",
        "EBH-5/6 qualification",
        "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_sqlite_findings_cover_annex",
        prod=False,
        prod_rationale="Qualification inventory — production paths referenced per touchpoint row",
    ),
    _m(
        "L3-REC-01",
        "reliability_recovery",
        "intergrax/runtime/resilience/dependency_attempt_execution_boundary.py",
        (
            "dependency_attempt_execution_boundary.py",
            "intergrax/contracts/reliability/resilience_policy.py",
        ),
        "Reliability admission",
        "ResiliencePolicy",
        "Execution composition",
        "tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py::test_no_recovery_god_object_in_production",
    ),
    _m(
        "L3-REC-02",
        "reliability_recovery",
        "intergrax/runtime/resilience/dependency_attempt_execution_boundary.py",
        (
            "dependency_attempt_execution_boundary.py",
            "intergrax/contracts/reliability/resilience_policy.py",
        ),
        "Reliability admission",
        "ResiliencePolicy",
        "Execution composition",
        "tests/unit/runtime/architecture/test_w3_c4_decision_durable_recovery_canonical_wiring.py",
    ),
    _m(
        "L3-REC-03",
        "reliability_recovery",
        "intergrax/runtime/resilience/local_provider_retry_budget.py",
        (
            "local_provider_retry_budget.py",
            "intergrax/contracts/retry_budget.py",
        ),
        "Retry boundary",
        "Bounded retry budget",
        "ToolRuntime / Execution",
        "tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py",
    ),
    _m(
        "L3-OBS-01",
        "observability_diagnostics",
        "intergrax/applications/_shared/runtime_event_delivery_wiring.py",
        (
            "RuntimeEventBus",
            "BoundedEventSink",
            "OtlpTransportPort",
        ),
        "Observability (non-authoritative)",
        "ObservabilityExportPayload",
        "Application observability wiring",
        "tests/unit/runtime/events/test_enterprise_scale_resilience_w5_b2_composition_wiring.py::test_published_event_reaches_bounded_sink",
    ),
    _m(
        "L3-OBS-02",
        "observability_diagnostics",
        "intergrax/runtime/observability/runtime_event_bus.py",
        (
            "runtime_event_bus.py",
            "intergrax/contracts/observability/runtime_event.py",
        ),
        "Event bus delivery",
        "RuntimeEvent",
        "Runtime observability wiring",
        "tests/unit/runtime/observability/test_enterprise_scale_resilience_w5_g_profile_activation.py::test_export_failure_isolated_from_execution_plane",
    ),
    _m(
        "L3-OBS-03",
        "observability_diagnostics",
        "intergrax/runtime/diagnostics/diagnostic_orchestrator.py",
        (
            "diagnostic_orchestrator.py",
            "intergrax/contracts/diagnostics/",
        ),
        "Diagnostics interpret-only",
        "Diagnostic ports",
        "Diagnostic runtime wiring",
        "tests/qualification/control_planes/ctrl_x/test_ctrl_x_r2_middleware_composition.py",
    ),
    _m(
        "L3-TRC-01",
        "events_traceability",
        "intergrax/applications/_shared/runtime_event_delivery_wiring.py",
        (
            "event delivery wiring",
            "TRACE-X sanctioned root registry",
        ),
        "Trace spine",
        "RuntimeEvent correlation fields",
        "Application + TRACE-X qual",
        "tests/qualification/trace_x/test_trace_x_p4_r5_closed_world.py::test_txp4r5_q04_production_composition_registry_parity",
    ),
    _m(
        "L3-TRC-02",
        "events_traceability",
        "intergrax/runtime/observability/export/runtime_event_export_sink.py",
        (
            "runtime_event_export_sink.py",
            "EventExportSinkPort",
        ),
        "Export sink",
        "EventExportSinkPort",
        "Observability composition",
        "tests/unit/runtime/events/test_enterprise_scale_resilience_w5_b2_composition_wiring.py",
    ),
    _m(
        "L3-TRC-03",
        "events_traceability",
        "tests/qualification/trace_x/_trace_x_p4_production_composition_registry.py",
        (
            "PRODUCTION_COMPOSITION_SITE_REGISTRY",
            "sanctioned root parity tests",
        ),
        "TRACE-X qualification",
        "RegisteredProductionCompositionSite",
        "TRACE-X P4 qual",
        "tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py::test_ebh6_exit_sanctioned_composition_roots_non_empty",
    ),
    _m(
        "L3-AGD-01",
        "agent_distribution",
        "agents/external_contractor_adapter/external_contractor_adapter_agent.py",
        (
            "Tier-2 agent module",
            "intergrax/contracts/agent/",
        ),
        "Agent distribution tier",
        "Agent contract ports",
        "Application host roster",
        "tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py::test_tier0_must_not_import_agents_or_applications",
    ),
    _m(
        "L3-AGD-02",
        "agent_distribution",
        "intergrax/applications/_shared/registry_projection.py",
        (
            "registry_projection.py",
            "intergrax/contracts/agent_distribution/",
            "intergrax/applications/_shared/deploy_activate_pipeline.py",
        ),
        "Registry read boundary",
        "Distribution revision contracts",
        "Agent distribution activation",
        "tests/integration/agent_distribution/test_canonical_agent_lifecycle_architecture_gate.py",
    ),
    _m(
        "L3-AGD-03",
        "agent_distribution",
        "intergrax/agent_distribution/catalog.py",
        (
            "intergrax/agent_distribution/catalog.py",
            "intergrax/agent_distribution/effective_roster.py",
        ),
        "Agent registry",
        "Registry projection",
        "Distribution composition",
        "tests/integration/agent_distribution/test_canonical_agent_lifecycle_architecture_gate.py",
    ),
    _m(
        "L3-RAG-01",
        "rag_memory",
        "intergrax/applications/_shared/rag_runtime_bridge.py",
        (
            "rag_runtime_bridge.py",
            "intergrax/rag/",
            "intergrax/integrations/",
        ),
        "RAG vs Integrations",
        "RAG retrieval contracts",
        "Application RAG wiring",
        "tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py::test_ebh5_exit_inventory_complete",
    ),
    _m(
        "L3-RAG-02",
        "rag_memory",
        "intergrax/memory/user_profile_store.py",
        (
            "memory tier",
            "intergrax/contracts/memory/",
        ),
        "Memory semantic owner",
        "MemoryControlPlane",
        "Application memory composition",
        "tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_layer_evidence_annex_non_empty",
    ),
    _m(
        "L3-RAG-03",
        "rag_memory",
        "intergrax/rag/retrieval/retrieval_service.py",
        (
            "retrieval_service.py",
            "Integration vectorstore materialization",
        ),
        "RAG retrieval",
        "RetrievalService contract",
        "Integration-owned providers",
        "tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py",
    ),
    _m(
        "L3-HST-01",
        "runtime_hosting",
        "intergrax/runtime/bootstrap/context_bootstrap.py",
        (
            "context_bootstrap.py",
            "intergrax/contracts/runtime/hosting_profile.py",
        ),
        "Hosting bootstrap",
        "HostingProfile",
        "Runtime hosting composition",
        "tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py::test_ebh6_exit_sanctioned_composition_roots_non_empty",
    ),
    _m(
        "L3-HST-02",
        "runtime_hosting",
        "intergrax/runtime/bootstrap/context_bootstrap.py",
        (
            "context_bootstrap.py",
            "CE orchestrator entry",
        ),
        "Context engineering hosting",
        "ContextEngineering contracts",
        "CE composition",
        "tests/qualification/control_planes/ctrl_x/catalog.py",
    ),
    _m(
        "L3-HST-03",
        "runtime_hosting",
        "intergrax/applications/_shared/runtime_config_bridge.py",
        (
            "runtime_config_bridge.py",
            "HostProfile / materialize_runtime_config",
        ),
        "Host profile loading",
        "HostProfile loader",
        "Configuration composition",
        "tests/qualification/ebh_6/test_ebh_6_config_x_head_reconciliation.py",
    ),
    _m(
        "L3-TLC-01",
        "tools_skills_capabilities",
        "intergrax/tools/registry/wiring.py",
        (
            "tools/registry/wiring.py",
            "intergrax/contracts/tools/",
        ),
        "Tool registry",
        "ToolRuntime SPI",
        "Tool composition root",
        "tests/unit/runtime/architecture/test_ds_plugin_architecture_gates.py",
    ),
    _m(
        "L3-TLC-02",
        "tools_skills_capabilities",
        "intergrax/runtime/skills/skill_resolver.py",
        (
            "skill_resolver.py",
            "SkillManifest",
        ),
        "Skills plane",
        "SkillResolver",
        "Agent contract assembly",
        "tests/unit/runtime/architecture/test_hardening_5_plugin_architecture_gate.py",
    ),
    _m(
        "L3-TLC-03",
        "tools_skills_capabilities",
        "intergrax/runtime/execution/tool_runtime.py",
        (
            "tool_runtime.py",
            "governance admission",
        ),
        "Tool execution",
        "Tool intent contracts",
        "ToolRuntime composition",
        "tests/qualification/control_planes/ctrl_x/test_ctrl_x_r2_middleware_composition.py",
    ),
    _m(
        "L3-BGI-01",
        "background_workers_infra_adapters",
        "intergrax/integrations/providers/message_bus/celery/bundle.py",
        (
            "celery worker adapter",
            "Integration factory",
        ),
        "Background work via integration",
        "MessageBus provider bundle",
        "Integration composition",
        "tests/qualification/ebh_5/test_ebh_5_reachability_gates.py",
    ),
    _m(
        "L3-BGI-02",
        "background_workers_infra_adapters",
        "intergrax/runtime/transport/bootstrap.py",
        (
            "transport bootstrap",
            "queue adapter selection via profile",
        ),
        "Transport / queue ingress",
        "Transport bootstrap owner",
        "Runtime transport root",
        "tests/qualification/ebh_5/test_ebh_5_reachability_gates.py",
    ),
    _m(
        "L3-BGI-03",
        "background_workers_infra_adapters",
        "intergrax/integrations/providers/",
        (
            "provider bundles",
            "infrastructure adapters at Integration edge only",
        ),
        "Integration adapters",
        "Provider bundle pattern",
        "Integration factory",
        "tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py::test_ebh6_exit_closed_world_surfaces",
    ),
)


@lru_cache(maxsize=1)
def all_l3_reachability_records() -> tuple[L3ReachabilityRecord, ...]:
    return _L3_MECHANISMS


def l3_records_by_layer(layer: str) -> tuple[L3ReachabilityRecord, ...]:
    return tuple(r for r in all_l3_reachability_records() if r.layer == layer)


def mechanism_ids_for_layer(layer: str) -> tuple[str, ...]:
    return tuple(r.mechanism_id for r in l3_records_by_layer(layer))


def count_l3_aj_outcomes() -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in all_l3_reachability_records():
        for ctrl in rec.aj_controls:
            key = ctrl.outcome.value
            counts[key] = counts.get(key, 0) + 1
    return counts


def assert_l3_reachability_complete() -> None:
    assert len(all_l3_reachability_records()) >= 48
    for layer in EBH6_AUDIT_LAYERS:
        records = l3_records_by_layer(layer)
        assert len(records) >= 3, layer
        for rec in records:
            assert rec.entrypoint.strip(), rec.mechanism_id
            assert len(rec.call_chain) >= 1, rec.mechanism_id
            assert rec.semantic_owner.strip()
            assert rec.canonical_contract.strip()
            assert rec.composition_owner.strip()
            assert len(rec.aj_controls) == 10, rec.mechanism_id
            for ctrl in rec.aj_controls:
                assert ctrl.evidence_ref.strip(), rec.mechanism_id
                if ctrl.outcome == AjControlOutcome.NA_WITH_EVIDENCE:
                    assert ctrl.evidence_ref.startswith("N/A — WITH EVIDENCE:"), rec.mechanism_id
                if ctrl.outcome == AjControlOutcome.PASS_CANDIDATE:
                    assert "group=AJG-" in ctrl.evidence_ref, f"{rec.mechanism_id}/{ctrl.detection_class}"
