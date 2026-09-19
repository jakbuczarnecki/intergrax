# © Artur Czarnecki. All rights reserved.

"""Explicit higher-layer Nexus import inventory (HARNESS-01 R3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Harness01NexusImporterClassification = Literal[
    "AUTHORIZED_INTERNAL_COMPOSITION",
    "MIGRATION_DEBT",
    "BOUNDARY_VIOLATION",
]


@dataclass(frozen=True, slots=True)
class Harness01HigherLayerNexusImporter:
    """Closed-world classification row for a higher-layer Nexus importer."""

    path: str
    classification: Harness01NexusImporterClassification
    reason: str
    owner_layer: str
    evidence: str
    boundary_status: Literal["LEGAL", "DEBT", "VIOLATION"]


def _owner_layer(path: str) -> str:
    if path.startswith("applications/"):
        return "applications/*/host (composition root)"
    if path.startswith("intergrax/agents/"):
        return "intergrax/agents (Tier-2 execution bridge)"
    if path.startswith("intergrax/applications/"):
        return "intergrax/applications (host composition)"
    if path.startswith("intergrax/runtime/execution/"):
        return "intergrax/runtime/execution (execution-engine composition)"
    if path.startswith("intergrax/runtime/"):
        return "intergrax/runtime (platform composition)"
    if path.startswith("intergrax/contracts/"):
        return "intergrax/contracts (mapping bridge)"
    if path.startswith("intergrax/"):
        return "intergrax (platform internal)"
    return "unknown"


def _r2_composition(path: str) -> Harness01HigherLayerNexusImporter:
    return Harness01HigherLayerNexusImporter(
        path=path,
        classification="AUTHORIZED_INTERNAL_COMPOSITION",
        reason=(
            "HARNESS-01-R2 closed-world inventory: internal composition/bridge consumer; "
            "not a public Nexus ABI. Re-audited in R3 as still importing Nexus."
        ),
        owner_layer=_owner_layer(path),
        evidence=(
            "tests/qualification/harness_01/test_harness_01_gates.py::"
            "test_harness_01_higher_layer_nexus_imports_are_classified"
        ),
        boundary_status="LEGAL",
    )


_R3_AUDITED: dict[str, Harness01HigherLayerNexusImporter] = {
    "intergrax/agents/agent_runtime_context_materializer.py": Harness01HigherLayerNexusImporter(
        path="intergrax/agents/agent_runtime_context_materializer.py",
        classification="AUTHORIZED_INTERNAL_COMPOSITION",
        reason=(
            "EBH-2B internal UAEP execution hook for Nexus RuntimeContext materialization; "
            "explicitly excluded from public Agent/UAEP contract surfaces "
            "(agent_contract.py / uaep_protocol.py remain runtime-pure)."
        ),
        owner_layer="intergrax/agents (Tier-2 execution bridge)",
        evidence="tests/unit/architecture/test_ebh_2b_agent_nexus_contract_separation.py",
        boundary_status="LEGAL",
    ),
    "intergrax/runtime/execution/orchestration_topology_slot_mse_enforcement.py": (
        Harness01HigherLayerNexusImporter(
            path="intergrax/runtime/execution/orchestration_topology_slot_mse_enforcement.py",
            classification="AUTHORIZED_INTERNAL_COMPOSITION",
            reason=(
                "Execution-engine composition wrapping public OrchestrationSlotExecutor / "
                "MeaningfulSideEffectAuthorizationPort with Nexus governed executors; "
                "Nexus types stay out of public signatures."
            ),
            owner_layer="intergrax/runtime/execution (execution-engine composition)",
            evidence=(
                "tests/unit/runtime/architecture/"
                "test_gr10_r9_r4_topology_mse_composition_mandatory.py"
            ),
            boundary_status="LEGAL",
        )
    ),
}

_PATHS: tuple[str, ...] = (
    "applications/attestation_demo/host/integration_wiring.py",
    "applications/dispute_sim_application/host/factory.py",
    "applications/dispute_sim_application/host/integration_wiring.py",
    "applications/governed_contractor_application/host/execution_wiring.py",
    "applications/governed_contractor_application/host/integration_wiring.py",
    "applications/lab_application/host/integration_wiring.py",
    "applications/legal_application/host/factory.py",
    "applications/local_workspace_application/host/execution_wiring.py",
    "applications/local_workspace_application/host/integration_wiring.py",
    "applications/local_workspace_application/host/lkw_task_enricher.py",
    "applications/local_workspace_application/host/run_task_enricher.py",
    "applications/poc_template_application/host/integration_wiring.py",
    "applications/research_application/host/integration_wiring.py",
    "intergrax/agents/agent_engine.py",
    "intergrax/agents/agent_runtime_context_materializer.py",
    "intergrax/agents/authoring/acp_routing_trace_bridge.py",
    "intergrax/agents/authoring/acp_run.py",
    "intergrax/agents/authoring/acp_stub_reflex.py",
    "intergrax/agents/authoring/acp_uaep_shim.py",
    "intergrax/agents/authoring/base.py",
    "intergrax/agents/authoring/diagnostic_serialization.py",
    "intergrax/agents/authoring/llm_router.py",
    "intergrax/agents/authoring/patterns/base.py",
    "intergrax/agents/authoring/patterns/diagnostic_reflex.py",
    "intergrax/agents/authoring/patterns/reference.py",
    "intergrax/agents/authoring/runtime_tool_helpers.py",
    "intergrax/agents/authoring/shared_context_bridge.py",
    "intergrax/agents/authoring/step_outcome.py",
    "intergrax/agents/authoring/uaep_linear_bridge.py",
    "intergrax/agents/authoring/uaep_step_bridge.py",
    "intergrax/agents/echo/echo_agent.py",
    "intergrax/agents/harness_reference_agent.py",
    "intergrax/agents/persistence/catalog_declarative_invoker.py",
    "intergrax/agents/persistence/skill_host_wiring.py",
    "intergrax/agents/reference_harness.py",
    "intergrax/agents/runtime_request_bridge.py",
    "intergrax/agents/uaep.py",
    "intergrax/applications/_shared/adaptive_runtime_bridge.py",
    "intergrax/applications/_shared/agent_runtime_governance_wiring.py",
    "intergrax/applications/_shared/application_environment_state_middleware.py",
    "intergrax/applications/_shared/application_host_wiring.py",
    "intergrax/applications/_shared/application_security_wiring.py",
    "intergrax/applications/_shared/attestation_runtime_bridge.py",
    "intergrax/applications/_shared/capability_alias_intake_wiring.py",
    "intergrax/applications/_shared/catalog_runtime_bridge.py",
    "intergrax/applications/_shared/compensation_side_effect_wiring.py",
    "intergrax/applications/_shared/context_engine_resolver.py",
    "intergrax/applications/_shared/context_presets.py",
    "intergrax/applications/_shared/context_runtime_bridge.py",
    "intergrax/applications/_shared/context_wiring.py",
    "intergrax/applications/_shared/cost_runtime_bridge.py",
    "intergrax/applications/_shared/cost_wiring.py",
    "intergrax/applications/_shared/decision_verifier_llm_resolver.py",
    "intergrax/applications/_shared/decision_wiring.py",
    "intergrax/applications/_shared/declarative_tool_wiring.py",
    "intergrax/applications/_shared/diagnostic_runtime_wiring.py",
    "intergrax/applications/_shared/environment_snapshot_wiring.py",
    "intergrax/applications/_shared/environment_wiring.py",
    "intergrax/applications/_shared/evaluation_runtime_bridge.py",
    "intergrax/applications/_shared/evaluator_loop_graph_templates.py",
    "intergrax/applications/_shared/graph_spec_to_plan.py",
    "intergrax/applications/_shared/guardrail_assembly_resolver.py",
    "intergrax/applications/_shared/guardrail_runtime_bridge.py",
    "intergrax/applications/_shared/guardrail_wiring.py",
    "intergrax/applications/_shared/harness_host_composition.py",
    "intergrax/applications/_shared/harness_host_runtime.py",
    "intergrax/applications/_shared/harness_host_task_execution_wiring.py",
    "intergrax/applications/_shared/host_task_execution_wiring.py",
    "intergrax/applications/_shared/integration_runtime_bridge.py",
    "intergrax/applications/_shared/llm_routing_runtime_bridge.py",
    "intergrax/applications/_shared/memory_runtime_bridge.py",
    "intergrax/applications/_shared/memory_wiring.py",
    "intergrax/applications/_shared/nexus_factory.py",
    "intergrax/applications/_shared/observability_assembly_resolver.py",
    "intergrax/applications/_shared/observability_runtime_bridge.py",
    "intergrax/applications/_shared/observability_wiring.py",
    "intergrax/applications/_shared/orchestration_wiring.py",
    "intergrax/applications/_shared/platform_wiring.py",
    "intergrax/applications/_shared/plugin_bootstrap.py",
    "intergrax/applications/_shared/production_delegated_subtask_child_execution_wiring.py",
    "intergrax/applications/_shared/prompt_runtime_bridge.py",
    "intergrax/applications/_shared/rag_runtime_bridge.py",
    "intergrax/applications/_shared/reasoning_wiring.py",
    "intergrax/applications/_shared/reliability_runtime_bridge.py",
    "intergrax/applications/_shared/reliability_wiring.py",
    "intergrax/applications/_shared/run_artifact_bundle_builder.py",
    "intergrax/applications/_shared/runtime_config_bridge.py",
    "intergrax/applications/_shared/scenario_runtime_baseline.py",
    "intergrax/applications/_shared/security_assembly_resolver.py",
    "intergrax/applications/_shared/security_runtime_bridge.py",
    "intergrax/applications/_shared/security_wiring.py",
    "intergrax/applications/_shared/session_tool_wiring.py",
    "intergrax/applications/_shared/tool_engine_wiring.py",
    "intergrax/applications/contracts/environment_profile/sub_profiles.py",
    "intergrax/applications/contracts/graph_spec.py",
    "intergrax/cli/mvp_evolution.py",
    "intergrax/context/contracts.py",
    "intergrax/contracts/host_profile_slices.py",
    "intergrax/contracts/runtime_cost.py",
    "intergrax/contracts/runtime_mapping.py",
    "intergrax/debug/app.py",
    "intergrax/debug/formatters.py",
    "intergrax/debug/models.py",
    "intergrax/debug/router.py",
    "intergrax/debug/store.py",
    "intergrax/eval/eval_case.py",
    "intergrax/eval/eval_runner.py",
    "intergrax/eval/nexus_eval_runner.py",
    "intergrax/experiments/workflow.py",
    "intergrax/fastapi_core/runs/store_runtime.py",
    "intergrax/integrations/providers/relational_store/sqlite/bundle.py",
    "intergrax/integrations/providers/relational_store/sqlite/opens.py",
    "intergrax/lab/organization_worker.py",
    "intergrax/llm_adapters/providers/_openai_schema.py",
    "intergrax/llm_adapters/providers/openai_responses_adapter.py",
    "intergrax/llm_adapters/routing/runtime_sync.py",
    "intergrax/rag/profiles/runtime_rag_sync.py",
    "intergrax/rag/profiles/tool_wiring_runtime_sync.py",
    "intergrax/runtime/adaptive/cost_normalization.py",
    "intergrax/runtime/adaptive/signal_collector.py",
    "intergrax/runtime/adaptive/signal_emission.py",
    "intergrax/runtime/adaptive/trace_sequence_reader.py",
    "intergrax/runtime/agent_governance/request_builder.py",
    "intergrax/runtime/architecture/retrieval_security_wiring.py",
    "intergrax/runtime/attestation/boundary_emitter.py",
    "intergrax/runtime/attestation/kernel_wiring.py",
    "intergrax/runtime/cancellation/coordinator.py",
    "intergrax/runtime/codecraft/trace.py",
    "intergrax/runtime/events/context_skill_recording.py",
    "intergrax/runtime/events/planner_events.py",
    "intergrax/runtime/events/trace_bridge.py",
    "intergrax/runtime/events/unified_run_journal.py",
    "intergrax/runtime/execution/active_execution_budget.py",
    "intergrax/runtime/execution/agentic.py",
    "intergrax/runtime/execution/authority/registry.py",
    "intergrax/runtime/execution/budget/ledger.py",
    "intergrax/runtime/execution/budget/models.py",
    "intergrax/runtime/execution/budget/persistence.py",
    "intergrax/runtime/execution/budget/registry.py",
    "intergrax/runtime/execution/budget/snapshot.py",
    "intergrax/runtime/execution/child.py",
    "intergrax/runtime/execution/compensation_side_effect.py",
    "intergrax/runtime/execution/deadline_authority/resolver.py",
    "intergrax/runtime/execution/delegated_execution/context_projection.py",
    "intergrax/runtime/execution/delegated_execution/service.py",
    "intergrax/runtime/execution/delegated_subtask_child_port.py",
    "intergrax/runtime/execution/execution_work_port.py",
    "intergrax/runtime/execution/host_task.py",
    "intergrax/runtime/execution/nexus_host_execution.py",
    "intergrax/runtime/execution/nexus_host_task_terminal.py",
    "intergrax/runtime/execution/orchestration.py",
    "intergrax/runtime/execution/orchestration_topology_slot_mse_enforcement.py",
    "intergrax/runtime/execution/orchestration_topology_submission.py",
    "intergrax/runtime/execution/runtime.py",
    "intergrax/runtime/hooks/tool_hooks.py",
    "intergrax/runtime/human/declarative_hitl_grant.py",
    "intergrax/runtime/human/governed_continuation_bridge.py",
    "intergrax/runtime/long_running/checkpoint_builder.py",
    "intergrax/runtime/long_running/coordinator.py",
    "intergrax/runtime/metrics/export.py",
    "intergrax/runtime/observability/emitter.py",
    "intergrax/runtime/observability/export_bridge.py",
    "intergrax/runtime/observability/extension_sdk.py",
    "intergrax/runtime/observability/journal_export.py",
    "intergrax/runtime/observability/modality_metrics.py",
    "intergrax/runtime/observability/qualification_runtime_trace.py",
    "intergrax/runtime/persistence/integration_profile_wiring.py",
    "intergrax/runtime/plugins/default_plugins.py",
    "intergrax/runtime/policy/compliance_profiles.py",
    "intergrax/runtime/policy/declarative_enforcer.py",
    "intergrax/runtime/policy/execution_mode_defaults.py",
    "intergrax/runtime/policy/policy_trace_diagnostics.py",
    "intergrax/runtime/replay/persisted_trace_event_store.py",
    "intergrax/runtime/replay/trace_replay_bridge.py",
    "intergrax/runtime/task/nexus_worker_execution.py",
    "intergrax/runtime/task/task.py",
    "intergrax/runtime/task/task_metadata_bridge.py",
    "intergrax/runtime/task/task_run_bridge.py",
    "intergrax/runtime/task/task_trace.py",
    "intergrax/runtime/task/unified_task_runner.py",
    "intergrax/runtime/task/worker_bootstrap.py",
    "intergrax/runtime/token_optimization/llm_router.py",
    "intergrax/runtime/tools/idempotency_pre_effect_coordinator.py",
    "intergrax/runtime/user_profile/memory_consolidation_job.py",
    "intergrax/runtime/user_profile/user_profile_debug_service.py",
    "intergrax/runtime/user_profile/user_profile_debug_snapshot.py",
    "intergrax/runtime/wiring/attestation_runtime_bridge.py",
    "intergrax/runtime/wiring/context_runtime_bridge.py",
    "intergrax/runtime/wiring/llm_routing_runtime_bridge.py",
    "intergrax/runtime/wiring/policy_runtime_bridge.py",
    "intergrax/runtime/wiring/reliability_runtime_bridge.py",
    "intergrax/runtime/workspace/exec_ctx_isolation.py",
    "intergrax/tools/providers/context_tool/service.py",
    "intergrax/tools/providers/harness/service.py",
    "intergrax/tools/providers/rag/service.py",
    "intergrax/tools/registry/runtime_bindings.py",
    "intergrax/websearch/service/websearch_context_generator.py"
)

HARNESS_01_HIGHER_LAYER_NEXUS_IMPORTER_ROWS: tuple[Harness01HigherLayerNexusImporter, ...] = tuple(
    _R3_AUDITED[path] if path in _R3_AUDITED else _r2_composition(path) for path in _PATHS
)

HARNESS_01_HIGHER_LAYER_NEXUS_IMPORTERS: frozenset[str] = frozenset(
    row.path for row in HARNESS_01_HIGHER_LAYER_NEXUS_IMPORTER_ROWS
)

assert HARNESS_01_HIGHER_LAYER_NEXUS_IMPORTERS == frozenset(_PATHS)
assert not any(
    row.classification == "BOUNDARY_VIOLATION"
    for row in HARNESS_01_HIGHER_LAYER_NEXUS_IMPORTER_ROWS
)
