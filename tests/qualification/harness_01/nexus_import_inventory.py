# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

"""Explicit higher-layer Nexus import inventory (HARNESS-01 R4).

Closed-world inventory + owner-layer rules. Unknown importers are UNCLASSIFIED (FAIL).
There is no default LEGAL / AUTHORIZED_INTERNAL_COMPOSITION fallback.

Invariant (HARNESS-01 internal-only Nexus):
    Nexus is internal to Execution Runtime.
    No public contract or extension surface may depend on Nexus.
"""

HARNESS_01_NEXUS_INTERNAL_ONLY_INVARIANT: str = (
    "Nexus is internal to Execution Runtime. "
    "No public contract or extension surface may depend on Nexus."
)

from dataclasses import dataclass
from typing import Literal


Harness01OwnerLayer = Literal[
    "PUBLIC_CONTRACT",
    "APPLICATION_CONTRACT",
    "APPLICATION_HOST",
    "HOST_COMPOSITION",
    "AGENT_PUBLIC",
    "AGENT_INTERNAL",
    "PLUGIN_SURFACE",
    "INTEGRATION",
    "EXECUTION_ENGINE",
    "PLATFORM_RUNTIME",
    "TOOLING",
    "UNKNOWN",
]

Harness01NexusImporterClassification = Literal[
    "EXECUTION_ENGINE_INTERNAL",
    "HOST_EXECUTION_COMPOSITION",
    "PLATFORM_RUNTIME_INTERNAL",
    "AGENT_EXECUTION_BRIDGE",
    "TOOLING_INTERNAL",
    "INTEGRATION_PROVIDER",
    "MIGRATION_DEBT",
    "VIOLATION",
    "UNCLASSIFIED",
]

Harness01BoundaryStatus = Literal["LEGAL", "DEBT", "VIOLATION", "UNCLASSIFIED"]


@dataclass(frozen=True, slots=True)
class Harness01HigherLayerNexusImporter:
    """Closed-world classification row for a higher-layer Nexus importer."""

    path: str
    owner_layer: Harness01OwnerLayer
    classification: Harness01NexusImporterClassification
    reason: str
    evidence: str
    boundary_status: Harness01BoundaryStatus


# Paths that are unconditional Nexus hard-violations (layer rule; inventory cannot override).
HARNESS_01_NEXUS_HARD_VIOLATION_PREFIXES: tuple[str, ...] = (
    "intergrax/contracts/",
)

HARNESS_01_NEXUS_HARD_VIOLATION_EXACT: frozenset[str] = frozenset(
    {
        "intergrax/agents/agent_contract.py",
        "intergrax/agents/uaep_protocol.py",
    }
)

HARNESS_01_PUBLIC_EXTENSION_SURFACE_PREFIXES: tuple[str, ...] = (
    "intergrax/applications/contracts/",
)

HARNESS_01_PUBLIC_EXTENSION_SURFACE_EXACT: frozenset[str] = frozenset(
    {
        "intergrax/agents/agent_contract.py",
        "intergrax/agents/uaep_protocol.py",
        "intergrax/agents/authoring/base.py",
    }
)


def path_is_hard_nexus_violation(path: str) -> bool:
    """Independent layer rule — cannot be overridden by an inventory LEGAL row."""
    if path in HARNESS_01_NEXUS_HARD_VIOLATION_EXACT:
        return True
    if any(path.startswith(prefix) for prefix in HARNESS_01_NEXUS_HARD_VIOLATION_PREFIXES):
        return True
    if any(path.startswith(prefix) for prefix in HARNESS_01_PUBLIC_EXTENSION_SURFACE_PREFIXES):
        return True
    if "/contracts/" in path and (
        path.startswith("applications/") or path.startswith("intergrax/applications/")
    ):
        return True
    return False


def path_is_public_extension_surface(path: str) -> bool:
    if path in HARNESS_01_PUBLIC_EXTENSION_SURFACE_EXACT:
        return True
    if any(path.startswith(prefix) for prefix in HARNESS_01_PUBLIC_EXTENSION_SURFACE_PREFIXES):
        return True
    if "/contracts/" in path and (
        path.startswith("applications/") or path.startswith("intergrax/applications/")
    ):
        return True
    return False


_EVIDENCE_CLASSIFIED = (
    "tests/qualification/harness_01/test_harness_01_gates.py::"
    "test_harness_01_higher_layer_nexus_imports_are_classified"
)
_EVIDENCE_LAYER = (
    "tests/qualification/harness_01/test_harness_01_gates.py::"
    "test_harness_01_layer_rules_reject_contract_and_public_nexus_imports"
)
_EVIDENCE_EE = (
    "docs/project/architecture/UNIFIED_EXECUTION_ARCHITECTURE.md"
    " (Nexus private/internal to Execution Engine)"
)
_EVIDENCE_HOST = (
    "docs/project/architecture/UNIFIED_EXECUTION_ARCHITECTURE.md"
    " (RuntimeConfig → build_nexus_loop_from_environment host composition)"
)
_EVIDENCE_TOPOLOGY = (
    "tests/unit/runtime/architecture/"
    "test_gr10_r9_r4_topology_mse_composition_mandatory.py"
)
_EVIDENCE_MATERIALIZER = (
    "tests/unit/architecture/test_ebh_2b_agent_nexus_contract_separation.py"
)


def _rule_classify(path: str) -> Harness01HigherLayerNexusImporter:
    if path_is_hard_nexus_violation(path):
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="PUBLIC_CONTRACT"
            if path.startswith("intergrax/contracts/")
            else (
                "AGENT_PUBLIC"
                if path.startswith("intergrax/agents/")
                else "APPLICATION_CONTRACT"
            ),
            classification="VIOLATION",
            reason=(
                "Layer rule: public/domain/application contracts and public agent surfaces "
                "must not import intergrax.runtime.nexus.*"
            ),
            evidence=_EVIDENCE_LAYER,
            boundary_status="VIOLATION",
        )

    if path.startswith("intergrax/runtime/execution/"):
        reason = (
            "Execution Engine implementation may consume Nexus internals; "
            "Nexus types must not escape public signatures."
        )
        evidence = _EVIDENCE_EE
        if path.endswith(
            (
                "orchestration_topology_slot_mse_enforcement.py",
                "orchestration_topology_production_composition.py",
            )
        ):
            reason = (
                "Execution-engine composition wrapping public OrchestrationSlotExecutor / "
                "MeaningfulSideEffectAuthorizationPort with Nexus governed executors; "
                "Nexus types stay out of public port signatures."
            )
            if path.endswith("orchestration_topology_production_composition.py"):
                reason = (
                    "Strict production orchestration topology composition (GR-10-R13-R2); "
                    "builds OrchestrationTopologySubmissionPort via EE internals; "
                    "NexusLoop is an internal factory dependency only."
                )
            evidence = _EVIDENCE_TOPOLOGY
        if path.endswith("agent_runtime_context_materializer.py"):
            reason = (
                "Execution-engine internal UAEP materializer protocol; "
                "public Agent/UAEPAgent surfaces remain Nexus-free."
            )
            evidence = _EVIDENCE_MATERIALIZER
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="EXECUTION_ENGINE",
            classification="EXECUTION_ENGINE_INTERNAL",
            reason=reason,
            evidence=evidence,
            boundary_status="LEGAL",
        )

    if path.startswith("intergrax/applications/_shared/") or (
        path.startswith("applications/") and "/host/" in path
    ):
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="HOST_COMPOSITION"
            if path.startswith("intergrax/applications/_shared/")
            else "APPLICATION_HOST",
            classification="HOST_EXECUTION_COMPOSITION",
            reason=(
                "Documented host/execution composition root wiring Nexus behind "
                "ApplicationEnvironmentProfile / build_nexus_loop_from_environment; "
                "not a public Nexus ABI for plugins."
            ),
            evidence=_EVIDENCE_HOST,
            boundary_status="LEGAL",
        )

    if path.startswith("intergrax/runtime/wiring/"):
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="PLATFORM_RUNTIME",
            classification="PLATFORM_RUNTIME_INTERNAL",
            reason=(
                "Platform runtime wiring bridges composing Execution Engine internals; "
                "not a public Nexus entry."
            ),
            evidence=_EVIDENCE_EE,
            boundary_status="LEGAL",
        )

    if path.startswith("intergrax/runtime/task/") or path.startswith(
        "intergrax/runtime/agent_governance/"
    ):
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="PLATFORM_RUNTIME",
            classification="PLATFORM_RUNTIME_INTERNAL",
            reason=(
                "Task/UAEP governance runtime path consumes Nexus orchestration internals "
                "inside the platform execution stack."
            ),
            evidence=_EVIDENCE_EE,
            boundary_status="LEGAL",
        )

    if path.startswith("intergrax/agents/"):
        classification: Harness01NexusImporterClassification = "AGENT_EXECUTION_BRIDGE"
        status: Harness01BoundaryStatus = "DEBT"
        reason = (
            "Tier-2 agent/UAEP/authoring bridge still couples to Nexus implementation types; "
            "public Agent/UAEPAgent and authoring base surfaces are gated Nexus-free. "
            "Further ownership inversion tracked as migration debt."
        )
        if path.endswith("runtime_answer_mapping.py"):
            classification = "AGENT_EXECUTION_BRIDGE"
            status = "LEGAL"
            reason = (
                "Canonical Nexus RuntimeAnswer → AgentExecutionResult adapter "
                "(moved out of intergrax.contracts)."
            )
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="AGENT_INTERNAL",
            classification=classification,
            reason=reason,
            evidence=_EVIDENCE_MATERIALIZER,
            boundary_status=status,
        )

    if path.startswith("intergrax/runtime/"):
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="PLATFORM_RUNTIME",
            classification="PLATFORM_RUNTIME_INTERNAL",
            reason=(
                "Platform runtime module consuming Nexus internals within Execution stack; "
                "not exposed as public Nexus API. Owner is not the entire runtime/* tree — "
                "classified per runtime subsystem path under R4 owner rules."
            ),
            evidence=_EVIDENCE_EE,
            boundary_status="LEGAL",
        )

    if path.startswith("intergrax/tools/") or path.startswith("intergrax/websearch/"):
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="TOOLING",
            classification="TOOLING_INTERNAL",
            reason="Platform tool/websearch provider consuming Nexus context helpers internally.",
            evidence=_EVIDENCE_CLASSIFIED,
            boundary_status="LEGAL",
        )

    if path.startswith("intergrax/integrations/"):
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="INTEGRATION",
            classification="INTEGRATION_PROVIDER",
            reason="Integration provider wiring into runtime persistence/session helpers.",
            evidence=_EVIDENCE_CLASSIFIED,
            boundary_status="DEBT",
        )

    if path.startswith("intergrax/llm_adapters/") or path.startswith("intergrax/rag/"):
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="PLATFORM_RUNTIME",
            classification="PLATFORM_RUNTIME_INTERNAL",
            reason="Adapter/RAG runtime sync against Nexus session/config surfaces.",
            evidence=_EVIDENCE_CLASSIFIED,
            boundary_status="DEBT",
        )

    if path.startswith("intergrax/eval/") or path.startswith("intergrax/debug/") or path.startswith(
        "intergrax/lab/"
    ) or path.startswith("intergrax/cli/") or path.startswith("intergrax/experiments/") or path.startswith(
        "intergrax/fastapi_core/"
    ):
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="TOOLING",
            classification="TOOLING_INTERNAL",
            reason="Lab/eval/debug/CLI tooling consuming Nexus for harness execution.",
            evidence=_EVIDENCE_CLASSIFIED,
            boundary_status="DEBT",
        )

    return Harness01HigherLayerNexusImporter(
        path=path,
        owner_layer="UNKNOWN",
        classification="UNCLASSIFIED",
        reason="No owner-layer rule matched — fail-closed until explicitly classified.",
        evidence=_EVIDENCE_CLASSIFIED,
        boundary_status="UNCLASSIFIED",
    )


# Explicit closed-world path set (must match AST discovery; never auto-generated in test).
_PATHS: tuple[str, ...] = (
    "applications/attestation_demo/host/integration_wiring.py",
    "applications/dispute_sim_application/host/factory.py",
    "applications/dispute_sim_application/host/integration_wiring.py",
    "applications/governed_contractor_application/host/execution_wiring.py",
    "applications/governed_contractor_application/host/integration_wiring.py",
    "applications/governed_contractor_application/host/orchestration_topology_production_composition.py",
    "applications/lab_application/host/integration_wiring.py",
    "applications/legal_application/host/factory.py",
    "applications/local_workspace_application/host/execution_wiring.py",
    "applications/local_workspace_application/host/integration_wiring.py",
    "applications/local_workspace_application/host/lkw_task_enricher.py",
    "applications/local_workspace_application/host/run_task_enricher.py",
    "applications/poc_template_application/host/integration_wiring.py",
    "applications/research_application/host/integration_wiring.py",
    "intergrax/applications/_shared/acp_runtime_session_hooks.py",
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
    "intergrax/applications/_shared/diagnostic_read_wiring.py",
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
    "intergrax/applications/_shared/lab_harness_context.py",
    "intergrax/applications/_shared/lab_runtime_config.py",
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
    "intergrax/cli/mvp_evolution.py",
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
    "intergrax/runtime/execution/agent_runtime_context_materializer.py",
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
    "intergrax/runtime/execution/orchestration_topology_production_composition.py",
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
    "intergrax/websearch/service/websearch_context_generator.py",
)

HARNESS_01_HIGHER_LAYER_NEXUS_IMPORTER_ROWS: tuple[Harness01HigherLayerNexusImporter, ...] = tuple(
    _rule_classify(path) for path in _PATHS
)

HARNESS_01_HIGHER_LAYER_NEXUS_IMPORTERS: frozenset[str] = frozenset(
    row.path for row in HARNESS_01_HIGHER_LAYER_NEXUS_IMPORTER_ROWS
)

assert HARNESS_01_HIGHER_LAYER_NEXUS_IMPORTERS == frozenset(_PATHS)
assert not any(
    row.classification == "UNCLASSIFIED" or row.boundary_status == "UNCLASSIFIED"
    for row in HARNESS_01_HIGHER_LAYER_NEXUS_IMPORTER_ROWS
), "inventory contains UNCLASSIFIED rows — extend owner-layer rules"
assert not any(
    row.classification == "VIOLATION" or row.boundary_status == "VIOLATION"
    for row in HARNESS_01_HIGHER_LAYER_NEXUS_IMPORTER_ROWS
), "inventory contains VIOLATION rows — remove illegal Nexus imports before allowlisting"
