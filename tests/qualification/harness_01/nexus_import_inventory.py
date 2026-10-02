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
_EVIDENCE_SUSPENDED = (
    "tests/unit/runtime/architecture/test_uca6c_r6_architecture_gates.py"
    " (Execution-owned suspended-operation / reentry; Nexus-internal imports)"
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
        if "/suspended_operation/" in path:
            reason = (
                "Execution Engine suspended-operation implementation; "
                "Nexus imports are implementation-private; "
                "public Execution contracts remain Nexus-free; "
                "RuntimeToolInvoker physical enforcement stays Nexus-owned."
            )
            evidence = _EVIDENCE_SUSPENDED
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="EXECUTION_ENGINE",
            classification="EXECUTION_ENGINE_INTERNAL",
            reason=reason,
            evidence=evidence,
            boundary_status="LEGAL",
        )

    if path.startswith("intergrax/applications/_shared/"):
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="HOST_COMPOSITION",
            classification="HOST_EXECUTION_COMPOSITION",
            reason=(
                "Temporary host/shared composition debt tracked for HARNESS-01-R5-W6 — "
                "Host Composition Convergence; must converge to neutral Execution-owned "
                "composition boundary; not a public Nexus API."
            ),
            evidence=_EVIDENCE_HOST,
            boundary_status="DEBT",
        )

    if path.startswith("applications/") and "/host/" in path:
        reason = (
            "Temporary host composition debt tracked for HARNESS-01-R5-W6 — "
            "Host Composition Convergence; must converge to neutral Execution-owned "
            "composition boundary; not a public Nexus API."
        )
        if path.endswith("orchestration_topology_production_composition.py"):
            reason = (
                "Temporary host composition debt (direct NexusLoop coupling in host module); "
                "tracked for HARNESS-01-R5-W6 — Host Composition Convergence; "
                "must converge to neutral Execution-owned composition boundary; "
                "not a public Nexus API."
            )
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="APPLICATION_HOST",
            classification="HOST_EXECUTION_COMPOSITION",
            reason=reason,
            evidence=_EVIDENCE_HOST,
            boundary_status="DEBT",
        )

    if path.startswith("intergrax/runtime/wiring/"):
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="PLATFORM_RUNTIME",
            classification="PLATFORM_RUNTIME_INTERNAL",
            reason=(
                "Non-EE runtime Nexus coupling tracked for HARNESS-01-R5-W5 — "
                "Runtime Non-EE Boundary Convergence; not a final legal Nexus owner-zone."
            ),
            evidence=_EVIDENCE_EE,
            boundary_status="DEBT",
        )

    if path.startswith("intergrax/runtime/persistence/"):
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="PLATFORM_RUNTIME",
            classification="PLATFORM_RUNTIME_INTERNAL",
            reason=(
                "Runtime persistence/session composition coupling to Nexus-backed helpers; "
                "tracked for STATE-X / runtime convergence — not a public Nexus API."
            ),
            evidence=_EVIDENCE_EE,
            boundary_status="DEBT",
        )

    if path.startswith("intergrax/runtime/task/") or path.startswith(
        "intergrax/runtime/agent_governance/"
    ):
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="PLATFORM_RUNTIME",
            classification="PLATFORM_RUNTIME_INTERNAL",
            reason=(
                "Non-EE runtime Nexus coupling tracked for HARNESS-01-R5-W5 — "
                "Runtime Non-EE Boundary Convergence; not a final legal Nexus owner-zone."
            ),
            evidence=_EVIDENCE_EE,
            boundary_status="DEBT",
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

    if path.startswith("intergrax/runtime/token_optimization/proofs/"):
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="PLATFORM_RUNTIME",
            classification="PLATFORM_RUNTIME_INTERNAL",
            reason=(
                "LLM live proof tooling Nexus coupling tracked for HARNESS-01-R5-W4 — "
                "RAG/LLM/Integrations Nexus Dependency Inversion; not a final legal owner-zone."
            ),
            evidence=_EVIDENCE_CLASSIFIED,
            boundary_status="DEBT",
        )

    if path.startswith("intergrax/runtime/"):
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="PLATFORM_RUNTIME",
            classification="PLATFORM_RUNTIME_INTERNAL",
            reason=(
                "Non-EE runtime Nexus coupling tracked for HARNESS-01-R5-W5 — "
                "Runtime Non-EE Boundary Convergence; not a final legal Nexus owner-zone."
            ),
            evidence=_EVIDENCE_EE,
            boundary_status="DEBT",
        )

    if path.startswith("intergrax/tools/") or path.startswith("intergrax/websearch/"):
        return Harness01HigherLayerNexusImporter(
            path=path,
            owner_layer="TOOLING",
            classification="TOOLING_INTERNAL",
            reason=(
                "Tools/WebSearch Nexus coupling tracked for HARNESS-01-R5-W3 — "
                "Tools & WebSearch Nexus Dependency Inversion; not a final legal owner-zone."
            ),
            evidence=_EVIDENCE_CLASSIFIED,
            boundary_status="DEBT",
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
    "intergrax/runtime/execution/_orchestration_backend_access.py",
    "intergrax/runtime/execution/active_execution_budget.py",
    "intergrax/runtime/execution/agent_catalog_tool_dispatch.py",
    "intergrax/runtime/execution/agent_in_memory_session_factory.py",
    "intergrax/runtime/execution/agent_runtime_context.py",
    "intergrax/runtime/execution/agent_runtime_context_materializer.py",
    "intergrax/runtime/execution/agentic.py",
    "intergrax/runtime/execution/application_acp_session_composition.py",
    "intergrax/runtime/execution/application_declarative_tool_composition.py",
    "intergrax/runtime/execution/application_environment_context_composition.py",
    "intergrax/runtime/execution/application_graph_spec_to_plan.py",
    "intergrax/runtime/execution/atomic_planner_round_bridge.py",
    "intergrax/runtime/execution/authority/registry.py",
    "intergrax/runtime/execution/b6_r2_host_orchestration_structural_proofs.py",
    "intergrax/runtime/execution/budget/ledger.py",
    "intergrax/runtime/execution/budget/models.py",
    "intergrax/runtime/execution/budget/persistence.py",
    "intergrax/runtime/execution/budget/registry.py",
    "intergrax/runtime/execution/budget/snapshot.py",
    "intergrax/runtime/execution/catalog_runtime_config_bridge.py",
    "intergrax/runtime/execution/checkpoint_execution_graph_bridge.py",
    "intergrax/runtime/execution/child.py",
    "intergrax/runtime/execution/compensation_side_effect.py",
    "intergrax/runtime/execution/deadline_authority/resolver.py",
    "intergrax/runtime/execution/debug_lab_nexus_loop.py",
    "intergrax/runtime/execution/delegated_execution/context_projection.py",
    "intergrax/runtime/execution/delegated_execution/service.py",
    "intergrax/runtime/execution/delegated_subtask_child_port.py",
    "intergrax/runtime/execution/environment_orchestration_materialization.py",
    "intergrax/runtime/execution/evaluator_loop_composition.py",
    "intergrax/runtime/execution/execution_bound_catalog_tool_composition.py",
    "intergrax/runtime/execution/execution_mode_runtime_policies.py",
    "intergrax/runtime/execution/execution_work_port.py",
    "intergrax/runtime/execution/harness_task_execution_port.py",
    "intergrax/runtime/execution/host_observability_composition.py",
    "intergrax/runtime/execution/host_orchestration_application_wiring_applier.py",
    "intergrax/runtime/execution/host_orchestration_environment_spec_builder.py",
    "intergrax/runtime/execution/host_orchestration_loop_init_spec.py",
    "intergrax/runtime/execution/host_orchestration_planner_classifier_wiring.py",
    "intergrax/runtime/execution/host_runtime_config.py",
    "intergrax/runtime/execution/host_task.py",
    "intergrax/runtime/execution/host_validation_composition.py",
    "intergrax/runtime/execution/human_continuation_composition.py",
    "intergrax/runtime/execution/idempotency_runtime_state_bridge.py",
    "intergrax/runtime/execution/lab_reference_agent_runtime.py",
    "intergrax/runtime/execution/llm_routing_surface_composition.py",
    "intergrax/runtime/execution/nexus_host_execution.py",
    "intergrax/runtime/execution/nexus_host_execution_from_wiring_target.py",
    "intergrax/runtime/execution/nexus_host_task_terminal.py",
    "intergrax/runtime/execution/orchestration.py",
    "intergrax/runtime/execution/orchestration_topology_production_composition.py",
    "intergrax/runtime/execution/orchestration_topology_slot_mse_enforcement.py",
    "intergrax/runtime/execution/orchestration_topology_submission.py",
    "intergrax/runtime/execution/reasoning_tool_planning_composition.py",
    "intergrax/runtime/execution/run_artifact_bundle_composition.py",
    "intergrax/runtime/execution/run_trace_store_factories.py",
    "intergrax/runtime/execution/runtime.py",
    "intergrax/runtime/execution/runtime_state.py",
    "intergrax/runtime/execution/scenario_host_diagnostic_wiring.py",
    "intergrax/runtime/execution/session_host_composition.py",
    "intergrax/runtime/execution/suspended_operation/agent_governance_reentry_grant.py",
    "intergrax/runtime/execution/suspended_operation/authority_sequential_pause.py",
    "intergrax/runtime/execution/suspended_operation/authorized_resume_reentry.py",
    "intergrax/runtime/execution/suspended_operation/claim_lifecycle_wiring.py",
    "intergrax/runtime/execution/suspended_operation/composition.py",
    "intergrax/runtime/execution/suspended_operation/governed_request.py",
    "intergrax/runtime/execution/suspended_operation/hitl_resume_claim_preparation.py",
    "intergrax/runtime/execution/suspended_operation/pause_required.py",
    "intergrax/runtime/execution/suspended_operation/reentry_coordinator.py",
    "intergrax/runtime/execution/tool_engine_hook_composition.py",
    "intergrax/runtime/execution/tool_planning_service_bridge.py",
    "intergrax/runtime/execution/trace_persistence_models_bridge.py",
    "intergrax/runtime/execution/trace_store_debug_access.py",
    "intergrax/runtime/execution/worker_host_task_execution_composition.py",
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
