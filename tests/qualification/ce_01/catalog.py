# © Artur Czarnecki. All rights reserved.

"""CE-01 CE-Q1..CE-Q15 evidence catalog."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Ce01QEvidence:
    q_id: str
    title: str
    pytest_node_ids: tuple[str, ...]


def _nid(test_file: str, test_name: str) -> str:
    return f"tests/qualification/ce_01/{test_file}::{test_name}"


def _ext(path: str, test_name: str) -> str:
    return f"{path}::{test_name}"


CE_01_Q_CATALOG: tuple[Ce01QEvidence, ...] = (
    Ce01QEvidence(
        "CE-Q1",
        "Single canonical model-facing path: execution surfaces delegate to ContextEngine.assemble",
        (
            _nid("test_ce_01_gates.py", "test_ce_q1_canonical_context_engine_entry_surfaces"),
            _ext(
                "tests/integration/runtime/test_context_engine_paths.py",
                "test_graph_path_emits_context_assembled_with_engine_id",
            ),
            _ext(
                "tests/unit/runtime/nexus/context/test_mem_xint4_single_context_composition.py",
                "test_fail_closed_without_context_engine_when_sources_active",
            ),
        ),
    ),
    Ce01QEvidence(
        "CE-Q2",
        "Typed ContextAssemblyRequest / AssembledContext ABI (not dict message soup)",
        (_nid("test_ce_01_gates.py", "test_ce_q2_typed_context_contracts_are_assembly_abi"),),
    ),
    Ce01QEvidence(
        "CE-Q3",
        "Tenant fail-closed before assembly accepts foreign scope",
        (
            _nid("test_ce_01_gates.py", "test_ce_q3_foreign_tenant_fragment_rejected"),
            _ext(
                "tests/unit/context/test_mem_xint5r_authority_and_policy_replaceability.py",
                "test_scope_isolation_rejects_foreign_tenant",
            ),
        ),
    ),
    Ce01QEvidence(
        "CE-Q4",
        "Memory reaches CE through typed provider inputs / read boundary",
        (
            _ext(
                "tests/unit/runtime/nexus/context/test_mem_xint4_single_context_composition.py",
                "test_ltm_fragment_appears_once_in_ce_assembly",
            ),
            _ext("tests/unit/context/test_mem_xint6r_typed_source_boundary.py", "test_builtin_has_zero_semantic_handle_reads"),
        ),
    ),
    Ce01QEvidence(
        "CE-Q5",
        "Retrieval/RAG staged via typed chunk inputs, not vector SDK inside CE core",
        (
            _ext(
                "tests/unit/runtime/nexus/context/test_mem_xint4_single_context_composition.py",
                "test_rag_step_stages_chunks_without_message_injection",
            ),
            _ext("tests/unit/context/test_legacy_bridge_providers.py", "test_fragments_from_rag_chunks_emits_citations"),
        ),
    ),
    Ce01QEvidence(
        "CE-Q6",
        "Tool evidence enters through typed iterative tool blocks",
        (
            _ext(
                "tests/unit/runtime/nexus/context/test_mem_xint4_single_context_composition.py",
                "test_run_tools_context_stages_tool_output_for_ce_not_system_injection",
            ),
            _ext("tests/unit/context/test_legacy_bridge_providers.py", "test_fragments_from_websearch_and_tool_output"),
        ),
    ),
    Ce01QEvidence(
        "CE-Q7",
        "Governance/policy authority cannot be self-assigned by external providers",
        (
            _ext(
                "tests/unit/context/test_mem_xint5r_authority_and_policy_replaceability.py",
                "test_external_provider_cannot_self_assign_system_context",
            ),
            _ext(
                "tests/unit/context/test_mem_xint5r2_hard_policy_invariant_envelope.py",
                "test_attack_authority_system_context_to_unassigned",
            ),
        ),
    ),
    Ce01QEvidence(
        "CE-Q8",
        "Explicit precedence: higher authority retained under conflict policy",
        (
            _nid("test_ce_01_gates.py", "test_ce_q8_authority_precedence_under_ranking"),
            _ext(
                "tests/unit/context/test_mem_xint5r_authority_and_policy_replaceability.py",
                "test_exact_dedup_retention_uses_explicit_authority_only",
            ),
        ),
    ),
    Ce01QEvidence(
        "CE-Q9",
        "Custom context source pluggable via ContextPluginRegistry",
        (
            _nid("test_ce_01_gates.py", "test_ce_q9_custom_provider_without_engine_core_change"),
            _ext("tests/unit/context/test_context_plugin_registry.py", "test_registry_add_list_unregister_provider"),
        ),
    ),
    Ce01QEvidence(
        "CE-Q10",
        "Custom selection/assembly policy strategies injectable",
        (
            _ext(
                "tests/unit/context/test_mem_xint5r_authority_and_policy_replaceability.py",
                "test_custom_ranker_strategy_still_used_via_execute_kwarg",
            ),
            _ext(
                "tests/unit/context/test_mem_xint5r2_hard_policy_invariant_envelope.py",
                "test_positive_custom_ranker",
            ),
        ),
    ),
    Ce01QEvidence(
        "CE-Q11",
        "Provider-neutral Tier-0 context contracts (no vendor SDK imports)",
        (_nid("test_ce_01_gates.py", "test_ce_q11_context_tier0_vendor_import_gate"),),
    ),
    Ce01QEvidence(
        "CE-Q12",
        "Model adapter serializes messages only; no memory/RAG fetch in adapter core",
        (_nid("test_ce_01_gates.py", "test_ce_q12_llm_adapter_core_no_memory_or_retrieval_fetch"),),
    ),
    Ce01QEvidence(
        "CE-Q13",
        "Provenance preserved on AssembledContext after assembly",
        (_nid("test_ce_01_gates.py", "test_ce_q13_assembled_context_carries_provenance_fields"),),
    ),
    Ce01QEvidence(
        "CE-Q14",
        "CE core avoids dynamic service-locator ABI and forbidden integration tokens",
        (
            _nid("test_ce_01_gates.py", "test_ce_q14_ce_core_forbidden_integration_gate"),
            _ext("tests/unit/context/test_context_tier0_import_boundary.py", "test_context_tier0_import_boundary_script"),
        ),
    ),
    Ce01QEvidence(
        "CE-Q15",
        "No alternate production prompt/context pipelines bypassing ContextEngine",
        (
            _nid("test_ce_01_gates.py", "test_ce_q15_nexus_canonical_paths_forbid_direct_prompt_injection"),
            _ext(
                "tests/unit/runtime/nexus/context/test_mem_xint4_single_context_composition.py",
                "test_ast_guards_no_canonical_direct_injection_helpers",
            ),
        ),
    ),
)
