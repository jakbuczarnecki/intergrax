# © Artur Czarnecki. All rights reserved.

"""PLUG-03 qualification matrix — surfaces, evidence nodes, levels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Plug03EvidenceRef:
    pytest_node_id: str
    kinds: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Plug03SurfaceEvidence:
    surface: str
    classification: str
    contract_owner: str
    selection_mechanism: str
    canonical_consumer: str
    level: str
    status: str
    evidence: tuple[Plug03EvidenceRef, ...]
    notes: str = ""

    @property
    def pytest_node_ids(self) -> tuple[str, ...]:
        return tuple(ref.pytest_node_id for ref in self.evidence)


def _nid(path: str, test_name: str) -> str:
    return f"{path}::{test_name}"


def _gate(test_name: str) -> str:
    return _nid("tests/qualification/plug_03/test_plug_03_gates.py", test_name)


def _ref(node_id: str, *kinds: str) -> Plug03EvidenceRef:
    return Plug03EvidenceRef(node_id, kinds)


PLUG_03_SURFACE_MATRIX: tuple[Plug03SurfaceEvidence, ...] = (
    Plug03SurfaceEvidence(
        "Tools / ToolPlugin",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.tools",
        "ToolProfile.enabled_bundles + catalog bootstrap",
        "Tool registry → RuntimeToolInvoker / RegistryToolExecutor",
        "Q4",
        "PASS",
        (
            _ref(
                _nid(
                    "tests/integration/platform_plugins/test_plugin8_dual_mode_tool_e2e.py",
                    "test_external_reference_wheel_end_to_end",
                ),
                "CANONICAL_CONSUMPTION",
                "DEFAULT_BYPASS",
            ),
            _ref(
                _gate("test_tools_profile_selection_executes_custom_not_catalog_default"),
                "SELECTION",
                "CANONICAL_CONSUMPTION",
            ),
            _ref(
                _gate("test_tools_discovered_but_unselected_not_in_execution_registry"),
                "SELECTION",
                "FAIL_CLOSED",
            ),
        ),
    ),
    Plug03SurfaceEvidence(
        "ToolInvocationPattern",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.tools.invocation_pattern",
        "RuntimeConfig.tool_invocation_pattern_id → resolve_invocation_pattern",
        "Nexus tool loop (internal bridge from public pattern)",
        "Q4",
        "PASS",
        (
            _ref(
                _nid(
                    "tests/unit/runtime/nexus/tools/test_tool_invocation_registry.py",
                    "test_resolve_invocation_pattern_prefers_entry_point",
                ),
                "SELECTION",
            ),
            _ref(
                _nid(
                    "tests/unit/runtime/nexus/tools/test_plug_02_r1_public_invocation_pattern_evidence.py",
                    "test_public_pattern_single_invoke_via_bounded_tool_loop",
                ),
                "CANONICAL_CONSUMPTION",
                "DEFAULT_BYPASS",
            ),
            _ref(
                _gate("test_canonical_resolver_selects_custom_pattern_without_shipped_default"),
                "SELECTION",
                "DEFAULT_BYPASS",
            ),
            _ref(
                _gate("test_explicit_missing_tool_invocation_pattern_id_fails_closed"),
                "FAIL_CLOSED",
            ),
            _ref(
                _nid(
                    "tests/unit/runtime/nexus/tools/test_tool_invocation_registry.py",
                    "test_resolve_invocation_pattern_explicit_missing_id_fails_closed",
                ),
                "FAIL_CLOSED",
            ),
        ),
    ),
    Plug03SurfaceEvidence(
        "Skills / SkillPlugin",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.skills",
        "SkillProfile.enabled_bundles",
        "AgentRegistry → allowed_tools → RuntimeToolGateway",
        "Q4",
        "PASS",
        (
            _ref(
                _nid("tests/unit/skills/test_external_skill_plugin.py", "test_external_skill_plugin_merges_allowed_tools"),
                "SELECTION",
            ),
            _ref(
                _gate("test_plug03_custom_skill_enables_canonical_tool_execution"),
                "CANONICAL_CONSUMPTION",
                "DEFAULT_BYPASS",
            ),
            _ref(
                _gate("test_plug03_without_custom_skill_tool_not_allowed"),
                "DEFAULT_BYPASS",
            ),
        ),
    ),
    Plug03SurfaceEvidence(
        "Integrations",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.integrations",
        "IntegrationProfile category binding slug",
        "profile.resolve(category)",
        "Q4",
        "PASS",
        (
            _ref(
                _nid(
                    "tests/unit/integrations/test_external_integration_entry_point.py",
                    "test_fixture_integration_resolves_via_entry_point",
                ),
                "SELECTION",
                "CANONICAL_CONSUMPTION",
            ),
            _ref(
                _gate("test_integration_discovered_but_unselected_keeps_default_binding"),
                "SELECTION",
                "FAIL_CLOSED",
            ),
            _ref(
                _gate("test_integration_explicit_slug_activates_fixture_provider"),
                "SELECTION",
                "CANONICAL_CONSUMPTION",
                "DEFAULT_BYPASS",
            ),
        ),
    ),
    Plug03SurfaceEvidence(
        "Context — token counter",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.context",
        "ContextPluginRegistry.set_token_counter",
        "DefaultNexusContextEngine.assemble (internal consumer)",
        "Q4",
        "PASS",
        (
            _ref(
                _nid("tests/qualification/ce_02/test_ce_02_gates.py", "test_ce2_q4_custom_token_counter_injection"),
                "SELECTION",
                "CANONICAL_CONSUMPTION",
                "DEFAULT_BYPASS",
            ),
        ),
    ),
    Plug03SurfaceEvidence(
        "Context — budget / compaction / degradation",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.context",
        "ContextPluginRegistry strategy setters",
        "DefaultNexusContextEngine.assemble",
        "Q4",
        "PASS",
        (
            _ref(
                _nid("tests/qualification/ce_02/test_ce_02_gates.py", "test_ce2_q5_custom_model_budget_policy"),
                "SELECTION",
                "CANONICAL_CONSUMPTION",
            ),
            _ref(
                _nid("tests/qualification/ce_02/test_ce_02_gates.py", "test_ce2_q6_custom_compaction_strategy"),
                "SELECTION",
                "CANONICAL_CONSUMPTION",
            ),
            _ref(
                _nid("tests/qualification/ce_02/test_ce_02_gates.py", "test_ce2_q7_custom_degradation_policy"),
                "SELECTION",
                "CANONICAL_CONSUMPTION",
            ),
            _ref(
                _gate("test_context_custom_compaction_default_strategy_not_invoked"),
                "SELECTION",
                "DEFAULT_BYPASS",
            ),
        ),
    ),
    Plug03SurfaceEvidence(
        "Memory — UserProfileStore",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.memory.contracts",
        "MemoryStorePlugin catalog + materialize_user_profile_store",
        "Memory control plane recall/remember",
        "Q4",
        "PASS",
        (
            _ref(
                _nid(
                    "tests/integration/memory/e2e/test_mem_ent15_plugin_replaceability.py",
                    "test_reference_and_plugin_providers_equivalent_semantic_recall",
                ),
                "CANONICAL_CONSUMPTION",
                "DEFAULT_BYPASS",
            ),
            _ref(
                _nid("tests/unit/memory/test_memory_store_resolver.py", "test_materialize_external_user_profile_store"),
                "SELECTION",
            ),
            _ref(
                _nid(
                    "tests/unit/memory/test_memory_store_resolver.py",
                    "test_resolve_memory_platform_wiring_explicit_unknown_id_fails",
                ),
                "FAIL_CLOSED",
            ),
        ),
    ),
    Plug03SurfaceEvidence(
        "Memory — SessionStorage",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.memory.contracts",
        "memory_profile.session_storage_plugin_id + materialize_session_storage",
        "SessionManager (create / append / read history)",
        "Q4",
        "PASS",
        (
            _ref(
                _nid(
                    "tests/unit/memory/test_memory_store_resolver.py",
                    "test_resolve_memory_platform_wiring_explicit_session_storage_only",
                ),
                "SELECTION",
            ),
            _ref(
                _gate("test_plug03_session_storage_canonical_session_manager_consumer"),
                "CANONICAL_CONSUMPTION",
                "DEFAULT_BYPASS",
            ),
            _ref(
                _gate("test_memory_session_storage_fixture_does_not_import_nexus"),
                "ADMISSION",
            ),
            _ref(
                _nid(
                    "tests/unit/memory/test_memory_store_resolver.py",
                    "test_resolve_memory_platform_wiring_explicit_unknown_id_fails",
                ),
                "FAIL_CLOSED",
            ),
        ),
    ),
    Plug03SurfaceEvidence(
        "RAG — chunker",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.rag.document_splitters",
        "ChunkingStrategyRegistry + RagProfile",
        "IngestPipeline / ingest flow",
        "Q4",
        "PASS",
        (
            _ref(
                _nid(
                    "tests/unit/rag/test_rag_plugin_discovery.py",
                    "test_external_chunker_entry_point_flows_through_ingest_and_retrieval",
                ),
                "CANONICAL_CONSUMPTION",
                "DEFAULT_BYPASS",
                "SELECTION",
            ),
        ),
    ),
    Plug03SurfaceEvidence(
        "RAG — retriever",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.rag.retrievers",
        "RetrieverRegistry + profile",
        "RetrievalService",
        "Q4",
        "PASS",
        (
            _ref(
                _nid(
                    "tests/unit/rag/test_rag_plugin_discovery.py",
                    "test_external_retriever_entry_point_uses_retrieval_service",
                ),
                "CANONICAL_CONSUMPTION",
                "DEFAULT_BYPASS",
                "SELECTION",
            ),
        ),
    ),
    Plug03SurfaceEvidence(
        "RAG — reranker",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.rag.rerankers",
        "RerankerRegistry + profile",
        "RetrievalService",
        "Q4",
        "PASS",
        (
            _ref(
                _nid(
                    "tests/unit/rag/test_rag_plugin_discovery.py",
                    "test_external_reranker_entry_point_uses_retrieval_service",
                ),
                "CANONICAL_CONSUMPTION",
                "DEFAULT_BYPASS",
                "SELECTION",
            ),
        ),
    ),
    Plug03SurfaceEvidence(
        "RAG — embedding / document handler",
        "HOST_COMPOSED_EXTENSION",
        "intergrax.rag",
        "Bootstrap factories / host RagProfile",
        "Ingest pipeline components",
        "Q2",
        "GAP",
        (),
        notes="EP-qualified for chunker/retriever/reranker; embedding/doc handler remain host-selected factories.",
    ),
    Plug03SurfaceEvidence(
        "SecurityDefensePlugin",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.security / runtime.security",
        "Application environment profile + defense registry",
        "PluginSecurityDefenseMiddleware security hook",
        "Q4",
        "PASS",
        (
            _ref(
                _nid(
                    "tests/unit/applications/test_security_plugin_adoption_wiring.py",
                    "test_strict_wire_application_environment_allows_valid_security_plugin",
                ),
                "ADMISSION",
                "SELECTION",
            ),
            _ref(
                _gate("test_plug03_security_defense_canonical_hook_invokes_custom_plugin"),
                "CANONICAL_CONSUMPTION",
            ),
        ),
    ),
    Plug03SurfaceEvidence(
        "PolicyRuleHandler",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.policy",
        "Policy plugin loader + catalog resolve",
        "DeclarativePolicyEnforcer / tool policy pipeline",
        "Q4",
        "PASS",
        (
            _ref(
                _nid(
                    "tests/unit/runtime/policy/rules/test_policy_plugin_contribution.py",
                    "test_end_to_end_handler_admission_to_catalog_resolve",
                ),
                "ADMISSION",
                "SELECTION",
            ),
            _ref(
                _gate("test_plug03_policy_pipeline_custom_handler_changes_decision"),
                "CANONICAL_CONSUMPTION",
            ),
        ),
    ),
    Plug03SurfaceEvidence(
        "Vendor Knowledge external provider",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.runtime.vendor_knowledge (host EP)",
        "Provider registry + host composition",
        "Live read / workspace knowledge paths",
        "Q3",
        "PASS",
        (
            _ref(
                _nid(
                    "tests/unit/runtime/vendor_knowledge/test_acme_reference_plugin.py",
                    "test_entry_point_discovery_loads_reference_contribution",
                ),
                "DISCOVERY",
                "ADMISSION",
            ),
            _ref(
                _nid(
                    "tests/unit/runtime/vendor_knowledge/test_acme_reference_plugin.py",
                    "test_reference_factory_creates_integration_from_credential_ref",
                ),
                "SELECTION",
            ),
        ),
        notes=(
            "Canonical workspace live/indexed read E2E harness repaired (SyntaxError); "
            "full E2E proof tracked separately — composition/selection qualified at Q3."
        ),
    ),
    Plug03SurfaceEvidence(
        "RuntimePlugin",
        "HOST_COMPOSED_EXTENSION",
        "intergrax.runtime.plugins",
        "Host composition bootstrap",
        "Runtime hook coordinator",
        "Q3",
        "PASS",
        (
            _ref(
                _nid("tests/unit/runtime/plugins/test_plugin_bootstrap.py", "test_bootstrap_runtime_plugins_registers_shutdown"),
                "CANONICAL_CONSUMPTION",
            ),
        ),
        notes="Not setuptools EP; host-embedded only.",
    ),
    Plug03SurfaceEvidence(
        "Agents",
        "HOST_COMPOSED_EXTENSION",
        "intergrax.contracts / runtime.registry",
        "AgentRegistry.register (host)",
        "Task routing / agent selection",
        "Q3",
        "PASS",
        (
            _ref(
                _nid("tests/unit/skills/test_external_skill_plugin.py", "test_external_skill_plugin_merges_allowed_tools"),
                "SELECTION",
            ),
        ),
    ),
    Plug03SurfaceEvidence(
        "Token optimization",
        "NOT_EXTENSIBLE",
        "intergrax.runtime.token_optimization",
        "descriptor-only",
        "advisory integration",
        "Q0",
        "NOT RUNTIME-PLUGINABLE",
        (),
        notes="No public replacement EP by current contract.",
    ),
    Plug03SurfaceEvidence(
        "Nexus encapsulation gate",
        "INTERNAL_EXTENSION_POINT",
        "intergrax.runtime.nexus",
        "n/a",
        "n/a",
        "Q4",
        "PASS",
        (
            _ref(
                _gate("test_public_external_plugin_packages_do_not_import_nexus"),
                "FAIL_CLOSED",
            ),
            _ref(
                _nid(
                    "tests/unit/platform_plugins/test_plug_02_extension_boundary_gates.py",
                    "test_tools_public_contract_modules_do_not_import_nexus",
                ),
                "FAIL_CLOSED",
            ),
        ),
    ),
)

PLUG_03_MAPPED_NODE_IDS: tuple[str, ...] = tuple(
    node_id for row in PLUG_03_SURFACE_MATRIX for node_id in row.pytest_node_ids
)

PLUG_03_Q4_PUBLIC_REPLACEMENT_REQUIRED_KINDS: frozenset[str] = frozenset(
    {
        "SELECTION",
        "CANONICAL_CONSUMPTION",
        "DEFAULT_BYPASS",
    }
)

PLUG_03_EXPLICIT_SELECTION_SURFACES: frozenset[str] = frozenset(
    {
        "ToolInvocationPattern",
        "Integrations",
        "Memory — SessionStorage",
        "Memory — UserProfileStore",
    }
)

PLUG_03_Q4_CHAIN_PARTICIPATION_SURFACES: frozenset[str] = frozenset(
    {
        "SecurityDefensePlugin",
        "PolicyRuleHandler",
    }
)
