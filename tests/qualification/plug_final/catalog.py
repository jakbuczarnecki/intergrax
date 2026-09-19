# © Artur Czarnecki. All rights reserved.

"""PLUG-FINAL enterprise surface matrix — classification, Q-level, evidence pointers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlugFinalEvidenceRef:
    pytest_node_id: str
    kinds: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlugFinalSurfaceRow:
    surface: str
    classification: str
    contract_owner: str
    discovery: str
    admission: str
    selection: str
    canonical_consumer: str
    evidence_domain: str
    final_q: str
    status: str
    evidence: tuple[PlugFinalEvidenceRef, ...] = ()
    notes: str = ""

    @property
    def pytest_node_ids(self) -> tuple[str, ...]:
        return tuple(ref.pytest_node_id for ref in self.evidence)


def _nid(path: str, test_name: str) -> str:
    return f"{path}::{test_name}"


def _gate03(test_name: str) -> str:
    return _nid("tests/qualification/plug_03/test_plug_03_gates.py", test_name)


def _gate04(test_name: str) -> str:
    return _nid("tests/qualification/plug_04/test_plug_04_gates.py", test_name)


def _ref(node_id: str, *kinds: str) -> PlugFinalEvidenceRef:
    return PlugFinalEvidenceRef(node_id, kinds)


PLUG_FINAL_SURFACE_MATRIX: tuple[PlugFinalSurfaceRow, ...] = (
    PlugFinalSurfaceRow(
        "Tools / ToolPlugin",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.tools",
        "register_plugins_with_report / EP intergrax.tools",
        "ToolPlugin + platform qualification",
        "ToolProfile.enabled_bundles",
        "Tool registry → RuntimeToolInvoker",
        "platform_plugin_evidence.tools",
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
            _ref(_gate03("test_tools_profile_selection_executes_custom_not_catalog_default"), "SELECTION"),
            _ref(_gate03("test_tools_discovered_but_unselected_not_in_execution_registry"), "FAIL_CLOSED"),
        ),
    ),
    PlugFinalSurfaceRow(
        "ToolInvocationPattern",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.tools.invocation_pattern",
        "EP intergrax.tool_invocation_patterns",
        "ToolInvocationPattern protocol",
        "RuntimeConfig.tool_invocation_pattern_id",
        "Nexus bounded bridge → RuntimeToolInvoker",
        "runtime selection (not app evidence domain)",
        "Q4",
        "PASS",
        (
            _ref(
                _nid(
                    "tests/unit/runtime/nexus/tools/test_plug_02_r1_public_invocation_pattern_evidence.py",
                    "test_public_pattern_single_invoke_via_bounded_tool_loop",
                ),
                "CANONICAL_CONSUMPTION",
                "DEFAULT_BYPASS",
            ),
            _ref(_gate03("test_explicit_missing_tool_invocation_pattern_id_fails_closed"), "FAIL_CLOSED"),
        ),
    ),
    PlugFinalSurfaceRow(
        "Skills / SkillPlugin",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.skills",
        "register_plugins_with_report",
        "SkillPlugin validation",
        "SkillProfile.enabled_bundles",
        "AgentRegistry → RuntimeToolGateway",
        "platform_plugin_evidence.skills",
        "Q4",
        "PASS",
        (
            _ref(_gate03("test_plug03_custom_skill_enables_canonical_tool_execution"), "CANONICAL_CONSUMPTION"),
            _ref(_gate03("test_plug03_without_custom_skill_tool_not_allowed"), "DEFAULT_BYPASS"),
        ),
    ),
    PlugFinalSurfaceRow(
        "Integrations",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.integrations",
        "register_plugins_with_report",
        "IntegrationPlugin validation",
        "IntegrationProfile category slug",
        "profile.resolve(category)",
        "platform_plugin_evidence.integrations",
        "Q4",
        "PASS",
        (
            _ref(_gate03("test_integration_explicit_slug_activates_fixture_provider"), "CANONICAL_CONSUMPTION"),
            _ref(_gate03("test_integration_discovered_but_unselected_keeps_default_binding"), "FAIL_CLOSED"),
        ),
    ),
    PlugFinalSurfaceRow(
        "Context — token / budget / compaction / degradation",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.context",
        "register_plugins_with_report",
        "Context plugin validation",
        "ContextPluginRegistry setters",
        "DefaultNexusContextEngine.assemble (internal consumer)",
        "platform_plugin_evidence.context",
        "Q4",
        "PASS",
        (
            _ref(
                _nid("tests/qualification/ce_02/test_ce_02_gates.py", "test_ce2_q4_custom_token_counter_injection"),
                "CANONICAL_CONSUMPTION",
            ),
            _ref(_gate03("test_context_custom_compaction_default_strategy_not_invoked"), "DEFAULT_BYPASS"),
        ),
    ),
    PlugFinalSurfaceRow(
        "Memory — UserProfileStore",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.memory.contracts",
        "EP intergrax.memory_stores + classifier",
        "Memory store classifier",
        "explicit store id / profile",
        "Memory control plane recall/remember",
        "platform_plugin_evidence.memory",
        "Q4",
        "PASS",
        (
            _ref(
                _nid(
                    "tests/integration/memory/e2e/test_mem_ent15_plugin_replaceability.py",
                    "test_reference_and_plugin_providers_equivalent_semantic_recall",
                ),
                "CANONICAL_CONSUMPTION",
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
    PlugFinalSurfaceRow(
        "Memory — SessionStorage",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.memory.contracts",
        "EP intergrax.memory_stores",
        "Session storage plugin contract",
        "memory_profile.session_storage_plugin_id",
        "SessionManager",
        "platform_plugin_evidence.memory",
        "Q4",
        "PASS",
        (
            _ref(_gate03("test_plug03_session_storage_canonical_session_manager_consumer"), "CANONICAL_CONSUMPTION"),
            _ref(_gate03("test_memory_session_storage_fixture_does_not_import_nexus"), "ADMISSION"),
        ),
    ),
    PlugFinalSurfaceRow(
        "RAG — chunker",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.rag.document_splitters",
        "register_plugins_with_report",
        "BaseChunkingStrategy",
        "RagProfile / registry",
        "IngestPipeline",
        "rag_chunkers (tenant bootstrap)",
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
            ),
            _ref(
                _nid(
                    "tests/unit/applications/test_rag_authoritative_plugin_evidence.py",
                    "test_wire_application_environment_lazy_host_does_not_materialize_rag_plugins",
                ),
                "EVIDENCE",
            ),
        ),
    ),
    PlugFinalSurfaceRow(
        "RAG — retriever",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.rag.retrievers",
        "register_plugins_with_report",
        "BaseRetriever",
        "RagProfile / registry",
        "RetrievalService",
        "rag_retrievers (tenant bootstrap)",
        "Q4",
        "PASS",
        (
            _ref(
                _nid(
                    "tests/unit/rag/test_rag_plugin_discovery.py",
                    "test_external_retriever_entry_point_uses_retrieval_service",
                ),
                "CANONICAL_CONSUMPTION",
            ),
        ),
    ),
    PlugFinalSurfaceRow(
        "RAG — reranker",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.rag.rerankers",
        "register_plugins_with_report",
        "BaseReranker",
        "RagProfile / registry",
        "RetrievalService",
        "rag_rerankers (tenant bootstrap)",
        "Q4",
        "PASS",
        (
            _ref(
                _nid(
                    "tests/unit/rag/test_rag_plugin_discovery.py",
                    "test_external_reranker_entry_point_uses_retrieval_service",
                ),
                "CANONICAL_CONSUMPTION",
            ),
        ),
    ),
    PlugFinalSurfaceRow(
        "RAG — embedding",
        "HOST_COMPOSED_EXTENSION",
        "intergrax.rag",
        "host factory",
        "host profile",
        "RagProfile embedding selection",
        "Ingest / vector store wiring",
        "profile only",
        "Q2",
        "INTENTIONAL",
        notes="Host-composed; no EP Q4 replacement contract.",
    ),
    PlugFinalSurfaceRow(
        "RAG — document handler",
        "HOST_COMPOSED_EXTENSION",
        "intergrax.rag",
        "host factory",
        "host profile",
        "RagProfile handler selection",
        "Ingest pipeline",
        "profile only",
        "Q2",
        "INTENTIONAL",
    ),
    PlugFinalSurfaceRow(
        "SecurityDefensePlugin",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.security",
        "defense_plugin_loader",
        "domain priority / hook semantics",
        "Application environment profile",
        "PluginSecurityDefenseMiddleware",
        "platform_plugin_evidence.security",
        "Q4",
        "PASS",
        (
            _ref(_gate03("test_plug03_security_defense_canonical_hook_invokes_custom_plugin"), "CANONICAL_CONSUMPTION"),
        ),
        notes="Chain semantics; no global default replacement required.",
    ),
    PlugFinalSurfaceRow(
        "PolicyRuleHandler",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.policy",
        "policy plugin_loader",
        "handler provenance binding",
        "catalog resolve + profile",
        "DeclarativePolicyEnforcer",
        "platform_plugin_evidence.policy",
        "Q4",
        "PASS",
        (
            _ref(_gate03("test_plug03_policy_pipeline_custom_handler_changes_decision"), "CANONICAL_CONSUMPTION"),
        ),
    ),
    PlugFinalSurfaceRow(
        "Vendor Knowledge provider",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.runtime.vendor_knowledge",
        "contribution_catalog EP",
        "contribution factory validation",
        "host opt-in registry",
        "workspace knowledge read paths",
        "not in ApplicationPlatformPluginEvidence",
        "Q3",
        "NON-BLOCKING DEBT",
        (
            _ref(
                _nid(
                    "tests/unit/runtime/vendor_knowledge/test_acme_reference_plugin.py",
                    "test_entry_point_discovery_loads_reference_contribution",
                ),
                "DISCOVERY",
            ),
        ),
        notes="Canonical live read E2E not Q4-certified; honest Q3.",
    ),
    PlugFinalSurfaceRow(
        "Decision — strategies",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.contracts.decision_strategy",
        "EP intergrax.decision_strategies",
        "production qualification + manifest binding",
        "profile allowlist / hybrid bindings",
        "Decision runtime composition",
        "application decision composition reports",
        "Q3",
        "PASS",
        (
            _ref(
                _nid("tests/unit/runtime/test_decision_plugin_composition.py", "test_valid_strategy_plugin_registers"),
                "ADMISSION",
            ),
            _ref(
                _nid(
                    "tests/unit/applications/test_application_decision_composition.py",
                    "test_selected_plugin_stage_merges_into_pipeline",
                ),
                "SELECTION",
            ),
            _ref(
                _nid(
                    "tests/unit/applications/test_application_decision_composition.py",
                    "test_selected_but_missing_plugin_fails_closed",
                ),
                "FAIL_CLOSED",
            ),
        ),
        notes="Registry/admission/selection qualified; not full cross-run Q4 artifact proof.",
    ),
    PlugFinalSurfaceRow(
        "Decision — verification stages",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.contracts.decision_verification_stage",
        "EP intergrax.decision_verification_stages",
        "stage kind validation",
        "profile selection",
        "Verification pipeline composition",
        "application decision composition",
        "Q3",
        "PASS",
        (
            _ref(
                _nid(
                    "tests/unit/applications/test_application_decision_composition.py",
                    "test_installed_but_not_selected_keeps_plugin_stages_inactive",
                ),
                "FAIL_CLOSED",
            ),
        ),
    ),
    PlugFinalSurfaceRow(
        "Decision — artifact kinds",
        "PUBLIC_EXTERNAL_PLUGIN",
        "intergrax.contracts.decision_artifact_registry",
        "EP intergrax.decision_artifact_kinds",
        "artifact kind validation",
        "registry registration",
        "Decision artifact registry",
        "application decision composition",
        "Q3",
        "PASS",
        (
            _ref(
                _nid("tests/unit/runtime/test_decision_plugin_composition.py", "test_valid_artifact_kind_registers"),
                "ADMISSION",
            ),
        ),
    ),
    PlugFinalSurfaceRow(
        "Decision — exposure selection strategies",
        "INTERNAL_EXTENSION_POINT",
        "intergrax.runtime.execution",
        "EP intergrax.decision_exposure_selection_strategies",
        "runtime registry admission",
        "execution profile",
        "Decision exposure selection registry",
        "runtime-only",
        "Q2",
        "INTENTIONAL",
        notes="Runtime-internal EP; not Tier-3 application evidence domain.",
    ),
    PlugFinalSurfaceRow(
        "RuntimePlugin",
        "HOST_COMPOSED_EXTENSION",
        "intergrax.runtime.plugins",
        "host bootstrap only",
        "host validation",
        "host embed list",
        "Runtime hook coordinator",
        "host wiring",
        "Q3",
        "PASS",
        (
            _ref(
                _nid("tests/unit/runtime/plugins/test_plugin_bootstrap.py", "test_bootstrap_runtime_plugins_registers_shutdown"),
                "CANONICAL_CONSUMPTION",
            ),
        ),
    ),
    PlugFinalSurfaceRow(
        "Agents",
        "HOST_COMPOSED_EXTENSION",
        "intergrax.contracts / runtime.registry",
        "host AgentRegistry.register",
        "host contract validation",
        "host routing profile",
        "Task routing / agent selection",
        "n/a",
        "Q3",
        "PASS",
    ),
    PlugFinalSurfaceRow(
        "Token optimization",
        "NOT_EXTENSIBLE",
        "intergrax.runtime.token_optimization",
        "n/a",
        "n/a",
        "descriptor advisory only",
        "advisory integration",
        "n/a",
        "Q0",
        "INTENTIONAL",
    ),
    PlugFinalSurfaceRow(
        "Nexus encapsulation (meta-gate)",
        "INTERNAL_EXTENSION_POINT",
        "intergrax.runtime.nexus",
        "n/a",
        "n/a",
        "n/a",
        "internal execution engine",
        "n/a",
        "n/a",
        "PASS",
        (
            _ref(_gate03("test_public_external_plugin_packages_do_not_import_nexus"), "GATE"),
            _ref(_gate04("test_core_plugins_package_does_not_import_nexus"), "GATE"),
            _ref(
                _nid(
                    "tests/unit/platform_plugins/test_plug_02_extension_boundary_gates.py",
                    "test_tools_public_contract_modules_do_not_import_nexus",
                ),
                "GATE",
            ),
        ),
    ),
)

PLUG_FINAL_Q4_PUBLIC_SURFACES: frozenset[str] = frozenset(
    row.surface
    for row in PLUG_FINAL_SURFACE_MATRIX
    if row.final_q == "Q4" and row.classification == "PUBLIC_EXTERNAL_PLUGIN"
)

PLUG_FINAL_MAPPED_NODE_IDS: tuple[str, ...] = tuple(
    node_id for row in PLUG_FINAL_SURFACE_MATRIX for node_id in row.pytest_node_ids
)

PLUG_FINAL_STATIC_GATE_NODE_IDS: tuple[str, ...] = (
    _nid("tests/qualification/plug_final/test_plug_final_gates.py", "test_plug_final_matrix_covers_inventory_surfaces"),
    _nid("tests/qualification/plug_final/test_plug_final_gates.py", "test_plug_final_q4_public_rows_have_evidence"),
    _nid("tests/qualification/plug_final/test_plug_final_gates.py", "test_application_platform_plugin_evidence_is_immutable_metadata"),
    _nid("tests/qualification/plug_final/test_plug_final_gates.py", "test_plug_final_catalog_evidence_nodes_collect"),
)
