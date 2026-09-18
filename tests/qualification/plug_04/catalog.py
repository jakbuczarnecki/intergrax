# © Artur Czarnecki. All rights reserved.

"""PLUG-04 coordination matrix — surfaces, levels, gate evidence."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Plug04SurfaceRow:
    surface: str
    ep_group: str
    domain: str
    classification: str
    shared_discovery: str
    typed_report: str
    admission: str
    determinism: str
    app_evidence: str
    level: str
    status: str


def _nid(path: str, test_name: str) -> str:
    return f"{path}::{test_name}"


PLUG_04_SURFACE_MATRIX: tuple[Plug04SurfaceRow, ...] = (
    Plug04SurfaceRow(
        "Tools",
        "intergrax.tools",
        "tools",
        "PUBLIC_EXTERNAL",
        "register_plugins_with_report",
        "DomainPluginLoadReport",
        "ToolPlugin validation",
        "sorted accepted/rejected/failed",
        "platform_plugin_evidence.tools",
        "D4",
        "PASS",
    ),
    Plug04SurfaceRow(
        "Skills",
        "intergrax.skills",
        "skills",
        "PUBLIC_EXTERNAL",
        "register_plugins_with_report",
        "DomainPluginLoadReport",
        "SkillPlugin validation",
        "sorted accepted/rejected/failed",
        "platform_plugin_evidence.skills",
        "D4",
        "PASS",
    ),
    Plug04SurfaceRow(
        "Integrations",
        "intergrax.integrations",
        "integrations",
        "PUBLIC_EXTERNAL",
        "register_plugins_with_report",
        "DomainPluginLoadReport",
        "IntegrationPlugin validation",
        "sorted accepted/rejected/failed",
        "platform_plugin_evidence.integrations",
        "D4",
        "PASS",
    ),
    Plug04SurfaceRow(
        "Memory stores",
        "intergrax.memory_stores",
        "memory",
        "PUBLIC_EXTERNAL",
        "load_entry_point_targets + classifier",
        "DomainPluginLoadReport",
        "Memory store classifier",
        "sorted accepted/rejected/failed",
        "platform_plugin_evidence.memory",
        "D4",
        "PASS",
    ),
    Plug04SurfaceRow(
        "Context",
        "intergrax.context",
        "context",
        "PUBLIC_EXTERNAL",
        "register_plugins_with_report",
        "DomainPluginLoadReport",
        "Context plugin validation",
        "sorted accepted/rejected/failed",
        "platform_plugin_evidence.context",
        "D4",
        "PASS",
    ),
    Plug04SurfaceRow(
        "RAG chunkers",
        "intergrax.rag.chunkers",
        "rag",
        "PUBLIC_EXTERNAL",
        "register_plugins_with_report",
        "DomainPluginLoadReport",
        "BaseChunkingStrategy",
        "sorted accepted/rejected/failed",
        "platform_plugin_evidence.rag_chunkers (after tenant RAG bootstrap)",
        "D4",
        "PASS",
    ),
    Plug04SurfaceRow(
        "RAG retrievers",
        "intergrax.rag.retrievers",
        "rag",
        "PUBLIC_EXTERNAL",
        "register_plugins_with_report",
        "DomainPluginLoadReport",
        "BaseRetriever contract",
        "sorted accepted/rejected/failed",
        "platform_plugin_evidence.rag_retrievers (after tenant RAG bootstrap)",
        "D4",
        "PASS",
    ),
    Plug04SurfaceRow(
        "RAG rerankers",
        "intergrax.rag.rerankers",
        "rag",
        "PUBLIC_EXTERNAL",
        "register_plugins_with_report",
        "DomainPluginLoadReport",
        "BaseReranker contract",
        "sorted accepted/rejected/failed",
        "platform_plugin_evidence.rag_rerankers (after tenant RAG bootstrap)",
        "D4",
        "PASS",
    ),
    Plug04SurfaceRow(
        "Tool invocation patterns",
        "intergrax.tool_invocation_patterns",
        "tools",
        "PUBLIC_EXTERNAL",
        "iter_entry_point_specs + load",
        "resolve errors / EP load",
        "ToolInvocationPattern type",
        "sorted EP specs",
        "runtime selection evidence",
        "D3",
        "PASS",
    ),
    Plug04SurfaceRow(
        "Security defenses",
        "intergrax.security_defenses",
        "security",
        "PUBLIC_EXTERNAL",
        "defense_plugin_loader",
        "DomainPluginLoadReport",
        "domain priority/hook semantics",
        "sorted accepted/rejected/failed",
        "platform_plugin_evidence.security",
        "D4",
        "PASS",
    ),
    Plug04SurfaceRow(
        "Policy rules",
        "intergrax.policy_rules",
        "policy",
        "PUBLIC_EXTERNAL",
        "policy plugin_loader",
        "DomainPluginLoadReport",
        "handler/provenance binding",
        "sorted accepted/rejected/failed",
        "platform_plugin_evidence.policy",
        "D4",
        "PASS",
    ),
    Plug04SurfaceRow(
        "Vendor knowledge",
        "intergrax.vendor_knowledge.providers",
        "vendor_knowledge",
        "Q3_HOST_OPT_IN",
        "contribution_catalog",
        "VendorKnowledgePluginLoadError",
        "domain contribution factory",
        "sorted entry point names",
        "not in ApplicationPlatformPluginEvidence",
        "D2",
        "PASS",
    ),
    Plug04SurfaceRow(
        "RAG embedding",
        "n/a",
        "rag",
        "HOST_COMPOSED",
        "host factory",
        "n/a",
        "host profile",
        "n/a",
        "profile selection only",
        "D1",
        "PASS",
    ),
    Plug04SurfaceRow(
        "RAG document handler",
        "n/a",
        "rag",
        "HOST_COMPOSED",
        "host factory",
        "n/a",
        "host profile",
        "n/a",
        "profile selection only",
        "D1",
        "PASS",
    ),
)

PLUG_04_GATE_NODE_IDS: tuple[str, ...] = (
    _nid("tests/qualification/plug_04/test_plug_04_gates.py", "test_core_plugins_package_does_not_import_nexus"),
    _nid("tests/qualification/plug_04/test_plug_04_gates.py", "test_q4_rag_surfaces_at_least_d3_in_matrix"),
    _nid("tests/qualification/plug_04/test_plug_04_gates.py", "test_application_evidence_includes_integrations_and_rag_when_enabled"),
    _nid(
        "tests/unit/core/plugins/test_plug_02_integration_catalog_admission.py",
        "test_integration_report_captures_invalid_sibling_without_crashing",
    ),
    _nid(
        "tests/unit/core/plugins/test_plug_02_integration_catalog_admission.py",
        "test_integration_report_ordering_is_deterministic",
    ),
    _nid(
        "tests/unit/rag/bootstrap/test_rag_entry_point_load.py",
        "test_rag_chunker_report_rejects_invalid_sibling_without_crashing",
    ),
    _nid(
        "tests/unit/rag/bootstrap/test_rag_entry_point_load.py",
        "test_collect_rag_evidence_discovery_disabled_returns_empty_reports",
    ),
)
