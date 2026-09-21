# © Artur Czarnecki. All rights reserved.

"""Runtime composition for Tier-3 platform plugin bootstrap evidence."""

from __future__ import annotations

from intergrax.applications.contracts.platform_plugin_evidence import (
    PLATFORM_PLUGIN_DOMAIN_CONTEXT,
    PLATFORM_PLUGIN_DOMAIN_INTEGRATIONS,
    PLATFORM_PLUGIN_DOMAIN_MEMORY,
    PLATFORM_PLUGIN_DOMAIN_POLICY,
    PLATFORM_PLUGIN_DOMAIN_RAG_CHUNKERS,
    PLATFORM_PLUGIN_DOMAIN_RAG_RERANKERS,
    PLATFORM_PLUGIN_DOMAIN_RAG_RETRIEVERS,
    PLATFORM_PLUGIN_DOMAIN_SECURITY,
    PLATFORM_PLUGIN_DOMAIN_SKILLS,
    PLATFORM_PLUGIN_DOMAIN_TOOLS,
    ApplicationPlatformPluginEvidence,
)
from intergrax.core.plugins.admission import DomainPluginLoadReport
from intergrax.core.plugins.discovery import EP_MEMORY_STORES
from intergrax.rag.bootstrap.entry_point_load import RagPluginLoadEvidence
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle


def build_application_platform_plugin_evidence(
    *,
    memory_report: DomainPluginLoadReport,
    context_report: DomainPluginLoadReport,
    security_report: DomainPluginLoadReport,
    tools_report: DomainPluginLoadReport,
    skills_report: DomainPluginLoadReport,
    policy_bundle: RuntimePolicyBundle,
    integrations_report: DomainPluginLoadReport,
    rag_plugin_load_evidence: RagPluginLoadEvidence | None = None,
) -> ApplicationPlatformPluginEvidence:
    """Compose application evidence from the same domain wiring invocations."""
    reports: dict[str, DomainPluginLoadReport] = {
        PLATFORM_PLUGIN_DOMAIN_MEMORY: memory_report,
        PLATFORM_PLUGIN_DOMAIN_CONTEXT: context_report,
        PLATFORM_PLUGIN_DOMAIN_SECURITY: security_report,
        PLATFORM_PLUGIN_DOMAIN_TOOLS: tools_report,
        PLATFORM_PLUGIN_DOMAIN_SKILLS: skills_report,
        PLATFORM_PLUGIN_DOMAIN_INTEGRATIONS: integrations_report,
    }
    if rag_plugin_load_evidence is not None:
        reports[PLATFORM_PLUGIN_DOMAIN_RAG_CHUNKERS] = (
            rag_plugin_load_evidence.chunker_report
        )
        reports[PLATFORM_PLUGIN_DOMAIN_RAG_RETRIEVERS] = (
            rag_plugin_load_evidence.retriever_report
        )
        reports[PLATFORM_PLUGIN_DOMAIN_RAG_RERANKERS] = (
            rag_plugin_load_evidence.reranker_report
        )
    declarative_runtime = policy_bundle.declarative_policy_runtime
    if declarative_runtime is not None:
        reports[PLATFORM_PLUGIN_DOMAIN_POLICY] = declarative_runtime.load_report
    return ApplicationPlatformPluginEvidence.from_domain_reports(reports)


def empty_memory_platform_plugin_evidence() -> ApplicationPlatformPluginEvidence:
    """Deterministic baseline when only Memory domain participates."""
    return ApplicationPlatformPluginEvidence.from_domain_reports(
        {
            PLATFORM_PLUGIN_DOMAIN_MEMORY: DomainPluginLoadReport.empty(EP_MEMORY_STORES),
        },
    )


__all__ = [
    "build_application_platform_plugin_evidence",
    "empty_memory_platform_plugin_evidence",
]
