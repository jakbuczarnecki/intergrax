# © Artur Czarnecki. All rights reserved.

"""Tier-3 skill catalog wiring (Phase R-Skill.4, SK-PRESET.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.core.catalog_bootstrap import CatalogBootstrapResult, bootstrap_catalogs
from intergrax.core.plugin_env import discover_plugins_enabled
from intergrax.core.plugins.admission import DomainPluginLoadReport
from intergrax.skills.registry import SkillProfile, SkillRegistry, build_registry_from_profile


@dataclass(frozen=True)
class ApplicationSkillWiring:
    profile: SkillProfile
    registry: SkillRegistry


class SkillCatalogBootstrapError(ValueError):
    """Raised when STRICT skill plugin bootstrap evidence is not acceptable."""


def skill_plugin_bootstrap_errors(report: DomainPluginLoadReport) -> tuple[str, ...]:
    errors: list[str] = []
    for item in report.failed:
        errors.append(f"skill plugin load failed: {item.spec.name}: {item.error}")
    for item in report.rejected:
        if item.fail_closed:
            errors.append(
                "skill plugin admission rejected: "
                f"{item.spec.name}: {item.reason_code.value}",
            )
    if not errors:
        errors.append("skill plugin bootstrap admission is not acceptable")
    return tuple(errors)


def assert_strict_skill_bootstrap_acceptable(
    env: ApplicationEnvironmentProfile,
    report: DomainPluginLoadReport,
) -> None:
    if env.execution_mode is not ExecutionMode.STRICT:
        return
    if report.critical_bootstrap_acceptable:
        return
    raise SkillCatalogBootstrapError("; ".join(skill_plugin_bootstrap_errors(report)))


def build_application_skill_wiring(
    profile: SkillProfile,
    *,
    catalog_bootstrap: CatalogBootstrapResult | None = None,
) -> ApplicationSkillWiring:
    skill_bundle_ids = tuple(profile.enabled_bundles) if profile.enabled_bundles else None
    if catalog_bootstrap is None:
        bootstrap_catalogs(
            register_shipped=True,
            skill_bundle_ids=skill_bundle_ids,
            discover_entry_points=discover_plugins_enabled(),
        )
    registry = build_registry_from_profile(profile)
    return ApplicationSkillWiring(profile=profile, registry=registry)


def harness_platform_skill_profile() -> SkillProfile:
    """Platform-only harness skills (no domain bundles)."""
    return SkillProfile(enabled_bundles=["harness"])


def lab_skill_profile() -> SkillProfile:
    """Lab harness preset: harness + domain + universal packs."""
    return SkillProfile(
        enabled_bundles=[
            "harness",
            "legal",
            "research",
            "rag",
            "workspace",
            "memory",
            "knowledge",
        ]
    )


def research_skill_profile() -> SkillProfile:
    return SkillProfile(enabled_bundles=["research", "rag", "browser"])


def legal_skill_profile() -> SkillProfile:
    return SkillProfile(enabled_bundles=["legal", "rag", "knowledge", "workspace"])


def knowledge_skill_profile() -> SkillProfile:
    return SkillProfile(enabled_bundles=["knowledge"])


def rag_skill_profile() -> SkillProfile:
    return SkillProfile(enabled_bundles=["rag"])


def ops_skill_profile() -> SkillProfile:
    return SkillProfile(enabled_bundles=["ops", "dev", "workspace"])


def platform_skill_profile() -> SkillProfile:
    """Intergrax assistant hub: concierge + universal retrieval packs."""
    return SkillProfile(enabled_bundles=["platform", "rag", "memory", "research"])


def lkw_skill_profile() -> SkillProfile:
    """Local Knowledge Workspace: ingest, Q&A, workspace authoring."""
    return SkillProfile(enabled_bundles=["rag", "workspace", "memory", "knowledge"])


def local_workspace_product_skill_profile() -> SkillProfile:
    """LKW production host: index, search, and synthesize product skills only."""
    return SkillProfile(enabled_bundles=["local"])


def dispute_skill_profile() -> SkillProfile:
    """Dispute simulation: legal research, memory scratchpad, citation synthesis."""
    return SkillProfile(enabled_bundles=["legal", "rag", "memory", "research"])


def sandbox_skill_profile() -> SkillProfile:
    """Coding agents: sandbox exec with shadow workspace IO."""
    return SkillProfile(enabled_bundles=["sandbox", "workspace"])


def hitl_skill_profile() -> SkillProfile:
    """Governed human-in-the-loop approval workflows."""
    return SkillProfile(enabled_bundles=["hitl", "notify"])


def graph_skill_profile() -> SkillProfile:
    """Knowledge graph traversal with RAG grounding."""
    return SkillProfile(enabled_bundles=["graph", "rag"])


def storage_skill_profile() -> SkillProfile:
    """Object storage sync with workspace artifact import/export."""
    return SkillProfile(enabled_bundles=["storage", "workspace"])


def message_bus_skill_profile() -> SkillProfile:
    """Async background tasks via message bus queue."""
    return SkillProfile(enabled_bundles=["message_bus"])


def cache_skill_profile() -> SkillProfile:
    """Session KV cache with task memory fallback."""
    return SkillProfile(enabled_bundles=["cache", "memory"])


def eval_skill_profile() -> SkillProfile:
    """Eval harness: score logging and trace correlation."""
    return SkillProfile(enabled_bundles=["eval", "ops"])


def modality_skill_profile() -> SkillProfile:
    """Voice and vision modality pipelines for lab hosts."""
    return SkillProfile(enabled_bundles=["modality", "rag"])


def notify_skill_profile() -> SkillProfile:
    """Deferred and immediate notification scheduling."""
    return SkillProfile(enabled_bundles=["notify"])


def cost_skill_profile() -> SkillProfile:
    """Run budget and quota governance."""
    return SkillProfile(enabled_bundles=["cost"])


def identity_skill_profile() -> SkillProfile:
    """Identity verification and tenancy resolution."""
    return SkillProfile(enabled_bundles=["identity"])


def health_skill_profile() -> SkillProfile:
    """Integration health probes for operator hosts."""
    return SkillProfile(enabled_bundles=["health"])


def context_skill_profile() -> SkillProfile:
    """Context token planning and summarization."""
    return SkillProfile(enabled_bundles=["context", "memory"])


def agent_roster_skill_profile() -> SkillProfile:
    """Agent roster introspection and skill resolution."""
    return SkillProfile(enabled_bundles=["agent", "platform"])


def vector_store_skill_profile() -> SkillProfile:
    """Vector store administration and health checks."""
    return SkillProfile(enabled_bundles=["vector_store", "rag"])


def crm_skill_profile() -> SkillProfile:
    """CRM account and support ticket lookup."""
    return SkillProfile(enabled_bundles=["crm"])


def billing_skill_profile() -> SkillProfile:
    """Usage metering and run cost correlation."""
    return SkillProfile(enabled_bundles=["billing", "cost"])


def metrics_skill_profile() -> SkillProfile:
    """Runtime metrics with trace correlation."""
    return SkillProfile(enabled_bundles=["metrics", "ops"])


def catalog_skill_profile() -> SkillProfile:
    """Tool catalog introspection and skill resolution."""
    return SkillProfile(enabled_bundles=["catalog", "agent"])


def interaction_skill_profile() -> SkillProfile:
    """User interaction session handling and input capture."""
    return SkillProfile(enabled_bundles=["interaction", "memory"])


def ml_skill_profile() -> SkillProfile:
    """ML predict/explain pipelines for lab hosts."""
    return SkillProfile(enabled_bundles=["ml", "modality"])


def openai_skill_profile() -> SkillProfile:
    """OpenAI vector store admin and file_search."""
    return SkillProfile(enabled_bundles=["openai", "knowledge"])


def http_skill_profile() -> SkillProfile:
    """HTTP API client with observability correlation."""
    return SkillProfile(enabled_bundles=["http", "ops"])


def jira_skill_profile() -> SkillProfile:
    """Jira-native task navigation complementing generic dev bundle."""
    return SkillProfile(enabled_bundles=["jira", "dev"])


def gitlab_skill_profile() -> SkillProfile:
    """GitLab issue creation complementing generic dev bundle."""
    return SkillProfile(enabled_bundles=["gitlab", "dev"])


def code_skill_profile() -> SkillProfile:
    """Controlled code/script execution with sandbox listing."""
    return SkillProfile(enabled_bundles=["code", "sandbox"])


def filesystem_skill_profile() -> SkillProfile:
    """Local filesystem IO for trusted operator hosts only."""
    return SkillProfile(enabled_bundles=["filesystem"])


def cloud_platform_skill_profile() -> SkillProfile:
    """Cloud platform resolution and health checks."""
    return SkillProfile(enabled_bundles=["cloud_platform", "health"])


def oncall_skill_profile() -> SkillProfile:
    """On-call SRE: runbooks, log tail, incident ack, metrics."""
    return SkillProfile(enabled_bundles=["ops", "metrics", "notify", "hitl"])


def legal_ops_skill_profile() -> SkillProfile:
    """Legal ops: redline, regulatory scan, obligation tracking."""
    return SkillProfile(enabled_bundles=["legal", "rag", "workspace", "memory"])


def research_lab_skill_profile() -> SkillProfile:
    """Research lab: deep dive, validation, report compile, web cache."""
    return SkillProfile(enabled_bundles=["research", "rag", "workspace", "browser"])


def data_platform_skill_profile() -> SkillProfile:
    """Data platform: SQL, records, schema docs, pipeline probes."""
    return SkillProfile(enabled_bundles=["data", "health", "workspace"])


def sandbox_dev_skill_profile() -> SkillProfile:
    """Sandbox development: test runner, refactor loop, code exec."""
    return SkillProfile(enabled_bundles=["sandbox", "code", "workspace"])
