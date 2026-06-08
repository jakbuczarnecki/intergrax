# © Artur Czarnecki. All rights reserved.

"""Tier-3 skill catalog wiring (Phase R-Skill.4, SK-PRESET.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.core.catalog_bootstrap import bootstrap_catalogs
from intergrax.core.plugin_env import discover_plugins_enabled
from intergrax.skills.registry import SkillProfile, SkillRegistry, build_registry_from_profile


@dataclass(frozen=True)
class ApplicationSkillWiring:
    profile: SkillProfile
    registry: SkillRegistry


def build_application_skill_wiring(profile: SkillProfile) -> ApplicationSkillWiring:
    skill_bundle_ids = tuple(profile.enabled_bundles) if profile.enabled_bundles else None
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
