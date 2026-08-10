"""Canonical contribution-driven composition of Vendor Knowledge plugins."""

from __future__ import annotations

from intergrax.runtime.vendor_knowledge.contribution_catalog import (
    build_default_vendor_knowledge_contribution_catalog,
    build_vendor_knowledge_source_plugin_registry,
)
from intergrax.runtime.vendor_knowledge.google_workspace_contribution import (
    build_google_workspace_vendor_knowledge_contribution,
)
from intergrax.runtime.vendor_knowledge.jira_contribution import (
    build_jira_vendor_knowledge_contribution,
)
from intergrax.runtime.vendor_knowledge.confluence_contribution import (
    build_confluence_vendor_knowledge_contribution,
)
from intergrax.runtime.vendor_knowledge.plugin import (
    VendorKnowledgeSourcePlugin,
    VendorKnowledgeSourcePluginRegistry,
)

def build_google_workspace_vendor_knowledge_source_plugins() -> (
    tuple[VendorKnowledgeSourcePlugin, ...]
):
    return build_google_workspace_vendor_knowledge_contribution().source_plugins


def build_jira_vendor_knowledge_source_plugin() -> VendorKnowledgeSourcePlugin:
    return build_jira_vendor_knowledge_contribution().source_plugins[0]


def build_confluence_vendor_knowledge_source_plugin() -> VendorKnowledgeSourcePlugin:
    return build_confluence_vendor_knowledge_contribution().source_plugins[0]


def build_default_vendor_knowledge_source_plugin_registry(
    *,
    discover_entry_points: bool = False,
) -> (
    VendorKnowledgeSourcePluginRegistry
):
    """Build every implemented source kind through contributions."""
    catalog = build_default_vendor_knowledge_contribution_catalog(
        discover_entry_points=discover_entry_points,
    )
    return build_vendor_knowledge_source_plugin_registry(catalog)


__all__ = [
    "build_confluence_vendor_knowledge_source_plugin",
    "build_default_vendor_knowledge_source_plugin_registry",
    "build_google_workspace_vendor_knowledge_source_plugins",
    "build_jira_vendor_knowledge_source_plugin",
]
