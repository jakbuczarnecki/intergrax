# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical composition of implemented Vendor Knowledge source adapters."""

from __future__ import annotations

from intergrax.runtime.vendor_knowledge.adapters import (
    register_confluence_pages_knowledge_adapter,
    register_google_workspace_calendar_knowledge_adapter,
    register_google_workspace_docs_knowledge_adapter,
    register_google_workspace_drive_knowledge_adapter,
    register_google_workspace_sheets_knowledge_adapter,
    register_jira_issues_knowledge_adapter,
    register_msgraph_calendar_knowledge_adapter,
    register_msgraph_drive_knowledge_adapter,
    register_msgraph_mail_knowledge_adapter,
    register_msgraph_teams_channel_knowledge_adapter,
    register_msgraph_teams_chat_knowledge_adapter,
    register_slack_conversation_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry


def build_default_vendor_knowledge_adapter_registry() -> KnowledgeAdapterRegistry:
    """Build the deterministic registry of all currently implemented adapters."""
    registry = KnowledgeAdapterRegistry()
    register_msgraph_drive_knowledge_adapter(registry)
    register_msgraph_mail_knowledge_adapter(registry)
    register_msgraph_teams_channel_knowledge_adapter(registry)
    register_msgraph_teams_chat_knowledge_adapter(registry)
    register_msgraph_calendar_knowledge_adapter(registry)
    register_slack_conversation_knowledge_adapter(registry)
    register_google_workspace_drive_knowledge_adapter(registry)
    register_google_workspace_docs_knowledge_adapter(registry)
    register_google_workspace_sheets_knowledge_adapter(registry)
    register_google_workspace_calendar_knowledge_adapter(registry)
    register_jira_issues_knowledge_adapter(registry)
    register_confluence_pages_knowledge_adapter(registry)
    return registry


__all__ = ["build_default_vendor_knowledge_adapter_registry"]
