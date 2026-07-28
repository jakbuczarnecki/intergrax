# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vendor-specific knowledge source adapters (explicit registration only)."""

from intergrax.runtime.vendor_knowledge.adapters.confluence_pages import (
    ConfluencePagesKnowledgeAdapter,
    register_confluence_pages_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.jira_issues import (
    JiraIssuesKnowledgeAdapter,
    register_jira_issues_knowledge_adapter,
)

__all__ = [
    "ConfluencePagesKnowledgeAdapter",
    "JiraIssuesKnowledgeAdapter",
    "register_confluence_pages_knowledge_adapter",
    "register_jira_issues_knowledge_adapter",
]
