# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vendor-specific knowledge source adapters (explicit registration only)."""

from intergrax.runtime.vendor_knowledge.adapters.jira_issues import (
    JiraIssuesKnowledgeAdapter,
    register_jira_issues_knowledge_adapter,
)

__all__ = [
    "JiraIssuesKnowledgeAdapter",
    "register_jira_issues_knowledge_adapter",
]
