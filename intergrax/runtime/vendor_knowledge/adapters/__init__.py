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
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_calendar import (
    MSGRAPH_CALENDAR_CURSOR_VERSION,
    MSGRAPH_CALENDAR_SCOPE_TYPE,
    MsGraphCalendarKnowledgeAdapter,
    encode_msgraph_calendar_scope_id,
    register_msgraph_calendar_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_drive import (
    MSGRAPH_DRIVE_CURSOR_VERSION,
    MSGRAPH_DRIVE_SCOPE_TYPE,
    MsGraphDriveKnowledgeAdapter,
    register_msgraph_drive_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_mail import (
    MSGRAPH_MAIL_CURSOR_VERSION,
    MSGRAPH_MAIL_SCOPE_TYPE,
    MsGraphMailKnowledgeAdapter,
    encode_msgraph_mail_folder_scope_id,
    register_msgraph_mail_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_channel import (
    MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION,
    MSGRAPH_TEAMS_CHANNEL_SCOPE_TYPE,
    MsGraphTeamsChannelKnowledgeAdapter,
    encode_msgraph_teams_channel_scope_id,
    register_msgraph_teams_channel_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_chat import (
    MSGRAPH_TEAMS_CHAT_CURSOR_VERSION,
    MSGRAPH_TEAMS_CHAT_SCOPE_TYPE,
    MsGraphTeamsChatKnowledgeAdapter,
    encode_msgraph_teams_chat_scope_id,
    register_msgraph_teams_chat_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.slack_conversation import (
    SLACK_CONVERSATION_CURSOR_VERSION,
    SLACK_CONVERSATION_SCOPE_TYPE,
    SlackConversationKnowledgeAdapter,
    encode_slack_conversation_scope_id,
    register_slack_conversation_knowledge_adapter,
)

__all__ = [
    "ConfluencePagesKnowledgeAdapter",
    "JiraIssuesKnowledgeAdapter",
    "MSGRAPH_CALENDAR_CURSOR_VERSION",
    "MSGRAPH_CALENDAR_SCOPE_TYPE",
    "MSGRAPH_DRIVE_CURSOR_VERSION",
    "MSGRAPH_DRIVE_SCOPE_TYPE",
    "MSGRAPH_MAIL_CURSOR_VERSION",
    "MSGRAPH_MAIL_SCOPE_TYPE",
    "MSGRAPH_TEAMS_CHANNEL_CURSOR_VERSION",
    "MSGRAPH_TEAMS_CHANNEL_SCOPE_TYPE",
    "MSGRAPH_TEAMS_CHAT_CURSOR_VERSION",
    "MSGRAPH_TEAMS_CHAT_SCOPE_TYPE",
    "SLACK_CONVERSATION_CURSOR_VERSION",
    "SLACK_CONVERSATION_SCOPE_TYPE",
    "MsGraphCalendarKnowledgeAdapter",
    "MsGraphDriveKnowledgeAdapter",
    "MsGraphMailKnowledgeAdapter",
    "MsGraphTeamsChannelKnowledgeAdapter",
    "MsGraphTeamsChatKnowledgeAdapter",
    "SlackConversationKnowledgeAdapter",
    "encode_msgraph_calendar_scope_id",
    "encode_msgraph_mail_folder_scope_id",
    "encode_msgraph_teams_channel_scope_id",
    "encode_msgraph_teams_chat_scope_id",
    "encode_slack_conversation_scope_id",
    "register_confluence_pages_knowledge_adapter",
    "register_jira_issues_knowledge_adapter",
    "register_msgraph_calendar_knowledge_adapter",
    "register_msgraph_drive_knowledge_adapter",
    "register_msgraph_mail_knowledge_adapter",
    "register_msgraph_teams_channel_knowledge_adapter",
    "register_msgraph_teams_chat_knowledge_adapter",
    "register_slack_conversation_knowledge_adapter",
]
