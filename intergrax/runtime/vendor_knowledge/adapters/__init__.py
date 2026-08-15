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
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_docs import (
    GOOGLE_DOCS_CURSOR_VERSION,
    GOOGLE_DOCS_DOCUMENT_SCOPE_TYPE,
    GOOGLE_DOCS_ITEM_METADATA_VERSION,
    GOOGLE_DOCS_STRUCTURED_RECORD_MIME_TYPE,
    GOOGLE_DOCS_STRUCTURED_RECORD_SCHEMA,
    GoogleWorkspaceDocsKnowledgeAdapter,
    register_google_workspace_docs_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_sheets import (
    GOOGLE_SHEETS_CURSOR_VERSION,
    GOOGLE_SHEETS_ITEM_METADATA_VERSION,
    GOOGLE_SHEETS_SPREADSHEET_SCOPE_TYPE,
    GOOGLE_SHEETS_STRUCTURED_RECORD_MIME_TYPE,
    GOOGLE_SHEETS_STRUCTURED_RECORD_SCHEMA,
    GoogleWorkspaceSheetsKnowledgeAdapter,
    register_google_workspace_sheets_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_drive import (
    GOOGLE_DRIVE_CURSOR_VERSION,
    GOOGLE_DRIVE_ITEM_METADATA_VERSION,
    GOOGLE_DRIVE_SHARED_DRIVE_SCOPE_TYPE,
    GOOGLE_DRIVE_USER_SCOPE_ID,
    GOOGLE_DRIVE_USER_SCOPE_TYPE,
    GoogleWorkspaceDriveKnowledgeAdapter,
    register_google_workspace_drive_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_calendar import (
    GOOGLE_CALENDAR_CURSOR_VERSION,
    GOOGLE_CALENDAR_ITEM_METADATA_VERSION,
    GOOGLE_CALENDAR_SCOPE_TYPE,
    GOOGLE_CALENDAR_STRUCTURED_RECORD_MIME_TYPE,
    GOOGLE_CALENDAR_STRUCTURED_RECORD_SCHEMA,
    GoogleWorkspaceCalendarKnowledgeAdapter,
    register_google_workspace_calendar_knowledge_adapter,
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
    "GOOGLE_DOCS_CURSOR_VERSION",
    "GOOGLE_DOCS_DOCUMENT_SCOPE_TYPE",
    "GOOGLE_DOCS_ITEM_METADATA_VERSION",
    "GOOGLE_DOCS_STRUCTURED_RECORD_MIME_TYPE",
    "GOOGLE_DOCS_STRUCTURED_RECORD_SCHEMA",
    "GoogleWorkspaceDocsKnowledgeAdapter",
    "GOOGLE_SHEETS_CURSOR_VERSION",
    "GOOGLE_SHEETS_ITEM_METADATA_VERSION",
    "GOOGLE_SHEETS_SPREADSHEET_SCOPE_TYPE",
    "GOOGLE_SHEETS_STRUCTURED_RECORD_MIME_TYPE",
    "GOOGLE_SHEETS_STRUCTURED_RECORD_SCHEMA",
    "GoogleWorkspaceSheetsKnowledgeAdapter",
    "GOOGLE_DRIVE_CURSOR_VERSION",
    "GOOGLE_DRIVE_ITEM_METADATA_VERSION",
    "GOOGLE_DRIVE_SHARED_DRIVE_SCOPE_TYPE",
    "GOOGLE_DRIVE_USER_SCOPE_ID",
    "GOOGLE_DRIVE_USER_SCOPE_TYPE",
    "GoogleWorkspaceDriveKnowledgeAdapter",
    "GOOGLE_CALENDAR_CURSOR_VERSION",
    "GOOGLE_CALENDAR_ITEM_METADATA_VERSION",
    "GOOGLE_CALENDAR_SCOPE_TYPE",
    "GOOGLE_CALENDAR_STRUCTURED_RECORD_MIME_TYPE",
    "GOOGLE_CALENDAR_STRUCTURED_RECORD_SCHEMA",
    "GoogleWorkspaceCalendarKnowledgeAdapter",
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
    "register_google_workspace_docs_knowledge_adapter",
    "register_google_workspace_sheets_knowledge_adapter",
    "register_google_workspace_drive_knowledge_adapter",
    "register_google_workspace_calendar_knowledge_adapter",
    "register_confluence_pages_knowledge_adapter",
    "register_jira_issues_knowledge_adapter",
    "register_msgraph_calendar_knowledge_adapter",
    "register_msgraph_drive_knowledge_adapter",
    "register_msgraph_mail_knowledge_adapter",
    "register_msgraph_teams_channel_knowledge_adapter",
    "register_msgraph_teams_chat_knowledge_adapter",
    "register_slack_conversation_knowledge_adapter",
]
