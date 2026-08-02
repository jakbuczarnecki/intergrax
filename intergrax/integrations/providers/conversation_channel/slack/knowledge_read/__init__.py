# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.common import (
    ABSOLUTE_MESSAGE_MAX_CHARS,
    DEFAULT_MESSAGE_MAX_CHARS,
    MAX_HISTORY_REPLY_PAGE_LIMIT,
    MAX_INVENTORY_PAGE_LIMIT,
    SLACK_CONVERSATION_SOURCE_KIND,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.errors import (
    SlackConversationContentTooLarge,
    SlackConversationMessageChanged,
    SlackConversationMessageNotFound,
    SlackConversationReadConfigurationError,
    SlackConversationReadError,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.models import (
    SlackConversationExactMessageResult,
    SlackConversationFileReference,
    SlackConversationInventoryPage,
    SlackConversationKind,
    SlackConversationMessage,
    SlackConversationMessagePage,
    SlackConversationPointWindow,
    SlackConversationSourceWindow,
    SlackConversationSummary,
    validate_slack_conversation_message,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.reader import (
    SlackConversationKnowledgeReadClient,
    SlackConversationKnowledgeReader,
    compute_slack_conversation_message_revision,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.timestamp import (
    compare_slack_timestamps,
    validate_slack_timestamp,
)

__all__ = [
    "ABSOLUTE_MESSAGE_MAX_CHARS",
    "DEFAULT_MESSAGE_MAX_CHARS",
    "MAX_HISTORY_REPLY_PAGE_LIMIT",
    "MAX_INVENTORY_PAGE_LIMIT",
    "SLACK_CONVERSATION_SOURCE_KIND",
    "SlackConversationContentTooLarge",
    "SlackConversationExactMessageResult",
    "SlackConversationFileReference",
    "SlackConversationInventoryPage",
    "SlackConversationKind",
    "SlackConversationKnowledgeReadClient",
    "SlackConversationKnowledgeReader",
    "SlackConversationMessage",
    "SlackConversationMessageChanged",
    "SlackConversationMessageNotFound",
    "SlackConversationMessagePage",
    "SlackConversationPointWindow",
    "SlackConversationReadConfigurationError",
    "SlackConversationReadError",
    "SlackConversationSourceWindow",
    "SlackConversationSummary",
    "compare_slack_timestamps",
    "compute_slack_conversation_message_revision",
    "validate_slack_conversation_message",
    "validate_slack_timestamp",
]
