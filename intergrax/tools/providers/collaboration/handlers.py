# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.collaboration.contracts import (
    CollaborationGetMessageInput,
    CollaborationGetMessageOutput,
    CollaborationGetUserInput,
    CollaborationGetUserOutput,
    CollaborationListCalendarInput,
    CollaborationListCalendarOutput,
    CollaborationListMessagesInput,
    CollaborationListMessagesOutput,
    CollaborationSendMailInput,
    CollaborationSendMailOutput,
)
from intergrax.tools.providers.collaboration.service import (
    collaboration_get_message,
    collaboration_get_user,
    collaboration_list_calendar,
    collaboration_list_messages,
    collaboration_send_mail,
)


class CollaborationSendMailHandler(ServiceToolHandler[CollaborationSendMailInput, CollaborationSendMailOutput]):
    _service = collaboration_send_mail


class CollaborationListMessagesHandler(
    ServiceToolHandler[CollaborationListMessagesInput, CollaborationListMessagesOutput]
):
    _service = collaboration_list_messages


class CollaborationGetMessageHandler(ServiceToolHandler[CollaborationGetMessageInput, CollaborationGetMessageOutput]):
    _service = collaboration_get_message


class CollaborationListCalendarHandler(
    ServiceToolHandler[CollaborationListCalendarInput, CollaborationListCalendarOutput]
):
    _service = collaboration_list_calendar


class CollaborationGetUserHandler(ServiceToolHandler[CollaborationGetUserInput, CollaborationGetUserOutput]):
    _service = collaboration_get_user
