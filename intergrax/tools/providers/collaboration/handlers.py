# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.collaboration.contracts import (
    CollaborationCreateEventInput,
    CollaborationCreateEventOutput,
    CollaborationGetMessageInput,
    CollaborationGetMessageOutput,
    CollaborationGetUserInput,
    CollaborationGetUserOutput,
    CollaborationListCalendarInput,
    CollaborationListCalendarOutput,
    CollaborationListMessagesInput,
    CollaborationListMessagesOutput,
    CollaborationReplyMessageInput,
    CollaborationReplyMessageOutput,
    CollaborationSendMailInput,
    CollaborationSendMailOutput,
)
from intergrax.tools.providers.collaboration.service import (
    collaboration_create_event,
    collaboration_get_message,
    collaboration_get_user,
    collaboration_list_calendar,
    collaboration_list_messages,
    collaboration_reply_message,
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


class CollaborationReplyMessageHandler(
    ServiceToolHandler[CollaborationReplyMessageInput, CollaborationReplyMessageOutput]
):
    _service = collaboration_reply_message


class CollaborationCreateEventHandler(
    ServiceToolHandler[CollaborationCreateEventInput, CollaborationCreateEventOutput]
):
    _service = collaboration_create_event
