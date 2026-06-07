# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite, MailMessage
from intergrax.tools.providers.collaboration.contracts import (
    CollaborationCalendarEventOutput,
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
    CollaborationMailMessageOutput,
    CollaborationReplyMessageInput,
    CollaborationReplyMessageOutput,
    CollaborationSendMailInput,
    CollaborationSendMailOutput,
    CollaborationUserOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

COLLABORATION_SEND_MAIL_TOOL_ID = "collaboration.send_mail"
COLLABORATION_LIST_MESSAGES_TOOL_ID = "collaboration.list_messages"
COLLABORATION_GET_MESSAGE_TOOL_ID = "collaboration.get_message"
COLLABORATION_LIST_CALENDAR_TOOL_ID = "collaboration.list_calendar"
COLLABORATION_GET_USER_TOOL_ID = "collaboration.get_user"
COLLABORATION_REPLY_MESSAGE_TOOL_ID = "collaboration.reply_message"
COLLABORATION_CREATE_EVENT_TOOL_ID = "collaboration.create_event"


def _require_suite(ctx: ToolWiringContext) -> CollaborationSuite:
    suite = ctx.collaboration_suite
    if suite is None:
        raise RuntimeError("collaboration_suite_not_configured")
    return suite


def collaboration_send_mail(ctx: ToolWiringContext, params: CollaborationSendMailInput) -> CollaborationSendMailOutput:
    recipients = [item.strip() for item in params.to if item.strip()]
    _require_suite(ctx).send_mail(
        params.user_id.strip(),
        subject=params.subject,
        body=params.body,
        to=recipients,
    )
    return CollaborationSendMailOutput(sent=True, recipient_count=len(recipients))


def _mail_output(message: MailMessage) -> CollaborationMailMessageOutput:
    return CollaborationMailMessageOutput(
        id=message.id,
        subject=message.subject,
        body_preview=message.body_preview,
        from_address=message.from_address,
        received_at=message.received_at,
    )


def collaboration_list_messages(
    ctx: ToolWiringContext,
    params: CollaborationListMessagesInput,
) -> CollaborationListMessagesOutput:
    result = _require_suite(ctx).list_messages(
        params.user_id.strip(),
        folder=params.folder,
        limit=params.limit,
    )
    messages = [_mail_output(item) for item in result.messages]
    return CollaborationListMessagesOutput(messages=messages, total=result.total or len(messages))


def collaboration_get_message(
    ctx: ToolWiringContext,
    params: CollaborationGetMessageInput,
) -> CollaborationGetMessageOutput:
    message = _require_suite(ctx).get_message(params.user_id.strip(), params.message_id.strip())
    return CollaborationGetMessageOutput(message=_mail_output(message))


def collaboration_list_calendar(
    ctx: ToolWiringContext,
    params: CollaborationListCalendarInput,
) -> CollaborationListCalendarOutput:
    result = _require_suite(ctx).list_calendar_events(
        params.user_id.strip(),
        start=params.start,
        end=params.end,
        limit=params.limit,
    )
    events = [
        CollaborationCalendarEventOutput(
            id=item.id,
            subject=item.subject,
            start=item.start,
            end=item.end,
            location=item.location,
            organizer=item.organizer,
        )
        for item in result.events
    ]
    return CollaborationListCalendarOutput(events=events, total=result.total or len(events))


def collaboration_get_user(ctx: ToolWiringContext, params: CollaborationGetUserInput) -> CollaborationGetUserOutput:
    user = _require_suite(ctx).get_user(params.user_id.strip())
    return CollaborationGetUserOutput(
        user=CollaborationUserOutput(
            id=user.id,
            display_name=user.display_name,
            email=user.email,
        )
    )


def collaboration_reply_message(
    ctx: ToolWiringContext,
    params: CollaborationReplyMessageInput,
) -> CollaborationReplyMessageOutput:
    _require_suite(ctx).reply_message(
        params.user_id.strip(),
        params.message_id.strip(),
        body=params.body,
    )
    return CollaborationReplyMessageOutput(replied=True)


def collaboration_create_event(
    ctx: ToolWiringContext,
    params: CollaborationCreateEventInput,
) -> CollaborationCreateEventOutput:
    event = _require_suite(ctx).create_event(
        params.user_id.strip(),
        subject=params.subject,
        start=params.start,
        end=params.end,
        location=params.location,
        attendees=params.attendees,
    )
    return CollaborationCreateEventOutput(
        event=CollaborationCalendarEventOutput(
            id=event.id,
            subject=event.subject,
            start=event.start,
            end=event.end,
            location=event.location,
            organizer=event.organizer,
        )
    )
