"""Microsoft Graph live capability registration."""

from __future__ import annotations

from intergrax.runtime.vendor_knowledge.live import (
    LiveCapabilityExecutionResultV1,
)
from intergrax.runtime.vendor_knowledge.live.registration import (
    LiveRegistrationBundleV1,
)
from intergrax.runtime.vendor_knowledge.live.schemas import (
    SchemaRegistrationV1,
    SchemaRoleV1,
)

from .drive import (
    MSGRAPH_DRIVE_LIST_REQUEST_SCHEMA_REF,
    MSGRAPH_DRIVE_LIST_RESULT_SCHEMA_REF,
    MsGraphDriveListLiveHandlerV1,
    MsGraphDriveListLiveRequestV1,
    build_msgraph_drive_list_descriptor,
)
from .mail import (
    MSGRAPH_MAIL_LIST_REQUEST_SCHEMA_REF,
    MSGRAPH_MAIL_LIST_RESULT_SCHEMA_REF,
    MsGraphMailListLiveHandlerV1,
    MsGraphMailListLiveRequestV1,
    build_msgraph_mail_list_descriptor,
)
from .teams_channel import (
    MSGRAPH_TEAMS_CHANNEL_LIST_REQUEST_SCHEMA_REF,
    MSGRAPH_TEAMS_CHANNEL_LIST_RESULT_SCHEMA_REF,
    MsGraphTeamsChannelListLiveHandlerV1,
    MsGraphTeamsChannelListLiveRequestV1,
    build_msgraph_teams_channel_list_descriptor,
)
from .teams_chat import (
    MSGRAPH_TEAMS_CHAT_LIST_REQUEST_SCHEMA_REF,
    MSGRAPH_TEAMS_CHAT_LIST_RESULT_SCHEMA_REF,
    MsGraphTeamsChatListLiveHandlerV1,
    MsGraphTeamsChatListLiveRequestV1,
    build_msgraph_teams_chat_list_descriptor,
)
from .calendar import (
    MSGRAPH_CALENDAR_LIST_REQUEST_SCHEMA_REF,
    MSGRAPH_CALENDAR_LIST_RESULT_SCHEMA_REF,
    MsGraphCalendarListLiveHandlerV1,
    MsGraphCalendarListLiveRequestV1,
    build_msgraph_calendar_list_descriptor,
)


def build_msgraph_drive_live_registration_bundles() -> (
    tuple[LiveRegistrationBundleV1, ...]
):
    """Return exactly the complete supported Microsoft Graph Drive bundle."""

    handler = MsGraphDriveListLiveHandlerV1()
    descriptor = build_msgraph_drive_list_descriptor()
    return (
        LiveRegistrationBundleV1(
            descriptor=descriptor,
            handler=handler,
            request_schema=SchemaRegistrationV1(
                schema_ref=MSGRAPH_DRIVE_LIST_REQUEST_SCHEMA_REF,
                role=SchemaRoleV1.REQUEST,
                model=MsGraphDriveListLiveRequestV1,
                contract_version="1",
            ),
            result_schema=SchemaRegistrationV1(
                schema_ref=MSGRAPH_DRIVE_LIST_RESULT_SCHEMA_REF,
                role=SchemaRoleV1.RESULT,
                model=LiveCapabilityExecutionResultV1,
                contract_version="1",
            ),
        ),
    )


def build_msgraph_live_registration_bundles() -> (
    tuple[LiveRegistrationBundleV1, ...]
):
    """Return the complete deterministic Microsoft Graph live family."""

    mail_descriptor = build_msgraph_mail_list_descriptor()
    teams_channel_descriptor = build_msgraph_teams_channel_list_descriptor()
    teams_chat_descriptor = build_msgraph_teams_chat_list_descriptor()
    calendar_descriptor = build_msgraph_calendar_list_descriptor()
    return (
        *build_msgraph_drive_live_registration_bundles(),
        LiveRegistrationBundleV1(
            descriptor=mail_descriptor,
            handler=MsGraphMailListLiveHandlerV1(),
            request_schema=SchemaRegistrationV1(
                schema_ref=MSGRAPH_MAIL_LIST_REQUEST_SCHEMA_REF,
                role=SchemaRoleV1.REQUEST,
                model=MsGraphMailListLiveRequestV1,
                contract_version="1",
            ),
            result_schema=SchemaRegistrationV1(
                schema_ref=MSGRAPH_MAIL_LIST_RESULT_SCHEMA_REF,
                role=SchemaRoleV1.RESULT,
                model=LiveCapabilityExecutionResultV1,
                contract_version="1",
            ),
        ),
        LiveRegistrationBundleV1(
            descriptor=teams_channel_descriptor,
            handler=MsGraphTeamsChannelListLiveHandlerV1(),
            request_schema=SchemaRegistrationV1(
                schema_ref=MSGRAPH_TEAMS_CHANNEL_LIST_REQUEST_SCHEMA_REF,
                role=SchemaRoleV1.REQUEST,
                model=MsGraphTeamsChannelListLiveRequestV1,
                contract_version="1",
            ),
            result_schema=SchemaRegistrationV1(
                schema_ref=MSGRAPH_TEAMS_CHANNEL_LIST_RESULT_SCHEMA_REF,
                role=SchemaRoleV1.RESULT,
                model=LiveCapabilityExecutionResultV1,
                contract_version="1",
            ),
        ),
        LiveRegistrationBundleV1(
            descriptor=teams_chat_descriptor,
            handler=MsGraphTeamsChatListLiveHandlerV1(),
            request_schema=SchemaRegistrationV1(
                schema_ref=MSGRAPH_TEAMS_CHAT_LIST_REQUEST_SCHEMA_REF,
                role=SchemaRoleV1.REQUEST,
                model=MsGraphTeamsChatListLiveRequestV1,
                contract_version="1",
            ),
            result_schema=SchemaRegistrationV1(
                schema_ref=MSGRAPH_TEAMS_CHAT_LIST_RESULT_SCHEMA_REF,
                role=SchemaRoleV1.RESULT,
                model=LiveCapabilityExecutionResultV1,
                contract_version="1",
            ),
        ),
        LiveRegistrationBundleV1(
            descriptor=calendar_descriptor,
            handler=MsGraphCalendarListLiveHandlerV1(),
            request_schema=SchemaRegistrationV1(
                schema_ref=MSGRAPH_CALENDAR_LIST_REQUEST_SCHEMA_REF,
                role=SchemaRoleV1.REQUEST,
                model=MsGraphCalendarListLiveRequestV1,
                contract_version="1",
            ),
            result_schema=SchemaRegistrationV1(
                schema_ref=MSGRAPH_CALENDAR_LIST_RESULT_SCHEMA_REF,
                role=SchemaRoleV1.RESULT,
                model=LiveCapabilityExecutionResultV1,
                contract_version="1",
            ),
        ),
    )
