"""Microsoft Graph Vendor Knowledge live capabilities."""

from .drive import (
    MSGRAPH_DRIVE_LIST_CAPABILITY_ID,
    MSGRAPH_DRIVE_LIST_REQUEST_SCHEMA_REF,
    MSGRAPH_DRIVE_LIST_RESULT_SCHEMA_REF,
    MsGraphDriveListLiveHandlerV1,
    MsGraphDriveListLiveRequestV1,
    build_msgraph_drive_list_descriptor,
)
from .mail import (
    MSGRAPH_MAIL_LIST_CAPABILITY_ID,
    MSGRAPH_MAIL_LIST_REQUEST_SCHEMA_REF,
    MSGRAPH_MAIL_LIST_RESULT_SCHEMA_REF,
    MsGraphMailListLiveHandlerV1,
    MsGraphMailListLiveRequestV1,
    build_msgraph_mail_list_descriptor,
)
from .registration import (
    build_msgraph_drive_live_registration_bundles,
    build_msgraph_live_registration_bundles,
)

__all__ = [
    "MSGRAPH_DRIVE_LIST_CAPABILITY_ID",
    "MSGRAPH_DRIVE_LIST_REQUEST_SCHEMA_REF",
    "MSGRAPH_DRIVE_LIST_RESULT_SCHEMA_REF",
    "MSGRAPH_MAIL_LIST_CAPABILITY_ID",
    "MSGRAPH_MAIL_LIST_REQUEST_SCHEMA_REF",
    "MSGRAPH_MAIL_LIST_RESULT_SCHEMA_REF",
    "MsGraphDriveListLiveHandlerV1",
    "MsGraphDriveListLiveRequestV1",
    "MsGraphMailListLiveHandlerV1",
    "MsGraphMailListLiveRequestV1",
    "build_msgraph_drive_list_descriptor",
    "build_msgraph_drive_live_registration_bundles",
    "build_msgraph_live_registration_bundles",
    "build_msgraph_mail_list_descriptor",
]
