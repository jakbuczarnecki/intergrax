"""Microsoft Graph live capability registration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_inventory import (
    MSGRAPH_CALENDAR_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.drive import (
    MSGRAPH_DRIVE_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_folders import (
    MSGRAPH_MAIL_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (
    MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
    MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
)
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
from intergrax.runtime.vendor_knowledge.plugin import (
    VendorKnowledgeMode,
    VendorKnowledgeModeCapability,
    VendorKnowledgeSourceIdentity,
    VendorKnowledgeSourcePlugin,
)

from .calendar import (
    MSGRAPH_CALENDAR_LIST_REQUEST_SCHEMA_REF,
    MSGRAPH_CALENDAR_LIST_RESULT_SCHEMA_REF,
    MsGraphCalendarListLiveHandlerV1,
    MsGraphCalendarListLiveRequestV1,
    build_msgraph_calendar_list_descriptor,
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


def _build_msgraph_vendor_knowledge_source_plugin(
    *,
    source_kind: str,
    live_runtime_ref: str,
    durable_runtime_ref: str,
    indexed_runtime_ref: str | None = None,
) -> VendorKnowledgeSourcePlugin:
    """Compose one Graph source declaration from existing runtimes."""
    live_bundles = build_msgraph_live_registration_bundles()
    live_capability_refs = tuple(
        bundle.descriptor.capability_id
        for bundle in live_bundles
        if bundle.descriptor.source_kind == source_kind
    )
    identity = VendorKnowledgeSourceIdentity(
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_category=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=source_kind,
    )
    capabilities = [
        VendorKnowledgeModeCapability(
            mode=VendorKnowledgeMode.DURABLE,
            contract_version="vendor-knowledge.durable.v1",
            operations=("inventory", "snapshot", "incremental", "reconciliation", "exact_fetch"),
            runtime_ref=durable_runtime_ref,
            constraints={"application_sink": "platform_foundation"},
        ),
        VendorKnowledgeModeCapability(
            mode=VendorKnowledgeMode.LIVE,
            contract_version="vendor-knowledge.live.v1",
            operations=("list",),
            runtime_ref=live_runtime_ref,
            capability_refs=live_capability_refs,
            constraints={"read_only": True, "bounded": True},
        ),
    ]
    if indexed_runtime_ref is not None:
        capabilities.append(
            VendorKnowledgeModeCapability(
                mode=VendorKnowledgeMode.INDEXED,
                contract_version="vendor-knowledge.indexed.v1",
                operations=("eligible", "materialize", "publish", "index"),
                runtime_ref=indexed_runtime_ref,
                constraints={"application_proof": "vk4"},
            )
        )
    return VendorKnowledgeSourcePlugin(
        identity=identity,
        capabilities=tuple(capabilities),
    )


def build_msgraph_drive_vendor_knowledge_source_plugin() -> VendorKnowledgeSourcePlugin:
    return _build_msgraph_vendor_knowledge_source_plugin(
        source_kind=MSGRAPH_DRIVE_SOURCE_KIND,
        live_runtime_ref="live-registration:ms365_graph:drive",
        durable_runtime_ref="knowledge-adapter:ms365_graph:collaboration_suite:drive",
    )


def build_msgraph_mail_vendor_knowledge_source_plugin() -> VendorKnowledgeSourcePlugin:
    return _build_msgraph_vendor_knowledge_source_plugin(
        source_kind=MSGRAPH_MAIL_SOURCE_KIND,
        live_runtime_ref="live-registration:ms365_graph:mail",
        durable_runtime_ref="knowledge-adapter:ms365_graph:collaboration_suite:mail",
        indexed_runtime_ref="indexed-source:ms365_graph:mail",
    )


def build_msgraph_teams_channel_vendor_knowledge_source_plugin() -> VendorKnowledgeSourcePlugin:
    return _build_msgraph_vendor_knowledge_source_plugin(
        source_kind=MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
        live_runtime_ref="live-registration:ms365_graph:teams_channel",
        durable_runtime_ref="knowledge-adapter:ms365_graph:collaboration_suite:teams_channel",
    )


def build_msgraph_teams_chat_vendor_knowledge_source_plugin() -> VendorKnowledgeSourcePlugin:
    return _build_msgraph_vendor_knowledge_source_plugin(
        source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
        live_runtime_ref="live-registration:ms365_graph:teams_chat",
        durable_runtime_ref="knowledge-adapter:ms365_graph:collaboration_suite:teams_chat",
        indexed_runtime_ref="indexed-source:ms365_graph:teams_chat",
    )


def build_msgraph_calendar_vendor_knowledge_source_plugin() -> VendorKnowledgeSourcePlugin:
    return _build_msgraph_vendor_knowledge_source_plugin(
        source_kind=MSGRAPH_CALENDAR_SOURCE_KIND,
        live_runtime_ref="live-registration:ms365_graph:calendar",
        durable_runtime_ref="knowledge-adapter:ms365_graph:collaboration_suite:calendar",
    )
