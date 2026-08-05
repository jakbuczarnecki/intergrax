from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (
    MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_channel import (
    MSGRAPH_TEAMS_CHANNEL_SCOPE_TYPE,
    encode_msgraph_teams_channel_scope_id,
)
from intergrax.runtime.vendor_knowledge.live import (
    EffectiveLiveCallBudgetV1,
    KnowledgeQueryAudienceV1,
    LiveCapabilityExecutionContextV1,
    LiveCapabilityExecutionResultV1,
    LiveExecutionOutcomeV1,
    LiveResultRetentionV1,
    ValidatedLiveCapabilityCallV1,
)
from intergrax.runtime.vendor_knowledge.live.identity import (
    LiveOperationV1,
    parse_capability_id,
)
from intergrax.runtime.vendor_knowledge.live.ms365_graph import (
    MSGRAPH_TEAMS_CHANNEL_LIST_CAPABILITY_ID,
    MSGRAPH_TEAMS_CHANNEL_LIST_REQUEST_SCHEMA_REF,
    MSGRAPH_TEAMS_CHANNEL_LIST_RESULT_SCHEMA_REF,
    MsGraphTeamsChannelListLiveHandlerV1,
    MsGraphTeamsChannelListLiveRequestV1,
    build_msgraph_live_registration_bundles,
    build_msgraph_teams_channel_list_descriptor,
)
from intergrax.runtime.vendor_knowledge.live.registration import (
    publish_live_registration_bundles,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeChange,
    KnowledgeChangeKind,
    KnowledgeCursor,
    KnowledgePage,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)
_SCOPE = encode_msgraph_teams_channel_scope_id(
    team_remote_id="team-1",
    channel_remote_id="channel-1",
)


class _FakeTeamsChannelAdapter:
    def __init__(self, page: KnowledgePage) -> None:
        self.page = page
        self.calls: list[dict[str, object]] = []

    async def read_page(
        self,
        *,
        integration: object,
        source: object,
        cursor: object,
        limit: int,
    ) -> KnowledgePage:
        self.calls.append(
            {
                "integration": integration,
                "source": source,
                "cursor": cursor,
                "limit": limit,
            }
        )
        return self.page


def _integration() -> Ms365GraphCollaborationSuiteIntegration:
    return Ms365GraphCollaborationSuiteIntegration.from_client(object())


def _call(
    *,
    request: object | None = None,
    remote_resource_id: str | None = _SCOPE,
    **identity_overrides: object,
) -> ValidatedLiveCapabilityCallV1:
    values: dict[str, Any] = {
        "call_id": "call-teams-channel-1",
        "capability_id": MSGRAPH_TEAMS_CHANNEL_LIST_CAPABILITY_ID,
        "contract_version": "1",
        "connection_ref": "connection-1",
        "live_access_binding_id": "binding-1",
        "remote_resource_id": remote_resource_id,
        "validated_request": (
            request
            if request is not None
            else MsGraphTeamsChannelListLiveRequestV1()
        ),
        "effective_budget": EffectiveLiveCallBudgetV1(
            max_live_calls=1,
            max_total_duration_ms=10_000,
            max_result_items=1,
            max_result_bytes=10_000,
            max_provider_pages=1,
            max_provider_requests=1,
            max_upstream_items=1,
            max_provider_page_size=1,
            max_content_bytes_per_item=4_096,
        ),
        "audience_context_ref": None,
        "provider_id": MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        "integration_kind": IntegrationCategory.COLLABORATION_SUITE,
        "source_kind": MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
    }
    values.update(identity_overrides)
    return ValidatedLiveCapabilityCallV1(**values)


def _context() -> LiveCapabilityExecutionContextV1:
    return LiveCapabilityExecutionContextV1(
        run_id="run-1",
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        audience=KnowledgeQueryAudienceV1.PERSONAL,
        started_at=_NOW,
        deadline_monotonic=100,
        retention=LiveResultRetentionV1.EPHEMERAL,
    )


def test_teams_channel_registration_uses_canonical_identity_and_schema_refs() -> None:
    bundles = build_msgraph_live_registration_bundles()
    published = publish_live_registration_bundles(bundles)
    capability_ids = tuple(bundle.descriptor.capability_id for bundle in bundles)

    assert capability_ids == (
        "vendor.ms365_graph.drive.list",
        "vendor.ms365_graph.mail.list",
        "vendor.ms365_graph.teams_channel.list",
        "vendor.ms365_graph.teams_chat.list",
        "vendor.ms365_graph.calendar.list",
    )
    provider_id, source_kind, operation = parse_capability_id(
        MSGRAPH_TEAMS_CHANNEL_LIST_CAPABILITY_ID
    )
    assert provider_id == MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID
    assert source_kind == MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND
    assert operation is LiveOperationV1.LIST
    assert (
        published.schemas.resolve_request(
            MSGRAPH_TEAMS_CHANNEL_LIST_REQUEST_SCHEMA_REF, "1"
        )
        is MsGraphTeamsChannelListLiveRequestV1
    )
    assert (
        published.schemas.resolve_result(
            MSGRAPH_TEAMS_CHANNEL_LIST_RESULT_SCHEMA_REF, "1"
        )
        is LiveCapabilityExecutionResultV1
    )
    previous_operation = "root.list"
    previous_schema_segment = "/root/list/"
    assert all(previous_operation not in value for value in capability_ids)
    assert previous_schema_segment not in MSGRAPH_TEAMS_CHANNEL_LIST_REQUEST_SCHEMA_REF
    assert previous_schema_segment not in MSGRAPH_TEAMS_CHANNEL_LIST_RESULT_SCHEMA_REF


def test_teams_channel_registration_exposes_only_canonical_list_operation() -> None:
    published = publish_live_registration_bundles(build_msgraph_live_registration_bundles())

    for operation in ("search", "read", "thread.read", "child.read", "content.read"):
        with pytest.raises(LookupError, match="live_capability_unavailable"):
            published.resolve_handler(
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                capability_id=f"vendor.ms365_graph.teams_channel.{operation}",
                contract_version="1",
            )
    with pytest.raises(LookupError, match="live_capability_unavailable"):
        published.resolve_handler(
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            capability_id="vendor.ms365_graph.teams_channel.root.list",
            contract_version="1",
        )


def test_teams_channel_descriptor_states_exact_one_root_post_limit() -> None:
    descriptor = build_msgraph_teams_channel_list_descriptor()

    assert descriptor.max_result_items == 1
    assert descriptor.max_upstream_items == 1
    assert descriptor.max_provider_page_size == 1
    assert descriptor.max_provider_pages == 1
    assert descriptor.max_provider_requests == 1
    assert (
        "The v1 Teams Channel list capability returns at most one root post."
        in (MsGraphTeamsChannelListLiveHandlerV1.__doc__ or "")
    )
    assert (
        "It does not list replies or all channel messages."
        in (MsGraphTeamsChannelListLiveHandlerV1.__doc__ or "")
    )


def test_teams_channel_request_is_strict_immutable_and_zero_field() -> None:
    request = MsGraphTeamsChannelListLiveRequestV1.model_validate({})

    assert request.model_fields == {}
    with pytest.raises(ValidationError):
        MsGraphTeamsChannelListLiveRequestV1.model_validate({"limit": 1})
    with pytest.raises(ValidationError):
        request.unexpected = True


@pytest.mark.asyncio
async def test_teams_channel_handler_calls_adapter_once_with_fixed_root_bound() -> None:
    adapter = _FakeTeamsChannelAdapter(KnowledgePage())
    result = await MsGraphTeamsChannelListLiveHandlerV1(adapter=adapter).execute(
        integration=_integration(),
        call=_call(),
        context=_context(),
    )

    assert result.normalized_outcome is LiveExecutionOutcomeV1.COMPLETED
    assert result.items == ()
    assert len(adapter.calls) == 1
    call = adapter.calls[0]
    assert call["cursor"] is None
    assert call["limit"] == 1
    source = call["source"]
    assert source.scope.remote_scope_id == _SCOPE  # type: ignore[union-attr]
    assert source.scope.remote_scope_type == MSGRAPH_TEAMS_CHANNEL_SCOPE_TYPE  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_teams_channel_handler_marks_provider_more_as_truncated_without_replies() -> None:
    change = KnowledgeChange(kind=KnowledgeChangeKind.DELETED, remote_id="root-1")
    page = KnowledgePage(
        changes=(change,),
        next_cursor=KnowledgeCursor(value="provider-cursor", version="teams-channel"),
        has_more=True,
    )
    adapter = _FakeTeamsChannelAdapter(page)
    result = await MsGraphTeamsChannelListLiveHandlerV1(adapter=adapter).execute(
        integration=_integration(),
        call=_call(),
        context=_context(),
    )

    assert result.normalized_outcome is LiveExecutionOutcomeV1.TRUNCATED
    assert result.truncated is True
    assert result.item_count == 1
    assert "root post" in result.items[0].safe_display_name
    assert "repl" not in result.items[0].content.lower()
