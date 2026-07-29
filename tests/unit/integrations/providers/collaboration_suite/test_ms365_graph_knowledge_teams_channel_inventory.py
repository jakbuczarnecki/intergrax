# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Microsoft Graph Teams Channel knowledge-read inventory surface."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.providers.collaboration_suite.ms365_graph.adapter import (
    _Ms365GraphCollaborationSuite,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.client import GraphRestClient
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    DEFAULT_GRAPH_BASE_URL,
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeTransport,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (
    MsGraphTeamsChannel,
    MsGraphTeamsChannelPage,
    MsGraphTeamsChannelMembershipType,
    MsGraphTeamsChannelsReader,
    parse_msgraph_teams_channel,
    validate_msgraph_teams_channel,
    validate_msgraph_teams_channel_page,
    validate_msgraph_teams_channels_continuation,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_TEAM_ID = "team-abc-123"
_OTHER_TEAM_ID = "other-team-456"
_CHANNEL_ID = "channel-abc-123"
_OTHER_CHANNEL_ID = "other-channel-456"
_OPAQUE_CHANNEL_ID = "channels/messages/allMembers/replies"
_QUOTED_TEAM = quote(_TEAM_ID, safe="")
_QUOTED_OTHER_TEAM = quote(_OTHER_TEAM_ID, safe="")
_SECRET_TOKEN = "secret-skiptoken-value"
_DISPLAY_NAME = "General Channel"
_HIDDEN_DISPLAY_NAME = "Hidden Display Name"
_TENANT_ID = "tenant-guid-0001"
_CREATED_AT_STR = "2026-01-01T00:00:00Z"
_UPDATED_AT_STR = "2026-01-02T00:00:00Z"
_CREATED_AT = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_UPDATED_AT = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
_ROOT_PATH = f"/teams/{_QUOTED_TEAM}/channels"
_CHANNEL_SELECT = (
    "id,displayName,description,createdDateTime,membershipType,isArchived,tenantId"
)
_PREFER_HEADER = {"Prefer": "include-unknown-enum-members"}
_SAFE_ERROR = "unexpected Microsoft Graph Teams channels response"
_REQUEST_ERROR = "invalid Microsoft Graph Teams channels request"
_CONT_ERROR = "invalid Microsoft Graph Teams channels continuation"
_VALIDATION_ERROR = "Microsoft Graph Teams Channel validation is not configured"
_MISSING = object()


def _config() -> Ms365GraphIntegrationConfig:
    return Ms365GraphIntegrationConfig(
        tenant_id="tenant-123",
        client_id="client-456",
        client_secret="secret",
        graph_base_url=_GRAPH_BASE,
    )


def _json_response(*, status_code: int = 200, payload: object | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload if payload is not None else {}
    response.raise_for_status = MagicMock()
    return response


def _next_link(*, path: str | None = None) -> str:
    resolved = path or f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels"
    return f"{resolved}?$skiptoken={_SECRET_TOKEN}"


def _page_payload(
    *,
    value: list[dict[str, Any]] | None = None,
    next_link: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"value": [] if value is None else value}
    if next_link is not None:
        payload["@odata.nextLink"] = next_link
    return payload


def _channel_payload(
    *,
    channel_id: str = _CHANNEL_ID,
    membership_type: str = "standard",
    created_at: str = _CREATED_AT_STR,
    is_archived: bool = False,
    display_name: str = _DISPLAY_NAME,
    description: str | None | object = _MISSING,
    tenant_id: str | None | object = _MISSING,
    extra_field: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": channel_id,
        "displayName": display_name,
        "membershipType": membership_type,
        "isArchived": is_archived,
        "createdDateTime": created_at,
    }
    if description is not _MISSING:
        payload["description"] = description
    if tenant_id is not _MISSING:
        payload["tenantId"] = tenant_id
    if extra_field is not None:
        payload["unknownField"] = extra_field
    return payload



def _reader(http: MagicMock) -> MsGraphTeamsChannelsReader:
    return MsGraphTeamsChannelsReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
    )


def _graph_client(http: MagicMock) -> GraphRestClient:
    return GraphRestClient(_config(), http_client=http)


def _parse_channel(payload: dict[str, Any]) -> MsGraphTeamsChannel:
    return parse_msgraph_teams_channel(payload, expected_team_id=_TEAM_ID)


def _valid_channel(**overrides: object) -> MsGraphTeamsChannel:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "remote_id": _CHANNEL_ID,
        "display_name": _DISPLAY_NAME,
        "description": None,
        "created_at": _CREATED_AT,
        "membership_type": MsGraphTeamsChannelMembershipType.STANDARD,
        "is_archived": False,
        "tenant_id": None,
    }
    defaults.update(overrides)
    return MsGraphTeamsChannel(**defaults)  # type: ignore[arg-type]


def _validate_page(page: MsGraphTeamsChannelPage) -> MsGraphTeamsChannelPage:
    return validate_msgraph_teams_channel_page(
        page,
        team_id=_TEAM_ID,
        graph_base_url=_GRAPH_BASE,
    )


def _assert_safe_provider_error(exc: BaseException) -> None:
    assert str(exc.value if isinstance(exc, pytest.ExceptionInfo) else exc) == _SAFE_ERROR
    cause = exc.value.__cause__ if isinstance(exc, pytest.ExceptionInfo) else exc.__cause__
    assert cause is None
    message = str(exc.value if isinstance(exc, pytest.ExceptionInfo) else exc)
    for forbidden in (
        _TEAM_ID,
        _CHANNEL_ID,
        _DISPLAY_NAME,
        _HIDDEN_DISPLAY_NAME,
        _TENANT_ID,
        "Authorization",
        "access token",
        "nextLink",
        _SECRET_TOKEN,
    ):
        assert forbidden not in message


# --- parser success ---


def test_parse_standard_channel() -> None:
    channel = _parse_channel(_channel_payload(membership_type="standard"))
    assert channel.membership_type is MsGraphTeamsChannelMembershipType.STANDARD
    assert channel.remote_id == _CHANNEL_ID


def test_parse_private_channel() -> None:
    channel = _parse_channel(_channel_payload(membership_type="private", channel_id=_OTHER_CHANNEL_ID))
    assert channel.membership_type is MsGraphTeamsChannelMembershipType.PRIVATE
    assert channel.remote_id == _OTHER_CHANNEL_ID


def test_parse_shared_channel() -> None:
    channel = _parse_channel(_channel_payload(membership_type="shared"))
    assert channel.membership_type is MsGraphTeamsChannelMembershipType.SHARED


def test_parse_unknown_future_membership_type() -> None:
    channel = _parse_channel(_channel_payload(membership_type="futureMembershipType"))
    assert channel.membership_type is MsGraphTeamsChannelMembershipType.UNKNOWN


def test_parse_description_present() -> None:
    channel = _parse_channel(_channel_payload(description=_DISPLAY_NAME))
    assert channel.description == _DISPLAY_NAME


def test_parse_description_absent() -> None:
    channel = _parse_channel(_channel_payload())
    assert channel.description is None


def test_parse_description_null() -> None:
    channel = _parse_channel(_channel_payload(description=None))
    assert channel.description is None


def test_parse_description_empty_becomes_none() -> None:
    channel = _parse_channel(_channel_payload(description="   "))
    assert channel.description is None


def test_parse_description_trimmed() -> None:
    channel = _parse_channel(_channel_payload(description=f"  {_DISPLAY_NAME}  "))
    assert channel.description == _DISPLAY_NAME


def test_parse_tenant_id_present() -> None:
    channel = _parse_channel(_channel_payload(tenant_id=_TENANT_ID))
    assert channel.tenant_id == _TENANT_ID


def test_parse_tenant_id_absent() -> None:
    channel = _parse_channel(_channel_payload())
    assert channel.tenant_id is None









def test_parse_timestamps_utc_from_z_suffix() -> None:
    channel = _parse_channel(_channel_payload())
    assert channel.created_at == _CREATED_AT
    assert channel.created_at.tzinfo == timezone.utc





def test_parse_opaque_channel_id() -> None:
    channel = _parse_channel(_channel_payload(channel_id=_OPAQUE_CHANNEL_ID))
    assert channel.remote_id == _OPAQUE_CHANNEL_ID


def test_parse_expected_team_id_preserved() -> None:
    channel = _parse_channel(_channel_payload())
    assert channel.team_remote_id == _TEAM_ID


def test_parse_unknown_provider_fields_not_retained() -> None:
    channel = _parse_channel(_channel_payload(extra_field="should-not-appear"))
    assert not hasattr(channel, "unknownField")
    dumped = channel.model_dump()
    assert "unknownField" not in dumped


def test_sensitive_fields_hidden_from_repr() -> None:
    channel = _valid_channel(description=_HIDDEN_DISPLAY_NAME, tenant_id=_TENANT_ID)
    channel_with_fields = _parse_channel(
        _channel_payload(description=_HIDDEN_DISPLAY_NAME, tenant_id=_TENANT_ID)
    )
    for item in (channel, channel_with_fields):
        rendered = repr(item)
        assert _HIDDEN_DISPLAY_NAME not in rendered
        assert _TENANT_ID not in rendered


def test_raw_provider_payload_not_stored() -> None:
    payload = _channel_payload()
    channel = _parse_channel(payload)
    assert not hasattr(channel, "__pydantic_extra__") or not channel.__pydantic_extra__


# --- malformed provider ---


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-dict",
        {},
        {"membershipType": "standard"},
        {"id": 123, "membershipType": "standard"},
        {"id": "", "membershipType": "standard"},
        {"id": _CHANNEL_ID},
        {"id": _CHANNEL_ID, "membershipType": 123},
        {"id": _CHANNEL_ID, "membershipType": ""},
        {
            "id": _CHANNEL_ID,
            "membershipType": "standard",
            "createdDateTime": _CREATED_AT_STR,
        },
        {
            "id": _CHANNEL_ID,
            "membershipType": "standard",
            "createdDateTime": _CREATED_AT_STR,
            "isArchived": 1,
        },
        {
            "id": _CHANNEL_ID,
            "membershipType": "standard",
            "createdDateTime": "not-a-datetime",
            "isArchived": False,
            "displayName": "General",
        },
        {
            "id": _CHANNEL_ID,
            "membershipType": "standard",
            "createdDateTime": "2026-01-01T00:00:00",
            "isArchived": False,
            "displayName": "General",
        },
        {
            "id": _CHANNEL_ID,
            "membershipType": "standard",
            "createdDateTime": _CREATED_AT_STR,
            "isArchived": False,
            "displayName": "bad\x00name",
        },
        {
            "id": _CHANNEL_ID,
            "membershipType": "standard",
            "createdDateTime": _CREATED_AT_STR,
            "isArchived": False,
            "displayName": "General",
            "tenantId": "",
        },
        {
            "id": _CHANNEL_ID,
            "membershipType": "standard",
            "createdDateTime": _CREATED_AT_STR,
            "isArchived": False,
            "displayName": "General",
            "description": "bad\x00desc",
        },
    ],
)
def test_malformed_provider_payload_rejected(payload: object) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        parse_msgraph_teams_channel(payload, expected_team_id=_TEAM_ID)
    _assert_safe_provider_error(exc)


def test_parse_description_over_limit_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        _parse_channel(_channel_payload(description="x" * 32769))


def test_parse_tenant_id_over_limit_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        _parse_channel(_channel_payload(tenant_id="x" * 2049))


def test_malformed_page_model_construct_missing_team_id() -> None:
    malformed = MsGraphTeamsChannelPage.model_construct(items=(_valid_channel(),))
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_wrong_team_id() -> None:
    malformed = MsGraphTeamsChannelPage.model_construct(
        team_remote_id=_OTHER_TEAM_ID,
        items=(_valid_channel(),),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    _assert_safe_provider_error(exc)


def test_validate_page_returns_new_instance() -> None:
    page = MsGraphTeamsChannelPage(team_remote_id=_TEAM_ID, items=(_valid_channel(),))
    validated = _validate_page(page)
    assert validated == page
    assert validated is not page
    assert validated.items[0] is not page.items[0]


# --- page model ---


def test_page_empty_tuple() -> None:
    page = MsGraphTeamsChannelPage(team_remote_id=_TEAM_ID, items=())
    assert page.items == ()
    assert page.has_more is False


def test_page_multiple_channels() -> None:
    page = MsGraphTeamsChannelPage(
        team_remote_id=_TEAM_ID,
        items=(
            _valid_channel(remote_id="c1"),
            _valid_channel(remote_id="c2", membership_type=MsGraphTeamsChannelMembershipType.PRIVATE),
        ),
    )
    assert len(page.items) == 2


def test_page_has_more_false() -> None:
    page = MsGraphTeamsChannelPage(team_remote_id=_TEAM_ID, items=())
    assert page.has_more is False


def test_page_has_more_true() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(),
    )
    page = MsGraphTeamsChannelPage(
        team_remote_id=_TEAM_ID,
        items=(),
        continuation=continuation,
    )
    assert page.has_more is True


def test_page_duplicate_channel_ids_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphTeamsChannelPage(
            team_remote_id=_TEAM_ID,
            items=(
                _valid_channel(remote_id="same"),
                _valid_channel(remote_id="same", membership_type=MsGraphTeamsChannelMembershipType.PRIVATE),
            ),
        )


def test_page_cross_team_item_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphTeamsChannelPage(
            team_remote_id=_TEAM_ID,
            items=(_valid_channel(team_remote_id=_OTHER_TEAM_ID),),
        )


def test_page_items_as_list_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphTeamsChannelPage(
            team_remote_id=_TEAM_ID,
            items=[_valid_channel()],  # type: ignore[arg-type]
        )


def test_page_item_wrong_type_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphTeamsChannelPage(
            team_remote_id=_TEAM_ID,
            items=("not-a-chat",),  # type: ignore[arg-type]
        )


def test_page_delta_continuation_rejected() -> None:
    delta = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_next_link(),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphTeamsChannelPage(
            team_remote_id=_TEAM_ID,
            items=(),
            continuation=delta,
        )


def test_token_hidden_from_repr() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(),
    )
    page = MsGraphTeamsChannelPage(
        team_remote_id=_TEAM_ID,
        items=(),
        continuation=continuation,
    )
    assert _SECRET_TOKEN not in repr(page)
    assert _SECRET_TOKEN not in repr(continuation)


@pytest.mark.parametrize(
    "chat_kwargs",
    [
        {"remote_id": None},
        {"membership_type": "group"},
        {"is_archived": "yes"},
        {"created_at": "2026-01-01T00:00:00Z"},
    ],
)
def test_malformed_channel_model_construct_rejected(
    chat_kwargs: dict[str, object],
) -> None:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "remote_id": _CHANNEL_ID,
        "display_name": _DISPLAY_NAME,
        "description": None,
        "tenant_id": None,
        "created_at": _CREATED_AT,
        "membership_type": MsGraphTeamsChannelMembershipType.STANDARD,
        "is_archived": False,
    }
    defaults.update(chat_kwargs)
    malformed = MsGraphTeamsChannel.model_construct(**defaults)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_teams_channel(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_missing_items() -> None:
    malformed = MsGraphTeamsChannelPage.model_construct(team_remote_id=_TEAM_ID)
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_items_as_list() -> None:
    malformed = MsGraphTeamsChannelPage.model_construct(
        team_id=_TEAM_ID,
        items=[_valid_channel()],
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_malformed_chat() -> None:
    bad_chat = MsGraphTeamsChannel.model_construct(
        team_remote_id="bad\x00id",
        remote_id=_CHANNEL_ID,
        display_name=_DISPLAY_NAME,
        membership_type=MsGraphTeamsChannelMembershipType.STANDARD,
        description=None,
        tenant_id=None,
        created_at=_CREATED_AT,
        is_archived=False,
    )
    malformed = MsGraphTeamsChannelPage.model_construct(
        team_id=_TEAM_ID,
        items=(bad_chat,),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_malformed_continuation() -> None:
    bad_continuation = MsGraphKnowledgeContinuation.model_construct(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_next_link(),
    )
    malformed = MsGraphTeamsChannelPage.model_construct(
        team_remote_id=_TEAM_ID,
        items=(_valid_channel(),),
        continuation=bad_continuation,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    _assert_safe_provider_error(exc)


def test_validate_page_rejects_wrong_team_id() -> None:
    page = MsGraphTeamsChannelPage(team_remote_id=_TEAM_ID, items=(_valid_channel(),))
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        validate_msgraph_teams_channel_page(
            page,
            team_id=_OTHER_TEAM_ID,
            graph_base_url=_GRAPH_BASE,
        )


# --- request tests ---


def test_request_path_select_and_prefer_header() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    _reader(http).read_teams_channels_page(
        team_id=_TEAM_ID,
        continuation=None,
    )
    call = http.get.call_args
    assert call.args[0] == _ROOT_PATH
    assert call.kwargs["params"]["$select"] == _CHANNEL_SELECT
    assert call.kwargs["headers"] == _PREFER_HEADER
    assert "$top" not in call.kwargs["params"]
    assert "$expand" not in call.kwargs["params"]
    assert "$filter" not in call.kwargs["params"]
    assert "/messages" not in call.args[0]


def test_empty_page_request() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    page = _reader(http).read_teams_channels_page(
        team_id=_TEAM_ID,
        continuation=None,
    )
    assert page.items == ()
    assert page.has_more is False


def test_paging_request_returns_continuation() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(
            value=[_channel_payload()],
            next_link=_next_link(),
        )
    )
    page = _reader(http).read_teams_channels_page(
        team_id=_TEAM_ID,
        continuation=None,
    )
    assert len(page.items) == 1
    assert page.has_more is True
    assert page.continuation is not None


def test_continuation_request_uses_full_url_without_params() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(),
    )
    _reader(http).read_teams_channels_page(
        team_id=_TEAM_ID,
        continuation=continuation,
    )
    assert http.get.call_args.args[0] == _next_link()
    assert "params" not in http.get.call_args.kwargs
    assert http.get.call_args.kwargs["headers"] == _PREFER_HEADER



@pytest.mark.parametrize("team_id", ["", "  ", "bad\x00id", 123])
def test_invalid_team_id_rejected_before_http(team_id: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_teams_channels_page(
            team_id=team_id,  # type: ignore[arg-type]
            continuation=None,
        )
    http.get.assert_not_called()


# --- continuation tests ---


def test_validate_continuation_same_user_slash_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(),
    )
    validated = validate_msgraph_teams_channels_continuation(
        continuation,
        team_id=_TEAM_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation


def test_validate_continuation_same_user_odata_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(
            path=(
                f"https://graph.microsoft.com/v1.0/teams('{_TEAM_ID}')/channels"
            )
        ),
    )
    validated = validate_msgraph_teams_channels_continuation(
        continuation,
        team_id=_TEAM_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation


def test_validate_continuation_case_insensitive_resource_names() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(
            path=f"https://graph.microsoft.com/v1.0/Teams/{_QUOTED_TEAM}/channels"
        ),
    )
    validated = validate_msgraph_teams_channels_continuation(
        continuation,
        team_id=_TEAM_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation


@pytest.mark.parametrize(
    "url",
    [
        _next_link(
            path=(
                f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_OTHER_TEAM}/channels"
            )
        ),
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/messages?$skiptoken={_SECRET_TOKEN}",
        (
            f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/mailFolders?"
            f"$skiptoken={_SECRET_TOKEN}"
        ),
        (
            f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
            f"{quote(_CHANNEL_ID, safe='')}/messages?$skiptoken={_SECRET_TOKEN}"
        ),
        (
            f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/delta?"
            f"$deltatoken={_SECRET_TOKEN}"
        ),
        (
            f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
            f"extra?$skiptoken={_SECRET_TOKEN}"
        ),
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/allChannels?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/incomingChannels?$skiptoken={_SECRET_TOKEN}",
        "https://graph.microsoft.com/v1.0/drives/drive-1/root/delta?$skiptoken=x",
    ],
)
def test_rejects_invalid_chats_continuation(url: str) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=url,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_teams_channels_continuation(
            continuation,
        team_id=_TEAM_ID,
        graph_base_url=_GRAPH_BASE,
        )
    assert _SECRET_TOKEN not in str(exc.value)
    assert _TEAM_ID not in str(exc.value)
    assert exc.value.__cause__ is None


def test_delta_continuation_rejected_in_validator() -> None:
    delta = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_next_link(),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        validate_msgraph_teams_channels_continuation(
            delta,
        team_id=_TEAM_ID,
        graph_base_url=_GRAPH_BASE,
        )


def test_invalid_continuation_rejected_before_http() -> None:
    http = MagicMock()
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(
            path=(
                f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_OTHER_TEAM}/channels"
            )
        ),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        _reader(http).read_teams_channels_page(
            team_id=_TEAM_ID,
            continuation=continuation,
        )
    http.get.assert_not_called()


# --- delegation ---


def test_graph_rest_client_delegates_teams_chats() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(value=[_channel_payload()])
    )
    page = _graph_client(http).read_teams_channels_page(team_id=_TEAM_ID)
    assert len(page.items) == 1
    assert page.items[0].remote_id == _CHANNEL_ID


def test_collaboration_suite_delegates_teams_chats() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    suite = _Ms365GraphCollaborationSuite(_graph_client(http))
    page = suite.read_teams_channels_page(team_id=_TEAM_ID)
    assert page.items == ()


def test_integration_delegates_teams_chats() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(_graph_client(http)),
        enabled=True,
    )
    page = integration.read_teams_channels_page(team_id=_TEAM_ID)
    assert page.items == ()


def test_transport_and_reader_share_injected_http_client() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    client = _graph_client(http)
    client.read_teams_channels_page(team_id=_TEAM_ID)
    assert client._knowledge_transport._http_client is http
    assert client._teams_channels_reader._transport._http_client is http
    http.get.assert_called_once()


def test_no_new_http_client_created() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    client = _graph_client(http)
    client.read_teams_channels_page(team_id=_TEAM_ID)
    assert client._http_client is http


def test_existing_drive_operations_still_work() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload={
            "value": [],
            "@odata.deltaLink": (
                f"https://graph.microsoft.com/v1.0/drives/{quote('drive-1', safe='')}/root/delta?"
                "$deltatoken=tok"
            ),
        }
    )
    client = _graph_client(http)
    page = client.read_drive_delta_page(drive_id="drive-1", limit=10)
    assert page.is_complete is True


def test_existing_list_messages_still_works() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload={"value": []})
    client = _graph_client(http)
    result = client.list_messages(_TEAM_ID, folder="inbox", limit=5)
    assert result.messages == []


def test_existing_get_message_still_works() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload={
            "id": "msg-1",
            "subject": "Hello",
            "bodyPreview": "Preview",
            "from": None,
            "receivedDateTime": "2026-01-01T00:00:00Z",
        }
    )
    client = _graph_client(http)
    message = client.get_message(_TEAM_ID, "msg-1")
    assert message.id == "msg-1"


class _CustomSuiteWithoutChannels(CollaborationSuite):
    def get_message(self, user_id: str, message_id: str):
        raise NotImplementedError

    def list_messages(self, user_id: str, *, folder: str = "inbox", limit: int = 25):
        raise NotImplementedError

    def send_mail(self, user_id: str, *, subject: str, body: str, to):
        raise NotImplementedError

    def list_calendar_events(self, user_id: str, *, start: str, end: str, limit: int = 50):
        raise NotImplementedError

    def get_user(self, user_id: str):
        raise NotImplementedError

    def reply_message(self, user_id: str, message_id: str, *, body: str) -> None:
        raise NotImplementedError

    def create_event(
        self,
        user_id: str,
        *,
        subject: str,
        start: str,
        end: str,
        location: str = "",
        attendees=(),
    ):
        raise NotImplementedError


class _CustomGraphChannelsClient(GraphRestClient):
    def __init__(self, page: MsGraphTeamsChannelPage, http: MagicMock) -> None:
        super().__init__(_config(), http_client=http)
        self._custom_page = page

    def read_teams_channels_page(
        self,
        *,
        team_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
            ) -> MsGraphTeamsChannelPage:
        return self._custom_page


class _CustomChannelsSuite(CollaborationSuite):
    def __init__(self, page: MsGraphTeamsChannelPage) -> None:
        self._page = page

    def read_teams_channels_page(
        self,
        *,
        team_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
            ) -> MsGraphTeamsChannelPage:
        return self._page

    def get_message(self, user_id: str, message_id: str):
        raise NotImplementedError

    def list_messages(self, user_id: str, *, folder: str = "inbox", limit: int = 25):
        raise NotImplementedError

    def send_mail(self, user_id: str, *, subject: str, body: str, to):
        raise NotImplementedError

    def list_calendar_events(self, user_id: str, *, start: str, end: str, limit: int = 50):
        raise NotImplementedError

    def get_user(self, user_id: str):
        raise NotImplementedError

    def reply_message(self, user_id: str, message_id: str, *, body: str) -> None:
        raise NotImplementedError

    def create_event(
        self,
        user_id: str,
        *,
        subject: str,
        start: str,
        end: str,
        location: str = "",
        attendees=(),
    ):
        raise NotImplementedError


def test_custom_client_without_channels_capability_fails() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomSuiteWithoutChannels(),
        enabled=True,
    )
    with pytest.raises(
        IntegrationConfigurationError,
        match="Microsoft Graph integration does not expose Teams channels capability",
    ):
        integration.read_teams_channels_page(team_id=_TEAM_ID)


def test_custom_client_malformed_page_rejected() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphChannelsClient(
                page=MsGraphTeamsChannelPage.model_construct(team_remote_id=_TEAM_ID),
                http=MagicMock(),
            )
        ),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        integration.read_teams_channels_page(team_id=_TEAM_ID)
    assert exc.value.__cause__ is None


def test_custom_client_valid_page_revalidated() -> None:
    supplied = MsGraphTeamsChannelPage(
        team_remote_id=_TEAM_ID,
        items=(_valid_channel(),),
    )
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphChannelsClient(page=supplied, http=MagicMock())
        ),
        enabled=True,
    )
    returned = integration.read_teams_channels_page(team_id=_TEAM_ID)
    assert returned == supplied
    assert returned is not supplied
    assert returned.items[0] is not supplied.items[0]


def test_custom_client_validation_not_configured() -> None:
    page = MsGraphTeamsChannelPage(team_remote_id=_TEAM_ID, items=())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomChannelsSuite(page=page),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_VALIDATION_ERROR):
        integration._graph_base_url_for_teams_channel_validation()


class _CountingChannelsClient(GraphRestClient):
    def __init__(self, page: MsGraphTeamsChannelPage, http: MagicMock) -> None:
        super().__init__(_config(), http_client=http)
        self._custom_page = page
        self.call_count = 0
        self.last_continuation: MsGraphKnowledgeContinuation | None = None

    def read_teams_channels_page(
        self,
        *,
        team_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
            ) -> MsGraphTeamsChannelPage:
        self.call_count += 1
        self.last_continuation = continuation
        return self._custom_page


@pytest.mark.parametrize(
    "continuation",
    [
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        ),
        MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=_next_link(path=f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/delta"),
        ),
        MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=_next_link(
                path=f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_OTHER_TEAM}/channels"
            ),
        ),
    ],
)
def test_integration_rejects_malformed_continuation_before_custom_call(
    continuation: MsGraphKnowledgeContinuation,
) -> None:
    page = MsGraphTeamsChannelPage(team_remote_id=_TEAM_ID, items=(_valid_channel(),))
    client = _CountingChannelsClient(page=page, http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        integration.read_teams_channels_page(
            team_id=_TEAM_ID,
            continuation=continuation,
        )
    assert client.call_count == 0
    assert _SECRET_TOKEN not in str(exc.value)


def test_integration_valid_continuation_calls_custom_client_once() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(path=f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels"),
    )
    page = MsGraphTeamsChannelPage(team_remote_id=_TEAM_ID, items=(_valid_channel(),))
    client = _CountingChannelsClient(page=page, http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    returned = integration.read_teams_channels_page(
        team_id=_TEAM_ID,
        continuation=continuation,
    )
    assert client.call_count == 1
    assert client.last_continuation == continuation
    assert client.last_continuation is not continuation
    assert client.last_continuation is not None
    assert client.last_continuation.url == continuation.url
    assert returned.items[0] is not page.items[0]


# --- security ---


def test_security_channel_repr_and_errors() -> None:
    channel = _valid_channel(description=_HIDDEN_DISPLAY_NAME, tenant_id=_TENANT_ID)
    assert _HIDDEN_DISPLAY_NAME not in repr(channel)
    assert _TENANT_ID not in repr(channel)
    assert _CHANNEL_ID in repr(channel)

    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR) as exc:
        _reader(MagicMock()).read_teams_channels_page(
            team_id="",
            continuation=None,
        )
    assert _TEAM_ID not in str(exc.value)

    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_teams_channels_continuation(
            MsGraphKnowledgeContinuation(
                kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_next_link(
                    path=(
                        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_OTHER_TEAM}/channels"
                    )
                ),
            ),
            team_id=_TEAM_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert _SECRET_TOKEN not in str(exc.value)
    assert _TEAM_ID not in str(exc.value)
