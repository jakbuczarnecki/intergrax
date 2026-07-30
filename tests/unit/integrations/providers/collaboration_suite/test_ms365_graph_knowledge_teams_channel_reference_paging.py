# © Artur Czarnecki. All rights reserved.

"""Reference-based Teams channel root and reply paging tests."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.collaboration_suite import CollaborationSuite
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    DEFAULT_GRAPH_BASE_URL,
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeTransport,
    MsGraphTeamsChannelBodyKind,
    MsGraphTeamsChannelImportance,
    MsGraphTeamsChannelMessage,
    MsGraphTeamsChannelMessageType,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (
    MsGraphTeamsChannelReference,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_messages import (
    MsGraphTeamsChannelMessageChanged,
    MsGraphTeamsChannelMessageKind,
    MsGraphTeamsChannelMessagesReader,
    MsGraphTeamsChannelMessageState,
    MsGraphTeamsChannelRootMessageReference,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_TEAM_ID = "team-abc-123"
_OTHER_TEAM_ID = "other-team-456"
_CHANNEL_ID = "channel-abc-123"
_OTHER_CHANNEL_ID = "other-channel-456"
_MESSAGE_ID = "root-msg-001"
_ROOT_MESSAGE_ID = "root-msg-001"
_REPLY_MESSAGE_ID = "reply-msg-002"
_OTHER_ROOT_MESSAGE_ID = "other-root-999"
_REPLY_ETAG = "reply-etag-2"
_ETAG = "etag-1"
_SECRET_TOKEN = "secret-skiptoken-value"
_QUOTED_TEAM_ID = quote(_TEAM_ID, safe="")
_QUOTED_OTHER_TEAM_ID = quote(_OTHER_TEAM_ID, safe="")
_QUOTED_CHANNEL = quote(_CHANNEL_ID, safe="")
_QUOTED_ROOT_MESSAGE_ID = quote(_ROOT_MESSAGE_ID, safe="")
_QUOTED_REPLY_MESSAGE_ID = quote(_REPLY_MESSAGE_ID, safe="")
_ROOT_MESSAGES_PATH = f"/teams/{_QUOTED_TEAM_ID}/channels/{_QUOTED_CHANNEL}/messages"
_ROOT_OBSERVATION_PATH = (
    f"/teams/{_QUOTED_TEAM_ID}/channels/{_QUOTED_CHANNEL}/messages/{_QUOTED_ROOT_MESSAGE_ID}"
)
_REPLIES_COLLECTION_PATH = f"{_ROOT_OBSERVATION_PATH}/replies"
_PREFER = {"Prefer": "include-unknown-enum-members"}
_REQUEST_ERROR = "invalid Microsoft Graph Teams channel messages request"
_CONT = "invalid Microsoft Graph Teams channel messages continuation"
_SAFE = "unexpected Microsoft Graph Teams channel messages response"
_REF_PAGING_ERROR = (
    "Microsoft Graph integration does not expose Teams channel reference paging capability"
)


def _config() -> Ms365GraphIntegrationConfig:
    return Ms365GraphIntegrationConfig(
        tenant_id="tenant-123",
        client_id="client-456",
        client_secret="secret",
        graph_base_url=_GRAPH_BASE,
    )


def _channel_reference() -> MsGraphTeamsChannelReference:
    return MsGraphTeamsChannelReference(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
    )


def _root_reference(**overrides: object) -> MsGraphTeamsChannelRootMessageReference:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "remote_id": _MESSAGE_ID,
        "revision": _ETAG,
        "state": MsGraphTeamsChannelMessageState.ACTIVE,
    }
    defaults.update(overrides)
    return MsGraphTeamsChannelRootMessageReference(**defaults)


def _valid_active_message(**overrides: object) -> MsGraphTeamsChannelMessage:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "thread_root_remote_id": _MESSAGE_ID,
        "message_kind": MsGraphTeamsChannelMessageKind.ROOT,
        "remote_id": _MESSAGE_ID,
        "revision": _ETAG,
        "state": MsGraphTeamsChannelMessageState.ACTIVE,
        "message_type": MsGraphTeamsChannelMessageType.MESSAGE,
        "importance": MsGraphTeamsChannelImportance.NORMAL,
        "created_at": datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        "last_modified_at": datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
        "body_kind": MsGraphTeamsChannelBodyKind.TEXT,
        "body_content": "Hello",
    }
    defaults.update(overrides)
    return MsGraphTeamsChannelMessage(**defaults)


def _active_message_payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": _MESSAGE_ID,
        "channelIdentity": {"teamId": _TEAM_ID, "channelId": _CHANNEL_ID},
        "etag": _ETAG,
        "messageType": "message",
        "createdDateTime": "2024-01-01T10:00:00Z",
        "lastModifiedDateTime": "2024-01-01T11:00:00Z",
        "deletedDateTime": None,
        "importance": "normal",
        "body": {"contentType": "text", "content": "Hello"},
        "from": {"user": {"id": "u1", "displayName": "Alice"}},
        "attachments": [],
        "mentions": [],
        "reactions": [],
    }
    base.update(overrides)
    return base


def _deleted_root_observation_payload() -> dict[str, Any]:
    return {
        "id": _MESSAGE_ID,
        "channelIdentity": {"teamId": _TEAM_ID, "channelId": _CHANNEL_ID},
        "etag": _ETAG,
        "messageType": "message",
        "createdDateTime": "2024-01-01T10:00:00Z",
        "lastModifiedDateTime": "2024-01-01T12:00:00Z",
        "deletedDateTime": "2024-01-01T12:00:00Z",
        "importance": "normal",
    }


def _active_reply_payload(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": _REPLY_MESSAGE_ID,
        "replyToId": _ROOT_MESSAGE_ID,
        "channelIdentity": {"teamId": _TEAM_ID, "channelId": _CHANNEL_ID},
        "etag": _REPLY_ETAG,
        "messageType": "message",
        "createdDateTime": "2024-01-01T10:00:00Z",
        "lastModifiedDateTime": "2024-01-01T11:00:00Z",
        "deletedDateTime": None,
        "importance": "normal",
        "body": {"contentType": "text", "content": "Reply body"},
        "from": {"user": {"id": "u2", "displayName": "Bob"}},
        "attachments": [],
        "mentions": [],
        "reactions": [],
    }
    base.update(overrides)
    return base


def _json_response(payload: object) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    return response


def _reader(http: MagicMock) -> MsGraphTeamsChannelMessagesReader:
    return MsGraphTeamsChannelMessagesReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
    )


def _root_next_link() -> str:
    return (
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM_ID}/channels/"
        f"{_QUOTED_CHANNEL}/messages?$skiptoken={_SECRET_TOKEN}"
    )


def _replies_next_link() -> str:
    return (
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM_ID}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{_QUOTED_ROOT_MESSAGE_ID}/replies?$skiptoken={_SECRET_TOKEN}"
    )


def _setup_replies_page_http(
    http: MagicMock,
    *,
    root_observation: dict[str, object] | None = None,
    reply_items: list[dict[str, object]] | None = None,
    next_link: str | None = None,
) -> None:
    observation = root_observation if root_observation is not None else _active_message_payload()
    replies_payload: dict[str, object] = {
        "value": reply_items if reply_items is not None else [_active_reply_payload()],
    }
    if next_link is not None:
        replies_payload["@odata.nextLink"] = next_link
    http.get.side_effect = [
        _json_response(payload=observation),
        _json_response(payload=replies_payload),
        _json_response(payload=observation),
    ]


# --- root paging by reference ---


def test_root_paging_by_reference_initial_success() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload={"value": [_active_message_payload()]},
    )
    page = _reader(http).read_teams_channel_root_messages_page_by_reference(
        channel=_channel_reference(),
        continuation=None,
        limit=25,
        max_chars_per_message=1000,
    )
    assert len(page.items) == 1
    http.get.assert_called_once()
    assert http.get.call_args.args[0] == _ROOT_MESSAGES_PATH
    assert http.get.call_args.kwargs["params"] == {"$top": 25}
    assert http.get.call_args.kwargs["headers"] == _PREFER


def test_root_paging_by_reference_url_quoting() -> None:
    opaque_team = "team/with/slash"
    opaque_channel = "channels/messages/allMembers"
    http = MagicMock()
    http.get.return_value = _json_response(payload={"value": []})
    _reader(http).read_teams_channel_root_messages_page_by_reference(
        channel=MsGraphTeamsChannelReference(
            team_remote_id=opaque_team,
            channel_remote_id=opaque_channel,
        ),
        continuation=None,
        limit=50,
        max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    expected = (
        f"/teams/{quote(opaque_team, safe='')}/channels/"
        f"{quote(opaque_channel, safe='')}/messages"
    )
    assert http.get.call_args.args[0] == expected


def test_root_paging_by_reference_continuation() -> None:
    http = MagicMock()
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_root_next_link(),
    )
    http.get.return_value = _json_response(payload={"value": [_active_message_payload()]})
    page = _reader(http).read_teams_channel_root_messages_page_by_reference(
        channel=_channel_reference(),
        continuation=continuation,
        limit=50,
        max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert len(page.items) == 1
    http.get.assert_called_once()
    assert http.get.call_args.args[0] == continuation.url


@pytest.mark.parametrize("limit", [0, 51, True, False])
def test_root_paging_by_reference_invalid_limit_before_http(limit: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_teams_channel_root_messages_page_by_reference(
            channel=_channel_reference(),
            continuation=None,
            limit=limit,  # type: ignore[arg-type]
            max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )
    http.get.assert_not_called()


def test_root_paging_by_reference_malformed_channel_reference_before_http() -> None:
    http = MagicMock()
    bad = MsGraphTeamsChannelReference.model_construct(
        team_remote_id="",
        channel_remote_id=_CHANNEL_ID,
    )
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_teams_channel_root_messages_page_by_reference(
            channel=bad,
            continuation=None,
            limit=50,
            max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )
    http.get.assert_not_called()


def test_root_paging_by_reference_team_mismatch_rejected() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload={
            "value": [
                _active_message_payload(
                    channelIdentity={"teamId": _OTHER_TEAM_ID, "channelId": _CHANNEL_ID},
                ),
            ],
        },
    )
    with pytest.raises(ValueError, match=_SAFE):
        _reader(http).read_teams_channel_root_messages_page_by_reference(
            channel=_channel_reference(),
            continuation=None,
            limit=50,
            max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )


def test_root_paging_by_reference_duplicate_root_ids_last_wins() -> None:
    http = MagicMock()
    second = _active_message_payload(body={"contentType": "text", "content": "Second"})
    http.get.return_value = _json_response(payload={"value": [_active_message_payload(), second]})
    page = _reader(http).read_teams_channel_root_messages_page_by_reference(
        channel=_channel_reference(),
        continuation=None,
        limit=50,
        max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert len(page.items) == 1
    assert page.items[0].body_content == "Second"


class _SuiteWithoutReferencePaging(CollaborationSuite):
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


def test_integration_missing_reference_paging_capability() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _SuiteWithoutReferencePaging(),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_REF_PAGING_ERROR):
        integration.read_teams_channel_root_messages_page_by_reference(
            channel=_channel_reference(),
            continuation=None,
            limit=50,
            max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )


# --- reply paging by reference ---


def test_reply_paging_by_reference_active_root_sequence() -> None:
    http = MagicMock()
    _setup_replies_page_http(http)
    page = _reader(http).read_teams_channel_replies_page_by_reference(
        root_message=_root_reference(),
        continuation=None,
        limit=25,
        max_chars_per_message=1000,
    )
    assert http.get.call_count == 3
    assert http.get.call_args_list[0].args[0] == _ROOT_OBSERVATION_PATH
    assert http.get.call_args_list[1].args[0] == _REPLIES_COLLECTION_PATH
    assert http.get.call_args_list[2].args[0] == _ROOT_OBSERVATION_PATH
    assert page.items[0].thread_root_remote_id == _ROOT_MESSAGE_ID
    assert page.root_message_revision == _ETAG


def test_reply_paging_by_reference_deleted_root() -> None:
    http = MagicMock()
    deleted_observation = _deleted_root_observation_payload()
    _setup_replies_page_http(http, root_observation=deleted_observation)
    page = _reader(http).read_teams_channel_replies_page_by_reference(
        root_message=_root_reference(state=MsGraphTeamsChannelMessageState.DELETED),
        continuation=None,
        limit=50,
        max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert len(page.items) == 1
    assert http.get.call_count == 3


def test_reply_paging_root_changed_before_replies_no_reply_request() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_active_message_payload(etag="changed-etag"),
    )
    with pytest.raises(MsGraphTeamsChannelMessageChanged):
        _reader(http).read_teams_channel_replies_page_by_reference(
            root_message=_root_reference(),
            continuation=None,
            limit=50,
            max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )
    http.get.assert_called_once()


def test_reply_paging_root_changed_after_replies_page_discarded() -> None:
    http = MagicMock()
    observation = _active_message_payload()
    changed_observation = _active_message_payload(etag="changed-after")
    http.get.side_effect = [
        _json_response(payload=observation),
        _json_response(payload={"value": [_active_reply_payload()]}),
        _json_response(payload=changed_observation),
    ]
    with pytest.raises(MsGraphTeamsChannelMessageChanged):
        _reader(http).read_teams_channel_replies_page_by_reference(
            root_message=_root_reference(),
            continuation=None,
            limit=50,
            max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )
    assert http.get.call_count == 3


def test_reply_paging_continuation_for_other_root_rejected_before_continuation_http() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_active_message_payload())
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM_ID}/channels/"
            f"{_QUOTED_CHANNEL}/messages/{quote(_OTHER_ROOT_MESSAGE_ID, safe='')}"
            f"/replies?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT):
        _reader(http).read_teams_channel_replies_page_by_reference(
            root_message=_root_reference(),
            continuation=continuation,
            limit=50,
            max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )
    http.get.assert_called_once()


def test_reply_paging_wrong_thread_root_in_response_rejected() -> None:
    http = MagicMock()
    _setup_replies_page_http(
        http,
        reply_items=[_active_reply_payload(replyToId=_OTHER_ROOT_MESSAGE_ID)],
    )
    with pytest.raises(ValueError, match=_SAFE):
        _reader(http).read_teams_channel_replies_page_by_reference(
            root_message=_root_reference(),
            continuation=None,
            limit=50,
            max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )


def test_reply_paging_root_item_in_replies_endpoint_rejected() -> None:
    http = MagicMock()
    _setup_replies_page_http(http, reply_items=[_active_message_payload()])
    with pytest.raises(ValueError, match=_SAFE):
        _reader(http).read_teams_channel_replies_page_by_reference(
            root_message=_root_reference(),
            continuation=None,
            limit=50,
            max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
        )


def test_backward_compat_full_channel_root_paging_still_works() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload={"value": [_active_message_payload()]})
    from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (
        parse_msgraph_teams_channel,
    )

    channel = parse_msgraph_teams_channel(
        {
            "id": _CHANNEL_ID,
            "displayName": "General",
            "membershipType": "standard",
            "isArchived": False,
            "createdDateTime": "2024-01-01T10:00:00Z",
        },
        expected_team_id=_TEAM_ID,
    )
    page = _reader(http).read_teams_channel_root_messages_page(
        channel=channel,
        continuation=None,
        limit=50,
        max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert len(page.items) == 1
    assert http.get.call_args.args[0] == _ROOT_MESSAGES_PATH


def test_backward_compat_full_root_message_replies_still_works() -> None:
    http = MagicMock()
    _setup_replies_page_http(http)
    page = _reader(http).read_teams_channel_replies_page(
        root_message=_valid_active_message(),
        continuation=None,
        limit=50,
        max_chars_per_message=DEFAULT_TEAMS_CHANNEL_MESSAGE_MAX_CHARS,
    )
    assert len(page.items) == 1
    assert http.get.call_count == 3
