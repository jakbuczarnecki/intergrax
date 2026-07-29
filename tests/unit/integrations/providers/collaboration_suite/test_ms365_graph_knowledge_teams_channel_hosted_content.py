# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Microsoft Graph Teams Channel knowledge-read hosted content surface."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
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
    ABSOLUTE_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES,
    DEFAULT_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeTransport,
    MsGraphTeamsChannelBodyKind,
    MsGraphTeamsChannelHostedContent,
    MsGraphTeamsChannelHostedContentBytes,
    MsGraphTeamsChannelHostedContentPage,
    MsGraphTeamsChannelHostedContentReader,
    MsGraphTeamsChannelHostedContentTooLarge,
    MsGraphTeamsChannelImportance,
    MsGraphTeamsChannelMessage,
    MsGraphTeamsChannelMessageChanged,
    MsGraphTeamsChannelMessageState,
    MsGraphTeamsChannelMessageKind,
    MsGraphTeamsChannelMessageType,
    parse_msgraph_teams_channel_hosted_content,
    validate_msgraph_teams_channel_hosted_content,
    validate_msgraph_teams_channel_hosted_content_bytes,
    validate_msgraph_teams_channel_hosted_content_page,
    validate_msgraph_teams_channel_hosted_contents_continuation,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_TEAM_ID = "team-abc-123"
_OTHER_TEAM_ID = "other@contoso.com"
_CHANNEL_ID = "channel-abc-123"
_OTHER_CHANNEL_ID = "other-channel-456"
_MESSAGE_ID = "root-msg-001"
_ROOT_MESSAGE_ID = "root-msg-001"
_REPLY_MESSAGE_ID = "reply-msg-002"
_OTHER_ROOT_MESSAGE_ID = "other-root-999"
_OTHER_MESSAGE_ID = "reply-msg-002"
_REVISION = "etag-secret-value"
_REPLY_REVISION = "reply-etag-value"
_OTHER_REVISION = "other-etag-value"
_HOSTED_CONTENT_ID = "hosted-content-001"
_OTHER_HOSTED_CONTENT_ID = "hosted-other-002"
_SECRET_TOKEN = "secret-skiptoken-value"
_QUOTED_TEAM = quote(_TEAM_ID, safe="")
_QUOTED_CHANNEL = quote(_CHANNEL_ID, safe="")
_QUOTED_OTHER_CHANNEL = quote(_OTHER_CHANNEL_ID, safe="")
_QUOTED_MESSAGE_ID = quote(_MESSAGE_ID, safe="")
_QUOTED_ROOT_MESSAGE_ID = quote(_ROOT_MESSAGE_ID, safe="")
_QUOTED_REPLY_MESSAGE_ID = quote(_REPLY_MESSAGE_ID, safe="")
_QUOTED_OTHER_MESSAGE_ID = quote(_OTHER_MESSAGE_ID, safe="")
_QUOTED_HOSTED_CONTENT_ID = quote(_HOSTED_CONTENT_ID, safe="")
_HOSTED_CONTENTS_PATH = (
    f"/teams/{_QUOTED_TEAM}/channels/{_QUOTED_CHANNEL}/messages/"
    f"{_QUOTED_MESSAGE_ID}/hostedContents"
)
_VALUE_PATH = (
    f"/teams/{_QUOTED_TEAM}/channels/{_QUOTED_CHANNEL}/messages/{_QUOTED_MESSAGE_ID}"
    f"/hostedContents/{_QUOTED_HOSTED_CONTENT_ID}/$value"
)
_OBSERVATION_PATH = (
    f"/teams/{_QUOTED_TEAM}/channels/{_QUOTED_CHANNEL}/messages/{_QUOTED_MESSAGE_ID}"
)
_REPLY_OBSERVATION_PATH = (
    f"/teams/{_QUOTED_TEAM}/channels/{_QUOTED_CHANNEL}/messages/{_QUOTED_ROOT_MESSAGE_ID}"
    f"/replies/{_QUOTED_REPLY_MESSAGE_ID}"
)
_REPLY_HOSTED_CONTENTS_PATH = (
    f"/teams/{_QUOTED_TEAM}/channels/{_QUOTED_CHANNEL}/messages/{_QUOTED_ROOT_MESSAGE_ID}"
    f"/replies/{_QUOTED_REPLY_MESSAGE_ID}/hostedContents"
)
_REPLY_VALUE_PATH = (
    f"/teams/{_QUOTED_TEAM}/channels/{_QUOTED_CHANNEL}/messages/{_QUOTED_ROOT_MESSAGE_ID}"
    f"/replies/{_QUOTED_REPLY_MESSAGE_ID}/hostedContents/{_QUOTED_HOSTED_CONTENT_ID}/$value"
)
_PREFER_HEADER = {"Prefer": "include-unknown-enum-members"}
_SAFE_ERROR = "unexpected Microsoft Graph Teams hosted content response"
_REQUEST_ERROR = "invalid Microsoft Graph Teams hosted content request"
_CONT_ERROR = "invalid Microsoft Graph Teams hosted content continuation"
_INVALID_RESPONSE = "Microsoft Graph Teams hosted content response is invalid"
_CAPABILITY_ERROR = (
    "Microsoft Graph integration does not expose Teams channel hosted content capability"
)
_VALIDATION_ERROR = "Microsoft Graph Teams Channel validation is not configured"


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


def _hosted_contents_next_link() -> str:
    return (
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{quote(_MESSAGE_ID, safe='')}/hostedContents?$skiptoken={_SECRET_TOKEN}"
    )


def _odata_hosted_contents_next_link(
    message_id: str = _MESSAGE_ID,
    channel_id: str = _CHANNEL_ID,
    team_id: str = _TEAM_ID,
) -> str:
    escaped_message = message_id.replace("'", "''")
    escaped_channel = channel_id.replace("'", "''")
    escaped_team = team_id.replace("'", "''")
    return (
        f"https://graph.microsoft.com/v1.0/teams('{escaped_team}')/channels('{escaped_channel}')"
        f"/messages('{escaped_message}')/hostedContents?$skiptoken={_SECRET_TOKEN}"
    )


def _slash_hosted_contents_next_link(message_id: str, channel_id: str = _CHANNEL_ID) -> str:
    quoted_message = quote(message_id, safe="")
    quoted_channel = quote(channel_id, safe="")
    return (
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
        f"{quoted_channel}/messages/{quoted_message}/hostedContents?$skiptoken={_SECRET_TOKEN}"
    )


def _odata_percent_encoded_hosted_contents_next_link(
    message_id: str,
    channel_id: str = _CHANNEL_ID,
    team_id: str = _TEAM_ID,
) -> str:
    escaped_message = message_id.replace("'", "''")
    escaped_channel = channel_id.replace("'", "''")
    escaped_team = team_id.replace("'", "''")
    encoded_message = quote(escaped_message, safe="")
    encoded_channel = quote(escaped_channel, safe="")
    encoded_team = quote(escaped_team, safe="")
    return (
        f"https://graph.microsoft.com/v1.0/teams('{encoded_team}')/channels('{encoded_channel}')"
        f"/messages('{encoded_message}')/hostedContents?$skiptoken={_SECRET_TOKEN}"
    )


def _page_payload(
    *,
    value: list[dict[str, Any]] | None = None,
    next_link: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"value": [] if value is None else value}
    if next_link is not None:
        payload["@odata.nextLink"] = next_link
    return payload


def _observation_payload(
    *,
    message_id: str = _MESSAGE_ID,
    channel_id: str = _CHANNEL_ID,
    revision: str = _REVISION,
    deleted: bool = False,
) -> dict[str, Any]:
    return {
        "id": message_id,
        "channelIdentity": {"teamId": _TEAM_ID, "channelId": channel_id},
        "etag": revision,
        "deletedDateTime": "2024-06-01T12:00:00Z" if deleted else None,
    }


def _hosted_content_payload(
    *,
    hosted_id: str = _HOSTED_CONTENT_ID,
    include_content_bytes_key: bool = False,
    include_content_type_key: bool = False,
    content_bytes_null: bool = True,
    content_type_null: bool = True,
    content_bytes: object = "base64payload",
    content_type: object = "image/png",
) -> dict[str, Any]:
    payload: dict[str, Any] = {"id": hosted_id}
    if include_content_bytes_key:
        payload["contentBytes"] = None if content_bytes_null else content_bytes
    if include_content_type_key:
        payload["contentType"] = None if content_type_null else content_type
    return payload


class _FakeStreamContext:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: tuple[bytes, ...] = (b"hello",),
        iter_raises: Exception | None = None,
    ) -> None:
        self._status_code = status_code
        self._headers = {} if headers is None else headers
        self._chunks = chunks
        self._iter_raises = iter_raises

    def __enter__(self) -> MagicMock:
        response = MagicMock()
        response.status_code = self._status_code
        response.headers = self._headers

        def _iter_bytes() -> Iterator[bytes]:
            if self._iter_raises is not None:
                raise self._iter_raises
            yield from self._chunks

        response.iter_bytes = _iter_bytes
        return response

    def __exit__(self, *args: object) -> None:
        return None


class _BrokenHeaderMapping(Mapping[str, str]):
    def __getitem__(self, key: str) -> str:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> Iterator[tuple[str, str]]:
        raise RuntimeError("broken headers")


def _valid_active_message(**overrides: object) -> MsGraphTeamsChannelMessage:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "thread_root_remote_id": _MESSAGE_ID,
        "message_kind": MsGraphTeamsChannelMessageKind.ROOT,
        "remote_id": _MESSAGE_ID,
        "revision": _REVISION,
        "state": MsGraphTeamsChannelMessageState.ACTIVE,
        "message_type": MsGraphTeamsChannelMessageType.MESSAGE,
        "importance": MsGraphTeamsChannelImportance.NORMAL,
        "created_at": datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
        "last_modified_at": datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
        "body_kind": MsGraphTeamsChannelBodyKind.TEXT,
        "body_content": "hello",
    }
    defaults.update(overrides)
    return MsGraphTeamsChannelMessage(**defaults)


def _valid_active_reply(**overrides: object) -> MsGraphTeamsChannelMessage:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "thread_root_remote_id": _ROOT_MESSAGE_ID,
        "message_kind": MsGraphTeamsChannelMessageKind.REPLY,
        "remote_id": _REPLY_MESSAGE_ID,
        "revision": _REPLY_REVISION,
        "state": MsGraphTeamsChannelMessageState.ACTIVE,
        "message_type": MsGraphTeamsChannelMessageType.MESSAGE,
        "importance": MsGraphTeamsChannelImportance.NORMAL,
        "created_at": datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
        "last_modified_at": datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
        "body_kind": MsGraphTeamsChannelBodyKind.TEXT,
        "body_content": "reply body",
    }
    defaults.update(overrides)
    return MsGraphTeamsChannelMessage(**defaults)


def _reply_observation_payload(
    *,
    message_id: str = _REPLY_MESSAGE_ID,
    root_id: str = _ROOT_MESSAGE_ID,
    revision: str = _REPLY_REVISION,
    deleted: bool = False,
) -> dict[str, Any]:
    return {
        "id": message_id,
        "replyToId": root_id,
        "channelIdentity": {"teamId": _TEAM_ID, "channelId": _CHANNEL_ID},
        "etag": revision,
        "deletedDateTime": "2024-06-01T12:00:00Z" if deleted else None,
    }


def _valid_reply_hosted_content(**overrides: object) -> MsGraphTeamsChannelHostedContent:
    return MsGraphTeamsChannelHostedContent(
        **_valid_hosted_content_kwargs(
            message_remote_id=_REPLY_MESSAGE_ID,
            thread_root_remote_id=_ROOT_MESSAGE_ID,
            message_kind=MsGraphTeamsChannelMessageKind.REPLY,
            message_revision=_REPLY_REVISION,
            **overrides,
        )
    )


def _valid_reply_hosted_content_page(**overrides: object) -> MsGraphTeamsChannelHostedContentPage:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "message_remote_id": _REPLY_MESSAGE_ID,
        "thread_root_remote_id": _ROOT_MESSAGE_ID,
        "message_kind": MsGraphTeamsChannelMessageKind.REPLY,
        "message_revision": _REPLY_REVISION,
        "items": (_valid_reply_hosted_content(),),
        "continuation": None,
    }
    defaults.update(overrides)
    return MsGraphTeamsChannelHostedContentPage(**defaults)


def _reply_hosted_contents_next_link(
    reply_id: str = _REPLY_MESSAGE_ID,
    root_id: str = _ROOT_MESSAGE_ID,
    channel_id: str = _CHANNEL_ID,
    team_id: str = _TEAM_ID,
) -> str:
    quoted_reply = quote(reply_id, safe="")
    quoted_root = quote(root_id, safe="")
    quoted_channel = quote(channel_id, safe="")
    quoted_team = quote(team_id, safe="")
    return (
        f"https://graph.microsoft.com/v1.0/teams/{quoted_team}/channels/"
        f"{quoted_channel}/messages/{quoted_root}/replies/{quoted_reply}/hostedContents"
        f"?$skiptoken={_SECRET_TOKEN}"
    )


def _odata_reply_hosted_contents_next_link(
    reply_id: str = _REPLY_MESSAGE_ID,
    root_id: str = _ROOT_MESSAGE_ID,
    channel_id: str = _CHANNEL_ID,
    team_id: str = _TEAM_ID,
) -> str:
    escaped_reply = reply_id.replace("'", "''")
    escaped_root = root_id.replace("'", "''")
    escaped_channel = channel_id.replace("'", "''")
    escaped_team = team_id.replace("'", "''")
    return (
        f"https://graph.microsoft.com/v1.0/teams('{escaped_team}')/channels('{escaped_channel}')"
        f"/messages('{escaped_root}')/replies('{escaped_reply}')/hostedContents"
        f"?$skiptoken={_SECRET_TOKEN}"
    )


def _slash_reply_hosted_contents_next_link(reply_id: str, root_id: str = _ROOT_MESSAGE_ID) -> str:
    quoted_reply = quote(reply_id, safe="")
    quoted_root = quote(root_id, safe="")
    return (
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{quoted_root}/replies/{quoted_reply}/hostedContents"
        f"?$skiptoken={_SECRET_TOKEN}"
    )


def _odata_percent_encoded_reply_hosted_contents_next_link(
    reply_id: str,
    root_id: str = _ROOT_MESSAGE_ID,
) -> str:
    escaped_reply = reply_id.replace("'", "''")
    escaped_root = root_id.replace("'", "''")
    encoded_reply = quote(escaped_reply, safe="")
    encoded_root = quote(escaped_root, safe="")
    return (
        f"https://graph.microsoft.com/v1.0/teams('{_TEAM_ID}')/channels('{_CHANNEL_ID}')"
        f"/messages('{encoded_root}')/replies('{encoded_reply}')/hostedContents"
        f"?$skiptoken={_SECRET_TOKEN}"
    )


def _setup_reply_hosted_contents_page(
    http: MagicMock,
    *,
    items: list[dict[str, Any]] | None = None,
    next_link: str | None = None,
) -> None:
    observation = _reply_observation_payload()
    hosted_payload = _page_payload(
        value=items if items is not None else [_hosted_content_payload()],
        next_link=next_link,
    )
    http.get.side_effect = [
        _json_response(payload=observation),
        _json_response(payload=hosted_payload),
        _json_response(payload=observation),
    ]


def _setup_reply_hosted_content_bytes(
    http: MagicMock,
    *,
    file_bytes: bytes = b"hello-world",
    hosted_content: MsGraphTeamsChannelHostedContent | None = None,
    content_type: str | None = "application/octet-stream",
) -> MsGraphTeamsChannelHostedContent:
    if hosted_content is None:
        hosted_content = _valid_reply_hosted_content()
    observation = _reply_observation_payload()
    http.get.side_effect = [
        _json_response(payload=observation),
        _json_response(payload=observation),
    ]
    headers: dict[str, str] = {"Content-Length": str(len(file_bytes))}
    if content_type is not None:
        headers["Content-Type"] = content_type
    http.stream.return_value = _FakeStreamContext(
        headers=headers,
        chunks=(file_bytes,),
    )
    return hosted_content


def _valid_deleted_message(**overrides: object) -> MsGraphTeamsChannelMessage:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "thread_root_remote_id": _MESSAGE_ID,
        "message_kind": MsGraphTeamsChannelMessageKind.ROOT,
        "remote_id": _MESSAGE_ID,
        "revision": _REVISION,
        "state": MsGraphTeamsChannelMessageState.DELETED,
        "message_type": MsGraphTeamsChannelMessageType.MESSAGE,
        "importance": MsGraphTeamsChannelImportance.NORMAL,
        "created_at": datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
        "last_modified_at": datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
        "deleted_at": datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return MsGraphTeamsChannelMessage(**defaults)


def _valid_hosted_content_kwargs(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "message_remote_id": _MESSAGE_ID,
        "thread_root_remote_id": _MESSAGE_ID,
        "message_kind": MsGraphTeamsChannelMessageKind.ROOT,
        "message_revision": _REVISION,
        "remote_id": _HOSTED_CONTENT_ID,
    }
    defaults.update(overrides)
    return defaults


def _valid_hosted_content(**overrides: object) -> MsGraphTeamsChannelHostedContent:
    return MsGraphTeamsChannelHostedContent(**_valid_hosted_content_kwargs(**overrides))


def _valid_hosted_content_page(**overrides: object) -> MsGraphTeamsChannelHostedContentPage:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "message_remote_id": _MESSAGE_ID,
        "thread_root_remote_id": _MESSAGE_ID,
        "message_kind": MsGraphTeamsChannelMessageKind.ROOT,
        "message_revision": _REVISION,
        "items": (_valid_hosted_content(),),
        "continuation": None,
    }
    defaults.update(overrides)
    return MsGraphTeamsChannelHostedContentPage(**defaults)


def _valid_hosted_content_bytes(
    data: bytes = b"hello-world",
    **overrides: object,
) -> MsGraphTeamsChannelHostedContentBytes:
    defaults: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "message_remote_id": _MESSAGE_ID,
        "thread_root_remote_id": _MESSAGE_ID,
        "message_kind": MsGraphTeamsChannelMessageKind.ROOT,
        "message_revision": _REVISION,
        "hosted_content_remote_id": _HOSTED_CONTENT_ID,
        "content_type": "application/octet-stream",
        "data": data,
        "size_bytes": len(data),
        "content_hash": hashlib.sha256(data).hexdigest(),
    }
    defaults.update(overrides)
    return MsGraphTeamsChannelHostedContentBytes(**defaults)


def _reader(http: MagicMock) -> MsGraphTeamsChannelHostedContentReader:
    return MsGraphTeamsChannelHostedContentReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
        graph_http_client=http,
    )


def _graph_client(http: MagicMock) -> GraphRestClient:
    return GraphRestClient(_config(), http_client=http)


def _parse_hosted_content(payload: dict[str, Any]) -> MsGraphTeamsChannelHostedContent:
    return parse_msgraph_teams_channel_hosted_content(payload, message=_valid_active_message())


def _setup_hosted_contents_page(
    http: MagicMock,
    *,
    items: list[dict[str, Any]] | None = None,
    next_link: str | None = None,
) -> None:
    observation = _observation_payload()
    hosted_payload = _page_payload(
        value=items if items is not None else [_hosted_content_payload()],
        next_link=next_link,
    )
    http.get.side_effect = [
        _json_response(payload=observation),
        _json_response(payload=hosted_payload),
        _json_response(payload=observation),
    ]


def _setup_hosted_content_bytes(
    http: MagicMock,
    *,
    file_bytes: bytes = b"hello-world",
    hosted_content: MsGraphTeamsChannelHostedContent | None = None,
    content_type: str | None = "application/octet-stream",
) -> MsGraphTeamsChannelHostedContent:
    if hosted_content is None:
        hosted_content = _valid_hosted_content()
    observation = _observation_payload()
    http.get.side_effect = [
        _json_response(payload=observation),
        _json_response(payload=observation),
    ]
    headers: dict[str, str] = {"Content-Length": str(len(file_bytes))}
    if content_type is not None:
        headers["Content-Type"] = content_type
    http.stream.return_value = _FakeStreamContext(
        headers=headers,
        chunks=(file_bytes,),
    )
    return hosted_content


# --- parser: inventory ---


def test_parse_hosted_content_with_id_only() -> None:
    hosted = _parse_hosted_content(_hosted_content_payload())
    assert hosted.remote_id == _HOSTED_CONTENT_ID
    assert hosted.message_revision == _REVISION
    assert hosted.channel_remote_id == _CHANNEL_ID
    assert hosted.message_remote_id == _MESSAGE_ID


def test_parse_hosted_content_null_content_bytes_and_content_type_keys() -> None:
    hosted = _parse_hosted_content(
        _hosted_content_payload(
            include_content_bytes_key=True,
            include_content_type_key=True,
            content_bytes_null=True,
            content_type_null=True,
        )
    )
    assert hosted.remote_id == _HOSTED_CONTENT_ID


@pytest.mark.parametrize(
    "content_bytes",
    ["base64payload", b"bytes", 123, ""],
)
def test_parse_hosted_content_rejects_non_null_content_bytes(content_bytes: object) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _parse_hosted_content(
            _hosted_content_payload(
                include_content_bytes_key=True,
                content_bytes_null=False,
                content_bytes=content_bytes,
            )
        )
    assert exc.value.__cause__ is None


@pytest.mark.parametrize(
    "content_type",
    ["image/png", "application/pdf", "", "   ", 123],
)
def test_parse_hosted_content_rejects_non_null_content_type(content_type: object) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _parse_hosted_content(
            _hosted_content_payload(
                include_content_type_key=True,
                content_type_null=False,
                content_type=content_type,
            )
        )
    assert exc.value.__cause__ is None


def test_parse_hosted_content_deleted_message_rejected() -> None:
    with pytest.raises(MsGraphTeamsChannelMessageChanged):
        parse_msgraph_teams_channel_hosted_content(
            _hosted_content_payload(),
            message=_valid_deleted_message(),
        )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"id": ""},
        {"id": _HOSTED_CONTENT_ID, "contentBytes": "inline"},
        {"id": _HOSTED_CONTENT_ID, "contentType": "text/plain"},
        {"id": "\x00bad"},
        {"id": 123},
    ],
)
def test_parse_malformed_provider_payload(payload: object) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        parse_msgraph_teams_channel_hosted_content(payload, message=_valid_active_message())
    assert exc.value.__cause__ is None
    assert _HOSTED_CONTENT_ID not in str(exc.value)


# --- model and deep validation ---


def test_validate_hosted_content_returns_new_instance() -> None:
    original = _valid_hosted_content()
    validated = validate_msgraph_teams_channel_hosted_content(original)
    assert validated == original
    assert validated is not original


@pytest.mark.parametrize(
    "kwargs",
    [
        {"remote_id": None},
        {"message_revision": None},
        {"remote_id": ""},
    ],
)
def test_model_construct_malformed_hosted_content_rejected(kwargs: dict[str, object]) -> None:
    malformed = MsGraphTeamsChannelHostedContent.model_construct(
        **{**_valid_hosted_content_kwargs(), **kwargs}
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_teams_channel_hosted_content(malformed)
    assert exc.value.__cause__ is None


def test_validate_hosted_content_page_returns_new_instances() -> None:
    original = _valid_hosted_content_page()
    validated = validate_msgraph_teams_channel_hosted_content_page(
        original,
        message=_valid_active_message(),
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == original
    assert validated is not original
    assert validated.items[0] is not original.items[0]


@pytest.mark.parametrize(
    "page_kwargs",
    [
        {"items": None},
        {"nested_malformed": True},
        {"continuation": "bad"},
    ],
)
def test_model_construct_malformed_page_rejected(page_kwargs: dict[str, object]) -> None:
    if page_kwargs.get("items") is None:
        malformed = MsGraphTeamsChannelHostedContentPage.model_construct(
            team_remote_id=_TEAM_ID,
            channel_remote_id=_CHANNEL_ID,
            message_remote_id=_MESSAGE_ID,
            message_revision=_REVISION,
        )
    elif page_kwargs.get("nested_malformed"):
        malformed = MsGraphTeamsChannelHostedContentPage.model_construct(
            team_remote_id=_TEAM_ID,
            channel_remote_id=_CHANNEL_ID,
            message_remote_id=_MESSAGE_ID,
            message_revision=_REVISION,
            items=(MsGraphTeamsChannelHostedContent.model_construct(),),
        )
    else:
        malformed = MsGraphTeamsChannelHostedContentPage.model_construct(
            team_remote_id=_TEAM_ID,
            channel_remote_id=_CHANNEL_ID,
            message_remote_id=_MESSAGE_ID,
            message_revision=_REVISION,
            items=(_valid_hosted_content(),),
            continuation=page_kwargs["continuation"],
        )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_teams_channel_hosted_content_page(
            malformed,
            message=_valid_active_message(),
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None


def test_hosted_content_page_model_rejects_duplicate_remote_ids() -> None:
    item = _valid_hosted_content()
    page = MsGraphTeamsChannelHostedContentPage.model_construct(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        message_remote_id=_MESSAGE_ID,
        thread_root_remote_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        message_revision=_REVISION,
        items=(item, item),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        validate_msgraph_teams_channel_hosted_content_page(
            page,
            message=_valid_active_message(),
            graph_base_url=_GRAPH_BASE,
        )


def test_validate_page_rejects_stale_message_revision() -> None:
    page = MsGraphTeamsChannelHostedContentPage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        message_remote_id=_MESSAGE_ID,
        thread_root_remote_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        message_revision=_OTHER_REVISION,
        items=(_valid_hosted_content(message_revision=_OTHER_REVISION),),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        validate_msgraph_teams_channel_hosted_content_page(
            page,
            message=_valid_active_message(),
            graph_base_url=_GRAPH_BASE,
        )


def test_validate_hosted_content_bytes_returns_new_instance() -> None:
    data = b"content-bytes"
    original = _valid_hosted_content_bytes(data=data)
    validated = validate_msgraph_teams_channel_hosted_content_bytes(
        original,
        message=_valid_active_message(),
        hosted_content=_valid_hosted_content(),
        max_bytes=1024,
    )
    assert validated == original
    assert validated is not original


@pytest.mark.parametrize(
    "kwargs,error_type",
    [
        ({"data": b"x"}, ValueError),
        ({"size_bytes": 99}, ValueError),
        ({"content_hash": "not-a-valid-sha256-hash"}, ValueError),
        ({"hosted_content_remote_id": _OTHER_HOSTED_CONTENT_ID}, ValueError),
        ({"message_revision": _OTHER_REVISION}, ValueError),
        ({"data": bytearray(b"abc")}, ValueError),
        ({"size_bytes": True}, ValueError),
        ({"content_type": ""}, ValueError),
    ],
)
def test_model_construct_malformed_hosted_content_bytes_rejected(
    kwargs: dict[str, object],
    error_type: type[BaseException],
) -> None:
    data = b"abc"
    base = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "message_remote_id": _MESSAGE_ID,
        "message_revision": _REVISION,
        "hosted_content_remote_id": _HOSTED_CONTENT_ID,
        "data": data,
        "size_bytes": len(data),
        "content_hash": hashlib.sha256(data).hexdigest(),
    }
    malformed = MsGraphTeamsChannelHostedContentBytes.model_construct(**{**base, **kwargs})
    with pytest.raises(
        error_type,
        match=_SAFE_ERROR if error_type is ValueError else "changed",
    ) as exc:
        validate_msgraph_teams_channel_hosted_content_bytes(
            malformed,
            message=_valid_active_message(),
            hosted_content=_valid_hosted_content(),
            max_bytes=1024,
        )
    assert exc.value.__cause__ is None


@pytest.mark.parametrize(
    "missing_field",
    ["data", "size_bytes", "content_hash", "hosted_content_remote_id"],
)
def test_model_construct_missing_hosted_content_bytes_field_rejected(missing_field: str) -> None:
    data = b"abc"
    base: dict[str, object] = {
        "team_remote_id": _TEAM_ID,
        "channel_remote_id": _CHANNEL_ID,
        "message_remote_id": _MESSAGE_ID,
        "message_revision": _REVISION,
        "hosted_content_remote_id": _HOSTED_CONTENT_ID,
        "data": data,
        "size_bytes": len(data),
        "content_hash": hashlib.sha256(data).hexdigest(),
    }
    del base[missing_field]
    malformed = MsGraphTeamsChannelHostedContentBytes.model_construct(**base)
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_teams_channel_hosted_content_bytes(
            malformed,
            message=_valid_active_message(),
            hosted_content=_valid_hosted_content(),
            max_bytes=1024,
        )
    assert exc.value.__cause__ is None


def test_validate_hosted_content_bytes_too_large() -> None:
    data = b"toolarge"
    content = _valid_hosted_content_bytes(data=data)
    with pytest.raises(MsGraphTeamsChannelHostedContentTooLarge):
        validate_msgraph_teams_channel_hosted_content_bytes(
            content,
            message=_valid_active_message(),
            hosted_content=_valid_hosted_content(),
            max_bytes=3,
        )


# --- requests and headers ---


def test_initial_request_path_and_headers() -> None:
    http = MagicMock()
    _setup_hosted_contents_page(http)
    _reader(http).read_teams_channel_hosted_contents_page(
        message=_valid_active_message(),
        continuation=None,
    )
    list_call = http.get.call_args_list[1]
    assert list_call.args[0] == _HOSTED_CONTENTS_PATH
    assert list_call.kwargs.get("params") is None
    assert list_call.kwargs["headers"] == _PREFER_HEADER


def test_continuation_request_uses_full_url_without_params() -> None:
    http = MagicMock()
    observation = _observation_payload()
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_hosted_contents_next_link(),
    )
    http.get.side_effect = [
        _json_response(payload=observation),
        _json_response(payload=_page_payload()),
        _json_response(payload=observation),
    ]
    _reader(http).read_teams_channel_hosted_contents_page(
        message=_valid_active_message(),
        continuation=continuation,
    )
    continuation_call = http.get.call_args_list[1]
    assert continuation_call.args[0] == _hosted_contents_next_link()
    assert continuation_call.kwargs.get("params") is None
    assert continuation_call.kwargs["headers"] == _PREFER_HEADER


def test_observation_requests_use_prefer_header() -> None:
    http = MagicMock()
    _setup_hosted_contents_page(http)
    _reader(http).read_teams_channel_hosted_contents_page(
        message=_valid_active_message(),
        continuation=None,
    )
    pre_observation = http.get.call_args_list[0]
    post_observation = http.get.call_args_list[2]
    assert pre_observation.args[0] == _OBSERVATION_PATH
    assert pre_observation.kwargs["headers"] == _PREFER_HEADER
    assert post_observation.args[0] == _OBSERVATION_PATH
    assert post_observation.kwargs["headers"] == _PREFER_HEADER


@pytest.mark.parametrize("max_bytes", [0, ABSOLUTE_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES + 1, True, "10", None])
def test_invalid_max_bytes_rejected_before_http(max_bytes: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_teams_channel_hosted_content_bytes(
            message=_valid_active_message(),
            hosted_content=_valid_hosted_content(),
            max_bytes=max_bytes,  # type: ignore[arg-type]
        )
    http.get.assert_not_called()
    http.stream.assert_not_called()


# --- continuation validation ---


def test_validate_continuation_accepts_next_page_slash_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_hosted_contents_next_link(),
    )
    validated = validate_msgraph_teams_channel_hosted_contents_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        thread_root_id=_MESSAGE_ID,
        message_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation
    assert validated.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE
    assert validated.url == continuation.url


def test_validate_continuation_accepts_odata_key_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_hosted_contents_next_link(),
    )
    validated = validate_msgraph_teams_channel_hosted_contents_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        thread_root_id=_MESSAGE_ID,
        message_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_continuation_accepts_literal_with_escaped_quotes() -> None:
    message_id = "msg'quote'part"
    channel_id = "channel'quote'part"
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_hosted_contents_next_link(message_id, channel_id),
    )
    validated = validate_msgraph_teams_channel_hosted_contents_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=channel_id,
        thread_root_id=message_id,
        message_id=message_id,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_continuation_accepts_uppercase_resource_names() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/TEAMS/{_QUOTED_TEAM}/CHANNELS/"
            f"{_QUOTED_CHANNEL}/MESSAGES/{_QUOTED_MESSAGE_ID}/HOSTEDCONTENTS?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    validated = validate_msgraph_teams_channel_hosted_contents_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        thread_root_id=_MESSAGE_ID,
        message_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_continuation_accepts_percent_encoded_message_literal() -> None:
    message_id = "msg/special"
    encoded = quote(message_id, safe="")
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
            f"{_QUOTED_CHANNEL}/messages('{encoded}')/hostedContents?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    validated = validate_msgraph_teams_channel_hosted_contents_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        thread_root_id=message_id,
        message_id=message_id,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


@pytest.mark.parametrize(
    "message_id",
    [
        "msg-delta-hosted",
        "msg-DELTA-hosted",
        "opaque-delta-only",
        "opaque-hosted-only",
    ],
)
def test_validate_continuation_accepts_opaque_message_id_with_reserved_substrings(
    message_id: str,
) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_slash_hosted_contents_next_link(message_id),
    )
    validated = validate_msgraph_teams_channel_hosted_contents_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        thread_root_id=message_id,
        message_id=message_id,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_continuation_accepts_odata_literal_with_quote_percent_and_delta() -> None:
    message_id = "msg'delta/special"
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_percent_encoded_hosted_contents_next_link(message_id),
    )
    validated = validate_msgraph_teams_channel_hosted_contents_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        thread_root_id=message_id,
        message_id=message_id,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


@pytest.mark.parametrize(
    "url",
    [
        f"https://graph.microsoft.com/v1.0/teams/{quote(_OTHER_TEAM_ID, safe='')}/"
        f"chats/{_QUOTED_CHANNEL}/messages/{quote(_MESSAGE_ID, safe='')}/hostedContents?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
        f"{_QUOTED_OTHER_CHANNEL}/messages/{quote(_MESSAGE_ID, safe='')}/hostedContents?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{_QUOTED_OTHER_MESSAGE_ID}/hostedContents?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{quote(_MESSAGE_ID, safe='')}/hostedContents/delta?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/messages/"
        f"{_QUOTED_MESSAGE_ID}/hostedContents?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{quote(_MESSAGE_ID, safe='')}/hostedContents/"
        f"{_QUOTED_HOSTED_CONTENT_ID}/$value",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{quote(_MESSAGE_ID, safe='')}/hostedContents/extra?$skiptoken={_SECRET_TOKEN}",
        "https://graph.microsoft.com/v1.0/drives/drive-1/root/children?$skiptoken=x",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels('{_QUOTED_CHANNEL}')"
        f"/messages('unterminated?$skiptoken={_SECRET_TOKEN}",
    ],
)
def test_validate_continuation_rejects_invalid_urls(url: str) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=url,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_teams_channel_hosted_contents_continuation(
            continuation,
            team_id=_TEAM_ID,
            channel_id=_CHANNEL_ID,
            thread_root_id=_MESSAGE_ID,
        message_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
            graph_base_url=_GRAPH_BASE,
        )
    assert _SECRET_TOKEN not in str(exc.value)
    assert _TEAM_ID not in str(exc.value)


def test_validate_continuation_rejects_delta_kind() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_hosted_contents_next_link(),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        validate_msgraph_teams_channel_hosted_contents_continuation(
            continuation,
            team_id=_TEAM_ID,
            channel_id=_CHANNEL_ID,
            thread_root_id=_MESSAGE_ID,
        message_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
            graph_base_url=_GRAPH_BASE,
        )


def test_validate_continuation_rejects_wrong_object_type() -> None:
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        validate_msgraph_teams_channel_hosted_contents_continuation(
            "bad",
            team_id=_TEAM_ID,
            channel_id=_CHANNEL_ID,
            thread_root_id=_MESSAGE_ID,
        message_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
            graph_base_url=_GRAPH_BASE,
        )


def _assert_malformed_continuation_rejected(continuation: object) -> None:
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_teams_channel_hosted_contents_continuation(
            continuation,
            team_id=_TEAM_ID,
            channel_id=_CHANNEL_ID,
            thread_root_id=_MESSAGE_ID,
        message_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None
    assert _SECRET_TOKEN not in str(exc.value)
    assert _TEAM_ID not in str(exc.value)
    assert _MESSAGE_ID not in str(exc.value)


@pytest.mark.parametrize(
    "continuation",
    [
        MsGraphKnowledgeContinuation.model_construct(),
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        ),
        MsGraphKnowledgeContinuation.model_construct(url=_hosted_contents_next_link()),
        MsGraphKnowledgeContinuation.model_construct(
            kind="next_page",
            url=_hosted_contents_next_link(),
        ),
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=123,
        ),
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url="",
        ),
    ],
)
def test_validate_continuation_rejects_model_construct_malformed(
    continuation: MsGraphKnowledgeContinuation,
) -> None:
    _assert_malformed_continuation_rejected(continuation)


def test_validate_page_rejects_delta_continuation_kind() -> None:
    malformed = MsGraphTeamsChannelHostedContentPage.model_construct(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        message_remote_id=_MESSAGE_ID,
        thread_root_remote_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        message_revision=_REVISION,
        items=(_valid_hosted_content(),),
        continuation=MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=_hosted_contents_next_link(),
        ),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        validate_msgraph_teams_channel_hosted_content_page(
            malformed,
            message=_valid_active_message(),
            graph_base_url=_GRAPH_BASE,
        )


# --- paging semantics ---


def test_first_page_with_next_page() -> None:
    http = MagicMock()
    _setup_hosted_contents_page(http, next_link=_hosted_contents_next_link())
    page = _reader(http).read_teams_channel_hosted_contents_page(
        message=_valid_active_message(),
        continuation=None,
    )
    assert page.continuation is not None
    assert page.continuation.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE


def test_final_page_without_continuation() -> None:
    http = MagicMock()
    _setup_hosted_contents_page(http)
    page = _reader(http).read_teams_channel_hosted_contents_page(
        message=_valid_active_message(),
        continuation=None,
    )
    assert page.continuation is None


def test_message_changed_before_list() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_observation_payload(revision=_OTHER_REVISION),
    )
    with pytest.raises(MsGraphTeamsChannelMessageChanged):
        _reader(http).read_teams_channel_hosted_contents_page(
            message=_valid_active_message(),
            continuation=None,
        )
    http.get.assert_called_once()


def test_message_changed_after_list() -> None:
    http = MagicMock()
    observation_before = _observation_payload()
    observation_after = _observation_payload(revision=_OTHER_REVISION)
    http.get.side_effect = [
        _json_response(payload=observation_before),
        _json_response(payload=_page_payload(value=[_hosted_content_payload()])),
        _json_response(payload=observation_after),
    ]
    with pytest.raises(MsGraphTeamsChannelMessageChanged):
        _reader(http).read_teams_channel_hosted_contents_page(
            message=_valid_active_message(),
            continuation=None,
        )


def test_read_teams_channel_hosted_contents_page_deleted_message_rejected() -> None:
    http = MagicMock()
    with pytest.raises(MsGraphTeamsChannelMessageChanged):
        _reader(http).read_teams_channel_hosted_contents_page(
            message=_valid_deleted_message(),
            continuation=None,
        )
    http.get.assert_not_called()


def test_read_teams_channel_hosted_content_bytes_stale_hosted_revision() -> None:
    http = MagicMock()
    hosted = _valid_hosted_content(message_revision=_OTHER_REVISION)
    with pytest.raises(MsGraphTeamsChannelMessageChanged):
        _reader(http).read_teams_channel_hosted_content_bytes(
            message=_valid_active_message(),
            hosted_content=hosted,
            max_bytes=1024,
        )
    http.get.assert_not_called()
    http.stream.assert_not_called()


# --- hosted content byte streaming ---


def test_download_small_content_with_sha256() -> None:
    data = b"small-file-content"
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=data)
    result = _reader(http).read_teams_channel_hosted_content_bytes(
        message=_valid_active_message(),
        hosted_content=hosted,
        max_bytes=1024,
    )
    assert isinstance(result, MsGraphTeamsChannelHostedContentBytes)
    assert result.data == data
    assert result.size_bytes == len(data)
    assert result.content_hash == hashlib.sha256(data).hexdigest()
    assert result.content_type == "application/octet-stream"
    assert result.message_revision == _REVISION
    assert result.hosted_content_remote_id == _HOSTED_CONTENT_ID


def test_download_empty_content() -> None:
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=b"")
    result = _reader(http).read_teams_channel_hosted_content_bytes(
        message=_valid_active_message(),
        hosted_content=hosted,
        max_bytes=1024,
    )
    assert result.data == b""
    assert result.size_bytes == 0


def test_download_multiple_chunks() -> None:
    data = b"hello-world"
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=data)
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": str(len(data))},
        chunks=(b"hello", b"-world"),
    )
    result = _reader(http).read_teams_channel_hosted_content_bytes(
        message=_valid_active_message(),
        hosted_content=hosted,
        max_bytes=1024,
    )
    assert result.data == data


def test_download_without_content_length() -> None:
    data = b"12345"
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=data)
    http.stream.return_value = _FakeStreamContext(chunks=(data,))
    result = _reader(http).read_teams_channel_hosted_content_bytes(
        message=_valid_active_message(),
        hosted_content=hosted,
        max_bytes=1024,
    )
    assert result.data == data


def test_download_content_length_exceeds_limit() -> None:
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=b"x")
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "100"},
        chunks=(b"x",),
    )
    with pytest.raises(MsGraphTeamsChannelHostedContentTooLarge):
        _reader(http).read_teams_channel_hosted_content_bytes(
            message=_valid_active_message(),
            hosted_content=hosted,
            max_bytes=10,
        )


def test_download_malformed_content_length() -> None:
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=b"x")
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "not-a-number"},
        chunks=(b"x",),
    )
    with pytest.raises(IntegrationDependencyError, match=_INVALID_RESPONSE):
        _reader(http).read_teams_channel_hosted_content_bytes(
            message=_valid_active_message(),
            hosted_content=hosted,
            max_bytes=1024,
        )


def test_download_bytes_exceed_limit_during_stream() -> None:
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=b"abcdefghij")
    http.stream.return_value = _FakeStreamContext(
        chunks=(b"a" * 5, b"b" * 10),
    )
    with pytest.raises(MsGraphTeamsChannelHostedContentTooLarge):
        _reader(http).read_teams_channel_hosted_content_bytes(
            message=_valid_active_message(),
            hosted_content=hosted,
            max_bytes=10,
        )


def test_download_content_length_mismatch() -> None:
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=b"short")
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "10"},
        chunks=(b"short",),
    )
    with pytest.raises(IntegrationDependencyError, match=_INVALID_RESPONSE):
        _reader(http).read_teams_channel_hosted_content_bytes(
            message=_valid_active_message(),
            hosted_content=hosted,
            max_bytes=1024,
        )


def test_download_chunk_not_bytes() -> None:
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=b"hello")

    class _BadChunkStream(_FakeStreamContext):
        def __enter__(self) -> MagicMock:
            response = super().__enter__()
            response.iter_bytes = lambda: iter(["not-bytes"])  # type: ignore[assignment]
            return response

    http.stream.return_value = _BadChunkStream(chunks=())
    with pytest.raises(IntegrationDependencyError, match=_INVALID_RESPONSE):
        _reader(http).read_teams_channel_hosted_content_bytes(
            message=_valid_active_message(),
            hosted_content=hosted,
            max_bytes=1024,
        )


@pytest.mark.parametrize("status_code", [206, 302, 500])
def test_download_bad_status(status_code: int) -> None:
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=b"x")
    http.stream.return_value = _FakeStreamContext(status_code=status_code, chunks=(b"x",))
    with pytest.raises(IntegrationDependencyError):
        _reader(http).read_teams_channel_hosted_content_bytes(
            message=_valid_active_message(),
            hosted_content=hosted,
            max_bytes=1024,
        )


@pytest.mark.parametrize("status_code", [401, 403])
def test_download_configuration_errors(status_code: int) -> None:
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=b"x")
    http.stream.return_value = _FakeStreamContext(status_code=status_code, chunks=(b"x",))
    with pytest.raises(IntegrationConfigurationError):
        _reader(http).read_teams_channel_hosted_content_bytes(
            message=_valid_active_message(),
            hosted_content=hosted,
            max_bytes=1024,
        )


def test_download_stream_transport_exception() -> None:
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=b"x")
    http.stream.side_effect = RuntimeError("stream failed")
    with pytest.raises(IntegrationDependencyError) as exc:
        _reader(http).read_teams_channel_hosted_content_bytes(
            message=_valid_active_message(),
            hosted_content=hosted,
            max_bytes=1024,
        )
    assert exc.value.__cause__ is None


def test_download_request_path_and_headers() -> None:
    data = b"bytes"
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=data)
    _reader(http).read_teams_channel_hosted_content_bytes(
        message=_valid_active_message(),
        hosted_content=hosted,
        max_bytes=1024,
    )
    call = http.stream.call_args
    assert call.args[1] == _VALUE_PATH
    assert call.kwargs["follow_redirects"] is False
    assert call.kwargs["headers"] == {"Accept": "application/octet-stream"}
    assert "params" not in call.kwargs


def test_download_integer_header_key_rejected() -> None:
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=b"hello")
    http.stream.return_value = _FakeStreamContext(
        headers={1: "5"},
        chunks=(b"hello",),
    )
    with pytest.raises(IntegrationDependencyError, match=_INVALID_RESPONSE) as exc:
        _reader(http).read_teams_channel_hosted_content_bytes(
            message=_valid_active_message(),
            hosted_content=hosted,
            max_bytes=1024,
        )
    assert exc.value.__cause__ is None


def test_download_duplicate_content_length_different_case_rejected() -> None:
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=b"hello")
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": "5", "content-length": "5"},
        chunks=(b"hello",),
    )
    with pytest.raises(IntegrationDependencyError, match=_INVALID_RESPONSE):
        _reader(http).read_teams_channel_hosted_content_bytes(
            message=_valid_active_message(),
            hosted_content=hosted,
            max_bytes=1024,
        )


def test_download_headers_items_raises_rejected() -> None:
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=b"hello")
    http.stream.return_value = _FakeStreamContext(
        headers=_BrokenHeaderMapping(),
        chunks=(b"hello",),
    )
    with pytest.raises(IntegrationDependencyError, match=_INVALID_RESPONSE):
        _reader(http).read_teams_channel_hosted_content_bytes(
            message=_valid_active_message(),
            hosted_content=hosted,
            max_bytes=1024,
        )


def test_message_changed_after_download_bytes_not_returned() -> None:
    http = MagicMock()
    data = b"hello"
    hosted = _valid_hosted_content()
    http.get.side_effect = [
        _json_response(payload=_observation_payload()),
        _json_response(payload=_observation_payload(revision=_OTHER_REVISION)),
    ]
    http.stream.return_value = _FakeStreamContext(
        headers={"Content-Length": str(len(data))},
        chunks=(data,),
    )
    with pytest.raises(MsGraphTeamsChannelMessageChanged):
        _reader(http).read_teams_channel_hosted_content_bytes(
            message=_valid_active_message(),
            hosted_content=hosted,
            max_bytes=1024,
        )


def test_observation_before_and_after_download() -> None:
    data = b"payload"
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=data)
    _reader(http).read_teams_channel_hosted_content_bytes(
        message=_valid_active_message(),
        hosted_content=hosted,
        max_bytes=1024,
    )
    assert http.get.call_count == 2
    assert http.get.call_args_list[0].args[0] == _OBSERVATION_PATH
    assert http.get.call_args_list[1].args[0] == _OBSERVATION_PATH
    assert http.stream.call_count == 1


# --- delegation ---


def test_graph_rest_client_delegates_hosted_contents_page() -> None:
    http = MagicMock()
    _setup_hosted_contents_page(http)
    page = _graph_client(http).read_teams_channel_hosted_contents_page(
        message=_valid_active_message(),
    )
    assert page.items[0].remote_id == _HOSTED_CONTENT_ID


def test_graph_rest_client_delegates_hosted_content_bytes() -> None:
    data = b"delegated"
    http = MagicMock()
    hosted = _setup_hosted_content_bytes(http, file_bytes=data)
    result = _graph_client(http).read_teams_channel_hosted_content_bytes(
        message=_valid_active_message(),
        hosted_content=hosted,
    )
    assert result.data == data


def test_collaboration_suite_delegates_hosted_contents() -> None:
    http = MagicMock()
    _setup_hosted_contents_page(http)
    suite = _Ms365GraphCollaborationSuite(_graph_client(http))
    page = suite.read_teams_channel_hosted_contents_page(message=_valid_active_message())
    assert page.items[0].remote_id == _HOSTED_CONTENT_ID


def test_transport_and_reader_share_injected_http_client() -> None:
    http = MagicMock()
    _setup_hosted_contents_page(http)
    client = _graph_client(http)
    client.read_teams_channel_hosted_contents_page(message=_valid_active_message())
    assert client._knowledge_transport._http_client is http
    assert client._teams_channel_hosted_content_reader._transport._http_client is http
    assert client._teams_channel_hosted_content_reader._graph_http_client is http


def test_no_new_http_client_created() -> None:
    http = MagicMock()
    _setup_hosted_contents_page(http)
    client = _graph_client(http)
    client.read_teams_channel_hosted_contents_page(message=_valid_active_message())
    assert client._http_client is http


class _CustomGraphTeamsChannelHostedContentClient(GraphRestClient):
    def __init__(self, page: MsGraphTeamsChannelHostedContentPage, http: MagicMock) -> None:
        super().__init__(_config(), http_client=http)
        self._custom_page = page

    def read_teams_channel_hosted_contents_page(
        self,
        *,
        message: MsGraphTeamsChannelMessage,
        continuation: MsGraphKnowledgeContinuation | None = None,
    ) -> MsGraphTeamsChannelHostedContentPage:
        return self._custom_page


class _CustomTeamsChatHostedContentReader:
    def read_teams_channel_hosted_contents_page(
        self,
        *,
        message: MsGraphTeamsChannelMessage,
        continuation: MsGraphKnowledgeContinuation | None = None,
    ) -> MsGraphTeamsChannelHostedContentPage:
        return _valid_hosted_content_page()

    def read_teams_channel_hosted_content_bytes(
        self,
        *,
        message: MsGraphTeamsChannelMessage,
        hosted_content: MsGraphTeamsChannelHostedContent,
        max_bytes: int = DEFAULT_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES,
    ) -> MsGraphTeamsChannelHostedContentBytes:
        data = b"custom-binary"
        return MsGraphTeamsChannelHostedContentBytes(
            team_remote_id=message.team_remote_id,
            channel_remote_id=message.channel_remote_id,
            message_remote_id=message.remote_id,
            thread_root_remote_id=message.thread_root_remote_id,
            message_kind=message.message_kind,
            message_revision=message.revision,
            hosted_content_remote_id=hosted_content.remote_id,
            content_type="application/octet-stream",
            data=data,
            size_bytes=len(data),
            content_hash=hashlib.sha256(data).hexdigest(),
        )


def test_custom_client_malformed_page_rejected() -> None:
    malformed_page = MsGraphTeamsChannelHostedContentPage.model_construct(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        message_remote_id=_MESSAGE_ID,
        thread_root_remote_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        message_revision=_REVISION,
        items=(MsGraphTeamsChannelHostedContent.model_construct(),),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_teams_channel_hosted_content_page(
            _CustomGraphTeamsChannelHostedContentClient(
                page=malformed_page,
                http=MagicMock(),
            ).read_teams_channel_hosted_contents_page(message=_valid_active_message()),
            message=_valid_active_message(),
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None


def test_custom_client_valid_page_revalidated() -> None:
    supplied = _valid_hosted_content_page()
    returned = validate_msgraph_teams_channel_hosted_content_page(
        _CustomGraphTeamsChannelHostedContentClient(page=supplied, http=MagicMock()).read_teams_channel_hosted_contents_page(
            message=_valid_active_message(),
        ),
        message=_valid_active_message(),
        graph_base_url=_GRAPH_BASE,
    )
    assert returned == supplied
    assert returned is not supplied
    assert returned.items[0] is not supplied.items[0]


def test_custom_client_rejects_malformed_continuation_without_url() -> None:
    malformed_page = MsGraphTeamsChannelHostedContentPage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        message_remote_id=_MESSAGE_ID,
        thread_root_remote_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        message_revision=_REVISION,
        items=(_valid_hosted_content(),),
        continuation=MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        ),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_teams_channel_hosted_content_page(
            malformed_page,
            message=_valid_active_message(),
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None


def test_custom_client_cross_message_continuation_rejected() -> None:
    wrong_continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
            f"{_QUOTED_CHANNEL}/messages/{_QUOTED_OTHER_MESSAGE_ID}/hostedContents?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    page = MsGraphTeamsChannelHostedContentPage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        message_remote_id=_MESSAGE_ID,
        thread_root_remote_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        message_revision=_REVISION,
        items=(_valid_hosted_content(),),
        continuation=wrong_continuation,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        validate_msgraph_teams_channel_hosted_content_page(
            page,
            message=_valid_active_message(),
            graph_base_url=_GRAPH_BASE,
        )


class _CustomSuiteWithoutHostedContentCapability(CollaborationSuite):
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


def test_custom_client_without_hosted_content_capability_fails() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomSuiteWithoutHostedContentCapability(),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CAPABILITY_ERROR):
        integration.read_teams_channel_hosted_contents_page(message=_valid_active_message())


class _CustomTeamsChatHostedContentSuite(CollaborationSuite):
    def __init__(self, page: MsGraphTeamsChannelHostedContentPage) -> None:
        self._page = page

    def read_teams_channel_hosted_contents_page(
        self,
        *,
        message: MsGraphTeamsChannelMessage,
        continuation: MsGraphKnowledgeContinuation | None = None,
    ) -> MsGraphTeamsChannelHostedContentPage:
        return self._page

    def read_teams_channel_hosted_content_bytes(
        self,
        *,
        message: MsGraphTeamsChannelMessage,
        hosted_content: MsGraphTeamsChannelHostedContent,
        max_bytes: int = DEFAULT_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES,
    ) -> MsGraphTeamsChannelHostedContentBytes:
        data = b"custom"
        return MsGraphTeamsChannelHostedContentBytes(
            team_remote_id=message.team_remote_id,
            channel_remote_id=message.channel_remote_id,
            message_remote_id=message.remote_id,
            thread_root_remote_id=message.thread_root_remote_id,
            message_kind=message.message_kind,
            message_revision=message.revision,
            hosted_content_remote_id=hosted_content.remote_id,
            content_type="application/octet-stream",
            data=data,
            size_bytes=len(data),
            content_hash=hashlib.sha256(data).hexdigest(),
        )

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


def test_integration_custom_client_malformed_page_rejected() -> None:
    malformed_page = MsGraphTeamsChannelHostedContentPage.model_construct(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        message_remote_id=_MESSAGE_ID,
        thread_root_remote_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        message_revision=_REVISION,
        items=(MsGraphTeamsChannelHostedContent.model_construct(),),
    )
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphTeamsChannelHostedContentClient(page=malformed_page, http=MagicMock())
        ),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        integration.read_teams_channel_hosted_contents_page(message=_valid_active_message())
    assert exc.value.__cause__ is None


def test_integration_custom_client_valid_page_revalidated() -> None:
    supplied = _valid_hosted_content_page()
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphTeamsChannelHostedContentClient(page=supplied, http=MagicMock())
        ),
        enabled=True,
    )
    returned = integration.read_teams_channel_hosted_contents_page(message=_valid_active_message())
    assert returned == supplied
    assert returned is not supplied
    assert returned.items[0] is not supplied.items[0]


def test_integration_rejects_malformed_continuation_without_url() -> None:
    malformed_page = MsGraphTeamsChannelHostedContentPage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        message_remote_id=_MESSAGE_ID,
        thread_root_remote_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        message_revision=_REVISION,
        items=(_valid_hosted_content(),),
        continuation=MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        ),
    )
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphTeamsChannelHostedContentClient(page=malformed_page, http=MagicMock())
        ),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        integration.read_teams_channel_hosted_contents_page(message=_valid_active_message())
    assert exc.value.__cause__ is None


def test_integration_custom_client_cross_message_continuation_rejected() -> None:
    wrong_continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
            f"{_QUOTED_CHANNEL}/messages/{_QUOTED_OTHER_MESSAGE_ID}/hostedContents?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    page = MsGraphTeamsChannelHostedContentPage(
        team_remote_id=_TEAM_ID,
        channel_remote_id=_CHANNEL_ID,
        message_remote_id=_MESSAGE_ID,
        thread_root_remote_id=_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.ROOT,
        message_revision=_REVISION,
        items=(_valid_hosted_content(),),
        continuation=wrong_continuation,
    )
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphTeamsChannelHostedContentClient(page=page, http=MagicMock())
        ),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        integration.read_teams_channel_hosted_contents_page(message=_valid_active_message())


class _CountingHostedContentClient(GraphRestClient):
    def __init__(self, page: MsGraphTeamsChannelHostedContentPage, http: MagicMock) -> None:
        super().__init__(_config(), http_client=http)
        self._custom_page = page
        self.call_count = 0
        self.last_continuation: MsGraphKnowledgeContinuation | None = None

    def read_teams_channel_hosted_contents_page(
        self,
        *,
        message: MsGraphTeamsChannelMessage,
        continuation: MsGraphKnowledgeContinuation | None = None,
    ) -> MsGraphTeamsChannelHostedContentPage:
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
            url=(
                f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
                f"{_QUOTED_CHANNEL}/messages/{quote(_MESSAGE_ID, safe='')}/hostedContents/delta"
            ),
        ),
        MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=(
                f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
                f"{_QUOTED_CHANNEL}/messages/{_QUOTED_OTHER_MESSAGE_ID}/hostedContents?$skiptoken={_SECRET_TOKEN}"
            ),
        ),
    ],
)
def test_integration_rejects_malformed_input_continuation_before_custom_call(
    continuation: MsGraphKnowledgeContinuation,
) -> None:
    page = _valid_hosted_content_page()
    client = _CountingHostedContentClient(page=page, http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        integration.read_teams_channel_hosted_contents_page(
            message=_valid_active_message(),
            continuation=continuation,
        )
    assert client.call_count == 0
    assert _SECRET_TOKEN not in str(exc.value)


def test_integration_valid_hosted_continuation_calls_custom_client_once() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_hosted_contents_next_link(),
    )
    page = _valid_hosted_content_page(continuation=continuation)
    client = _CountingHostedContentClient(page=page, http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    returned = integration.read_teams_channel_hosted_contents_page(
        message=_valid_active_message(),
        continuation=continuation,
    )
    assert client.call_count == 1
    assert client.last_continuation == continuation
    assert client.last_continuation is not continuation
    assert client.last_continuation is not None
    assert client.last_continuation.url == continuation.url
    assert returned.items[0] is not page.items[0]


def test_custom_client_validation_not_configured() -> None:
    page = _valid_hosted_content_page()
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomTeamsChatHostedContentSuite(page=page),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_VALIDATION_ERROR):
        integration._graph_base_url_for_teams_channel_validation()


def test_integration_rejects_deleted_message() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomTeamsChatHostedContentSuite(page=_valid_hosted_content_page()),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        integration.read_teams_channel_hosted_contents_page(message=_valid_deleted_message())


@pytest.mark.parametrize("max_bytes", [0, ABSOLUTE_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES + 1, True, "10", None])
def test_integration_custom_client_invalid_max_bytes_rejected(max_bytes: object) -> None:
    hosted = _valid_hosted_content()
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomTeamsChatHostedContentSuite(page=_valid_hosted_content_page()),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        integration.read_teams_channel_hosted_content_bytes(
            message=_valid_active_message(),
            hosted_content=hosted,
            max_bytes=max_bytes,  # type: ignore[arg-type]
        )


def test_integration_custom_client_valid_bytes_revalidated() -> None:
    hosted = _valid_hosted_content()
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomTeamsChatHostedContentSuite(page=_valid_hosted_content_page()),
        enabled=True,
    )
    result = integration.read_teams_channel_hosted_content_bytes(
        message=_valid_active_message(),
        hosted_content=hosted,
        max_bytes=1024,
    )
    assert result.data == b"custom"
    assert result.hosted_content_remote_id == _HOSTED_CONTENT_ID


# --- security ---


def test_security_repr_hides_sensitive_fields() -> None:
    hosted = _valid_hosted_content()
    rendered = repr(hosted)
    assert _REVISION not in rendered
    assert _HOSTED_CONTENT_ID not in rendered

    page = _valid_hosted_content_page(
        continuation=MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=_hosted_contents_next_link(),
        )
    )
    page_rendered = repr(page)
    assert _SECRET_TOKEN not in page_rendered
    assert "nextLink" not in page_rendered
    assert "skiptoken" not in page_rendered

    data = b"secret-bytes-payload"
    content = _valid_hosted_content_bytes(data=data)
    content_rendered = repr(content)
    assert data.decode() not in content_rendered
    assert _HOSTED_CONTENT_ID not in content_rendered


def test_default_max_bytes_constants() -> None:
    assert DEFAULT_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES == 10 * 1024 * 1024
    assert ABSOLUTE_TEAMS_CHANNEL_HOSTED_CONTENT_MAX_BYTES == 25 * 1024 * 1024


# --- Teams Channel reply hosted content ---


def test_parse_reply_hosted_content_metadata() -> None:
    hosted = parse_msgraph_teams_channel_hosted_content(
        _hosted_content_payload(),
        message=_valid_active_reply(),
    )
    assert hosted.message_kind is MsGraphTeamsChannelMessageKind.REPLY
    assert hosted.message_remote_id == _REPLY_MESSAGE_ID
    assert hosted.thread_root_remote_id == _ROOT_MESSAGE_ID
    assert hosted.message_revision == _REPLY_REVISION


def test_reply_hosted_contents_page_request_sequence() -> None:
    http = MagicMock()
    _setup_reply_hosted_contents_page(http)
    page = _reader(http).read_teams_channel_hosted_contents_page(
        message=_valid_active_reply(),
        continuation=None,
    )
    assert http.get.call_count == 3
    pre_observation = http.get.call_args_list[0]
    collection = http.get.call_args_list[1]
    post_observation = http.get.call_args_list[2]
    assert pre_observation.args[0] == _REPLY_OBSERVATION_PATH
    assert collection.args[0] == _REPLY_HOSTED_CONTENTS_PATH
    assert post_observation.args[0] == _REPLY_OBSERVATION_PATH
    assert collection.kwargs.get("params") is None
    assert pre_observation.kwargs["headers"] == _PREFER_HEADER
    assert collection.kwargs["headers"] == _PREFER_HEADER
    assert page.message_kind is MsGraphTeamsChannelMessageKind.REPLY
    assert page.message_remote_id == _REPLY_MESSAGE_ID
    assert page.thread_root_remote_id == _ROOT_MESSAGE_ID
    assert page.message_revision == _REPLY_REVISION
    assert page.items[0].message_kind is MsGraphTeamsChannelMessageKind.REPLY


def test_reply_hosted_content_bytes_request_sequence() -> None:
    data = b"reply-bytes"
    http = MagicMock()
    hosted = _setup_reply_hosted_content_bytes(http, file_bytes=data)
    result = _reader(http).read_teams_channel_hosted_content_bytes(
        message=_valid_active_reply(),
        hosted_content=hosted,
        max_bytes=1024,
    )
    assert http.get.call_args_list[0].args[0] == _REPLY_OBSERVATION_PATH
    assert http.get.call_args_list[1].args[0] == _REPLY_OBSERVATION_PATH
    stream_call = http.stream.call_args
    assert stream_call.args[1] == _REPLY_VALUE_PATH
    assert stream_call.kwargs["follow_redirects"] is False
    assert stream_call.kwargs["headers"] == {"Accept": "application/octet-stream"}
    assert "params" not in stream_call.kwargs
    assert result.data == data
    assert result.size_bytes == len(data)
    assert result.content_hash == hashlib.sha256(data).hexdigest()
    assert result.message_kind is MsGraphTeamsChannelMessageKind.REPLY
    assert result.message_remote_id == _REPLY_MESSAGE_ID
    assert result.thread_root_remote_id == _ROOT_MESSAGE_ID
    assert result.message_revision == _REPLY_REVISION


def test_validate_reply_hosted_continuation_accepts_slash_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_reply_hosted_contents_next_link(),
    )
    validated = validate_msgraph_teams_channel_hosted_contents_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        thread_root_id=_ROOT_MESSAGE_ID,
        message_id=_REPLY_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.REPLY,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_reply_hosted_continuation_accepts_odata_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_reply_hosted_contents_next_link(),
    )
    validated = validate_msgraph_teams_channel_hosted_contents_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        thread_root_id=_ROOT_MESSAGE_ID,
        message_id=_REPLY_MESSAGE_ID,
        message_kind=MsGraphTeamsChannelMessageKind.REPLY,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_reply_hosted_continuation_accepts_literal_with_escaped_quotes() -> None:
    reply_id = "reply'quote'part"
    root_id = "root'quote'part"
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_reply_hosted_contents_next_link(reply_id=reply_id, root_id=root_id),
    )
    validated = validate_msgraph_teams_channel_hosted_contents_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        thread_root_id=root_id,
        message_id=reply_id,
        message_kind=MsGraphTeamsChannelMessageKind.REPLY,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


@pytest.mark.parametrize(
    "reply_id",
    [
        "opaque-messages-replies",
        "opaque-replies-only",
        "opaque-hostedContents-replies",
        "messages",
        "replies",
        "members",
        "hostedContents",
        "teams",
        "channels",
        "users",
        "chats",
    ],
)
def test_validate_reply_hosted_continuation_accepts_opaque_reply_id_with_reserved_words(
    reply_id: str,
) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_slash_reply_hosted_contents_next_link(reply_id),
    )
    validated = validate_msgraph_teams_channel_hosted_contents_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        thread_root_id=_ROOT_MESSAGE_ID,
        message_id=reply_id,
        message_kind=MsGraphTeamsChannelMessageKind.REPLY,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_reply_hosted_continuation_accepts_percent_encoded_literal() -> None:
    reply_id = "reply'delta/special"
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_percent_encoded_reply_hosted_contents_next_link(reply_id),
    )
    validated = validate_msgraph_teams_channel_hosted_contents_continuation(
        continuation,
        team_id=_TEAM_ID,
        channel_id=_CHANNEL_ID,
        thread_root_id=_ROOT_MESSAGE_ID,
        message_id=reply_id,
        message_kind=MsGraphTeamsChannelMessageKind.REPLY,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_reply_hosted_continuation_rejects_root_hosted_path_for_reply_context() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_hosted_contents_next_link(),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        validate_msgraph_teams_channel_hosted_contents_continuation(
            continuation,
            team_id=_TEAM_ID,
            channel_id=_CHANNEL_ID,
            thread_root_id=_ROOT_MESSAGE_ID,
            message_id=_REPLY_MESSAGE_ID,
            message_kind=MsGraphTeamsChannelMessageKind.REPLY,
            graph_base_url=_GRAPH_BASE,
        )


def test_validate_root_hosted_continuation_rejects_reply_path_for_root_context() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_reply_hosted_contents_next_link(),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        validate_msgraph_teams_channel_hosted_contents_continuation(
            continuation,
            team_id=_TEAM_ID,
            channel_id=_CHANNEL_ID,
            thread_root_id=_ROOT_MESSAGE_ID,
            message_id=_ROOT_MESSAGE_ID,
            message_kind=MsGraphTeamsChannelMessageKind.ROOT,
            graph_base_url=_GRAPH_BASE,
        )


@pytest.mark.parametrize(
    "url",
    [
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{_QUOTED_ROOT_MESSAGE_ID}/hostedContents?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{quote(_OTHER_ROOT_MESSAGE_ID, safe='')}/replies/"
        f"{_QUOTED_REPLY_MESSAGE_ID}/hostedContents?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{_QUOTED_ROOT_MESSAGE_ID}/replies/"
        f"{quote('other-reply', safe='')}/hostedContents?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{_QUOTED_ROOT_MESSAGE_ID}/replies/"
        f"{_QUOTED_REPLY_MESSAGE_ID}/hostedContents/{_QUOTED_HOSTED_CONTENT_ID}/$value",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{_QUOTED_ROOT_MESSAGE_ID}/replies/"
        f"{_QUOTED_REPLY_MESSAGE_ID}/hostedContents/extra?$skiptoken={_SECRET_TOKEN}",
        f"https://graph.microsoft.com/v1.0/teams/{_QUOTED_TEAM}/channels/"
        f"{_QUOTED_CHANNEL}/messages/{_QUOTED_ROOT_MESSAGE_ID}/replies/"
        f"{_QUOTED_REPLY_MESSAGE_ID}/hostedContents/delta?$skiptoken={_SECRET_TOKEN}",
    ],
)
def test_validate_reply_hosted_continuation_rejects_invalid_urls(url: str) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=url,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_teams_channel_hosted_contents_continuation(
            continuation,
            team_id=_TEAM_ID,
            channel_id=_CHANNEL_ID,
            thread_root_id=_ROOT_MESSAGE_ID,
            message_id=_REPLY_MESSAGE_ID,
            message_kind=MsGraphTeamsChannelMessageKind.REPLY,
            graph_base_url=_GRAPH_BASE,
        )
    assert _SECRET_TOKEN not in str(exc.value)


def test_validate_reply_hosted_continuation_rejects_delta_kind() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_reply_hosted_contents_next_link(),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        validate_msgraph_teams_channel_hosted_contents_continuation(
            continuation,
            team_id=_TEAM_ID,
            channel_id=_CHANNEL_ID,
            thread_root_id=_ROOT_MESSAGE_ID,
            message_id=_REPLY_MESSAGE_ID,
            message_kind=MsGraphTeamsChannelMessageKind.REPLY,
            graph_base_url=_GRAPH_BASE,
        )


@pytest.mark.parametrize(
    "continuation",
    [
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        ),
        MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=_reply_hosted_contents_next_link(),
        ),
        MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=_reply_hosted_contents_next_link(reply_id="other-reply-999"),
        ),
    ],
)
def test_integration_rejects_malformed_reply_hosted_continuation_before_custom_call(
    continuation: MsGraphKnowledgeContinuation,
) -> None:
    page = _valid_reply_hosted_content_page()
    client = _CountingHostedContentClient(page=page, http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        integration.read_teams_channel_hosted_contents_page(
            message=_valid_active_reply(),
            continuation=continuation,
        )
    assert client.call_count == 0
    assert _SECRET_TOKEN not in str(exc.value)


def test_integration_valid_reply_hosted_continuation_calls_custom_client_once() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_reply_hosted_contents_next_link(),
    )
    page = _valid_reply_hosted_content_page(continuation=continuation)
    client = _CountingHostedContentClient(page=page, http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    returned = integration.read_teams_channel_hosted_contents_page(
        message=_valid_active_reply(),
        continuation=continuation,
    )
    assert client.call_count == 1
    assert client.last_continuation == continuation
    assert client.last_continuation is not continuation
    assert client.last_continuation is not None
    assert client.last_continuation.url == continuation.url
    assert returned.items[0] is not page.items[0]
    assert returned.message_kind is MsGraphTeamsChannelMessageKind.REPLY


@pytest.mark.parametrize(
    ("page_overrides", "message"),
    [
        (
            {
                "message_kind": MsGraphTeamsChannelMessageKind.ROOT,
                "message_remote_id": _ROOT_MESSAGE_ID,
                "thread_root_remote_id": _ROOT_MESSAGE_ID,
            },
            _valid_active_reply(),
        ),
        (
            {
                "thread_root_remote_id": _OTHER_ROOT_MESSAGE_ID,
                "items": (
                    MsGraphTeamsChannelHostedContent(
                        **_valid_hosted_content_kwargs(
                            message_remote_id=_REPLY_MESSAGE_ID,
                            thread_root_remote_id=_OTHER_ROOT_MESSAGE_ID,
                            message_kind=MsGraphTeamsChannelMessageKind.REPLY,
                            message_revision=_REPLY_REVISION,
                        )
                    ),
                ),
            },
            _valid_active_reply(),
        ),
        (
            {
                "message_remote_id": "wrong-reply",
                "items": (
                    MsGraphTeamsChannelHostedContent(
                        **_valid_hosted_content_kwargs(
                            message_remote_id="wrong-reply",
                            thread_root_remote_id=_ROOT_MESSAGE_ID,
                            message_kind=MsGraphTeamsChannelMessageKind.REPLY,
                            message_revision=_REPLY_REVISION,
                        )
                    ),
                ),
            },
            _valid_active_reply(),
        ),
        (
            {
                "message_revision": _OTHER_REVISION,
                "items": (
                    MsGraphTeamsChannelHostedContent(
                        **_valid_hosted_content_kwargs(
                            message_remote_id=_REPLY_MESSAGE_ID,
                            thread_root_remote_id=_ROOT_MESSAGE_ID,
                            message_kind=MsGraphTeamsChannelMessageKind.REPLY,
                            message_revision=_OTHER_REVISION,
                        )
                    ),
                ),
            },
            _valid_active_reply(),
        ),
    ],
)
def test_integration_custom_client_malformed_reply_hosted_page_rejected(
    page_overrides: dict[str, object],
    message: MsGraphTeamsChannelMessage,
) -> None:
    if page_overrides.get("message_kind") is MsGraphTeamsChannelMessageKind.ROOT:
        page = MsGraphTeamsChannelHostedContentPage.model_construct(
            team_remote_id=_TEAM_ID,
            channel_remote_id=_CHANNEL_ID,
            message_remote_id=_ROOT_MESSAGE_ID,
            thread_root_remote_id=_ROOT_MESSAGE_ID,
            message_kind=MsGraphTeamsChannelMessageKind.ROOT,
            message_revision=_REPLY_REVISION,
            items=(_valid_reply_hosted_content(),),
        )
    else:
        page = _valid_reply_hosted_content_page(**page_overrides)
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphTeamsChannelHostedContentClient(page=page, http=MagicMock())
        ),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        integration.read_teams_channel_hosted_contents_page(message=message)
    assert exc.value.__cause__ is None
