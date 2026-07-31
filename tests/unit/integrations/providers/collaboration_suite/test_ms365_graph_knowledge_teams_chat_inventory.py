# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Microsoft Graph Teams Chat knowledge-read inventory surface."""

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
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
    MsGraphTeamsChat,
    MsGraphTeamsChatMigrationMode,
    MsGraphTeamsChatPage,
    MsGraphTeamsChatType,
    MsGraphTeamsChatsReader,
    parse_msgraph_teams_chat,
    validate_msgraph_teams_chat,
    validate_msgraph_teams_chat_page,
    validate_msgraph_teams_chats_continuation,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_MAILBOX_USER_ID = "user@contoso.com"
_OTHER_MAILBOX_USER_ID = "other@contoso.com"
_CHAT_ID = "chat-abc-123"
_OTHER_CHAT_ID = "other-chat"
_OPAQUE_CHAT_ID = "19:meeting_abc@thread.v2/special+id"
_QUOTED_MAILBOX = quote(_MAILBOX_USER_ID, safe="")
_QUOTED_OTHER_MAILBOX = quote(_OTHER_MAILBOX_USER_ID, safe="")
_SECRET_TOKEN = "secret-skiptoken-value"
_TOPIC = "Project Discussion"
_HIDDEN_TOPIC = "Hidden Topic"
_TENANT_ID = "tenant-guid-0001"
_CREATED_AT_STR = "2026-01-01T00:00:00Z"
_UPDATED_AT_STR = "2026-01-02T00:00:00Z"
_CREATED_AT = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_UPDATED_AT = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
_ROOT_PATH = f"/users/{_QUOTED_MAILBOX}/chats"
_PREFER_HEADER = {"Prefer": "include-unknown-enum-members"}
_SAFE_ERROR = "unexpected Microsoft Graph Teams chats response"
_REQUEST_ERROR = "invalid Microsoft Graph Teams chats request"
_CONT_ERROR = "invalid Microsoft Graph Teams chats continuation"
_VALIDATION_ERROR = "Microsoft Graph Teams Chat validation is not configured"
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
    resolved = path or f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats"
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


def _chat_payload(
    *,
    chat_id: str = _CHAT_ID,
    chat_type: str = "oneOnOne",
    created_at: str = _CREATED_AT_STR,
    last_updated_at: str = _UPDATED_AT_STR,
    is_hidden: bool = False,
    topic: str | None | object = _MISSING,
    tenant_id: str | None | object = _MISSING,
    original_created_at: str | None | object = _MISSING,
    migration_mode: str | None | object = _MISSING,
    online_meeting_info: dict[str, Any] | None | object = _MISSING,
    extra_field: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": chat_id,
        "chatType": chat_type,
        "createdDateTime": created_at,
        "lastUpdatedDateTime": last_updated_at,
        "isHiddenForAllMembers": is_hidden,
    }
    if topic is not _MISSING:
        payload["topic"] = topic
    if tenant_id is not _MISSING:
        payload["tenantId"] = tenant_id
    if original_created_at is not _MISSING:
        payload["originalCreatedDateTime"] = original_created_at
    if migration_mode is not _MISSING:
        payload["migrationMode"] = migration_mode
    if online_meeting_info is not _MISSING:
        payload["onlineMeetingInfo"] = online_meeting_info
    if extra_field is not None:
        payload["unknownField"] = extra_field
    return payload


def _reader(http: MagicMock) -> MsGraphTeamsChatsReader:
    return MsGraphTeamsChatsReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
    )


def _graph_client(http: MagicMock) -> GraphRestClient:
    return GraphRestClient(_config(), http_client=http)


def _parse_chat(payload: dict[str, Any]) -> MsGraphTeamsChat:
    return parse_msgraph_teams_chat(payload, expected_mailbox_user_id=_MAILBOX_USER_ID)


def _valid_chat(**overrides: object) -> MsGraphTeamsChat:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "remote_id": _CHAT_ID,
        "chat_type": MsGraphTeamsChatType.ONE_ON_ONE,
        "topic": None,
        "tenant_id": None,
        "created_at": _CREATED_AT,
        "last_updated_at": _UPDATED_AT,
        "original_created_at": None,
        "is_hidden_for_all_members": False,
        "migration_mode": None,
        "has_online_meeting_info": False,
    }
    defaults.update(overrides)
    return MsGraphTeamsChat(**defaults)  # type: ignore[arg-type]


def _validate_page(page: MsGraphTeamsChatPage) -> MsGraphTeamsChatPage:
    return validate_msgraph_teams_chat_page(
        page,
        mailbox_user_id=_MAILBOX_USER_ID,
        graph_base_url=_GRAPH_BASE,
    )


def _assert_safe_provider_error(exc: BaseException) -> None:
    assert str(exc.value if isinstance(exc, pytest.ExceptionInfo) else exc) == _SAFE_ERROR
    cause = exc.value.__cause__ if isinstance(exc, pytest.ExceptionInfo) else exc.__cause__
    assert cause is None
    message = str(exc.value if isinstance(exc, pytest.ExceptionInfo) else exc)
    for forbidden in (
        _MAILBOX_USER_ID,
        _CHAT_ID,
        _TOPIC,
        _HIDDEN_TOPIC,
        _TENANT_ID,
        "Authorization",
        "access token",
        "nextLink",
        _SECRET_TOKEN,
    ):
        assert forbidden not in message


# --- parser success ---


def test_parse_one_on_one_chat() -> None:
    chat = _parse_chat(_chat_payload(chat_type="oneOnOne"))
    assert chat.chat_type is MsGraphTeamsChatType.ONE_ON_ONE
    assert chat.remote_id == _CHAT_ID


def test_parse_group_chat() -> None:
    chat = _parse_chat(_chat_payload(chat_type="group", chat_id=_OTHER_CHAT_ID))
    assert chat.chat_type is MsGraphTeamsChatType.GROUP
    assert chat.remote_id == _OTHER_CHAT_ID


def test_parse_meeting_chat() -> None:
    chat = _parse_chat(_chat_payload(chat_type="meeting"))
    assert chat.chat_type is MsGraphTeamsChatType.MEETING


def test_parse_unknown_future_chat_type() -> None:
    chat = _parse_chat(_chat_payload(chat_type="futureChatType"))
    assert chat.chat_type is MsGraphTeamsChatType.UNKNOWN


def test_parse_topic_present() -> None:
    chat = _parse_chat(_chat_payload(topic=_TOPIC))
    assert chat.topic == _TOPIC


def test_parse_topic_absent() -> None:
    chat = _parse_chat(_chat_payload())
    assert chat.topic is None


def test_parse_topic_null() -> None:
    chat = _parse_chat(_chat_payload(topic=None))
    assert chat.topic is None


def test_parse_topic_empty_becomes_none() -> None:
    chat = _parse_chat(_chat_payload(topic="   "))
    assert chat.topic is None


def test_parse_topic_trimmed() -> None:
    chat = _parse_chat(_chat_payload(topic=f"  {_TOPIC}  "))
    assert chat.topic == _TOPIC


def test_parse_tenant_id_present() -> None:
    chat = _parse_chat(_chat_payload(tenant_id=_TENANT_ID))
    assert chat.tenant_id == _TENANT_ID


def test_parse_tenant_id_absent() -> None:
    chat = _parse_chat(_chat_payload())
    assert chat.tenant_id is None


def test_parse_migration_in_progress() -> None:
    chat = _parse_chat(_chat_payload(migration_mode="inProgress"))
    assert chat.migration_mode is MsGraphTeamsChatMigrationMode.IN_PROGRESS


def test_parse_migration_completed() -> None:
    chat = _parse_chat(_chat_payload(migration_mode="completed"))
    assert chat.migration_mode is MsGraphTeamsChatMigrationMode.COMPLETED


def test_parse_unknown_migration_mode() -> None:
    chat = _parse_chat(_chat_payload(migration_mode="futureMode"))
    assert chat.migration_mode is MsGraphTeamsChatMigrationMode.UNKNOWN


def test_parse_migration_absent() -> None:
    chat = _parse_chat(_chat_payload())
    assert chat.migration_mode is None


def test_parse_online_meeting_info_present() -> None:
    chat = _parse_chat(
        _chat_payload(online_meeting_info={"calendarEventId": "evt-1"})
    )
    assert chat.has_online_meeting_info is True


def test_parse_online_meeting_info_absent() -> None:
    chat = _parse_chat(_chat_payload())
    assert chat.has_online_meeting_info is False


def test_parse_online_meeting_info_null() -> None:
    chat = _parse_chat(_chat_payload(online_meeting_info=None))
    assert chat.has_online_meeting_info is False


def test_parse_timestamps_utc_from_z_suffix() -> None:
    chat = _parse_chat(_chat_payload())
    assert chat.created_at == _CREATED_AT
    assert chat.last_updated_at == _UPDATED_AT
    assert chat.created_at.tzinfo == timezone.utc
    assert chat.last_updated_at.tzinfo == timezone.utc


def test_parse_timestamps_normalized_to_utc() -> None:
    chat = _parse_chat(
        _chat_payload(
            created_at="2026-01-01T02:00:00+02:00",
            last_updated_at="2026-01-02T14:00:00+02:00",
        )
    )
    assert chat.created_at == _CREATED_AT
    assert chat.last_updated_at == datetime(2026, 1, 2, 12, 0, 0, tzinfo=timezone.utc)


def test_parse_original_created_at_present() -> None:
    chat = _parse_chat(
        _chat_payload(original_created_at="2025-12-31T23:00:00Z")
    )
    assert chat.original_created_at == datetime(2025, 12, 31, 23, 0, 0, tzinfo=timezone.utc)


def test_parse_is_hidden_flag() -> None:
    chat = _parse_chat(_chat_payload(is_hidden=True))
    assert chat.is_hidden_for_all_members is True


def test_parse_opaque_chat_id() -> None:
    chat = _parse_chat(_chat_payload(chat_id=_OPAQUE_CHAT_ID))
    assert chat.remote_id == _OPAQUE_CHAT_ID


def test_parse_expected_mailbox_user_id_preserved() -> None:
    chat = _parse_chat(_chat_payload())
    assert chat.mailbox_user_id == _MAILBOX_USER_ID


def test_parse_unknown_provider_fields_not_retained() -> None:
    chat = _parse_chat(_chat_payload(extra_field="should-not-appear"))
    assert not hasattr(chat, "unknownField")
    dumped = chat.model_dump()
    assert "unknownField" not in dumped


def test_sensitive_fields_hidden_from_repr() -> None:
    chat = _valid_chat(topic=_HIDDEN_TOPIC, tenant_id=_TENANT_ID)
    chat_with_fields = _parse_chat(
        _chat_payload(topic=_HIDDEN_TOPIC, tenant_id=_TENANT_ID)
    )
    for item in (chat, chat_with_fields):
        rendered = repr(item)
        assert _HIDDEN_TOPIC not in rendered
        assert _TENANT_ID not in rendered


def test_raw_provider_payload_not_stored() -> None:
    payload = _chat_payload()
    chat = _parse_chat(payload)
    assert not hasattr(chat, "__pydantic_extra__") or not chat.__pydantic_extra__


# --- malformed provider ---


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-dict",
        {},
        {"chatType": "oneOnOne"},
        {"id": 123, "chatType": "oneOnOne"},
        {"id": "", "chatType": "oneOnOne"},
        {"id": _CHAT_ID},
        {"id": _CHAT_ID, "chatType": 123},
        {"id": _CHAT_ID, "chatType": ""},
        {"id": _CHAT_ID, "chatType": "oneOnOne", "createdDateTime": _CREATED_AT_STR},
        {
            "id": _CHAT_ID,
            "chatType": "oneOnOne",
            "createdDateTime": _CREATED_AT_STR,
            "lastUpdatedDateTime": _UPDATED_AT_STR,
        },
        {
            "id": _CHAT_ID,
            "chatType": "oneOnOne",
            "createdDateTime": _CREATED_AT_STR,
            "lastUpdatedDateTime": _UPDATED_AT_STR,
            "isHiddenForAllMembers": 1,
        },
        {
            "id": _CHAT_ID,
            "chatType": "oneOnOne",
            "createdDateTime": "not-a-datetime",
            "lastUpdatedDateTime": _UPDATED_AT_STR,
            "isHiddenForAllMembers": False,
        },
        {
            "id": _CHAT_ID,
            "chatType": "oneOnOne",
            "createdDateTime": "2026-01-01T00:00:00",
            "lastUpdatedDateTime": _UPDATED_AT_STR,
            "isHiddenForAllMembers": False,
        },
        {
            "id": _CHAT_ID,
            "chatType": "oneOnOne",
            "createdDateTime": _CREATED_AT_STR,
            "lastUpdatedDateTime": "2025-12-31T00:00:00Z",
            "isHiddenForAllMembers": False,
        },
        {
            "id": _CHAT_ID,
            "chatType": "oneOnOne",
            "createdDateTime": _CREATED_AT_STR,
            "lastUpdatedDateTime": _UPDATED_AT_STR,
            "isHiddenForAllMembers": False,
            "topic": "bad\x00topic",
        },
        {
            "id": _CHAT_ID,
            "chatType": "oneOnOne",
            "createdDateTime": _CREATED_AT_STR,
            "lastUpdatedDateTime": _UPDATED_AT_STR,
            "isHiddenForAllMembers": False,
            "tenantId": "",
        },
        {
            "id": _CHAT_ID,
            "chatType": "oneOnOne",
            "createdDateTime": _CREATED_AT_STR,
            "lastUpdatedDateTime": _UPDATED_AT_STR,
            "isHiddenForAllMembers": False,
            "onlineMeetingInfo": "not-a-dict",
        },
        {
            "id": _CHAT_ID,
            "chatType": "oneOnOne",
            "createdDateTime": _CREATED_AT_STR,
            "lastUpdatedDateTime": _UPDATED_AT_STR,
            "isHiddenForAllMembers": False,
            "migrationMode": "bad\x00mode",
        },
    ],
)
def test_malformed_provider_payload_rejected(payload: object) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        parse_msgraph_teams_chat(payload, expected_mailbox_user_id=_MAILBOX_USER_ID)
    _assert_safe_provider_error(exc)


def test_parse_topic_over_limit_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        _parse_chat(_chat_payload(topic="x" * 4097))


def test_parse_tenant_id_over_limit_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        _parse_chat(_chat_payload(tenant_id="x" * 2049))


def test_malformed_page_model_construct_missing_mailbox_user_id() -> None:
    malformed = MsGraphTeamsChatPage.model_construct(items=(_valid_chat(),))
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_wrong_mailbox_user_id() -> None:
    malformed = MsGraphTeamsChatPage.model_construct(
        mailbox_user_id=_OTHER_MAILBOX_USER_ID,
        items=(_valid_chat(),),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    _assert_safe_provider_error(exc)


def test_validate_page_returns_new_instance() -> None:
    page = MsGraphTeamsChatPage(mailbox_user_id=_MAILBOX_USER_ID, items=(_valid_chat(),))
    validated = _validate_page(page)
    assert validated == page
    assert validated is not page
    assert validated.items[0] is not page.items[0]


# --- page model ---


def test_page_empty_tuple() -> None:
    page = MsGraphTeamsChatPage(mailbox_user_id=_MAILBOX_USER_ID, items=())
    assert page.items == ()
    assert page.has_more is False


def test_page_multiple_chats() -> None:
    page = MsGraphTeamsChatPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        items=(
            _valid_chat(remote_id="c1"),
            _valid_chat(remote_id="c2", chat_type=MsGraphTeamsChatType.GROUP),
        ),
    )
    assert len(page.items) == 2


def test_page_has_more_false() -> None:
    page = MsGraphTeamsChatPage(mailbox_user_id=_MAILBOX_USER_ID, items=())
    assert page.has_more is False


def test_page_has_more_true() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(),
    )
    page = MsGraphTeamsChatPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        items=(),
        continuation=continuation,
    )
    assert page.has_more is True


def test_page_duplicate_chat_ids_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphTeamsChatPage(
            mailbox_user_id=_MAILBOX_USER_ID,
            items=(
                _valid_chat(remote_id="same"),
                _valid_chat(remote_id="same", chat_type=MsGraphTeamsChatType.GROUP),
            ),
        )


def test_page_cross_mailbox_item_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphTeamsChatPage(
            mailbox_user_id=_MAILBOX_USER_ID,
            items=(_valid_chat(mailbox_user_id=_OTHER_MAILBOX_USER_ID),),
        )


def test_page_items_as_list_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphTeamsChatPage(
            mailbox_user_id=_MAILBOX_USER_ID,
            items=[_valid_chat()],  # type: ignore[arg-type]
        )


def test_page_item_wrong_type_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphTeamsChatPage(
            mailbox_user_id=_MAILBOX_USER_ID,
            items=("not-a-chat",),  # type: ignore[arg-type]
        )


def test_page_delta_continuation_rejected() -> None:
    delta = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_next_link(),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphTeamsChatPage(
            mailbox_user_id=_MAILBOX_USER_ID,
            items=(),
            continuation=delta,
        )


def test_token_hidden_from_repr() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(),
    )
    page = MsGraphTeamsChatPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        items=(),
        continuation=continuation,
    )
    assert _SECRET_TOKEN not in repr(page)
    assert _SECRET_TOKEN not in repr(continuation)


@pytest.mark.parametrize(
    "chat_kwargs",
    [
        {"remote_id": None},
        {"chat_type": "group"},
        {"is_hidden_for_all_members": "yes"},
        {"created_at": "2026-01-01T00:00:00Z"},
    ],
)
def test_malformed_chat_model_construct_rejected(
    chat_kwargs: dict[str, object],
) -> None:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "remote_id": _CHAT_ID,
        "chat_type": MsGraphTeamsChatType.ONE_ON_ONE,
        "topic": None,
        "tenant_id": None,
        "created_at": _CREATED_AT,
        "last_updated_at": _UPDATED_AT,
        "original_created_at": None,
        "is_hidden_for_all_members": False,
        "migration_mode": None,
        "has_online_meeting_info": False,
    }
    defaults.update(chat_kwargs)
    malformed = MsGraphTeamsChat.model_construct(**defaults)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_teams_chat(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_missing_items() -> None:
    malformed = MsGraphTeamsChatPage.model_construct(mailbox_user_id=_MAILBOX_USER_ID)
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_items_as_list() -> None:
    malformed = MsGraphTeamsChatPage.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        items=[_valid_chat()],
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_malformed_chat() -> None:
    bad_chat = MsGraphTeamsChat.model_construct(
        mailbox_user_id="bad\x00id",
        remote_id=_CHAT_ID,
        chat_type=MsGraphTeamsChatType.ONE_ON_ONE,
        topic=None,
        tenant_id=None,
        created_at=_CREATED_AT,
        last_updated_at=_UPDATED_AT,
        original_created_at=None,
        is_hidden_for_all_members=False,
        migration_mode=None,
        has_online_meeting_info=False,
    )
    malformed = MsGraphTeamsChatPage.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
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
    malformed = MsGraphTeamsChatPage.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        items=(_valid_chat(),),
        continuation=bad_continuation,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    _assert_safe_provider_error(exc)


def test_validate_page_rejects_wrong_mailbox_user_id() -> None:
    page = MsGraphTeamsChatPage(mailbox_user_id=_MAILBOX_USER_ID, items=(_valid_chat(),))
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        validate_msgraph_teams_chat_page(
            page,
            mailbox_user_id=_OTHER_MAILBOX_USER_ID,
            graph_base_url=_GRAPH_BASE,
        )


# --- request tests ---


def test_request_path_top_and_prefer_header() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    _reader(http).read_chats_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        continuation=None,
        limit=50,
    )
    call = http.get.call_args
    assert call.args[0] == _ROOT_PATH
    assert call.kwargs["params"]["$top"] == 50
    assert call.kwargs["headers"] == _PREFER_HEADER
    assert "$select" not in call.kwargs["params"]
    assert "$expand" not in call.kwargs["params"]
    assert "$filter" not in call.kwargs["params"]
    assert "/messages" not in call.args[0]


def test_empty_page_request() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    page = _reader(http).read_chats_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        continuation=None,
        limit=50,
    )
    assert page.items == ()
    assert page.has_more is False


def test_paging_request_returns_continuation() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(
            value=[_chat_payload()],
            next_link=_next_link(),
        )
    )
    page = _reader(http).read_chats_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        continuation=None,
        limit=50,
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
    _reader(http).read_chats_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        continuation=continuation,
        limit=50,
    )
    assert http.get.call_args.args[0] == _next_link()
    assert "params" not in http.get.call_args.kwargs
    assert http.get.call_args.kwargs["headers"] == _PREFER_HEADER


@pytest.mark.parametrize("limit", [0, 51, True, "50"])
def test_invalid_limit_rejected_before_http(limit: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_chats_page(
            mailbox_user_id=_MAILBOX_USER_ID,
            continuation=None,
            limit=limit,  # type: ignore[arg-type]
        )
    http.get.assert_not_called()


@pytest.mark.parametrize("mailbox_user_id", ["", "  ", "bad\x00id", 123])
def test_invalid_mailbox_user_id_rejected_before_http(mailbox_user_id: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_chats_page(
            mailbox_user_id=mailbox_user_id,  # type: ignore[arg-type]
            continuation=None,
            limit=50,
        )
    http.get.assert_not_called()


# --- continuation tests ---


def test_validate_continuation_same_user_slash_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(),
    )
    validated = validate_msgraph_teams_chats_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation


def test_validate_continuation_same_user_odata_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(
            path=(
                f"https://graph.microsoft.com/v1.0/users('{_MAILBOX_USER_ID}')/chats"
            )
        ),
    )
    validated = validate_msgraph_teams_chats_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation


def test_validate_continuation_case_insensitive_resource_names() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(
            path=f"https://graph.microsoft.com/v1.0/Users/{_QUOTED_MAILBOX}/Chats"
        ),
    )
    validated = validate_msgraph_teams_chats_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation


@pytest.mark.parametrize(
    "url",
    [
        _next_link(
            path=(
                f"https://graph.microsoft.com/v1.0/users/{_QUOTED_OTHER_MAILBOX}/chats"
            )
        ),
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/messages?$skiptoken={_SECRET_TOKEN}",
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders?"
            f"$skiptoken={_SECRET_TOKEN}"
        ),
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats/"
            f"{quote(_CHAT_ID, safe='')}/messages?$skiptoken={_SECRET_TOKEN}"
        ),
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats/delta?"
            f"$deltatoken={_SECRET_TOKEN}"
        ),
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats/"
            f"extra?$skiptoken={_SECRET_TOKEN}"
        ),
        "https://graph.microsoft.com/v1.0/drives/drive-1/root/delta?$skiptoken=x",
    ],
)
def test_rejects_invalid_chats_continuation(url: str) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=url,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_teams_chats_continuation(
            continuation,
            mailbox_user_id=_MAILBOX_USER_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert _SECRET_TOKEN not in str(exc.value)
    assert _MAILBOX_USER_ID not in str(exc.value)
    assert exc.value.__cause__ is None


def test_delta_continuation_rejected_in_validator() -> None:
    delta = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_next_link(),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        validate_msgraph_teams_chats_continuation(
            delta,
            mailbox_user_id=_MAILBOX_USER_ID,
            graph_base_url=_GRAPH_BASE,
        )


def test_invalid_continuation_rejected_before_http() -> None:
    http = MagicMock()
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(
            path=(
                f"https://graph.microsoft.com/v1.0/users/{_QUOTED_OTHER_MAILBOX}/chats"
            )
        ),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        _reader(http).read_chats_page(
            mailbox_user_id=_MAILBOX_USER_ID,
            continuation=continuation,
            limit=50,
        )
    http.get.assert_not_called()


# --- delegation ---


def test_graph_rest_client_delegates_teams_chats() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(value=[_chat_payload()])
    )
    page = _graph_client(http).read_teams_chats_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert len(page.items) == 1
    assert page.items[0].remote_id == _CHAT_ID


def test_collaboration_suite_delegates_teams_chats() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    suite = _Ms365GraphCollaborationSuite(_graph_client(http))
    page = suite.read_teams_chats_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert page.items == ()


def test_integration_delegates_teams_chats() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(_graph_client(http)),
        enabled=True,
    )
    page = integration.read_teams_chats_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert page.items == ()


def test_transport_and_reader_share_injected_http_client() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    client = _graph_client(http)
    client.read_teams_chats_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert client._knowledge_transport._http_client is http
    assert client._teams_chats_reader._transport._http_client is http
    http.get.assert_called_once()


def test_no_new_http_client_created() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    client = _graph_client(http)
    client.read_teams_chats_page(mailbox_user_id=_MAILBOX_USER_ID)
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
    result = client.list_messages(_MAILBOX_USER_ID, folder="inbox", limit=5)
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
    message = client.get_message(_MAILBOX_USER_ID, "msg-1")
    assert message.id == "msg-1"


class _CustomSuiteWithoutChats(CollaborationSuite):
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


class _CustomGraphChatsClient(GraphRestClient):
    def __init__(self, page: MsGraphTeamsChatPage, http: MagicMock) -> None:
        super().__init__(_config(), http_client=http)
        self._custom_page = page

    def read_teams_chats_page(
        self,
        *,
        mailbox_user_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 50,
    ) -> MsGraphTeamsChatPage:
        return self._custom_page


class _CustomChatsSuite(CollaborationSuite):
    def __init__(self, page: MsGraphTeamsChatPage) -> None:
        self._page = page

    def read_teams_chats_page(
        self,
        *,
        mailbox_user_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 50,
    ) -> MsGraphTeamsChatPage:
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


def test_custom_client_without_chats_capability_fails() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomSuiteWithoutChats(),
        enabled=True,
    )
    with pytest.raises(
        IntegrationConfigurationError,
        match="Microsoft Graph integration does not expose Teams chats capability",
    ):
        integration.read_teams_chats_page(mailbox_user_id=_MAILBOX_USER_ID)


def test_custom_client_malformed_page_rejected() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphChatsClient(
                page=MsGraphTeamsChatPage.model_construct(mailbox_user_id=_MAILBOX_USER_ID),
                http=MagicMock(),
            )
        ),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        integration.read_teams_chats_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert exc.value.__cause__ is None


def test_custom_client_valid_page_revalidated() -> None:
    supplied = MsGraphTeamsChatPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        items=(_valid_chat(),),
    )
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphChatsClient(page=supplied, http=MagicMock())
        ),
        enabled=True,
    )
    returned = integration.read_teams_chats_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert returned == supplied
    assert returned is not supplied
    assert returned.items[0] is not supplied.items[0]


def test_custom_client_validation_not_configured() -> None:
    page = MsGraphTeamsChatPage(mailbox_user_id=_MAILBOX_USER_ID, items=())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomChatsSuite(page=page),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_VALIDATION_ERROR):
        integration._graph_base_url_for_teams_chat_validation()


class _CountingChatsClient(GraphRestClient):
    def __init__(self, page: MsGraphTeamsChatPage, http: MagicMock) -> None:
        super().__init__(_config(), http_client=http)
        self._custom_page = page
        self.call_count = 0
        self.last_continuation: MsGraphKnowledgeContinuation | None = None

    def read_teams_chats_page(
        self,
        *,
        mailbox_user_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 50,
    ) -> MsGraphTeamsChatPage:
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
            url=_next_link(path=f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats/delta"),
        ),
        MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=_next_link(
                path=f"https://graph.microsoft.com/v1.0/users/{_QUOTED_OTHER_MAILBOX}/chats"
            ),
        ),
    ],
)
def test_integration_rejects_malformed_continuation_before_custom_call(
    continuation: MsGraphKnowledgeContinuation,
) -> None:
    page = MsGraphTeamsChatPage(mailbox_user_id=_MAILBOX_USER_ID, items=(_valid_chat(),))
    client = _CountingChatsClient(page=page, http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        integration.read_teams_chats_page(
            mailbox_user_id=_MAILBOX_USER_ID,
            continuation=continuation,
            limit=50,
        )
    assert client.call_count == 0
    assert _SECRET_TOKEN not in str(exc.value)


def test_integration_valid_continuation_calls_custom_client_once() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(path=f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats"),
    )
    page = MsGraphTeamsChatPage(mailbox_user_id=_MAILBOX_USER_ID, items=(_valid_chat(),))
    client = _CountingChatsClient(page=page, http=MagicMock())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(client),
        enabled=True,
    )
    returned = integration.read_teams_chats_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        continuation=continuation,
        limit=50,
    )
    assert client.call_count == 1
    assert client.last_continuation == continuation
    assert client.last_continuation is not continuation
    assert client.last_continuation is not None
    assert client.last_continuation.url == continuation.url
    assert returned.items[0] is not page.items[0]


# --- security ---


# --- chat reference ---


def test_chat_reference_valid_construction() -> None:
    from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
        MsGraphTeamsChatReference,
        validate_msgraph_teams_chat_reference,
    )

    ref = MsGraphTeamsChatReference(
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_remote_id=_CHAT_ID,
    )
    assert ref.mailbox_user_id == _MAILBOX_USER_ID
    assert ref.chat_remote_id == _CHAT_ID
    validated = validate_msgraph_teams_chat_reference(ref)
    assert validated == ref
    assert validated is not ref


def test_chat_reference_mapping_validation() -> None:
    from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
        validate_msgraph_teams_chat_reference,
    )

    validated = validate_msgraph_teams_chat_reference(
        {"mailbox_user_id": _MAILBOX_USER_ID, "chat_remote_id": _CHAT_ID}
    )
    assert validated.mailbox_user_id == _MAILBOX_USER_ID
    assert validated.chat_remote_id == _CHAT_ID


def test_chat_reference_frozen_and_extra_key_rejected() -> None:
    from pydantic import ValidationError

    from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
        MsGraphTeamsChatReference,
    )

    ref = MsGraphTeamsChatReference(
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_remote_id=_CHAT_ID,
    )
    with pytest.raises(ValidationError):
        ref.mailbox_user_id = "other@contoso.com"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        MsGraphTeamsChatReference(
            mailbox_user_id=_MAILBOX_USER_ID,
            chat_remote_id=_CHAT_ID,
            topic=_TOPIC,
        )


def test_chat_reference_invalid_ids_rejected() -> None:
    from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
        validate_msgraph_teams_chat_reference,
    )

    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_teams_chat_reference(
            {"mailbox_user_id": "", "chat_remote_id": _CHAT_ID}
        )
    _assert_safe_provider_error(exc)
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_teams_chat_reference(
            {"mailbox_user_id": _MAILBOX_USER_ID, "chat_remote_id": ""}
        )
    _assert_safe_provider_error(exc)


def test_chat_reference_model_construct_corruption_rejected() -> None:
    from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
        MsGraphTeamsChatReference,
        validate_msgraph_teams_chat_reference,
    )

    malformed = MsGraphTeamsChatReference.model_construct(
        mailbox_user_id="bad\x00mailbox",
        chat_remote_id=_CHAT_ID,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_teams_chat_reference(malformed)
    _assert_safe_provider_error(exc)


def test_chat_reference_repr_hides_ids() -> None:
    from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
        MsGraphTeamsChatReference,
    )

    ref = MsGraphTeamsChatReference(
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_remote_id=_CHAT_ID,
    )
    rendered = repr(ref)
    assert _MAILBOX_USER_ID not in rendered
    assert _CHAT_ID not in rendered


def test_security_chat_repr_and_errors() -> None:
    chat = _valid_chat(topic=_HIDDEN_TOPIC, tenant_id=_TENANT_ID)
    assert _HIDDEN_TOPIC not in repr(chat)
    assert _TENANT_ID not in repr(chat)
    assert _CHAT_ID in repr(chat)

    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR) as exc:
        _reader(MagicMock()).read_chats_page(
            mailbox_user_id="",
            continuation=None,
            limit=50,
        )
    assert _MAILBOX_USER_ID not in str(exc.value)

    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_teams_chats_continuation(
            MsGraphKnowledgeContinuation(
                kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_next_link(
                    path=(
                        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_OTHER_MAILBOX}/chats"
                    )
                ),
            ),
            mailbox_user_id=_MAILBOX_USER_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert _SECRET_TOKEN not in str(exc.value)
    assert _MAILBOX_USER_ID not in str(exc.value)
