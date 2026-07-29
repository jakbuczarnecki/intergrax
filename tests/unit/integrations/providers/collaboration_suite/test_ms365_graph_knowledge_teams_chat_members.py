# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Microsoft Graph Teams Chat knowledge-read members surface."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    DEFAULT_GRAPH_BASE_URL,
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeTransport,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
    MsGraphTeamsChat,
    MsGraphTeamsChatChanged,
    MsGraphTeamsChatType,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_members import (
    MsGraphTeamsChatMember,
    MsGraphTeamsChatMemberKind,
    MsGraphTeamsChatMemberPage,
    MsGraphTeamsChatMemberRole,
    MsGraphTeamsChatMembersReader,
    parse_msgraph_teams_chat_member,
    validate_msgraph_teams_chat_member,
    validate_msgraph_teams_chat_member_page,
    validate_msgraph_teams_chat_members_continuation,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_MAILBOX_USER_ID = "user@contoso.com"
_OTHER_MAILBOX_USER_ID = "other@contoso.com"
_CHAT_ID = "19:chat-abc-123@thread.v2"
_OTHER_CHAT_ID = "19:other-chat@thread.v2"
_MEMBER_ID = "member-001"
_OTHER_MEMBER_ID = "member-other-002"
_PROVIDER_USER_ID = "aad-user-guid-123"
_OTHER_TENANT_ID = "other-tenant-guid-456"
_DISPLAY_NAME = "Alice Example"
_SECRET_DISPLAY_NAME = "secret-display-name"
_EMAIL = "alice@contoso.com"
_SECRET_EMAIL = "secret-email@contoso.com"
_QUOTED_MAILBOX = quote(_MAILBOX_USER_ID, safe="")
_QUOTED_OTHER_MAILBOX = quote(_OTHER_MAILBOX_USER_ID, safe="")
_QUOTED_CHAT = quote(_CHAT_ID, safe="")
_QUOTED_OTHER_CHAT = quote(_OTHER_CHAT_ID, safe="")
_SECRET_TOKEN = "secret-skiptoken-value"
_MEMBERS_PATH = f"/users/{_QUOTED_MAILBOX}/chats/{_QUOTED_CHAT}/members"
_OBSERVATION_PATH = f"/users/{_QUOTED_MAILBOX}/chats/{_QUOTED_CHAT}"
_SAFE_ERROR = "unexpected Microsoft Graph Teams chat members response"
_CONT_ERROR = "invalid Microsoft Graph Teams chat members continuation"
_CHANGED_ERROR = "Microsoft Graph Teams chat changed during read"
_PREFER_UNKNOWN_ENUM = {"Prefer": "include-unknown-enum-members"}
_VISIBLE_HISTORY = "2024-03-15T08:30:00Z"
_CHAT_LAST_UPDATED = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
_OTHER_CHAT_LAST_UPDATED = datetime(2024, 6, 2, 12, 0, tzinfo=timezone.utc)
_ODATA_AAD = "#microsoft.graph.aadUserConversationMember"
_ODATA_ANONYMOUS = "#microsoft.graph.anonymousGuestConversationMember"
_ODATA_MICROSOFT_ACCOUNT = "#microsoft.graph.microsoftAccountUserConversationMember"
_ODATA_SKYPE = "#microsoft.graph.skypeUserConversationMember"
_ODATA_SKYPE_FOR_BUSINESS = "#microsoft.graph.skypeForBusinessUserConversationMember"
_ODATA_ACS = "#microsoft.graph.azureCommunicationServicesUserConversationMember"
_ODATA_UNKNOWN = "#microsoft.graph.unknownFutureMemberType"


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


def _members_next_link() -> str:
    return (
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats/"
        f"{_QUOTED_CHAT}/members?$skiptoken={_SECRET_TOKEN}"
    )


def _odata_members_next_link(
    chat_id: str = _CHAT_ID,
    mailbox_user_id: str = _MAILBOX_USER_ID,
) -> str:
    escaped_chat = chat_id.replace("'", "''")
    quoted_mailbox = quote(mailbox_user_id, safe="")
    return (
        f"https://graph.microsoft.com/v1.0/users/{quoted_mailbox}/"
        f"chats('{escaped_chat}')/members?$skiptoken={_SECRET_TOKEN}"
    )


def _slash_members_next_link(chat_id: str, mailbox_user_id: str = _MAILBOX_USER_ID) -> str:
    quoted_chat = quote(chat_id, safe="")
    quoted_mailbox = quote(mailbox_user_id, safe="")
    return (
        f"https://graph.microsoft.com/v1.0/users/{quoted_mailbox}/chats/"
        f"{quoted_chat}/members?$skiptoken={_SECRET_TOKEN}"
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
    chat_id: str = _CHAT_ID,
    last_updated: str = "2024-06-01T12:00:00Z",
    chat_type: str = "group",
) -> dict[str, Any]:
    return {
        "id": chat_id,
        "chatType": chat_type,
        "createdDateTime": "2024-01-01T00:00:00Z",
        "lastUpdatedDateTime": last_updated,
        "isHiddenForAllMembers": False,
    }


def _member_payload(
    *,
    member_id: str = _MEMBER_ID,
    odata_type: str = _ODATA_AAD,
    roles: list[str] | None = None,
    user_id: str | None = _PROVIDER_USER_ID,
    tenant_id: str | None = None,
    display_name: str | None = None,
    email: str | None = None,
    visible_history: str | None = None,
    include_user_id: bool = True,
    include_tenant_id: bool = False,
    include_display_name: bool = False,
    include_email: bool = False,
    include_visible_history: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "@odata.type": odata_type,
        "id": member_id,
        "roles": roles if roles is not None else ["owner"],
    }
    if include_user_id:
        payload["userId"] = user_id
    if include_tenant_id:
        payload["tenantId"] = tenant_id
    if include_display_name:
        payload["displayName"] = display_name
    if include_email:
        payload["email"] = email
    if include_visible_history:
        payload["visibleHistoryStartDateTime"] = visible_history
    return payload


def _valid_chat(**overrides: object) -> MsGraphTeamsChat:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "remote_id": _CHAT_ID,
        "chat_type": MsGraphTeamsChatType.GROUP,
        "created_at": datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        "last_updated_at": _CHAT_LAST_UPDATED,
        "is_hidden_for_all_members": False,
        "has_online_meeting_info": False,
    }
    defaults.update(overrides)
    return MsGraphTeamsChat(**defaults)  # type: ignore[arg-type]


def _valid_member_kwargs(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "chat_remote_id": _CHAT_ID,
        "chat_revision": _CHAT_LAST_UPDATED,
        "remote_id": _MEMBER_ID,
        "member_kind": MsGraphTeamsChatMemberKind.AAD_USER,
        "provider_user_id": _PROVIDER_USER_ID,
        "tenant_id": None,
        "display_name": None,
        "email": None,
        "roles": (MsGraphTeamsChatMemberRole.OWNER,),
        "visible_history_start_at": None,
    }
    defaults.update(overrides)
    return defaults


def _valid_member(**overrides: object) -> MsGraphTeamsChatMember:
    return MsGraphTeamsChatMember(**_valid_member_kwargs(**overrides))


def _valid_member_page(**overrides: object) -> MsGraphTeamsChatMemberPage:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "chat_remote_id": _CHAT_ID,
        "chat_revision": _CHAT_LAST_UPDATED,
        "items": (_valid_member(),),
        "continuation": None,
    }
    defaults.update(overrides)
    return MsGraphTeamsChatMemberPage(**defaults)


def _reader(http: MagicMock) -> MsGraphTeamsChatMembersReader:
    return MsGraphTeamsChatMembersReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
    )


def _parse_member(payload: dict[str, Any]) -> MsGraphTeamsChatMember:
    return parse_msgraph_teams_chat_member(payload, chat=_valid_chat())


def _validate_page(page: MsGraphTeamsChatMemberPage) -> MsGraphTeamsChatMemberPage:
    return validate_msgraph_teams_chat_member_page(
        page,
        chat=_valid_chat(),
        graph_base_url=_GRAPH_BASE,
    )


def _setup_members_page(
    http: MagicMock,
    *,
    members: list[dict[str, Any]] | None = None,
    next_link: str | None = None,
) -> None:
    observation = _observation_payload()
    members_payload = _page_payload(
        value=members if members is not None else [_member_payload()],
        next_link=next_link,
    )
    http.get.side_effect = [
        _json_response(payload=observation),
        _json_response(payload=members_payload),
        _json_response(payload=observation),
    ]


def _assert_safe_provider_error(exc: BaseException) -> None:
    assert str(exc) == _SAFE_ERROR
    assert exc.__cause__ is None
    for forbidden in (
        _MAILBOX_USER_ID,
        _CHAT_ID,
        _MEMBER_ID,
        _DISPLAY_NAME,
        _EMAIL,
        _PROVIDER_USER_ID,
        "nextLink",
        _SECRET_TOKEN,
    ):
        assert forbidden not in str(exc)


# --- parser: member kinds ---


@pytest.mark.parametrize(
    ("odata_type", "kind"),
    [
        (_ODATA_AAD, MsGraphTeamsChatMemberKind.AAD_USER),
        (_ODATA_ANONYMOUS, MsGraphTeamsChatMemberKind.ANONYMOUS_GUEST),
        (_ODATA_MICROSOFT_ACCOUNT, MsGraphTeamsChatMemberKind.MICROSOFT_ACCOUNT),
        (_ODATA_SKYPE, MsGraphTeamsChatMemberKind.SKYPE_USER),
        (_ODATA_SKYPE_FOR_BUSINESS, MsGraphTeamsChatMemberKind.SKYPE_FOR_BUSINESS_USER),
        (_ODATA_ACS, MsGraphTeamsChatMemberKind.AZURE_COMMUNICATION_SERVICES_USER),
        (_ODATA_UNKNOWN, MsGraphTeamsChatMemberKind.UNKNOWN),
    ],
)
def test_parse_member_kinds(odata_type: str, kind: MsGraphTeamsChatMemberKind) -> None:
    member = _parse_member(_member_payload(odata_type=odata_type))
    assert member.member_kind is kind


def test_parse_member_roles_owner_guest_and_unknown() -> None:
    member = _parse_member(_member_payload(roles=["owner", "guest", "futureRole"]))
    assert member.roles == (
        MsGraphTeamsChatMemberRole.OWNER,
        MsGraphTeamsChatMemberRole.GUEST,
        MsGraphTeamsChatMemberRole.UNKNOWN,
    )


def test_parse_member_roles_deduplicated() -> None:
    member = _parse_member(_member_payload(roles=["owner", "owner", "guest", "guest"]))
    assert member.roles == (
        MsGraphTeamsChatMemberRole.OWNER,
        MsGraphTeamsChatMemberRole.GUEST,
    )


def test_parse_member_visible_history_start_at() -> None:
    member = _parse_member(
        _member_payload(
            include_visible_history=True,
            visible_history=_VISIBLE_HISTORY,
        )
    )
    assert member.visible_history_start_at == datetime(
        2024, 3, 15, 8, 30, tzinfo=timezone.utc
    )


def test_parse_member_cross_tenant_metadata() -> None:
    member = _parse_member(
        _member_payload(
            include_tenant_id=True,
            tenant_id=_OTHER_TENANT_ID,
            include_display_name=True,
            display_name=_DISPLAY_NAME,
            include_email=True,
            email=_EMAIL,
        )
    )
    assert member.tenant_id == _OTHER_TENANT_ID
    assert member.display_name == _DISPLAY_NAME
    assert member.email == _EMAIL
    assert member.provider_user_id == _PROVIDER_USER_ID
    assert member.chat_remote_id == _CHAT_ID
    assert member.chat_revision == _CHAT_LAST_UPDATED


def test_parse_member_null_user_id() -> None:
    member = _parse_member(_member_payload(include_user_id=True, user_id=None))
    assert member.provider_user_id is None


def test_parse_member_trimmed_display_name_and_email() -> None:
    member = _parse_member(
        _member_payload(
            include_display_name=True,
            display_name="  Alice  ",
            include_email=True,
            email="  alice@contoso.com  ",
        )
    )
    assert member.display_name == "Alice"
    assert member.email == "alice@contoso.com"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"@odata.type": _ODATA_AAD},
        {"@odata.type": _ODATA_AAD, "id": ""},
        {"@odata.type": _ODATA_AAD, "id": _MEMBER_ID},
        _member_payload() | {"roles": "owner"},
        _member_payload() | {"roles": [""]},
        _member_payload() | {"roles": [123]},
        _member_payload() | {"@odata.type": ""},
        _member_payload() | {"@odata.type": 123},
        _member_payload(include_user_id=True, user_id=""),
        _member_payload(include_tenant_id=True, tenant_id=""),
        _member_payload(include_tenant_id=True, tenant_id="   "),
        _member_payload(include_display_name=True, display_name="bad\x00name"),
        _member_payload(include_email=True, email=123),
        _member_payload(include_visible_history=True, visible_history="2024-06-01T12:00:00"),
        _member_payload() | {"id": "\x00bad"},
    ],
)
def test_parse_malformed_provider_payload(payload: object) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        parse_msgraph_teams_chat_member(payload, chat=_valid_chat())
    _assert_safe_provider_error(exc.value)


# --- model and deep validation ---


def test_validate_member_returns_new_instance() -> None:
    original = _valid_member(display_name=_SECRET_DISPLAY_NAME, email=_SECRET_EMAIL)
    validated = validate_msgraph_teams_chat_member(original)
    assert validated == original
    assert validated is not original


@pytest.mark.parametrize(
    "kwargs",
    [
        {"remote_id": None},
        {"member_kind": "aad_user"},
        {"chat_revision": None},
        {"roles": ("owner",)},
        {"roles": ("not-a-role",)},
        {"visible_history_start_at": datetime(2024, 6, 1, 12, 0)},
    ],
)
def test_model_construct_malformed_member_rejected(kwargs: dict[str, object]) -> None:
    malformed = MsGraphTeamsChatMember.model_construct(
        **{**_valid_member_kwargs(), **kwargs}
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_teams_chat_member(malformed)
    assert exc.value.__cause__ is None


def test_roles_deduplicated_at_model_boundary() -> None:
    member = MsGraphTeamsChatMember.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_remote_id=_CHAT_ID,
        chat_revision=_CHAT_LAST_UPDATED,
        remote_id=_MEMBER_ID,
        member_kind=MsGraphTeamsChatMemberKind.AAD_USER,
        roles=(
            MsGraphTeamsChatMemberRole.OWNER,
            MsGraphTeamsChatMemberRole.OWNER,
            MsGraphTeamsChatMemberRole.GUEST,
        ),
    )
    validated = validate_msgraph_teams_chat_member(member)
    assert validated.roles == (
        MsGraphTeamsChatMemberRole.OWNER,
        MsGraphTeamsChatMemberRole.GUEST,
    )


def test_validate_member_page_returns_new_instances() -> None:
    original = _valid_member_page()
    validated = _validate_page(original)
    assert validated == original
    assert validated is not original
    assert validated.items[0] is not original.items[0]


def test_member_page_model_rejects_duplicate_remote_ids() -> None:
    item = _valid_member()
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphTeamsChatMemberPage(
            mailbox_user_id=_MAILBOX_USER_ID,
            chat_remote_id=_CHAT_ID,
            chat_revision=_CHAT_LAST_UPDATED,
            items=(item, item),
        )


def test_page_cross_mailbox_item_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphTeamsChatMemberPage(
            mailbox_user_id=_MAILBOX_USER_ID,
            chat_remote_id=_CHAT_ID,
            chat_revision=_CHAT_LAST_UPDATED,
            items=(_valid_member(mailbox_user_id=_OTHER_MAILBOX_USER_ID),),
        )


def test_page_cross_chat_item_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphTeamsChatMemberPage(
            mailbox_user_id=_MAILBOX_USER_ID,
            chat_remote_id=_CHAT_ID,
            chat_revision=_CHAT_LAST_UPDATED,
            items=(_valid_member(chat_remote_id=_OTHER_CHAT_ID),),
        )


def test_page_stale_chat_revision_rejected() -> None:
    page = MsGraphTeamsChatMemberPage.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_remote_id=_CHAT_ID,
        chat_revision=_OTHER_CHAT_LAST_UPDATED,
        items=(_valid_member(),),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        _validate_page(page)


def test_page_items_as_list_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphTeamsChatMemberPage(
            mailbox_user_id=_MAILBOX_USER_ID,
            chat_remote_id=_CHAT_ID,
            chat_revision=_CHAT_LAST_UPDATED,
            items=[_valid_member()],  # type: ignore[arg-type]
        )


def test_page_delta_continuation_rejected() -> None:
    delta = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_members_next_link(),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphTeamsChatMemberPage(
            mailbox_user_id=_MAILBOX_USER_ID,
            chat_remote_id=_CHAT_ID,
            chat_revision=_CHAT_LAST_UPDATED,
            items=(),
            continuation=delta,
        )


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
        malformed = MsGraphTeamsChatMemberPage.model_construct(
            mailbox_user_id=_MAILBOX_USER_ID,
            chat_remote_id=_CHAT_ID,
            chat_revision=_CHAT_LAST_UPDATED,
        )
    elif page_kwargs.get("nested_malformed"):
        malformed = MsGraphTeamsChatMemberPage.model_construct(
            mailbox_user_id=_MAILBOX_USER_ID,
            chat_remote_id=_CHAT_ID,
            chat_revision=_CHAT_LAST_UPDATED,
            items=(MsGraphTeamsChatMember.model_construct(),),
        )
    else:
        malformed = MsGraphTeamsChatMemberPage.model_construct(
            mailbox_user_id=_MAILBOX_USER_ID,
            chat_remote_id=_CHAT_ID,
            chat_revision=_CHAT_LAST_UPDATED,
            items=(_valid_member(),),
            continuation=page_kwargs["continuation"],
        )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    assert exc.value.__cause__ is None


def test_validate_page_rejects_cross_chat_continuation() -> None:
    wrong_continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats/"
            f"{_QUOTED_OTHER_CHAT}/members?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    page = MsGraphTeamsChatMemberPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_remote_id=_CHAT_ID,
        chat_revision=_CHAT_LAST_UPDATED,
        items=(_valid_member(),),
        continuation=wrong_continuation,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        _validate_page(page)


# --- requests and observation ---


def test_initial_request_members_path_and_headers() -> None:
    http = MagicMock()
    _setup_members_page(http)
    _reader(http).read_members_page(chat=_valid_chat(), continuation=None)
    members_call = http.get.call_args_list[1]
    assert members_call.args[0] == _MEMBERS_PATH
    assert members_call.kwargs.get("params") is None
    assert members_call.kwargs["headers"] == _PREFER_UNKNOWN_ENUM


def test_observation_requests_before_and_after_members_list() -> None:
    http = MagicMock()
    _setup_members_page(http)
    _reader(http).read_members_page(chat=_valid_chat(), continuation=None)
    pre_observation = http.get.call_args_list[0]
    post_observation = http.get.call_args_list[2]
    assert pre_observation.args[0] == _OBSERVATION_PATH
    assert pre_observation.kwargs["headers"] == _PREFER_UNKNOWN_ENUM
    assert post_observation.args[0] == _OBSERVATION_PATH


def test_continuation_request_uses_full_url_without_params() -> None:
    http = MagicMock()
    observation = _observation_payload()
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_members_next_link(),
    )
    http.get.side_effect = [
        _json_response(payload=observation),
        _json_response(payload=_page_payload()),
        _json_response(payload=observation),
    ]
    _reader(http).read_members_page(chat=_valid_chat(), continuation=continuation)
    continuation_call = http.get.call_args_list[1]
    assert continuation_call.args[0] == _members_next_link()
    assert continuation_call.kwargs.get("params") is None
    assert continuation_call.kwargs["headers"] == _PREFER_UNKNOWN_ENUM


def test_first_page_with_continuation() -> None:
    http = MagicMock()
    _setup_members_page(http, next_link=_members_next_link())
    page = _reader(http).read_members_page(chat=_valid_chat(), continuation=None)
    assert page.continuation is not None
    assert page.continuation.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE


def test_final_page_without_continuation() -> None:
    http = MagicMock()
    _setup_members_page(http)
    page = _reader(http).read_members_page(chat=_valid_chat(), continuation=None)
    assert page.continuation is None
    assert len(page.items) == 1


def test_chat_changed_before_members_list() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_observation_payload(last_updated="2024-06-02T12:00:00Z"),
    )
    with pytest.raises(MsGraphTeamsChatChanged, match=_CHANGED_ERROR):
        _reader(http).read_members_page(chat=_valid_chat(), continuation=None)
    http.get.assert_called_once()


def test_chat_changed_after_members_list() -> None:
    http = MagicMock()
    observation_before = _observation_payload()
    observation_after = _observation_payload(last_updated="2024-06-02T12:00:00Z")
    http.get.side_effect = [
        _json_response(payload=observation_before),
        _json_response(payload=_page_payload(value=[_member_payload()])),
        _json_response(payload=observation_after),
    ]
    with pytest.raises(MsGraphTeamsChatChanged, match=_CHANGED_ERROR):
        _reader(http).read_members_page(chat=_valid_chat(), continuation=None)
    assert http.get.call_count == 3


def test_invalid_continuation_rejected_before_members_request() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_observation_payload())
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_OTHER_MAILBOX}/chats/"
            f"{_QUOTED_CHAT}/members?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        _reader(http).read_members_page(chat=_valid_chat(), continuation=continuation)
    http.get.assert_called_once()
    assert http.get.call_args.args[0] == _OBSERVATION_PATH


# --- continuation validation ---


def test_validate_continuation_accepts_next_page_slash_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_members_next_link(),
    )
    validated = validate_msgraph_teams_chat_members_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_id=_CHAT_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_continuation_accepts_odata_key_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_members_next_link(),
    )
    validated = validate_msgraph_teams_chat_members_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_id=_CHAT_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_continuation_accepts_chat_literal_with_escaped_quotes() -> None:
    chat_id = "19:chat'quote'part@thread.v2"
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_odata_members_next_link(chat_id=chat_id),
    )
    validated = validate_msgraph_teams_chat_members_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_id=chat_id,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_continuation_accepts_uppercase_resource_names() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/USERS/{_QUOTED_MAILBOX}/CHATS/"
            f"{_QUOTED_CHAT}/MEMBERS?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    validated = validate_msgraph_teams_chat_members_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_id=_CHAT_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


def test_validate_continuation_accepts_percent_encoded_chat_literal() -> None:
    chat_id = "19:special/chat@thread.v2"
    encoded = quote(chat_id, safe="")
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/"
            f"chats('{encoded}')/members?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    validated = validate_msgraph_teams_chat_members_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        chat_id=chat_id,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation
    assert validated is not continuation


@pytest.mark.parametrize(
    "url",
    [
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_OTHER_MAILBOX}/chats/"
            f"{_QUOTED_CHAT}/members?$skiptoken={_SECRET_TOKEN}"
        ),
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats/"
            f"{_QUOTED_OTHER_CHAT}/members?$skiptoken={_SECRET_TOKEN}"
        ),
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats/"
            f"{_QUOTED_CHAT}/members/delta?$skiptoken={_SECRET_TOKEN}"
        ),
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/messages/"
            f"{_QUOTED_CHAT}/members?$skiptoken={_SECRET_TOKEN}"
        ),
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats/"
            f"{_QUOTED_CHAT}/members/{quote(_MEMBER_ID, safe='')}"
        ),
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/chats/"
            f"{_QUOTED_CHAT}/members/extra?$skiptoken={_SECRET_TOKEN}"
        ),
        "https://graph.microsoft.com/v1.0/drives/drive-1/root/children?$skiptoken=x",
        (
            f"https://graph.microsoft.com/v1.0/users('{_QUOTED_MAILBOX}')"
            f"/chats('unterminated?$skiptoken={_SECRET_TOKEN}"
        ),
    ],
)
def test_validate_continuation_rejects_invalid_urls(url: str) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=url,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_teams_chat_members_continuation(
            continuation,
            mailbox_user_id=_MAILBOX_USER_ID,
            chat_id=_CHAT_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert _SECRET_TOKEN not in str(exc.value)
    assert _MAILBOX_USER_ID not in str(exc.value)
    assert exc.value.__cause__ is None


def test_validate_continuation_rejects_delta_kind() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_members_next_link(),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        validate_msgraph_teams_chat_members_continuation(
            continuation,
            mailbox_user_id=_MAILBOX_USER_ID,
            chat_id=_CHAT_ID,
            graph_base_url=_GRAPH_BASE,
        )


def test_validate_continuation_rejects_wrong_object_type() -> None:
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        validate_msgraph_teams_chat_members_continuation(
            "bad",
            mailbox_user_id=_MAILBOX_USER_ID,
            chat_id=_CHAT_ID,
            graph_base_url=_GRAPH_BASE,
        )


@pytest.mark.parametrize(
    "continuation",
    [
        MsGraphKnowledgeContinuation.model_construct(),
        MsGraphKnowledgeContinuation.model_construct(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        ),
        MsGraphKnowledgeContinuation.model_construct(url=_members_next_link()),
        MsGraphKnowledgeContinuation.model_construct(
            kind="next_page",
            url=_members_next_link(),
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
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_teams_chat_members_continuation(
            continuation,
            mailbox_user_id=_MAILBOX_USER_ID,
            chat_id=_CHAT_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert exc.value.__cause__ is None
    assert _SECRET_TOKEN not in str(exc.value)
    assert _MAILBOX_USER_ID not in str(exc.value)


# --- security ---


def test_security_repr_hides_sensitive_member_fields() -> None:
    member = _valid_member(
        display_name=_SECRET_DISPLAY_NAME,
        email=_SECRET_EMAIL,
        tenant_id=_OTHER_TENANT_ID,
    )
    rendered = repr(member)
    assert _SECRET_DISPLAY_NAME not in rendered
    assert _SECRET_EMAIL not in rendered
    assert _OTHER_TENANT_ID not in rendered
    assert _MEMBER_ID in rendered

    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_members_next_link(),
    )
    page = _valid_member_page(continuation=continuation)
    page_rendered = repr(page)
    assert _SECRET_TOKEN not in page_rendered
    assert "nextLink" not in page_rendered
    assert "skiptoken" not in page_rendered
