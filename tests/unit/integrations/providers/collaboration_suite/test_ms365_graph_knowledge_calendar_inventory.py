# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Microsoft Graph Calendar knowledge-read inventory surface."""

from __future__ import annotations

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
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_inventory import (
    MsGraphCalendar,
    MsGraphCalendarOnlineMeetingProvider,
    MsGraphCalendarPage,
    MsGraphCalendarsReader,
    parse_msgraph_calendar,
    validate_msgraph_calendar,
    validate_msgraph_calendar_page,
    validate_msgraph_calendars_continuation,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_MAILBOX_USER_ID = "user@contoso.com"
_OTHER_MAILBOX_USER_ID = "other@contoso.com"
_CALENDAR_ID = "calendar-abc-123"
_OTHER_CALENDAR_ID = "other-calendar"
_OPAQUE_CALENDAR_ID = "AAMkAGI2TGuLAAA=/special+id"
_QUOTED_MAILBOX = quote(_MAILBOX_USER_ID, safe="")
_QUOTED_OTHER_MAILBOX = quote(_OTHER_MAILBOX_USER_ID, safe="")
_SECRET_TOKEN = "secret-skiptoken-value"
_CALENDAR_NAME = "Calendar"
_HIDDEN_CALENDAR_NAME = "Hidden Calendar"
_CHANGE_KEY = "change-key-xyz"
_VALIDATION_ERROR = "Microsoft Graph Calendar validation is not configured"
_OWNER_ADDRESS = "owner@contoso.com"
_OWNER_NAME = "Owner User"
_ROOT_PATH = f"/users/{_QUOTED_MAILBOX}/calendars"
_SELECT = (
    "id,name,changeKey,isDefaultCalendar,canEdit,canShare,canViewPrivateItems,"
    "isRemovable,owner,allowedOnlineMeetingProviders,defaultOnlineMeetingProvider"
)
_SAFE_ERROR = "unexpected Microsoft Graph calendars response"
_REQUEST_ERROR = "invalid Microsoft Graph calendars request"
_CONT_ERROR = "invalid Microsoft Graph calendars continuation"
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
    resolved = path or f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars"
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


def _calendar_payload(
    *,
    calendar_id: str = _CALENDAR_ID,
    name: str = _CALENDAR_NAME,
    change_key: str = _CHANGE_KEY,
    is_default_calendar: bool = True,
    can_edit: bool = True,
    can_share: bool = True,
    can_view_private_items: bool = False,
    is_removable: bool = False,
    owner: dict[str, Any] | None | object = _MISSING,
    allowed_providers: list[str] | None = None,
    default_provider: str = "teamsForBusiness",
    extra_field: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": calendar_id,
        "name": name,
        "changeKey": change_key,
        "isDefaultCalendar": is_default_calendar,
        "canEdit": can_edit,
        "canShare": can_share,
        "canViewPrivateItems": can_view_private_items,
        "isRemovable": is_removable,
        "allowedOnlineMeetingProviders": (
            ["teamsForBusiness"] if allowed_providers is None else allowed_providers
        ),
        "defaultOnlineMeetingProvider": default_provider,
    }
    if owner is not _MISSING:
        payload["owner"] = owner
    if extra_field is not None:
        payload["unknownField"] = extra_field
    return payload


def _reader(http: MagicMock) -> MsGraphCalendarsReader:
    return MsGraphCalendarsReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
    )


def _graph_client(http: MagicMock) -> GraphRestClient:
    return GraphRestClient(_config(), http_client=http)


def _parse_calendar(payload: dict[str, Any]) -> MsGraphCalendar:
    return parse_msgraph_calendar(payload, expected_mailbox_user_id=_MAILBOX_USER_ID)


def _valid_calendar(**overrides: object) -> MsGraphCalendar:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "remote_id": _CALENDAR_ID,
        "name": _CALENDAR_NAME,
        "change_key": _CHANGE_KEY,
        "is_default_calendar": True,
        "can_edit": True,
        "can_share": True,
        "can_view_private_items": False,
        "is_removable": False,
        "owner": None,
        "allowed_online_meeting_providers": (
            MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
        ),
        "default_online_meeting_provider": MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
    }
    defaults.update(overrides)
    return MsGraphCalendar(**defaults)  # type: ignore[arg-type]


def _validate_page(page: MsGraphCalendarPage) -> MsGraphCalendarPage:
    return validate_msgraph_calendar_page(
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
        _CALENDAR_ID,
        _CALENDAR_NAME,
        _OWNER_ADDRESS,
        _OWNER_NAME,
        _CHANGE_KEY,
        "Authorization",
        "access token",
        "nextLink",
        _SECRET_TOKEN,
    ):
        assert forbidden not in message


# --- parser success ---


def test_parse_default_calendar() -> None:
    calendar = _parse_calendar(_calendar_payload(is_default_calendar=True))
    assert calendar.is_default_calendar is True
    assert calendar.remote_id == _CALENDAR_ID
    assert calendar.name == _CALENDAR_NAME


def test_parse_regular_non_default_calendar() -> None:
    calendar = _parse_calendar(
        _calendar_payload(
            calendar_id=_OTHER_CALENDAR_ID,
            name="Shared Calendar",
            is_default_calendar=False,
        )
    )
    assert calendar.is_default_calendar is False
    assert calendar.remote_id == _OTHER_CALENDAR_ID
    assert calendar.name == "Shared Calendar"


def test_parse_capability_flags() -> None:
    calendar = _parse_calendar(
        _calendar_payload(
            can_edit=False,
            can_share=True,
            can_view_private_items=True,
            is_removable=True,
        )
    )
    assert calendar.can_edit is False
    assert calendar.can_share is True
    assert calendar.can_view_private_items is True
    assert calendar.is_removable is True


def test_parse_owner_present() -> None:
    calendar = _parse_calendar(
        _calendar_payload(
            owner={"address": _OWNER_ADDRESS, "name": _OWNER_NAME}
        )
    )
    assert calendar.owner is not None
    assert calendar.owner.address == _OWNER_ADDRESS
    assert calendar.owner.display_name == _OWNER_NAME


def test_parse_nested_owner_email_address_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        _parse_calendar(
            _calendar_payload(
                owner={"emailAddress": {"address": _OWNER_ADDRESS, "name": _OWNER_NAME}}
            )
        )


def test_parse_owner_missing_address_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        _parse_calendar(_calendar_payload(owner={"name": _OWNER_NAME}))


def test_parse_owner_display_name_over_limit_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        _parse_calendar(
            _calendar_payload(
                owner={"address": _OWNER_ADDRESS, "name": "x" * 1025}
            )
        )


def test_parse_owner_display_name_control_chars_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        _parse_calendar(
            _calendar_payload(
                owner={"address": _OWNER_ADDRESS, "name": "bad\x00name"}
            )
        )


def test_parse_owner_absent() -> None:
    calendar = _parse_calendar(_calendar_payload())
    assert calendar.owner is None


def test_parse_owner_null() -> None:
    calendar = _parse_calendar(_calendar_payload(owner=None))
    assert calendar.owner is None


def test_parse_online_meeting_providers_mapped() -> None:
    calendar = _parse_calendar(
        _calendar_payload(
            allowed_providers=[
                "teamsForBusiness",
                "skypeForBusiness",
                "skypeForConsumer",
            ],
            default_provider="skypeForBusiness",
        )
    )
    assert calendar.allowed_online_meeting_providers == (
        MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
        MsGraphCalendarOnlineMeetingProvider.SKYPE_FOR_BUSINESS,
        MsGraphCalendarOnlineMeetingProvider.SKYPE_FOR_CONSUMER,
    )
    assert (
        calendar.default_online_meeting_provider
        is MsGraphCalendarOnlineMeetingProvider.SKYPE_FOR_BUSINESS
    )


def test_parse_unknown_online_meeting_provider_maps_to_unknown() -> None:
    calendar = _parse_calendar(
        _calendar_payload(
            allowed_providers=["futureProvider"],
            default_provider="futureProvider",
        )
    )
    assert calendar.allowed_online_meeting_providers == (
        MsGraphCalendarOnlineMeetingProvider.UNKNOWN,
    )
    assert (
        calendar.default_online_meeting_provider is MsGraphCalendarOnlineMeetingProvider.UNKNOWN
    )


def test_parse_providers_deduplicated() -> None:
    calendar = _parse_calendar(
        _calendar_payload(
            allowed_providers=["teamsForBusiness", "teamsForBusiness", "unknown"],
        )
    )
    assert calendar.allowed_online_meeting_providers == (
        MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
        MsGraphCalendarOnlineMeetingProvider.UNKNOWN,
    )


def test_parse_opaque_calendar_id() -> None:
    calendar = _parse_calendar(_calendar_payload(calendar_id=_OPAQUE_CALENDAR_ID))
    assert calendar.remote_id == _OPAQUE_CALENDAR_ID


def test_parse_name_and_change_key_trimmed() -> None:
    calendar = _parse_calendar(
        _calendar_payload(name="  Calendar  ", change_key="  key-1  ")
    )
    assert calendar.name == "Calendar"
    assert calendar.change_key == "key-1"


def test_parse_expected_mailbox_user_id_preserved() -> None:
    calendar = _parse_calendar(_calendar_payload())
    assert calendar.mailbox_user_id == _MAILBOX_USER_ID


def test_parse_unknown_provider_fields_not_retained() -> None:
    calendar = _parse_calendar(_calendar_payload(extra_field="should-not-appear"))
    assert not hasattr(calendar, "unknownField")
    dumped = calendar.model_dump()
    assert "unknownField" not in dumped


def test_sensitive_fields_hidden_from_repr() -> None:
    calendar = _valid_calendar(
        name=_HIDDEN_CALENDAR_NAME,
        change_key=_CHANGE_KEY,
        owner=_valid_calendar().owner,
    )
    calendar_with_owner = _parse_calendar(
        _calendar_payload(
            name=_HIDDEN_CALENDAR_NAME,
            owner={"address": _OWNER_ADDRESS, "name": _OWNER_NAME},
        )
    )
    for item in (calendar, calendar_with_owner):
        rendered = repr(item)
        assert _HIDDEN_CALENDAR_NAME not in rendered
        assert _CHANGE_KEY not in rendered
        assert _OWNER_ADDRESS not in rendered
        assert _OWNER_NAME not in rendered


def test_raw_provider_payload_not_stored() -> None:
    payload = _calendar_payload()
    calendar = _parse_calendar(payload)
    assert not hasattr(calendar, "__pydantic_extra__") or not calendar.__pydantic_extra__


# --- malformed provider ---


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-dict",
        {},
        {"name": _CALENDAR_NAME},
        {"id": 123, "name": _CALENDAR_NAME},
        {"id": "", "name": _CALENDAR_NAME},
        {"id": _CALENDAR_ID},
        {"id": _CALENDAR_ID, "name": 123},
        {"id": _CALENDAR_ID, "name": ""},
        {"id": _CALENDAR_ID, "name": "bad\x00name"},
        {
            "id": _CALENDAR_ID,
            "name": _CALENDAR_NAME,
            "changeKey": _CHANGE_KEY,
        },
        {
            "id": _CALENDAR_ID,
            "name": _CALENDAR_NAME,
            "changeKey": "",
        },
        {
            "id": _CALENDAR_ID,
            "name": _CALENDAR_NAME,
            "changeKey": _CHANGE_KEY,
            "isDefaultCalendar": 1,
        },
        {
            "id": _CALENDAR_ID,
            "name": _CALENDAR_NAME,
            "changeKey": _CHANGE_KEY,
            "isDefaultCalendar": True,
            "canEdit": "yes",
        },
        {
            "id": _CALENDAR_ID,
            "name": _CALENDAR_NAME,
            "changeKey": _CHANGE_KEY,
            "isDefaultCalendar": True,
            "canEdit": True,
            "canShare": True,
            "canViewPrivateItems": False,
            "isRemovable": False,
        },
        {
            "id": _CALENDAR_ID,
            "name": _CALENDAR_NAME,
            "changeKey": _CHANGE_KEY,
            "isDefaultCalendar": True,
            "canEdit": True,
            "canShare": True,
            "canViewPrivateItems": False,
            "isRemovable": False,
            "allowedOnlineMeetingProviders": "teamsForBusiness",
            "defaultOnlineMeetingProvider": "teamsForBusiness",
        },
        {
            "id": _CALENDAR_ID,
            "name": _CALENDAR_NAME,
            "changeKey": _CHANGE_KEY,
            "isDefaultCalendar": True,
            "canEdit": True,
            "canShare": True,
            "canViewPrivateItems": False,
            "isRemovable": False,
            "allowedOnlineMeetingProviders": ["teamsForBusiness", ""],
            "defaultOnlineMeetingProvider": "teamsForBusiness",
        },
        {
            "id": _CALENDAR_ID,
            "name": _CALENDAR_NAME,
            "changeKey": _CHANGE_KEY,
            "isDefaultCalendar": True,
            "canEdit": True,
            "canShare": True,
            "canViewPrivateItems": False,
            "isRemovable": False,
            "allowedOnlineMeetingProviders": [123],
            "defaultOnlineMeetingProvider": "teamsForBusiness",
        },
        {
            "id": _CALENDAR_ID,
            "name": _CALENDAR_NAME,
            "changeKey": _CHANGE_KEY,
            "isDefaultCalendar": True,
            "canEdit": True,
            "canShare": True,
            "canViewPrivateItems": False,
            "isRemovable": False,
            "allowedOnlineMeetingProviders": ["teamsForBusiness"],
            "defaultOnlineMeetingProvider": "",
        },
        {
            "id": _CALENDAR_ID,
            "name": _CALENDAR_NAME,
            "changeKey": _CHANGE_KEY,
            "isDefaultCalendar": True,
            "canEdit": True,
            "canShare": True,
            "canViewPrivateItems": False,
            "isRemovable": False,
            "allowedOnlineMeetingProviders": ["teamsForBusiness"],
            "defaultOnlineMeetingProvider": "teamsForBusiness",
            "owner": {"address": ""},
        },
        {
            "id": _CALENDAR_ID,
            "name": _CALENDAR_NAME,
            "changeKey": _CHANGE_KEY,
            "isDefaultCalendar": True,
            "canEdit": True,
            "canShare": True,
            "canViewPrivateItems": False,
            "isRemovable": False,
            "allowedOnlineMeetingProviders": ["teamsForBusiness"],
            "defaultOnlineMeetingProvider": "teamsForBusiness",
            "owner": 123,
        },
    ],
)
def test_malformed_provider_payload_rejected(payload: object) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        parse_msgraph_calendar(payload, expected_mailbox_user_id=_MAILBOX_USER_ID)
    _assert_safe_provider_error(exc)


def test_parse_empty_unknown_provider_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        _parse_calendar(_calendar_payload(allowed_providers=[""]))


def test_parse_control_character_provider_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        _parse_calendar(_calendar_payload(allowed_providers=["bad\x00provider"]))


def test_parse_provider_over_1024_chars_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        _parse_calendar(_calendar_payload(allowed_providers=["x" * 1025]))


def test_allowed_providers_deduplicated_at_model_boundary() -> None:
    calendar = MsGraphCalendar.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        remote_id=_CALENDAR_ID,
        name=_CALENDAR_NAME,
        change_key=_CHANGE_KEY,
        is_default_calendar=True,
        can_edit=True,
        can_share=True,
        can_view_private_items=False,
        is_removable=False,
        owner=None,
        allowed_online_meeting_providers=(
            MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
            MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
            MsGraphCalendarOnlineMeetingProvider.UNKNOWN,
        ),
        default_online_meeting_provider=MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
    )
    validated = validate_msgraph_calendar(calendar)
    assert validated.allowed_online_meeting_providers == (
        MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
        MsGraphCalendarOnlineMeetingProvider.UNKNOWN,
    )


def test_malformed_page_model_construct_missing_mailbox_user_id() -> None:
    malformed = MsGraphCalendarPage.model_construct(items=(_valid_calendar(),))
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_wrong_mailbox_user_id() -> None:
    malformed = MsGraphCalendarPage.model_construct(
        mailbox_user_id=_OTHER_MAILBOX_USER_ID,
        items=(_valid_calendar(),),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    _assert_safe_provider_error(exc)


def test_validate_page_returns_new_instance() -> None:
    page = MsGraphCalendarPage(mailbox_user_id=_MAILBOX_USER_ID, items=(_valid_calendar(),))
    validated = _validate_page(page)
    assert validated == page
    assert validated is not page
    assert validated.items[0] is not page.items[0]


# --- page model ---


def test_page_empty_tuple() -> None:
    page = MsGraphCalendarPage(mailbox_user_id=_MAILBOX_USER_ID, items=())
    assert page.items == ()
    assert page.has_more is False


def test_page_multiple_calendars() -> None:
    page = MsGraphCalendarPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        items=(
            _valid_calendar(remote_id="c1"),
            _valid_calendar(remote_id="c2", name="Shared"),
        ),
    )
    assert len(page.items) == 2


def test_page_has_more_false() -> None:
    page = MsGraphCalendarPage(mailbox_user_id=_MAILBOX_USER_ID, items=())
    assert page.has_more is False


def test_page_has_more_true() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(),
    )
    page = MsGraphCalendarPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        items=(),
        continuation=continuation,
    )
    assert page.has_more is True


def test_page_duplicate_calendar_ids_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphCalendarPage(
            mailbox_user_id=_MAILBOX_USER_ID,
            items=(
                _valid_calendar(remote_id="same"),
                _valid_calendar(remote_id="same", name="Other"),
            ),
        )


def test_page_cross_mailbox_item_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphCalendarPage(
            mailbox_user_id=_MAILBOX_USER_ID,
            items=(_valid_calendar(mailbox_user_id=_OTHER_MAILBOX_USER_ID),),
        )


def test_page_items_as_list_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphCalendarPage(
            mailbox_user_id=_MAILBOX_USER_ID,
            items=[_valid_calendar()],  # type: ignore[arg-type]
        )


def test_page_item_wrong_type_rejected() -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphCalendarPage(
            mailbox_user_id=_MAILBOX_USER_ID,
            items=("not-a-calendar",),  # type: ignore[arg-type]
        )


def test_page_delta_continuation_rejected() -> None:
    delta = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_next_link(),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphCalendarPage(
            mailbox_user_id=_MAILBOX_USER_ID,
            items=(),
            continuation=delta,
        )


def test_token_hidden_from_repr() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(),
    )
    page = MsGraphCalendarPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        items=(),
        continuation=continuation,
    )
    assert _SECRET_TOKEN not in repr(page)
    assert _SECRET_TOKEN not in repr(continuation)


@pytest.mark.parametrize(
    "calendar_kwargs",
    [
        {"remote_id": None},
        {"can_edit": "yes"},
        {"is_default_calendar": "true"},
        {"allowed_online_meeting_providers": ("not-an-enum",)},
    ],
)
def test_malformed_calendar_model_construct_rejected(
    calendar_kwargs: dict[str, object],
) -> None:
    defaults: dict[str, object] = {
        "mailbox_user_id": _MAILBOX_USER_ID,
        "remote_id": _CALENDAR_ID,
        "name": _CALENDAR_NAME,
        "change_key": _CHANGE_KEY,
        "is_default_calendar": True,
        "can_edit": True,
        "can_share": True,
        "can_view_private_items": False,
        "is_removable": False,
        "owner": None,
        "allowed_online_meeting_providers": (
            MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
        ),
        "default_online_meeting_provider": MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
    }
    defaults.update(calendar_kwargs)
    malformed = MsGraphCalendar.model_construct(**defaults)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_calendar(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_missing_items() -> None:
    malformed = MsGraphCalendarPage.model_construct(mailbox_user_id=_MAILBOX_USER_ID)
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_items_as_list() -> None:
    malformed = MsGraphCalendarPage.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        items=[_valid_calendar()],
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_malformed_calendar() -> None:
    bad_calendar = MsGraphCalendar.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        remote_id=_CALENDAR_ID,
        name=_CALENDAR_NAME,
        change_key=_CHANGE_KEY,
        is_default_calendar=True,
        can_edit=True,
        can_share=True,
        can_view_private_items=False,
        is_removable=False,
        allowed_online_meeting_providers=(
            MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
        ),
        default_online_meeting_provider=MsGraphCalendarOnlineMeetingProvider.TEAMS_FOR_BUSINESS,
        owner="not-an-owner",  # type: ignore[arg-type]
    )
    malformed = MsGraphCalendarPage.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        items=(bad_calendar,),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    _assert_safe_provider_error(exc)


def test_malformed_page_model_construct_malformed_continuation() -> None:
    bad_continuation = MsGraphKnowledgeContinuation.model_construct(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=_next_link(),
    )
    malformed = MsGraphCalendarPage.model_construct(
        mailbox_user_id=_MAILBOX_USER_ID,
        items=(_valid_calendar(),),
        continuation=bad_continuation,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        _validate_page(malformed)
    _assert_safe_provider_error(exc)


def test_validate_page_rejects_wrong_mailbox_user_id() -> None:
    page = MsGraphCalendarPage(mailbox_user_id=_MAILBOX_USER_ID, items=(_valid_calendar(),))
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        validate_msgraph_calendar_page(
            page,
            mailbox_user_id=_OTHER_MAILBOX_USER_ID,
            graph_base_url=_GRAPH_BASE,
        )


# --- request tests ---


def test_request_path_and_select() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    _reader(http).read_calendars_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        continuation=None,
        limit=50,
    )
    call = http.get.call_args
    assert call.args[0] == _ROOT_PATH
    assert call.kwargs["params"]["$top"] == 50
    assert call.kwargs["params"]["$select"] == _SELECT
    assert "$expand" not in call.kwargs["params"]
    assert "events" not in call.args[0]


def test_empty_page_request() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    page = _reader(http).read_calendars_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        continuation=None,
        limit=100,
    )
    assert page.items == ()
    assert page.has_more is False


def test_paging_request_returns_continuation() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(
            value=[_calendar_payload()],
            next_link=_next_link(),
        )
    )
    page = _reader(http).read_calendars_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        continuation=None,
        limit=100,
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
    _reader(http).read_calendars_page(
        mailbox_user_id=_MAILBOX_USER_ID,
        continuation=continuation,
        limit=100,
    )
    assert http.get.call_args.args[0] == _next_link()
    assert "params" not in http.get.call_args.kwargs


@pytest.mark.parametrize("limit", [0, 201, True, "50"])
def test_invalid_limit_rejected_before_http(limit: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_calendars_page(
            mailbox_user_id=_MAILBOX_USER_ID,
            continuation=None,
            limit=limit,  # type: ignore[arg-type]
        )
    http.get.assert_not_called()


@pytest.mark.parametrize("mailbox_user_id", ["", "  ", "bad\x00id", 123])
def test_invalid_mailbox_user_id_rejected_before_http(mailbox_user_id: object) -> None:
    http = MagicMock()
    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR):
        _reader(http).read_calendars_page(
            mailbox_user_id=mailbox_user_id,  # type: ignore[arg-type]
            continuation=None,
            limit=100,
        )
    http.get.assert_not_called()


# --- continuation tests ---


def test_validate_continuation_same_user_slash_path() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(),
    )
    validated = validate_msgraph_calendars_continuation(
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
                f"https://graph.microsoft.com/v1.0/users('{_MAILBOX_USER_ID}')/calendars"
            )
        ),
    )
    validated = validate_msgraph_calendars_continuation(
        continuation,
        mailbox_user_id=_MAILBOX_USER_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated == continuation


def test_validate_continuation_case_insensitive_resource_names() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_next_link(
            path=f"https://graph.microsoft.com/v1.0/Users/{_QUOTED_MAILBOX}/Calendars"
        ),
    )
    validated = validate_msgraph_calendars_continuation(
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
                f"https://graph.microsoft.com/v1.0/users/{_QUOTED_OTHER_MAILBOX}/calendars"
            )
        ),
        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/messages?$skiptoken={_SECRET_TOKEN}",
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/mailFolders?"
            f"$skiptoken={_SECRET_TOKEN}"
        ),
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
            f"{quote(_CALENDAR_ID, safe='')}/events?$skiptoken={_SECRET_TOKEN}"
        ),
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/delta?"
            f"$deltatoken={_SECRET_TOKEN}"
        ),
        (
            f"https://graph.microsoft.com/v1.0/users/{_QUOTED_MAILBOX}/calendars/"
            f"extra?$skiptoken={_SECRET_TOKEN}"
        ),
        "https://graph.microsoft.com/v1.0/drives/drive-1/root/delta?$skiptoken=x",
    ],
)
def test_rejects_invalid_calendars_continuation(url: str) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=url,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_calendars_continuation(
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
        validate_msgraph_calendars_continuation(
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
                f"https://graph.microsoft.com/v1.0/users/{_QUOTED_OTHER_MAILBOX}/calendars"
            )
        ),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        _reader(http).read_calendars_page(
            mailbox_user_id=_MAILBOX_USER_ID,
            continuation=continuation,
            limit=100,
        )
    http.get.assert_not_called()


# --- delegation ---


def test_graph_rest_client_delegates_calendars() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(value=[_calendar_payload()])
    )
    page = _graph_client(http).read_calendars_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert len(page.items) == 1
    assert page.items[0].remote_id == _CALENDAR_ID


def test_collaboration_suite_delegates_calendars() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    suite = _Ms365GraphCollaborationSuite(_graph_client(http))
    page = suite.read_calendars_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert page.items == ()


def test_integration_delegates_calendars() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(_graph_client(http)),
        enabled=True,
    )
    page = integration.read_calendars_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert page.items == ()


def test_transport_and_reader_share_injected_http_client() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    client = _graph_client(http)
    client.read_calendars_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert client._knowledge_transport._http_client is http
    assert client._calendars_reader._transport._http_client is http
    http.get.assert_called_once()


def test_no_new_http_client_created() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    client = _graph_client(http)
    client.read_calendars_page(mailbox_user_id=_MAILBOX_USER_ID)
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


class _CustomSuiteWithoutCalendars(CollaborationSuite):
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


class _CustomGraphCalendarsClient(GraphRestClient):
    def __init__(self, page: MsGraphCalendarPage, http: MagicMock) -> None:
        super().__init__(_config(), http_client=http)
        self._custom_page = page

    def read_calendars_page(
        self,
        *,
        mailbox_user_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphCalendarPage:
        return self._custom_page


class _CustomCalendarsSuite(CollaborationSuite):
    def __init__(self, page: MsGraphCalendarPage) -> None:
        self._page = page

    def read_calendars_page(
        self,
        *,
        mailbox_user_id: str,
        continuation: MsGraphKnowledgeContinuation | None = None,
        limit: int = 100,
    ) -> MsGraphCalendarPage:
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


def test_custom_client_without_calendars_capability_fails() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomSuiteWithoutCalendars(),
        enabled=True,
    )
    with pytest.raises(
        IntegrationConfigurationError,
        match="Microsoft Graph integration does not expose Calendar inventory capability",
    ):
        integration.read_calendars_page(mailbox_user_id=_MAILBOX_USER_ID)


def test_custom_client_malformed_page_rejected() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphCalendarsClient(
                page=MsGraphCalendarPage.model_construct(mailbox_user_id=_MAILBOX_USER_ID),
                http=MagicMock(),
            )
        ),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        integration.read_calendars_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert exc.value.__cause__ is None


def test_custom_client_valid_page_revalidated() -> None:
    supplied = MsGraphCalendarPage(
        mailbox_user_id=_MAILBOX_USER_ID,
        items=(_valid_calendar(),),
    )
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(
            _CustomGraphCalendarsClient(page=supplied, http=MagicMock())
        ),
        enabled=True,
    )
    returned = integration.read_calendars_page(mailbox_user_id=_MAILBOX_USER_ID)
    assert returned == supplied
    assert returned is not supplied
    assert returned.items[0] is not supplied.items[0]


def test_custom_client_validation_not_configured() -> None:
    page = MsGraphCalendarPage(mailbox_user_id=_MAILBOX_USER_ID, items=())
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomCalendarsSuite(page=page),
        enabled=True,
    )
    with pytest.raises(IntegrationConfigurationError, match=_VALIDATION_ERROR):
        integration._graph_base_url_for_calendar_validation()


# --- security ---


def test_security_calendar_repr_and_errors() -> None:
    calendar = _valid_calendar(name=_HIDDEN_CALENDAR_NAME, change_key=_CHANGE_KEY)
    assert _HIDDEN_CALENDAR_NAME not in repr(calendar)
    assert _CHANGE_KEY not in repr(calendar)
    assert _CALENDAR_ID in repr(calendar)

    with pytest.raises(IntegrationConfigurationError, match=_REQUEST_ERROR) as exc:
        _reader(MagicMock()).read_calendars_page(
            mailbox_user_id="",
            continuation=None,
            limit=100,
        )
    assert _MAILBOX_USER_ID not in str(exc.value)

    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_calendars_continuation(
            MsGraphKnowledgeContinuation(
                kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=_next_link(
                    path=(
                        f"https://graph.microsoft.com/v1.0/users/{_QUOTED_OTHER_MAILBOX}/calendars"
                    )
                ),
            ),
            mailbox_user_id=_MAILBOX_USER_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert _SECRET_TOKEN not in str(exc.value)
    assert _MAILBOX_USER_ID not in str(exc.value)
