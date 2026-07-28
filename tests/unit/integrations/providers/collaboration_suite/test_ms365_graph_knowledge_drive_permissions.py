# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Microsoft Graph Drive knowledge-read permissions surface."""

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
    MsGraphDriveItem,
    MsGraphDriveItemKind,
    MsGraphDriveLinkScope,
    MsGraphDrivePermission,
    MsGraphDrivePermissionKind,
    MsGraphDrivePermissionPage,
    MsGraphDrivePermissionPrincipal,
    MsGraphDrivePermissionPrincipalKind,
    MsGraphDrivePermissionsReader,
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeTransport,
    parse_msgraph_drive_permission,
    validate_msgraph_drive_permission,
    validate_msgraph_drive_permission_page,
    validate_msgraph_drive_permissions_continuation,
)

pytestmark = pytest.mark.unit

_GRAPH_BASE = DEFAULT_GRAPH_BASE_URL
_DRIVE_ID = "b!drive-id-with-special-chars"
_ITEM_ID = "item-abc-123"
_OTHER_DRIVE_ID = "other-drive"
_OTHER_ITEM_ID = "other-item"
_QUOTED_DRIVE_ID = quote(_DRIVE_ID, safe="")
_QUOTED_ITEM_ID = quote(_ITEM_ID, safe="")
_SECRET_TOKEN = "secret-skiptoken-value"
_USER_ID = "user-123"
_GROUP_ID = "group-456"
_APP_ID = "app-789"
_PERMISSIONS_PATH = f"/drives/{_QUOTED_DRIVE_ID}/items/{_QUOTED_ITEM_ID}/permissions"
_SELECT = (
    "id,roles,grantedToV2,grantedToIdentitiesV2,link,invitation,inheritedFrom,"
    "expirationDateTime,hasPassword"
)
_TS = "2026-05-29T10:15:30Z"
_SAFE_ERROR = "unexpected Microsoft Graph Drive permissions response"
_CONT_ERROR = "invalid Microsoft Graph Drive permissions continuation"


def _config() -> Ms365GraphIntegrationConfig:
    return Ms365GraphIntegrationConfig(
        tenant_id="tenant-123",
        client_id="client-456",
        client_secret="secret",
        graph_base_url=_GRAPH_BASE,
    )


def _drive_item(*, kind: MsGraphDriveItemKind = MsGraphDriveItemKind.FILE) -> MsGraphDriveItem:
    return MsGraphDriveItem(
        remote_id=_ITEM_ID,
        drive_id=_DRIVE_ID,
        parent_remote_id="parent-1",
        kind=kind,
        name="report.pdf",
        e_tag='"etag-1"',
        c_tag='"ctag-1"',
        size_bytes=42,
        mime_type="application/pdf",
        created_at=datetime(2026, 5, 29, 10, 15, 30, tzinfo=timezone.utc),
        last_modified_at=datetime(2026, 5, 29, 10, 15, 30, tzinfo=timezone.utc),
        web_url="https://contoso.sharepoint.com/file",
    )


def _json_response(*, status_code: int = 200, payload: object | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload if payload is not None else {}
    response.raise_for_status = MagicMock()
    return response


def _permissions_next_link() -> str:
    return (
        f"https://graph.microsoft.com/v1.0/drives/{_QUOTED_DRIVE_ID}/items/"
        f"{_QUOTED_ITEM_ID}/permissions?$skiptoken={_SECRET_TOKEN}"
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


def _reader(http: MagicMock) -> MsGraphDrivePermissionsReader:
    return MsGraphDrivePermissionsReader(
        config=_config(),
        transport=MsGraphKnowledgeTransport(config=_config(), http_client=http),
    )


def _graph_client(http: MagicMock) -> GraphRestClient:
    return GraphRestClient(_config(), http_client=http)


def _parse_permission(payload: dict[str, Any]):
    return parse_msgraph_drive_permission(
        payload,
        expected_drive_id=_DRIVE_ID,
        expected_item_id=_ITEM_ID,
    )


def _assert_safe_provider_error(exc: BaseException) -> None:
    assert str(exc.value if isinstance(exc, pytest.ExceptionInfo) else exc) == _SAFE_ERROR
    cause = exc.value.__cause__ if isinstance(exc, pytest.ExceptionInfo) else exc.__cause__
    assert cause is None
    message = str(exc.value if isinstance(exc, pytest.ExceptionInfo) else exc)
    for forbidden in (
        _USER_ID,
        _DRIVE_ID,
        _ITEM_ID,
        "shareId",
        "webUrl",
        "displayName",
        "loginName",
        "@example.com",
    ):
        assert forbidden not in message


def _valid_principal() -> MsGraphDrivePermissionPrincipal:
    return MsGraphDrivePermissionPrincipal(
        kind=MsGraphDrivePermissionPrincipalKind.USER,
        principal_id=_USER_ID,
    )


def _valid_direct_permission(**overrides: object) -> MsGraphDrivePermission:
    defaults: dict[str, object] = {
        "permission_id": "perm-direct",
        "roles": ("read",),
        "kind": MsGraphDrivePermissionKind.DIRECT,
        "principals": (_valid_principal(),),
        "projection_complete": True,
    }
    defaults.update(overrides)
    return MsGraphDrivePermission(**defaults)  # type: ignore[arg-type]


def _valid_link_permission(**overrides: object) -> MsGraphDrivePermission:
    defaults: dict[str, object] = {
        "permission_id": "perm-link",
        "roles": ("read",),
        "kind": MsGraphDrivePermissionKind.LINK,
        "link_scope": MsGraphDriveLinkScope.ANONYMOUS,
        "projection_complete": True,
    }
    defaults.update(overrides)
    return MsGraphDrivePermission(**defaults)  # type: ignore[arg-type]


# --- model shape invariants ---


def test_parse_direct_user_permission() -> None:
    perm = _parse_permission(
        {
            "id": "perm-user",
            "roles": ["read"],
            "grantedToV2": {"user": {"id": _USER_ID}},
        }
    )
    assert perm.kind is MsGraphDrivePermissionKind.DIRECT
    assert len(perm.principals) == 1
    assert perm.principals[0].kind is MsGraphDrivePermissionPrincipalKind.USER
    assert perm.principals[0].principal_id == _USER_ID
    assert perm.projection_complete is True
    assert perm.grants_read_access is True


def test_parse_direct_group_permission() -> None:
    perm = _parse_permission(
        {
            "id": "perm-group",
            "roles": ["write"],
            "grantedToV2": {"group": {"id": _GROUP_ID}},
        }
    )
    assert perm.principals[0].kind is MsGraphDrivePermissionPrincipalKind.GROUP
    assert perm.projection_complete is True


def test_parse_direct_application_permission() -> None:
    perm = _parse_permission(
        {
            "id": "perm-app",
            "roles": ["owner"],
            "grantedToV2": {"application": {"id": _APP_ID}},
        }
    )
    assert perm.principals[0].kind is MsGraphDrivePermissionPrincipalKind.APPLICATION


def test_parse_user_and_site_user_canonicalization() -> None:
    perm = _parse_permission(
        {
            "id": "perm-canonical-user",
            "roles": ["read"],
            "grantedToV2": {
                "user": {"id": _USER_ID},
                "siteUser": {"id": "site-user-ignored"},
            },
        }
    )
    assert len(perm.principals) == 1
    assert perm.principals[0].kind is MsGraphDrivePermissionPrincipalKind.USER
    assert perm.principals[0].principal_id == _USER_ID


def test_parse_sharepoint_group_and_site_group_canonicalization() -> None:
    perm = _parse_permission(
        {
            "id": "perm-canonical-sp",
            "roles": ["read"],
            "grantedToV2": {
                "sharePointGroup": {"id": "sp-group-1"},
                "siteGroup": {"id": "site-group-ignored"},
            },
        }
    )
    assert len(perm.principals) == 1
    assert perm.principals[0].kind is MsGraphDrivePermissionPrincipalKind.SHAREPOINT_GROUP


def test_parse_specific_users_link() -> None:
    perm = _parse_permission(
        {
            "id": "perm-users-link",
            "roles": ["read"],
            "link": {"scope": "users", "type": "view"},
            "grantedToIdentitiesV2": [{"user": {"id": _USER_ID}}],
        }
    )
    assert perm.kind is MsGraphDrivePermissionKind.LINK
    assert perm.link_scope is MsGraphDriveLinkScope.USERS
    assert perm.projection_complete is True


def test_parse_anonymous_read_link() -> None:
    perm = _parse_permission(
        {
            "id": "perm-anon",
            "roles": ["read"],
            "link": {"scope": "anonymous", "type": "view", "webUrl": "https://secret.example"},
        }
    )
    assert perm.link_scope is MsGraphDriveLinkScope.ANONYMOUS
    assert perm.grants_anonymous_read_access is True
    assert perm.projection_complete is True
    assert perm.link_type == "view"
    dumped = perm.model_dump()
    assert "webUrl" not in dumped


def test_parse_organization_read_link() -> None:
    perm = _parse_permission(
        {
            "id": "perm-org",
            "roles": ["read"],
            "link": {"scope": "organization", "type": "edit"},
        }
    )
    assert perm.grants_organization_read_access is True
    assert perm.projection_complete is True


def test_parse_existing_access_link_is_incomplete() -> None:
    perm = _parse_permission(
        {
            "id": "perm-existing",
            "roles": ["read"],
            "link": {"scope": "existingAccess", "type": "view"},
        }
    )
    assert perm.link_scope is MsGraphDriveLinkScope.EXISTING_ACCESS
    assert perm.projection_complete is False


def test_parse_unknown_link_scope() -> None:
    perm = _parse_permission(
        {
            "id": "perm-unknown-scope",
            "roles": ["read"],
            "link": {"scope": "futureScope", "type": "view"},
        }
    )
    assert perm.link_scope is MsGraphDriveLinkScope.UNKNOWN
    assert perm.projection_complete is False


def test_parse_unknown_role() -> None:
    perm = _parse_permission(
        {
            "id": "perm-unknown-role",
            "roles": ["customRole"],
            "grantedToV2": {"user": {"id": _USER_ID}},
        }
    )
    assert perm.projection_complete is False


def test_parse_redeemed_invitation() -> None:
    perm = _parse_permission(
        {
            "id": "perm-invite-redeemed",
            "roles": ["read"],
            "invitation": {"email": "guest@example.com"},
            "grantedToV2": {"user": {"id": _USER_ID}},
        }
    )
    assert perm.kind is MsGraphDrivePermissionKind.INVITATION
    assert perm.projection_complete is False
    assert "email" not in perm.model_dump()


def test_parse_unredeemed_invitation_without_principal() -> None:
    perm = _parse_permission(
        {
            "id": "perm-invite-open",
            "roles": ["read"],
            "invitation": {"email": "guest@example.com", "signInRequired": True},
        }
    )
    assert perm.kind is MsGraphDrivePermissionKind.INVITATION
    assert perm.principals == ()
    assert perm.projection_complete is False


def test_parse_deprecated_only_principal_data() -> None:
    perm = _parse_permission(
        {
            "id": "perm-deprecated",
            "roles": ["read"],
            "grantedTo": {
                "user": {
                    "id": _USER_ID,
                    "displayName": "Secret User",
                    "email": "secret@example.com",
                }
            },
        }
    )
    assert perm.principals == ()
    assert perm.projection_complete is False
    assert "displayName" not in perm.model_dump()


def test_parse_permission_without_known_facets() -> None:
    perm = _parse_permission({"id": "perm-unknown", "roles": ["read"]})
    assert perm.kind is MsGraphDrivePermissionKind.UNKNOWN
    assert perm.projection_complete is False


def test_empty_page_does_not_imply_private() -> None:
    page = MsGraphDrivePermissionPage(items=())
    assert page.has_anonymous_read_grant is False
    assert page.has_organization_read_grant is False
    assert page.acl_complete is False


def test_parse_inherited_from_present() -> None:
    perm = _parse_permission(
        {
            "id": "perm-inherited",
            "roles": ["read"],
            "grantedToV2": {"user": {"id": _USER_ID}},
            "inheritedFrom": {"id": "parent-item", "driveId": _DRIVE_ID, "path": "/ignored"},
        }
    )
    assert perm.inheritance_known is True
    assert perm.inherited_from_item_id == "parent-item"


def test_parse_inherited_from_absent() -> None:
    perm = _parse_permission(
        {
            "id": "perm-direct-inherit",
            "roles": ["read"],
            "grantedToV2": {"user": {"id": _USER_ID}},
        }
    )
    assert perm.inheritance_known is False
    assert perm.inherited_from_item_id is None


def test_parse_expiration_normalization() -> None:
    perm = _parse_permission(
        {
            "id": "perm-exp",
            "roles": ["read"],
            "link": {"scope": "anonymous", "type": "view"},
            "expirationDateTime": _TS,
        }
    )
    assert perm.expires_at == datetime(2026, 5, 29, 10, 15, 30, tzinfo=timezone.utc)


def test_parse_has_password_strict_bool() -> None:
    perm = _parse_permission(
        {
            "id": "perm-pw",
            "roles": ["read"],
            "link": {"scope": "anonymous", "type": "view"},
            "hasPassword": True,
        }
    )
    assert perm.has_password is True


def test_share_id_not_stored() -> None:
    perm = _parse_permission(
        {
            "id": "perm-share",
            "roles": ["read"],
            "link": {"scope": "anonymous", "type": "view"},
            "shareId": "must-not-appear",
        }
    )
    assert "shareId" not in perm.model_dump()


# --- malformed provider ---


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-dict",
        {"roles": ["read"]},
        {"id": 123, "roles": ["read"]},
        {"id": "perm", "roles": "read"},
        {"id": "perm", "roles": []},
        {"id": "perm", "roles": [123]},
        {"id": "perm", "roles": ["read"], "grantedToV2": None},
        {"id": "perm", "roles": ["read"], "grantedToIdentitiesV2": "bad"},
        {"id": "perm", "roles": ["read"], "grantedToV2": {"user": "bad"}},
        {"id": "perm", "roles": ["read"], "grantedToV2": {"user": {}}},
        {"id": "perm", "roles": ["read"], "grantedToV2": {"user": {"id": 1}}},
        {
            "id": "perm",
            "roles": ["read"],
            "grantedToV2": {"user": {"id": _USER_ID}, "group": {"id": _GROUP_ID}},
        },
        {"id": "perm", "roles": ["read"], "link": "bad"},
        {"id": "perm", "roles": ["read"], "invitation": "bad"},
        {
            "id": "perm",
            "roles": ["read"],
            "link": {"scope": "anonymous", "type": "view"},
            "invitation": {},
        },
        {"id": "perm", "roles": ["read"], "inheritedFrom": "bad"},
        {"id": "perm", "roles": ["read"], "inheritedFrom": {}},
        {
            "id": "perm",
            "roles": ["read"],
            "inheritedFrom": {"id": "parent", "driveId": _OTHER_DRIVE_ID},
        },
        {"id": "perm", "roles": ["read"], "expirationDateTime": "not-a-date"},
        {"id": "perm", "roles": ["read"], "hasPassword": "yes"},
    ],
)
def test_malformed_provider_payload_rejected(payload: object) -> None:
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        parse_msgraph_drive_permission(
            payload,
            expected_drive_id=_DRIVE_ID,
            expected_item_id=_ITEM_ID,
        )
    _assert_safe_provider_error(exc)


def test_duplicate_permission_ids_on_page_rejected() -> None:
    perm = _parse_permission(
        {"id": "dup", "roles": ["read"], "grantedToV2": {"user": {"id": _USER_ID}}}
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphDrivePermissionPage(items=(perm, perm))


# --- page and continuation ---


def test_last_page_without_continuation() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    page = _reader(http).read_permissions_page(item=_drive_item(), continuation=None)
    assert page.items == ()
    assert page.continuation is None
    assert page.has_more is False
    assert page.acl_complete is False
    assert page.inheritance_complete is False


def test_page_with_next_page_continuation() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(
            value=[{"id": "perm-1", "roles": ["read"], "grantedToV2": {"user": {"id": _USER_ID}}}],
            next_link=_permissions_next_link(),
        )
    )
    page = _reader(http).read_permissions_page(item=_drive_item(), continuation=None)
    assert page.has_more is True
    assert page.continuation is not None
    assert page.continuation.kind is MsGraphKnowledgeContinuationKind.NEXT_PAGE


def test_page_positive_evidence_and_unresolved_grants() -> None:
    anon = _parse_permission(
        {"id": "a", "roles": ["read"], "link": {"scope": "anonymous", "type": "view"}}
    )
    org = _parse_permission(
        {"id": "b", "roles": ["read"], "link": {"scope": "organization", "type": "view"}}
    )
    unresolved = _parse_permission({"id": "c", "roles": ["read"]})
    page = MsGraphDrivePermissionPage(items=(anon, org, unresolved))
    assert page.has_anonymous_read_grant is True
    assert page.has_organization_read_grant is True
    assert page.contains_unresolved_grants is True


def test_token_hidden_in_repr() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_permissions_next_link(),
    )
    page = MsGraphDrivePermissionPage(items=(), continuation=continuation)
    assert _SECRET_TOKEN not in repr(page)
    assert _SECRET_TOKEN not in repr(continuation)


def test_validate_permissions_continuation_same_drive_and_item() -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_permissions_next_link(),
    )
    validated = validate_msgraph_drive_permissions_continuation(
        continuation,
        drive_id=_DRIVE_ID,
        item_id=_ITEM_ID,
        graph_base_url=_GRAPH_BASE,
    )
    assert validated is continuation


@pytest.mark.parametrize(
    ("url",),
    [
        (
            f"https://graph.microsoft.com/v1.0/drives/{quote(_OTHER_DRIVE_ID, safe='')}/items/"
            f"{_QUOTED_ITEM_ID}/permissions?$skiptoken={_SECRET_TOKEN}",
        ),
        (
            f"https://graph.microsoft.com/v1.0/drives/{_QUOTED_DRIVE_ID}/items/"
            f"{quote(_OTHER_ITEM_ID, safe='')}/permissions?$skiptoken={_SECRET_TOKEN}",
        ),
        (
            f"https://graph.microsoft.com/v1.0/drives/{_QUOTED_DRIVE_ID}/root/delta?"
            f"$skiptoken={_SECRET_TOKEN}",
        ),
        (
            f"https://graph.microsoft.com/v1.0/drives/{_QUOTED_DRIVE_ID}/items/"
            f"{_QUOTED_ITEM_ID}/content",
        ),
        ("https://graph.microsoft.com/v1.0/users/user-1/messages?$skiptoken=x",),
    ],
)
def test_rejects_invalid_permissions_continuation(url: str) -> None:
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=url,
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR) as exc:
        validate_msgraph_drive_permissions_continuation(
            continuation,
            drive_id=_DRIVE_ID,
            item_id=_ITEM_ID,
            graph_base_url=_GRAPH_BASE,
        )
    assert _SECRET_TOKEN not in str(exc.value)
    assert exc.value.__cause__ is None


def test_delta_continuation_rejected_on_page_model() -> None:
    delta = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.DELTA,
        url=(
            f"https://graph.microsoft.com/v1.0/drives/{_QUOTED_DRIVE_ID}/items/"
            f"{_QUOTED_ITEM_ID}/permissions?$deltatoken=x"
        ),
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphDrivePermissionPage(items=(), continuation=delta)


def test_invalid_continuation_rejected_before_http() -> None:
    http = MagicMock()
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=(
            f"https://graph.microsoft.com/v1.0/drives/{quote(_OTHER_DRIVE_ID, safe='')}/items/"
            f"{_QUOTED_ITEM_ID}/permissions?$skiptoken={_SECRET_TOKEN}"
        ),
    )
    with pytest.raises(IntegrationConfigurationError, match=_CONT_ERROR):
        _reader(http).read_permissions_page(item=_drive_item(), continuation=continuation)
    http.get.assert_not_called()


# --- item boundary ---


@pytest.mark.parametrize(
    "item",
    [
        MsGraphDriveItem.model_construct(remote_id=None, drive_id=_DRIVE_ID, kind=MsGraphDriveItemKind.FILE),  # type: ignore[arg-type]
        MsGraphDriveItem.model_construct(remote_id=123, drive_id=_DRIVE_ID, kind=MsGraphDriveItemKind.FILE),  # type: ignore[arg-type]
        MsGraphDriveItem.model_construct(remote_id=_ITEM_ID, drive_id="", kind=MsGraphDriveItemKind.FILE),
        MsGraphDriveItem.model_construct(remote_id=_ITEM_ID, drive_id=_DRIVE_ID, kind="file"),  # type: ignore[arg-type]
        MsGraphDriveItem.model_construct(
            remote_id=_ITEM_ID,
            drive_id=_DRIVE_ID,
            kind=MsGraphDriveItemKind.DELETED,
            name="gone",
            last_modified_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        MsGraphDriveItem.model_construct(
            remote_id=_ITEM_ID,
            drive_id=_DRIVE_ID,
            kind=MsGraphDriveItemKind.FILE,
            parent_remote_id="bad\x00id",
            name="report.pdf",
            last_modified_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        MsGraphDriveItem.model_construct(
            remote_id=_ITEM_ID,
            drive_id=_DRIVE_ID,
            kind=MsGraphDriveItemKind.FILE,
            name="report.pdf",
            size_bytes="big",  # type: ignore[arg-type]
            last_modified_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
    ],
)
def test_invalid_item_rejected_before_http(item: MsGraphDriveItem) -> None:
    http = MagicMock()
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        _reader(http).read_permissions_page(item=item, continuation=None)
    http.get.assert_not_called()


@pytest.mark.parametrize("kind", [MsGraphDriveItemKind.FILE, MsGraphDriveItemKind.FOLDER])
def test_valid_file_or_folder_item_works(kind: MsGraphDriveItemKind) -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    item = _drive_item(kind=kind)
    page = _reader(http).read_permissions_page(item=item, continuation=None)
    assert page.items == ()


def test_initial_request_path_select_and_not_found_flag() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    _reader(http).read_permissions_page(item=_drive_item(), continuation=None)
    call = http.get.call_args
    assert call.args[0] == _PERMISSIONS_PATH
    assert call.kwargs["params"]["$select"] == _SELECT
    assert "$top" not in call.kwargs["params"]


def test_continuation_request_uses_full_url_without_params() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    continuation = MsGraphKnowledgeContinuation(
        kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
        url=_permissions_next_link(),
    )
    _reader(http).read_permissions_page(item=_drive_item(), continuation=continuation)
    assert http.get.call_args.args[0] == _permissions_next_link()
    assert "params" not in http.get.call_args.kwargs


# --- delegation ---


def test_graph_rest_client_delegates_permissions() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload=_page_payload(
            value=[{"id": "perm-1", "roles": ["read"], "grantedToV2": {"user": {"id": _USER_ID}}}]
        )
    )
    page = _graph_client(http).read_drive_permissions_page(item=_drive_item())
    assert len(page.items) == 1


def test_collaboration_suite_delegates_permissions() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    suite = _Ms365GraphCollaborationSuite(_graph_client(http))
    page = suite.read_drive_permissions_page(item=_drive_item())
    assert page.acl_complete is False


def test_integration_delegates_permissions() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _Ms365GraphCollaborationSuite(_graph_client(http)),
        enabled=True,
    )
    page = integration.read_drive_permissions_page(item=_drive_item())
    assert page.inheritance_complete is False


def test_transport_and_reader_share_injected_http_client() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(payload=_page_payload(value=[]))
    client = _graph_client(http)
    client.read_drive_permissions_page(item=_drive_item())
    assert client._knowledge_transport._http_client is http
    assert client._drive_permissions_reader._transport._http_client is http
    http.get.assert_called_once()


def test_drive_delta_still_works_after_permissions_wiring() -> None:
    http = MagicMock()
    http.get.return_value = _json_response(
        payload={
            "value": [],
            "@odata.deltaLink": (
                f"https://graph.microsoft.com/v1.0/drives/{_QUOTED_DRIVE_ID}/root/delta?"
                "$deltatoken=tok"
            ),
        }
    )
    client = _graph_client(http)
    page = client.read_drive_delta_page(drive_id=_DRIVE_ID, limit=10)
    assert page.is_complete is True


class _DriveDeltaOnlySuite(CollaborationSuite):
    def read_drive_delta_page(self, *, drive_id: str, continuation=None, limit: int = 100):
        raise NotImplementedError

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


def test_custom_delta_only_client_does_not_satisfy_permissions_protocol() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _DriveDeltaOnlySuite(),
        enabled=True,
    )
    with pytest.raises(
        IntegrationConfigurationError,
        match="Microsoft Graph integration does not expose Drive permissions capability",
    ):
        integration.read_drive_permissions_page(item=_drive_item())


# --- security ---


def test_models_repr_and_errors_exclude_secrets() -> None:
    anon = _parse_permission(
        {
            "id": "perm-safe",
            "roles": ["read"],
            "link": {"scope": "anonymous", "type": "view", "webUrl": "https://secret"},
        }
    )
    invite = _parse_permission(
        {
            "id": "perm-invite-safe",
            "roles": ["read"],
            "invitation": {"email": "guest@example.com"},
        }
    )
    text = repr(anon) + str(anon.model_dump()) + repr(invite) + str(invite.model_dump())
    for forbidden in (
        "shareId",
        "webUrl",
        "guest@example.com",
        "displayName",
        "loginName",
        "Authorization",
        "nextLink",
        _SECRET_TOKEN,
    ):
        assert forbidden not in text


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "kind": MsGraphDrivePermissionKind.DIRECT,
            "principals": (),
            "projection_complete": True,
        },
        {
            "kind": MsGraphDrivePermissionKind.DIRECT,
            "link_scope": MsGraphDriveLinkScope.ANONYMOUS,
            "projection_complete": False,
        },
        {
            "kind": MsGraphDrivePermissionKind.DIRECT,
            "link_type": "view",
            "projection_complete": False,
        },
        {
            "kind": MsGraphDrivePermissionKind.LINK,
            "link_scope": None,
            "projection_complete": False,
        },
        {
            "kind": MsGraphDrivePermissionKind.LINK,
            "link_scope": MsGraphDriveLinkScope.USERS,
            "principals": (),
            "projection_complete": True,
        },
        {
            "kind": MsGraphDrivePermissionKind.LINK,
            "link_scope": MsGraphDriveLinkScope.EXISTING_ACCESS,
            "projection_complete": True,
        },
        {
            "kind": MsGraphDrivePermissionKind.LINK,
            "link_scope": MsGraphDriveLinkScope.UNKNOWN,
            "projection_complete": True,
        },
        {
            "kind": MsGraphDrivePermissionKind.INVITATION,
            "projection_complete": True,
        },
        {
            "kind": MsGraphDrivePermissionKind.INVITATION,
            "link_scope": MsGraphDriveLinkScope.ANONYMOUS,
            "projection_complete": False,
        },
        {
            "kind": MsGraphDrivePermissionKind.INVITATION,
            "link_type": "view",
            "projection_complete": False,
        },
        {
            "kind": MsGraphDrivePermissionKind.UNKNOWN,
            "projection_complete": True,
        },
        {
            "kind": MsGraphDrivePermissionKind.UNKNOWN,
            "link_scope": MsGraphDriveLinkScope.ANONYMOUS,
            "projection_complete": False,
        },
        {
            "kind": MsGraphDrivePermissionKind.UNKNOWN,
            "link_type": "view",
            "projection_complete": False,
        },
        {
            "kind": MsGraphDrivePermissionKind.DIRECT,
            "roles": ("customRole",),
            "projection_complete": True,
        },
    ],
)
def test_permission_shape_invariants_rejected(kwargs: dict[str, object]) -> None:
    base = {
        "permission_id": "perm-shape",
        "roles": ("read",),
        "kind": MsGraphDrivePermissionKind.DIRECT,
        "principals": (_valid_principal(),),
        "projection_complete": False,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=_SAFE_ERROR):
        MsGraphDrivePermission(**base)  # type: ignore[arg-type]


def test_conservative_direct_permission_allowed() -> None:
    perm = MsGraphDrivePermission(
        permission_id="perm-conservative-direct",
        roles=("read",),
        kind=MsGraphDrivePermissionKind.DIRECT,
        principals=(),
        projection_complete=False,
    )
    assert perm.projection_complete is False


def test_conservative_anonymous_link_allowed() -> None:
    perm = MsGraphDrivePermission(
        permission_id="perm-conservative-anon",
        roles=("read",),
        kind=MsGraphDrivePermissionKind.LINK,
        link_scope=MsGraphDriveLinkScope.ANONYMOUS,
        projection_complete=False,
    )
    assert perm.projection_complete is False


def test_conservative_users_link_with_principals_allowed() -> None:
    perm = MsGraphDrivePermission(
        permission_id="perm-conservative-users",
        roles=("read",),
        kind=MsGraphDrivePermissionKind.LINK,
        link_scope=MsGraphDriveLinkScope.USERS,
        principals=(_valid_principal(),),
        projection_complete=False,
    )
    assert perm.projection_complete is False


# --- principal model_construct boundary ---


@pytest.mark.parametrize(
    "principal_kwargs",
    [
        {"principal_id": 123},  # type: ignore[dict-item]
        {"principal_id": ""},
        {"principal_id": "bad\x00id"},
        {"kind": "user"},  # type: ignore[dict-item]
        {"kind": None, "principal_id": _USER_ID},  # type: ignore[dict-item]
    ],
)
def test_malformed_principal_model_construct_rejected(
    principal_kwargs: dict[str, object],
) -> None:
    defaults: dict[str, object] = {
        "kind": MsGraphDrivePermissionPrincipalKind.USER,
        "principal_id": _USER_ID,
    }
    defaults.update(principal_kwargs)
    malformed = MsGraphDrivePermissionPrincipal.model_construct(**defaults)  # type: ignore[arg-type]
    permission = MsGraphDrivePermission.model_construct(
        permission_id="perm-bad-principal",
        roles=("read",),
        kind=MsGraphDrivePermissionKind.DIRECT,
        principals=(malformed,),
        projection_complete=True,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_drive_permission(permission)
    _assert_safe_provider_error(exc)


def test_malformed_principal_missing_principal_id_rejected() -> None:
    malformed = MsGraphDrivePermissionPrincipal.model_construct(
        kind=MsGraphDrivePermissionPrincipalKind.USER,
    )
    permission = MsGraphDrivePermission.model_construct(
        permission_id="perm-bad-principal",
        roles=("read",),
        kind=MsGraphDrivePermissionKind.DIRECT,
        principals=(malformed,),
        projection_complete=True,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_drive_permission(permission)
    _assert_safe_provider_error(exc)


# --- permission model_construct boundary ---


@pytest.mark.parametrize(
    "permission_kwargs",
    [
        {"permission_id": 123},  # type: ignore[dict-item]
        {"roles": ["read"]},  # type: ignore[dict-item]
        {"roles": (123,)},  # type: ignore[dict-item]
        {"kind": "direct"},  # type: ignore[dict-item]
        {"projection_complete": "true"},  # type: ignore[dict-item]
        {
            "kind": MsGraphDrivePermissionKind.DIRECT,
            "principals": (
                MsGraphDrivePermissionPrincipal.model_construct(
                    kind=MsGraphDrivePermissionPrincipalKind.USER,
                    principal_id="",
                ),
            ),
            "projection_complete": True,
        },
        {
            "kind": MsGraphDrivePermissionKind.INVITATION,
            "projection_complete": True,
        },
        {
            "kind": MsGraphDrivePermissionKind.UNKNOWN,
            "projection_complete": True,
        },
        {
            "kind": MsGraphDrivePermissionKind.LINK,
            "link_scope": MsGraphDriveLinkScope.EXISTING_ACCESS,
            "projection_complete": True,
        },
        {
            "kind": MsGraphDrivePermissionKind.LINK,
            "link_scope": MsGraphDriveLinkScope.USERS,
            "principals": (),
            "projection_complete": True,
        },
    ],
)
def test_malformed_permission_model_construct_rejected(
    permission_kwargs: dict[str, object],
) -> None:
    defaults: dict[str, object] = {
        "permission_id": "perm-bad",
        "roles": ("read",),
        "kind": MsGraphDrivePermissionKind.DIRECT,
        "principals": (_valid_principal(),),
        "projection_complete": True,
    }
    defaults.update(permission_kwargs)
    malformed = MsGraphDrivePermission.model_construct(**defaults)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_drive_permission(malformed)
    _assert_safe_provider_error(exc)


# --- page model_construct boundary ---


def _valid_permission_page_item() -> MsGraphDrivePermission:
    return _valid_direct_permission(permission_id="perm-page-1")


def test_permission_page_missing_items_rejected_safely() -> None:
    malformed = MsGraphDrivePermissionPage.model_construct()

    with pytest.raises(
        ValueError,
        match="unexpected Microsoft Graph Drive permissions response",
    ) as exc:
        validate_msgraph_drive_permission_page(malformed)

    assert exc.value.__cause__ is None


@pytest.mark.parametrize(
    ("items", "page_overrides"),
    [
        ([], {}),  # type: ignore[list-item]
        (("not-a-permission",), {}),  # type: ignore[list-item]
        (
            (
                MsGraphDrivePermission.model_construct(
                    permission_id="perm-malformed",
                    roles=("read",),
                    kind=MsGraphDrivePermissionKind.INVITATION,
                    projection_complete=True,
                ),
            ),
            {},
        ),
        ((_valid_permission_page_item(), _valid_permission_page_item()), {}),
        ((), {"continuation": "bad"}),  # type: ignore[dict-item]
        (
            (),
            {
                "continuation": MsGraphKnowledgeContinuation(
                    kind=MsGraphKnowledgeContinuationKind.DELTA,
                    url=_permissions_next_link(),
                )
            },
        ),
        ((), {"acl_complete": True}),  # type: ignore[dict-item]
        ((), {"inheritance_complete": True}),  # type: ignore[dict-item]
        ((), {"acl_complete": 0}),  # type: ignore[dict-item]
        ((), {"inheritance_complete": None}),  # type: ignore[dict-item]
    ],
)
def test_malformed_permission_page_model_construct_rejected(
    items: tuple[object, ...] | list[object],
    page_overrides: dict[str, object],
) -> None:
    defaults: dict[str, object] = {
        "items": items,
        "continuation": None,
        "acl_complete": False,
        "inheritance_complete": False,
    }
    defaults.update(page_overrides)
    malformed = MsGraphDrivePermissionPage.model_construct(**defaults)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        validate_msgraph_drive_permission_page(malformed)
    _assert_safe_provider_error(exc)


def test_validate_permission_returns_new_instance() -> None:
    original = _valid_direct_permission()
    validated = validate_msgraph_drive_permission(original)
    assert validated == original
    assert validated is not original


def test_validate_permission_page_returns_new_instance() -> None:
    original = MsGraphDrivePermissionPage(items=(_valid_permission_page_item(),))
    validated = validate_msgraph_drive_permission_page(original)
    assert validated == original
    assert validated is not original
    assert validated.items[0] is not original.items[0]


# --- custom-client integration boundary ---


class _CustomPermissionsSuite(CollaborationSuite):
    def __init__(self, *, page: MsGraphDrivePermissionPage) -> None:
        self._page = page

    def read_drive_permissions_page(
        self,
        *,
        item: MsGraphDriveItem,
        continuation: MsGraphKnowledgeContinuation | None = None,
    ) -> MsGraphDrivePermissionPage:
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


def test_custom_client_missing_items_page_rejected() -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomPermissionsSuite(page=MsGraphDrivePermissionPage.model_construct()),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        integration.read_drive_permissions_page(item=_drive_item())
    assert exc.value.__cause__ is None


def test_custom_client_valid_page_revalidated() -> None:
    supplied = MsGraphDrivePermissionPage(items=(_valid_permission_page_item(),))
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomPermissionsSuite(page=supplied),
        enabled=True,
    )
    returned = integration.read_drive_permissions_page(item=_drive_item())
    assert returned == supplied
    assert returned is not supplied
    assert returned.items[0] is not supplied.items[0]


@pytest.mark.parametrize(
    "page",
    [
        MsGraphDrivePermissionPage.model_construct(
            items=(_valid_permission_page_item(),),
            acl_complete=True,  # type: ignore[arg-type]
            inheritance_complete=False,
        ),
        MsGraphDrivePermissionPage.model_construct(
            items=(
                MsGraphDrivePermission.model_construct(
                    permission_id="perm-false-complete-unknown",
                    roles=("read",),
                    kind=MsGraphDrivePermissionKind.UNKNOWN,
                    projection_complete=True,
                ),
            )
        ),
        MsGraphDrivePermissionPage.model_construct(
            items=(
                MsGraphDrivePermission.model_construct(
                    permission_id="perm-false-complete-invite",
                    roles=("read",),
                    kind=MsGraphDrivePermissionKind.INVITATION,
                    projection_complete=True,
                ),
            )
        ),
        MsGraphDrivePermissionPage.model_construct(
            items=(
                MsGraphDrivePermission.model_construct(
                    permission_id="perm-bad-nested",
                    roles=("read",),
                    kind=MsGraphDrivePermissionKind.DIRECT,
                    principals=(
                        MsGraphDrivePermissionPrincipal.model_construct(
                            kind=MsGraphDrivePermissionPrincipalKind.USER,
                            principal_id="",
                        ),
                    ),
                    projection_complete=True,
                ),
            )
        ),
    ],
)
def test_custom_client_malformed_page_rejected(page: MsGraphDrivePermissionPage) -> None:
    integration = Ms365GraphCollaborationSuiteIntegration.from_client(
        _CustomPermissionsSuite(page=page),
        enabled=True,
    )
    with pytest.raises(ValueError, match=_SAFE_ERROR) as exc:
        integration.read_drive_permissions_page(item=_drive_item())
    _assert_safe_provider_error(exc)
