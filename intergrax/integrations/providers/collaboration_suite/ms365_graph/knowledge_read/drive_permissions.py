# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Drive knowledge-read: caller-visible sharing permissions for one item."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable
from urllib.parse import quote, unquote, urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    Ms365GraphIntegrationConfig,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.common import (
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
    MsGraphKnowledgeTransport,
    parse_msgraph_collection_page,
    validate_msgraph_continuation_url,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.drive import (
    MsGraphDriveItem,
    MsGraphDriveItemKind,
    validate_msgraph_drive_id,
    validate_msgraph_drive_item_id,
)

MSGRAPH_DRIVE_PERMISSIONS_SOURCE_KIND = "drive_permissions"

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_PERMISSIONS_RESPONSE = "unexpected Microsoft Graph Drive permissions response"
_INVALID_PERMISSIONS_CONTINUATION = "invalid Microsoft Graph Drive permissions continuation"
_MAX_MSGRAPH_ID_LEN = 1024
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_KNOWN_ROLES = frozenset({"read", "write", "owner"})
_READ_ACCESS_ROLES = frozenset({"read", "write", "owner"})

_PERMISSIONS_SELECT = (
    "id,roles,grantedToV2,grantedToIdentitiesV2,link,invitation,inheritedFrom,"
    "expirationDateTime,hasPassword"
)

_IDENTITY_FACET_KEYS = frozenset(
    {"user", "group", "application", "sharePointGroup", "siteUser", "siteGroup", "device"}
)
_INCOMPATIBLE_IDENTITY_PAIRS = frozenset(
    {
        frozenset({"user", "group"}),
        frozenset({"user", "application"}),
        frozenset({"group", "application"}),
        frozenset({"application", "device"}),
    }
)
_SCOPE_MAP = {
    "anonymous": "anonymous",
    "organization": "organization",
    "users": "users",
    "existingAccess": "existing_access",
}


class MsGraphDrivePermissionKind(StrEnum):
    DIRECT = "direct"
    LINK = "link"
    INVITATION = "invitation"
    UNKNOWN = "unknown"


class MsGraphDrivePermissionPrincipalKind(StrEnum):
    USER = "user"
    GROUP = "group"
    APPLICATION = "application"
    SHAREPOINT_GROUP = "sharepoint_group"
    SITE_USER = "site_user"
    SITE_GROUP = "site_group"
    DEVICE = "device"


class MsGraphDriveLinkScope(StrEnum):
    ANONYMOUS = "anonymous"
    ORGANIZATION = "organization"
    USERS = "users"
    EXISTING_ACCESS = "existing_access"
    UNKNOWN = "unknown"


def _validate_permission_opaque_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
    if len(trimmed) > _MAX_MSGRAPH_ID_LEN:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
    return trimmed


def _validate_roles_tuple(value: object) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
    if not value:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
    normalized: list[str] = []
    seen: set[str] = set()
    for role in value:
        if not isinstance(role, str):
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        trimmed = role.strip()
        if not trimmed:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        if trimmed not in seen:
            seen.add(trimmed)
            normalized.append(trimmed)
    return tuple(normalized)


class MsGraphDrivePermissionPrincipal(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    kind: MsGraphDrivePermissionPrincipalKind
    principal_id: str

    @field_validator("principal_id", mode="before")
    @classmethod
    def _validate_principal_id(cls, value: object) -> str:
        return _validate_permission_opaque_id(value)


class MsGraphDrivePermission(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    permission_id: str
    roles: tuple[str, ...]
    kind: MsGraphDrivePermissionKind

    principals: tuple[MsGraphDrivePermissionPrincipal, ...] = ()

    link_scope: MsGraphDriveLinkScope | None = None
    link_type: str | None = None

    inherited_from_item_id: str | None = None
    inheritance_known: bool = False

    expires_at: datetime | None = None
    has_password: bool | None = None

    projection_complete: bool

    @field_validator("permission_id", mode="before")
    @classmethod
    def _validate_permission_id(cls, value: object) -> str:
        return _validate_permission_opaque_id(value)

    @field_validator("roles", mode="before")
    @classmethod
    def _validate_roles(cls, value: object) -> tuple[str, ...]:
        return _validate_roles_tuple(value)

    @field_validator("principals", mode="before")
    @classmethod
    def _validate_principals(cls, value: object) -> tuple[MsGraphDrivePermissionPrincipal, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        validated: list[MsGraphDrivePermissionPrincipal] = []
        for principal in value:
            try:
                if isinstance(principal, MsGraphDrivePermissionPrincipal):
                    revalidated = MsGraphDrivePermissionPrincipal.model_validate(
                        principal.model_dump(mode="python")
                    )
                elif isinstance(principal, dict):
                    revalidated = MsGraphDrivePermissionPrincipal.model_validate(principal)
                else:
                    raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
            except (ValueError, TypeError, AttributeError, ValidationError):
                raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
            validated.append(revalidated)
        return tuple(validated)

    @field_validator("link_type", mode="before")
    @classmethod
    def _validate_link_type(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        trimmed = value.strip()
        if not trimmed:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        return trimmed

    @field_validator("inherited_from_item_id", mode="before")
    @classmethod
    def _validate_inherited_from_item_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return validate_msgraph_drive_item_id(value)

    @field_validator("expires_at", mode="before")
    @classmethod
    def _validate_expires_at(cls, value: object) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        if value.tzinfo is None:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        try:
            if value.utcoffset() is None:
                raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
            return value.astimezone(timezone.utc)
        except (ValueError, TypeError, OverflowError):
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None

    @field_validator("has_password", mode="before")
    @classmethod
    def _validate_has_password(cls, value: object) -> bool | None:
        if value is None:
            return None
        if type(value) is not bool:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_permission_shape(self) -> MsGraphDrivePermission:
        if not self.inheritance_known and self.inherited_from_item_id is not None:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        if self.inheritance_known and self.inherited_from_item_id is None:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)

        seen_principals: set[tuple[MsGraphDrivePermissionPrincipalKind, str]] = set()
        deduped: list[MsGraphDrivePermissionPrincipal] = []
        for principal in self.principals:
            key = (principal.kind, principal.principal_id)
            if key in seen_principals:
                continue
            seen_principals.add(key)
            deduped.append(principal)
        principals = tuple(deduped)
        needs_copy = principals != self.principals

        if not _roles_are_known(self.roles) and self.projection_complete:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)

        if self.kind is MsGraphDrivePermissionKind.DIRECT:
            if self.link_scope is not None or self.link_type is not None:
                raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
            if self.projection_complete:
                if not principals:
                    raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
                if not _roles_are_known(self.roles):
                    raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        elif self.kind is MsGraphDrivePermissionKind.LINK:
            if self.link_scope is None:
                raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
            if self.projection_complete:
                if not _roles_are_known(self.roles):
                    raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
                if self.link_scope is MsGraphDriveLinkScope.EXISTING_ACCESS:
                    raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
                if self.link_scope is MsGraphDriveLinkScope.UNKNOWN:
                    raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
                if self.link_scope is MsGraphDriveLinkScope.USERS and not principals:
                    raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        elif self.kind is MsGraphDrivePermissionKind.INVITATION:
            if self.projection_complete:
                raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
            if self.link_scope is not None or self.link_type is not None:
                raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        elif self.kind is MsGraphDrivePermissionKind.UNKNOWN:
            if self.projection_complete:
                raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
            if self.link_scope is not None or self.link_type is not None:
                raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)

        if needs_copy:
            return self.model_copy(update={"principals": principals})
        return self

    @property
    def grants_read_access(self) -> bool:
        return any(role in _READ_ACCESS_ROLES for role in self.roles)

    @property
    def grants_anonymous_read_access(self) -> bool:
        return (
            self.kind is MsGraphDrivePermissionKind.LINK
            and self.link_scope is MsGraphDriveLinkScope.ANONYMOUS
            and self.grants_read_access
        )

    @property
    def grants_organization_read_access(self) -> bool:
        return (
            self.kind is MsGraphDrivePermissionKind.LINK
            and self.link_scope is MsGraphDriveLinkScope.ORGANIZATION
            and self.grants_read_access
        )


class MsGraphDrivePermissionPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    items: tuple[MsGraphDrivePermission, ...]
    continuation: MsGraphKnowledgeContinuation | None = Field(default=None, repr=False)

    acl_complete: Literal[False] = False
    inheritance_complete: Literal[False] = False

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, value: object) -> tuple[MsGraphDrivePermission, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphDrivePermission):
                raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        return value

    @field_validator("continuation", mode="before")
    @classmethod
    def _validate_continuation(cls, value: object) -> MsGraphKnowledgeContinuation | None:
        if value is None:
            return None
        if not isinstance(value, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        if value.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_page_shape(self) -> MsGraphDrivePermissionPage:
        permission_ids = [item.permission_id for item in self.items]
        if len(permission_ids) != len(set(permission_ids)):
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        return self

    @property
    def has_more(self) -> bool:
        return self.continuation is not None

    @property
    def has_anonymous_read_grant(self) -> bool:
        return any(item.grants_anonymous_read_access for item in self.items)

    @property
    def has_organization_read_grant(self) -> bool:
        return any(item.grants_organization_read_access for item in self.items)

    @property
    def contains_unresolved_grants(self) -> bool:
        return any(not item.projection_complete for item in self.items)


@runtime_checkable
class MsGraphDrivePermissionsReadClient(Protocol):
    def read_drive_permissions_page(
        self,
        *,
        item: MsGraphDriveItem,
        continuation: MsGraphKnowledgeContinuation | None,
    ) -> MsGraphDrivePermissionPage:
        ...


def _facet_is_present(payload: dict[str, object], key: str) -> bool:
    if key not in payload:
        return False
    if payload[key] is None:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
    return True


def _parse_timezone_aware_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
    if trimmed.endswith("Z"):
        trimmed = f"{trimmed[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(trimmed)
    except ValueError:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
    if parsed.tzinfo is None:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
    return parsed.astimezone(timezone.utc)


def _parse_identity_facet_id(facet: object) -> str:
    if not isinstance(facet, dict):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
    raw_id = facet.get("id")
    return validate_msgraph_drive_id(raw_id)


def _facet_kind_for_key(key: str) -> MsGraphDrivePermissionPrincipalKind | None:
    mapping = {
        "user": MsGraphDrivePermissionPrincipalKind.USER,
        "group": MsGraphDrivePermissionPrincipalKind.GROUP,
        "application": MsGraphDrivePermissionPrincipalKind.APPLICATION,
        "sharePointGroup": MsGraphDrivePermissionPrincipalKind.SHAREPOINT_GROUP,
        "siteUser": MsGraphDrivePermissionPrincipalKind.SITE_USER,
        "siteGroup": MsGraphDrivePermissionPrincipalKind.SITE_GROUP,
        "device": MsGraphDrivePermissionPrincipalKind.DEVICE,
    }
    return mapping.get(key)


def _parse_identity_set(
    identity_set: object,
) -> tuple[tuple[MsGraphDrivePermissionPrincipal, ...], bool]:
    if not isinstance(identity_set, dict):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)

    present_known: dict[str, str] = {}
    has_unknown = False

    for key, value in identity_set.items():
        if key not in _IDENTITY_FACET_KEYS:
            has_unknown = True
            continue
        if value is None:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        principal_id = _parse_identity_facet_id(value)
        present_known[key] = principal_id

    present_keys = set(present_known)
    for pair in _INCOMPATIBLE_IDENTITY_PAIRS:
        if pair.issubset(present_keys):
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)

    principals: list[MsGraphDrivePermissionPrincipal] = []

    if "user" in present_known and "siteUser" in present_known:
        principals.append(
            MsGraphDrivePermissionPrincipal(
                kind=MsGraphDrivePermissionPrincipalKind.USER,
                principal_id=present_known["user"],
            )
        )
        present_keys.discard("siteUser")
        present_keys.discard("user")
    if "sharePointGroup" in present_known and "siteGroup" in present_known:
        principals.append(
            MsGraphDrivePermissionPrincipal(
                kind=MsGraphDrivePermissionPrincipalKind.SHAREPOINT_GROUP,
                principal_id=present_known["sharePointGroup"],
            )
        )
        present_keys.discard("siteGroup")
        present_keys.discard("sharePointGroup")

    for key in sorted(present_keys):
        kind = _facet_kind_for_key(key)
        if kind is None:
            continue
        principals.append(
            MsGraphDrivePermissionPrincipal(kind=kind, principal_id=present_known[key])
        )

    if not principals and not present_known and has_unknown:
        return (), True
    if not principals and present_known:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
    return tuple(principals), has_unknown


def _parse_v2_principals(
    payload: dict[str, object],
) -> tuple[tuple[MsGraphDrivePermissionPrincipal, ...], bool]:
    principals: list[MsGraphDrivePermissionPrincipal] = []
    has_unresolved = False

    if _facet_is_present(payload, "grantedToV2"):
        granted_to = payload["grantedToV2"]
        parsed, unresolved = _parse_identity_set(granted_to)
        principals.extend(parsed)
        has_unresolved = has_unresolved or unresolved

    if _facet_is_present(payload, "grantedToIdentitiesV2"):
        identities = payload["grantedToIdentitiesV2"]
        if not isinstance(identities, list):
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        for identity_set in identities:
            parsed, unresolved = _parse_identity_set(identity_set)
            principals.extend(parsed)
            has_unresolved = has_unresolved or unresolved

    deduped: list[MsGraphDrivePermissionPrincipal] = []
    seen: set[tuple[MsGraphDrivePermissionPrincipalKind, str]] = set()
    for principal in principals:
        key = (principal.kind, principal.principal_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(principal)
    return tuple(deduped), has_unresolved


def _has_deprecated_identity_fields(payload: dict[str, object]) -> bool:
    return _facet_is_present(payload, "grantedTo") or _facet_is_present(
        payload, "grantedToIdentities"
    )


def _has_grant_identity_fields(payload: dict[str, object]) -> bool:
    return (
        _facet_is_present(payload, "grantedToV2")
        or _facet_is_present(payload, "grantedToIdentitiesV2")
        or _has_deprecated_identity_fields(payload)
    )


def _roles_are_known(roles: tuple[str, ...]) -> bool:
    return all(role in _KNOWN_ROLES for role in roles)


def _compute_projection_complete(
    *,
    kind: MsGraphDrivePermissionKind,
    roles: tuple[str, ...],
    principals: tuple[MsGraphDrivePermissionPrincipal, ...],
    link_scope: MsGraphDriveLinkScope | None,
    has_unresolved_identity_facets: bool,
    deprecated_only: bool,
) -> bool:
    if not _roles_are_known(roles):
        return False
    if has_unresolved_identity_facets:
        return False
    if deprecated_only:
        return False

    if kind is MsGraphDrivePermissionKind.INVITATION:
        return False
    if kind is MsGraphDrivePermissionKind.UNKNOWN:
        return False

    if kind is MsGraphDrivePermissionKind.LINK:
        if link_scope is None:
            return False
        if link_scope is MsGraphDriveLinkScope.EXISTING_ACCESS:
            return False
        if link_scope is MsGraphDriveLinkScope.UNKNOWN:
            return False
        if link_scope in {MsGraphDriveLinkScope.ANONYMOUS, MsGraphDriveLinkScope.ORGANIZATION}:
            return True
        if link_scope is MsGraphDriveLinkScope.USERS:
            return len(principals) > 0
        return False

    if kind is MsGraphDrivePermissionKind.DIRECT:
        return len(principals) > 0

    return False


def _parse_link_scope(link_obj: dict[str, object]) -> tuple[MsGraphDriveLinkScope, str | None]:
    raw_scope = link_obj.get("scope")
    link_type: str | None = None
    if "type" in link_obj:
        raw_type = link_obj["type"]
        if raw_type is None:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        if not isinstance(raw_type, str):
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        trimmed_type = raw_type.strip()
        if not trimmed_type:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        link_type = trimmed_type

    if raw_scope is None:
        return MsGraphDriveLinkScope.UNKNOWN, link_type
    if not isinstance(raw_scope, str):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
    mapped = _SCOPE_MAP.get(raw_scope)
    if mapped is None:
        return MsGraphDriveLinkScope.UNKNOWN, link_type
    return MsGraphDriveLinkScope(mapped), link_type


def _parse_roles(payload: dict[str, object]) -> tuple[str, ...]:
    raw_roles = payload.get("roles")
    if not isinstance(raw_roles, list):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
    if not raw_roles:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
    normalized: list[str] = []
    seen: set[str] = set()
    for role in raw_roles:
        if not isinstance(role, str):
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        trimmed = role.strip()
        if not trimmed:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE)
        if trimmed not in seen:
            seen.add(trimmed)
            normalized.append(trimmed)
    return tuple(normalized)


def parse_msgraph_drive_permission(
    payload: object,
    *,
    expected_drive_id: str,
    expected_item_id: str,
) -> MsGraphDrivePermission:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None

    try:
        validated_drive_id = validate_msgraph_drive_id(expected_drive_id)
        validate_msgraph_drive_item_id(expected_item_id)
    except ValueError:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None

    try:
        permission_id = _validate_permission_opaque_id(payload.get("id"))
        roles = _parse_roles(payload)
    except ValueError:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None

    has_link = _facet_is_present(payload, "link")
    has_invitation = _facet_is_present(payload, "invitation")
    if has_link and has_invitation:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None

    if has_link:
        kind = MsGraphDrivePermissionKind.LINK
    elif has_invitation:
        kind = MsGraphDrivePermissionKind.INVITATION
    elif _has_grant_identity_fields(payload):
        kind = MsGraphDrivePermissionKind.DIRECT
    else:
        kind = MsGraphDrivePermissionKind.UNKNOWN

    link_scope: MsGraphDriveLinkScope | None = None
    link_type: str | None = None
    if has_link:
        link_obj = payload["link"]
        if not isinstance(link_obj, dict):
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
        link_scope, link_type = _parse_link_scope(link_obj)

    if has_invitation:
        invitation_obj = payload["invitation"]
        if not isinstance(invitation_obj, dict):
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None

    inheritance_known = False
    inherited_from_item_id: str | None = None
    if _facet_is_present(payload, "inheritedFrom"):
        inherited_from = payload["inheritedFrom"]
        if not isinstance(inherited_from, dict):
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
        try:
            inherited_from_item_id = validate_msgraph_drive_item_id(inherited_from.get("id"))
        except ValueError:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
        if "driveId" in inherited_from:
            drive_id_value = inherited_from["driveId"]
            if drive_id_value is None:
                raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
            try:
                inherited_drive_id = validate_msgraph_drive_id(drive_id_value)
            except ValueError:
                raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
            if inherited_drive_id != validated_drive_id:
                raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
        inheritance_known = True

    expires_at: datetime | None = None
    if "expirationDateTime" in payload:
        expires_at = _parse_timezone_aware_datetime(payload["expirationDateTime"])

    has_password: bool | None = None
    if "hasPassword" in payload:
        raw_has_password = payload["hasPassword"]
        if type(raw_has_password) is not bool:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
        has_password = raw_has_password

    has_v2 = _facet_is_present(payload, "grantedToV2") or _facet_is_present(
        payload, "grantedToIdentitiesV2"
    )
    deprecated_only = _has_deprecated_identity_fields(payload) and not has_v2

    principals: tuple[MsGraphDrivePermissionPrincipal, ...] = ()
    has_unresolved_identity_facets = False
    if has_v2:
        try:
            principals, has_unresolved_identity_facets = _parse_v2_principals(payload)
        except ValueError:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
    elif deprecated_only:
        principals = ()
        has_unresolved_identity_facets = False

    projection_complete = _compute_projection_complete(
        kind=kind,
        roles=roles,
        principals=principals,
        link_scope=link_scope,
        has_unresolved_identity_facets=has_unresolved_identity_facets,
        deprecated_only=deprecated_only,
    )
    if kind is MsGraphDrivePermissionKind.INVITATION:
        projection_complete = False

    return _safe_construct_permission(
        permission_id=permission_id,
        roles=roles,
        kind=kind,
        principals=principals,
        link_scope=link_scope,
        link_type=link_type,
        inherited_from_item_id=inherited_from_item_id,
        inheritance_known=inheritance_known,
        expires_at=expires_at,
        has_password=has_password,
        projection_complete=projection_complete,
    )


def _safe_construct_permission(**kwargs: object) -> MsGraphDrivePermission:
    try:
        return MsGraphDrivePermission(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None


def _safe_construct_permission_page(**kwargs: object) -> MsGraphDrivePermissionPage:
    try:
        return MsGraphDrivePermissionPage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None


def validate_msgraph_drive_permission(value: object) -> MsGraphDrivePermission:
    """Deep-revalidate a Drive permission instance against the full model contract."""
    if not isinstance(value, MsGraphDrivePermission):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
    try:
        return MsGraphDrivePermission.model_validate(value.model_dump(mode="python"))
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None


def validate_msgraph_drive_permission_page(value: object) -> MsGraphDrivePermissionPage:
    """Deep-revalidate a Drive permissions page and every nested permission."""
    if not isinstance(value, MsGraphDrivePermissionPage):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None

    try:
        raw_items = value.items
        raw_continuation = value.continuation
        raw_acl_complete = value.acl_complete
        raw_inheritance_complete = value.inheritance_complete
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None

    if type(raw_items) is not tuple:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None

    validated_items: list[MsGraphDrivePermission] = []
    for item in raw_items:
        if not isinstance(item, MsGraphDrivePermission):
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
        validated_items.append(validate_msgraph_drive_permission(item))

    permission_ids = [item.permission_id for item in validated_items]
    if len(permission_ids) != len(set(permission_ids)):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None

    continuation: MsGraphKnowledgeContinuation | None = None
    if raw_continuation is not None:
        if not isinstance(raw_continuation, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
        try:
            revalidated_continuation = MsGraphKnowledgeContinuation.model_validate(
                raw_continuation.model_dump(mode="python")
            )
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
        if revalidated_continuation.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
        continuation = revalidated_continuation

    if raw_acl_complete is not False:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
    if raw_inheritance_complete is not False:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None

    try:
        return MsGraphDrivePermissionPage(
            items=tuple(validated_items),
            continuation=continuation,
            acl_complete=False,
            inheritance_complete=False,
        )
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None


def _graph_base_path(graph_base_url: str) -> str:
    parsed_base = urlparse(graph_base_url)
    return parsed_base.path.rstrip("/") or "/"


def _extract_drive_and_item_from_permissions_path(
    path: str,
    *,
    graph_base_path: str,
) -> tuple[str, str] | None:
    normalized = path.rstrip("/") or "/"
    expected_prefix = f"{graph_base_path.rstrip('/')}/drives/"
    expected_suffix = "/permissions"

    if not normalized.startswith(expected_prefix):
        return None
    if not normalized.endswith(expected_suffix):
        return None

    middle = normalized[len(expected_prefix) : -len(expected_suffix)]
    parts = middle.split("/")
    if len(parts) != 3 or parts[1] != "items":
        return None
    drive_segment, _, item_segment = parts
    if not drive_segment or not item_segment:
        return None
    return unquote(drive_segment), unquote(item_segment)


def validate_msgraph_drive_permissions_continuation(
    continuation: object,
    *,
    drive_id: str,
    item_id: str,
    graph_base_url: str,
) -> MsGraphKnowledgeContinuation:
    if not isinstance(continuation, MsGraphKnowledgeContinuation):
        raise IntegrationConfigurationError(_INVALID_PERMISSIONS_CONTINUATION) from None
    if continuation.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
        raise IntegrationConfigurationError(_INVALID_PERMISSIONS_CONTINUATION) from None

    try:
        validated_url = validate_msgraph_continuation_url(
            continuation.url,
            graph_base_url=graph_base_url,
        )
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_PERMISSIONS_CONTINUATION) from None

    parsed = urlparse(validated_url)
    extracted = _extract_drive_and_item_from_permissions_path(
        parsed.path,
        graph_base_path=_graph_base_path(graph_base_url),
    )
    if extracted is None:
        raise IntegrationConfigurationError(_INVALID_PERMISSIONS_CONTINUATION) from None

    extracted_drive_id, extracted_item_id = extracted
    try:
        validated_drive_id = validate_msgraph_drive_id(drive_id)
        validated_item_id = validate_msgraph_drive_item_id(item_id)
        validated_extracted_drive_id = validate_msgraph_drive_id(extracted_drive_id)
        validated_extracted_item_id = validate_msgraph_drive_item_id(extracted_item_id)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_PERMISSIONS_CONTINUATION) from None

    if (
        validated_extracted_drive_id != validated_drive_id
        or validated_extracted_item_id != validated_item_id
    ):
        raise IntegrationConfigurationError(_INVALID_PERMISSIONS_CONTINUATION) from None

    return continuation


def _validate_permissions_item(item: object) -> MsGraphDriveItem:
    if not isinstance(item, MsGraphDriveItem):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
    try:
        validated = MsGraphDriveItem.model_validate(item.model_dump(mode="python"))
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
    if validated.kind is MsGraphDriveItemKind.DELETED:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
    if validated.kind not in {
        MsGraphDriveItemKind.FILE,
        MsGraphDriveItemKind.FOLDER,
        MsGraphDriveItemKind.PACKAGE,
        MsGraphDriveItemKind.OTHER,
    }:
        raise ValueError(_MALFORMED_PERMISSIONS_RESPONSE) from None
    return validated


class MsGraphDrivePermissionsReader:
    """Drive item permissions reader over the shared Graph knowledge transport."""

    def __init__(
        self,
        config: Ms365GraphIntegrationConfig,
        *,
        transport: MsGraphKnowledgeTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    def read_permissions_page(
        self,
        *,
        item: MsGraphDriveItem,
        continuation: MsGraphKnowledgeContinuation | None,
    ) -> MsGraphDrivePermissionPage:
        validated_item = _validate_permissions_item(item)
        validated_drive_id = validate_msgraph_drive_id(validated_item.drive_id)
        validated_item_id = validate_msgraph_drive_item_id(validated_item.remote_id)

        if continuation is None:
            quoted_drive_id = quote(validated_drive_id, safe="")
            quoted_item_id = quote(validated_item_id, safe="")
            path = f"/drives/{quoted_drive_id}/items/{quoted_item_id}/permissions"
            payload = self._transport.get_initial_json(
                path=path,
                params={"$select": _PERMISSIONS_SELECT},
                not_found_is_dependency=True,
            )
        else:
            validated_continuation = validate_msgraph_drive_permissions_continuation(
                continuation,
                drive_id=validated_drive_id,
                item_id=validated_item_id,
                graph_base_url=self._config.graph_base_url,
            )
            payload = self._transport.get_continuation_json(
                continuation=validated_continuation,
                not_found_is_dependency=True,
            )

        collection_page = parse_msgraph_collection_page(
            payload,
            graph_base_url=self._config.graph_base_url,
            delta_mode=False,
        )
        parsed_items = tuple(
            parse_msgraph_drive_permission(
                raw_item,
                expected_drive_id=validated_drive_id,
                expected_item_id=validated_item_id,
            )
            for raw_item in collection_page.items
        )
        return validate_msgraph_drive_permission_page(
            _safe_construct_permission_page(
                items=parsed_items,
                continuation=collection_page.continuation,
            )
        )
