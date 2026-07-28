# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft Graph Mail knowledge-read: mailbox folder enumeration for one known user."""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable
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

MSGRAPH_MAIL_SOURCE_KIND = "mail"

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_MAIL_FOLDERS_RESPONSE = "unexpected Microsoft Graph mail folders response"
_INVALID_MAIL_FOLDERS_REQUEST = "invalid Microsoft Graph mailbox folder request"
_INVALID_MAIL_FOLDERS_CONTINUATION = "invalid Microsoft Graph mail folders continuation"
_MAX_MSGRAPH_ID_LEN = 1024
_ASCII_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_MIN_FOLDER_LIMIT = 1
_MAX_FOLDER_LIMIT = 200

_MAIL_FOLDERS_SELECT = (
    "id,displayName,parentFolderId,childFolderCount,totalItemCount,unreadItemCount,isHidden"
)


def validate_msgraph_mailbox_user_id(value: object) -> str:
    return _validate_msgraph_opaque_id(value)


def validate_msgraph_mail_folder_id(value: object) -> str:
    return _validate_msgraph_opaque_id(value)


def _validate_msgraph_opaque_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
    if len(trimmed) > _MAX_MSGRAPH_ID_LEN:
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
    return trimmed


def _validate_display_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
    trimmed = value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
    if _ASCII_CONTROL.search(trimmed):
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
    if len(trimmed) > _MAX_MSGRAPH_ID_LEN:
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
    return trimmed


def _validate_non_negative_int(value: object) -> int:
    if type(value) is not int:
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
    if value < 0:
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
    return value


class MsGraphMailFolder(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    mailbox_user_id: str
    remote_id: str
    parent_remote_id: str | None = None

    display_name: str = Field(repr=False)

    child_folder_count: int
    total_item_count: int
    unread_item_count: int

    is_hidden: bool

    @field_validator("mailbox_user_id", mode="before")
    @classmethod
    def _validate_mailbox_user_id(cls, value: object) -> str:
        return validate_msgraph_mailbox_user_id(value)

    @field_validator("remote_id", mode="before")
    @classmethod
    def _validate_remote_id(cls, value: object) -> str:
        return validate_msgraph_mail_folder_id(value)

    @field_validator("parent_remote_id", mode="before")
    @classmethod
    def _validate_parent_remote_id(cls, value: object) -> str | None:
        if value is None:
            return None
        return validate_msgraph_mail_folder_id(value)

    @field_validator("display_name", mode="before")
    @classmethod
    def _validate_display_name_field(cls, value: object) -> str:
        return _validate_display_name(value)

    @field_validator("child_folder_count", "total_item_count", "unread_item_count", mode="before")
    @classmethod
    def _validate_counts(cls, value: object) -> int:
        return _validate_non_negative_int(value)

    @field_validator("is_hidden", mode="before")
    @classmethod
    def _validate_is_hidden(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_count_relationship(self) -> MsGraphMailFolder:
        if self.unread_item_count > self.total_item_count:
            raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
        return self


class MsGraphMailFolderPage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    items: tuple[MsGraphMailFolder, ...]
    continuation: MsGraphKnowledgeContinuation | None = Field(default=None, repr=False)

    @field_validator("items", mode="before")
    @classmethod
    def _validate_items(cls, value: object) -> tuple[MsGraphMailFolder, ...]:
        if type(value) is not tuple:
            raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
        for item in value:
            if not isinstance(item, MsGraphMailFolder):
                raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
        return value

    @field_validator("continuation", mode="before")
    @classmethod
    def _validate_continuation(cls, value: object) -> MsGraphKnowledgeContinuation | None:
        if value is None:
            return None
        if not isinstance(value, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
        if value.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
        return value

    @model_validator(mode="after")
    def _validate_page_shape(self) -> MsGraphMailFolderPage:
        remote_ids = [item.remote_id for item in self.items]
        if len(remote_ids) != len(set(remote_ids)):
            raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
        return self

    @property
    def has_more(self) -> bool:
        return self.continuation is not None


@runtime_checkable
class MsGraphMailFoldersReadClient(Protocol):
    def read_mail_folders_page(
        self,
        *,
        mailbox_user_id: str,
        parent_folder_id: str | None,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphMailFolderPage:
        ...


def _safe_construct_folder(**kwargs: object) -> MsGraphMailFolder:
    try:
        return MsGraphMailFolder(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None


def _safe_construct_folder_page(**kwargs: object) -> MsGraphMailFolderPage:
    try:
        return MsGraphMailFolderPage(**kwargs)
    except (ValueError, TypeError, ValidationError):
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None


def validate_msgraph_mail_folder(value: object) -> MsGraphMailFolder:
    """Deep-revalidate a Mail folder instance against the full model contract."""
    if not isinstance(value, MsGraphMailFolder):
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None
    try:
        return MsGraphMailFolder.model_validate(value.model_dump(mode="python"))
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None


def validate_msgraph_mail_folder_page(value: object) -> MsGraphMailFolderPage:
    """Deep-revalidate a Mail folders page and every nested folder."""
    if not isinstance(value, MsGraphMailFolderPage):
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None

    try:
        raw_items = value.items
        raw_continuation = value.continuation
    except (AttributeError, TypeError, ValueError):
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None

    if type(raw_items) is not tuple:
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None

    validated_items: list[MsGraphMailFolder] = []
    for item in raw_items:
        if not isinstance(item, MsGraphMailFolder):
            raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None
        validated_items.append(validate_msgraph_mail_folder(item))

    remote_ids = [item.remote_id for item in validated_items]
    if len(remote_ids) != len(set(remote_ids)):
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None

    continuation: MsGraphKnowledgeContinuation | None = None
    if raw_continuation is not None:
        if not isinstance(raw_continuation, MsGraphKnowledgeContinuation):
            raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None
        try:
            revalidated_continuation = MsGraphKnowledgeContinuation.model_validate(
                raw_continuation.model_dump(mode="python")
            )
        except (ValueError, TypeError, AttributeError, ValidationError):
            raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None
        if revalidated_continuation.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
            raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None
        continuation = revalidated_continuation

    try:
        return MsGraphMailFolderPage(items=tuple(validated_items), continuation=continuation)
    except (ValueError, TypeError, AttributeError, ValidationError):
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None


def _parse_required_non_negative_int(mapping: dict[str, object], key: str) -> int:
    if key not in mapping:
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
    return _validate_non_negative_int(mapping[key])


def _parse_required_bool(mapping: dict[str, object], key: str) -> bool:
    if key not in mapping:
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
    value = mapping[key]
    if type(value) is not bool:
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE)
    return value


def parse_msgraph_mail_folder(
    payload: object,
    *,
    expected_mailbox_user_id: str,
) -> MsGraphMailFolder:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None

    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(expected_mailbox_user_id)
    except ValueError:
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None

    try:
        remote_id = validate_msgraph_mail_folder_id(payload.get("id"))
        display_name = _validate_display_name(payload.get("displayName"))
        child_folder_count = _parse_required_non_negative_int(payload, "childFolderCount")
        total_item_count = _parse_required_non_negative_int(payload, "totalItemCount")
        unread_item_count = _parse_required_non_negative_int(payload, "unreadItemCount")
        is_hidden = _parse_required_bool(payload, "isHidden")
    except ValueError:
        raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None

    parent_remote_id: str | None = None
    if "parentFolderId" in payload:
        raw_parent = payload["parentFolderId"]
        if raw_parent is None:
            raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None
        try:
            parent_remote_id = validate_msgraph_mail_folder_id(raw_parent)
        except ValueError:
            raise ValueError(_MALFORMED_MAIL_FOLDERS_RESPONSE) from None

    return _safe_construct_folder(
        mailbox_user_id=validated_mailbox_user_id,
        remote_id=remote_id,
        parent_remote_id=parent_remote_id,
        display_name=display_name,
        child_folder_count=child_folder_count,
        total_item_count=total_item_count,
        unread_item_count=unread_item_count,
        is_hidden=is_hidden,
    )


def _graph_base_path(graph_base_url: str) -> str:
    parsed_base = urlparse(graph_base_url)
    return parsed_base.path.rstrip("/") or "/"


def _extract_mail_folders_path(
    path: str,
    *,
    graph_base_path: str,
) -> tuple[str, str | None] | None:
    """Return (mailbox_user_id, parent_folder_id_or_None) when path matches mail folders."""
    normalized = path.rstrip("/") or "/"
    expected_prefix = f"{graph_base_path.rstrip('/')}/users/"
    if not normalized.startswith(expected_prefix):
        return None

    remainder = normalized[len(expected_prefix) :]
    root_suffix = "/mailFolders"
    child_suffix = "/childFolders"

    if remainder.endswith(root_suffix) and not remainder.endswith(child_suffix):
        mailbox_segment = remainder[: -len(root_suffix)]
        if "/" in mailbox_segment or not mailbox_segment:
            return None
        return unquote(mailbox_segment), None

    if remainder.endswith(child_suffix):
        without_child = remainder[: -len(child_suffix)]
        parts = without_child.split("/")
        if len(parts) != 3 or parts[1] != "mailFolders" or not parts[0] or not parts[2]:
            return None
        return unquote(parts[0]), unquote(parts[2])

    return None


def validate_msgraph_mail_folders_continuation(
    continuation: object,
    *,
    mailbox_user_id: str,
    parent_folder_id: str | None,
    graph_base_url: str,
) -> MsGraphKnowledgeContinuation:
    if not isinstance(continuation, MsGraphKnowledgeContinuation):
        raise IntegrationConfigurationError(_INVALID_MAIL_FOLDERS_CONTINUATION) from None
    if continuation.kind is not MsGraphKnowledgeContinuationKind.NEXT_PAGE:
        raise IntegrationConfigurationError(_INVALID_MAIL_FOLDERS_CONTINUATION) from None

    try:
        validated_url = validate_msgraph_continuation_url(
            continuation.url,
            graph_base_url=graph_base_url,
        )
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_MAIL_FOLDERS_CONTINUATION) from None

    parsed = urlparse(validated_url)
    extracted = _extract_mail_folders_path(
        parsed.path,
        graph_base_path=_graph_base_path(graph_base_url),
    )
    if extracted is None:
        raise IntegrationConfigurationError(_INVALID_MAIL_FOLDERS_CONTINUATION) from None

    extracted_mailbox_user_id, extracted_parent_folder_id = extracted
    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(mailbox_user_id)
        validated_extracted_mailbox_user_id = validate_msgraph_mailbox_user_id(
            extracted_mailbox_user_id
        )
        if parent_folder_id is None:
            validated_parent_folder_id: str | None = None
        else:
            validated_parent_folder_id = validate_msgraph_mail_folder_id(parent_folder_id)
        if extracted_parent_folder_id is not None:
            validated_extracted_parent_folder_id = validate_msgraph_mail_folder_id(
                extracted_parent_folder_id
            )
        else:
            validated_extracted_parent_folder_id = None
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_MAIL_FOLDERS_CONTINUATION) from None

    if validated_extracted_mailbox_user_id != validated_mailbox_user_id:
        raise IntegrationConfigurationError(_INVALID_MAIL_FOLDERS_CONTINUATION) from None

    if validated_parent_folder_id is None:
        if validated_extracted_parent_folder_id is not None:
            raise IntegrationConfigurationError(_INVALID_MAIL_FOLDERS_CONTINUATION) from None
    elif validated_extracted_parent_folder_id != validated_parent_folder_id:
        raise IntegrationConfigurationError(_INVALID_MAIL_FOLDERS_CONTINUATION) from None

    return continuation


def _validate_folder_limit(limit: object) -> int:
    if type(limit) is not int:
        raise IntegrationConfigurationError(_INVALID_MAIL_FOLDERS_REQUEST)
    if limit < _MIN_FOLDER_LIMIT or limit > _MAX_FOLDER_LIMIT:
        raise IntegrationConfigurationError(_INVALID_MAIL_FOLDERS_REQUEST)
    return limit


def _validate_mailbox_request_input(
    *,
    mailbox_user_id: object,
    parent_folder_id: object,
) -> tuple[str, str | None]:
    try:
        validated_mailbox_user_id = validate_msgraph_mailbox_user_id(mailbox_user_id)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_MAIL_FOLDERS_REQUEST) from None

    if parent_folder_id is None:
        return validated_mailbox_user_id, None

    try:
        validated_parent_folder_id = validate_msgraph_mail_folder_id(parent_folder_id)
    except ValueError:
        raise IntegrationConfigurationError(_INVALID_MAIL_FOLDERS_REQUEST) from None

    return validated_mailbox_user_id, validated_parent_folder_id


class MsGraphMailFoldersReader:
    """Mailbox folder reader over the shared Graph knowledge transport."""

    def __init__(
        self,
        config: Ms365GraphIntegrationConfig,
        *,
        transport: MsGraphKnowledgeTransport,
    ) -> None:
        self._config = config
        self._transport = transport

    def read_folders_page(
        self,
        *,
        mailbox_user_id: str,
        parent_folder_id: str | None,
        continuation: MsGraphKnowledgeContinuation | None,
        limit: int,
    ) -> MsGraphMailFolderPage:
        validated_mailbox_user_id, validated_parent_folder_id = _validate_mailbox_request_input(
            mailbox_user_id=mailbox_user_id,
            parent_folder_id=parent_folder_id,
        )
        validated_limit = _validate_folder_limit(limit)

        if continuation is None:
            quoted_mailbox_user_id = quote(validated_mailbox_user_id, safe="")
            if validated_parent_folder_id is None:
                path = f"/users/{quoted_mailbox_user_id}/mailFolders"
            else:
                quoted_parent_folder_id = quote(validated_parent_folder_id, safe="")
                path = (
                    f"/users/{quoted_mailbox_user_id}/mailFolders/"
                    f"{quoted_parent_folder_id}/childFolders"
                )
            payload = self._transport.get_initial_json(
                path=path,
                params={
                    "$top": validated_limit,
                    "$select": _MAIL_FOLDERS_SELECT,
                    "includeHiddenFolders": "true",
                },
            )
        else:
            validated_continuation = validate_msgraph_mail_folders_continuation(
                continuation,
                mailbox_user_id=validated_mailbox_user_id,
                parent_folder_id=validated_parent_folder_id,
                graph_base_url=self._config.graph_base_url,
            )
            payload = self._transport.get_continuation_json(
                continuation=validated_continuation,
            )

        collection_page = parse_msgraph_collection_page(
            payload,
            graph_base_url=self._config.graph_base_url,
            delta_mode=False,
        )
        parsed_items = tuple(
            parse_msgraph_mail_folder(
                raw_item,
                expected_mailbox_user_id=validated_mailbox_user_id,
            )
            for raw_item in collection_page.items
        )
        return validate_msgraph_mail_folder_page(
            _safe_construct_folder_page(
                items=parsed_items,
                continuation=collection_page.continuation,
            )
        )
