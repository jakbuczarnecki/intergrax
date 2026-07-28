# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Private Confluence knowledge-read models and protocol (CONFLUENCE-KNOWLEDGE-ADAPTER-1)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONFLUENCE_PAGES_SOURCE_KIND = "pages"
CONFLUENCE_SPACE_SCOPE_TYPE = "confluence_space"
CONFLUENCE_PAGES_CURSOR_VERSION = "confluence.pages.cursor.v1"

_CONFLUENCE_NUMERIC_ID_RE = re.compile(r"^[1-9][0-9]*$")
_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)


def validate_confluence_space_id(space_id: str) -> str:
    cleaned = str(space_id).strip()
    if not _CONFLUENCE_NUMERIC_ID_RE.fullmatch(cleaned):
        raise ValueError("invalid Confluence space id")
    return cleaned


def validate_confluence_page_id(page_id: str) -> str:
    cleaned = str(page_id).strip()
    if not _CONFLUENCE_NUMERIC_ID_RE.fullmatch(cleaned):
        raise ValueError("invalid Confluence page id")
    return cleaned


class ConfluenceKnowledgePage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    remote_id: str
    space_id: str
    parent_id: str | None = None
    status: Literal["current"]
    title: str
    created_at: datetime
    version_number: int
    version_created_at: datetime
    storage_value: str | None = None
    web_url: str

    @field_validator("remote_id")
    @classmethod
    def _validate_remote_id(cls, value: str) -> str:
        return validate_confluence_page_id(value)

    @field_validator("space_id")
    @classmethod
    def _validate_space_id(cls, value: str) -> str:
        return validate_confluence_space_id(value)

    @field_validator("parent_id")
    @classmethod
    def _validate_parent_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_confluence_page_id(value)

    @field_validator("title", "web_url")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("field must be a non-empty string")
        return cleaned

    @field_validator("version_number")
    @classmethod
    def _validate_version_number(cls, value: int) -> int:
        if value < 1:
            raise ValueError("version_number must be >= 1")
        return value

    @field_validator("created_at", "version_created_at")
    @classmethod
    def _validate_timezone_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if value.utcoffset() is None:
            raise ValueError("timestamp must have a defined UTC offset")
        return value.astimezone(timezone.utc)


class ConfluenceKnowledgePagePage(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    pages: tuple[ConfluenceKnowledgePage, ...] = ()
    next_cursor: str | None = Field(default=None, repr=False)
    is_last: bool

    @field_validator("next_cursor", mode="before")
    @classmethod
    def _validate_next_cursor(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("next_cursor must be a string when provided")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("next_cursor must not be empty when provided")
        return cleaned

    @model_validator(mode="after")
    def _cursor_rules(self) -> ConfluenceKnowledgePagePage:
        if not self.is_last and not self.next_cursor:
            raise ValueError("next_cursor is required when is_last is False")
        if self.is_last and self.next_cursor is not None:
            raise ValueError("next_cursor must be None when is_last is True")
        seen_ids: set[str] = set()
        space_ids: set[str] = set()
        for page in self.pages:
            if page.remote_id in seen_ids:
                raise ValueError("duplicate page id on page")
            seen_ids.add(page.remote_id)
            space_ids.add(page.space_id)
        if len(space_ids) > 1:
            raise ValueError("all pages must belong to the same space")
        return self


@runtime_checkable
class ConfluenceKnowledgeReadClient(Protocol):
    def list_knowledge_pages(
        self,
        *,
        space_id: str,
        cursor: str | None,
        limit: int,
    ) -> ConfluenceKnowledgePagePage:
        ...

    def get_knowledge_page(
        self,
        *,
        page_id: str,
        version_number: int,
    ) -> ConfluenceKnowledgePage:
        ...


def _parse_timestamp(raw: object, *, field_name: str) -> datetime:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field_name} is required")
    text = raw.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    if len(text) >= 5 and text[-5] in "+-" and text[-3] != ":":
        text = f"{text[:-2]}:{text[-2:]}"
    try:
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} timestamp is invalid") from None
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def extract_confluence_knowledge_next_cursor(
    next_link: object,
    *,
    space_id: str,
) -> str:
    if not isinstance(next_link, str) or not next_link.strip():
        raise ValueError("unexpected Confluence knowledge response")
    parsed = urlparse(next_link.strip())
    if parsed.scheme or parsed.netloc or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("unexpected Confluence knowledge response")
    expected_path = f"/wiki/api/v2/spaces/{space_id}/pages"
    if parsed.path != expected_path:
        raise ValueError("unexpected Confluence knowledge response")
    cursor_values = parse_qs(parsed.query, keep_blank_values=False).get("cursor")
    if cursor_values is None or len(cursor_values) != 1:
        raise ValueError("unexpected Confluence knowledge response")
    cursor = cursor_values[0].strip()
    if not cursor:
        raise ValueError("unexpected Confluence knowledge response")
    return cursor


def _parse_inventory_page(
    payload: Mapping[str, Any],
    *,
    requested_space_id: str,
    page_url_builder: Any,
) -> ConfluenceKnowledgePage:
    if not isinstance(payload, dict):
        raise ValueError("page payload must be an object")
    remote_id_raw = payload.get("id")
    if remote_id_raw is None or not str(remote_id_raw).strip():
        raise ValueError("page id is required")
    status_raw = payload.get("status")
    if status_raw != "current":
        raise ValueError("page status must be current")
    title_raw = payload.get("title")
    if not isinstance(title_raw, str) or not title_raw.strip():
        raise ValueError("page title is required")
    space_id_raw = payload.get("spaceId")
    if space_id_raw is None or not str(space_id_raw).strip():
        raise ValueError("page spaceId is required")
    space_id = str(space_id_raw).strip()
    if space_id != requested_space_id:
        raise ValueError("page spaceId does not match requested space")
    created_at = _parse_timestamp(payload.get("createdAt"), field_name="createdAt")
    version_obj = payload.get("version")
    if not isinstance(version_obj, dict):
        raise ValueError("page version is required")
    version_number_raw = version_obj.get("number")
    if not isinstance(version_number_raw, int) or version_number_raw < 1:
        raise ValueError("page version number is required")
    version_created_at = _parse_timestamp(
        version_obj.get("createdAt"),
        field_name="version.createdAt",
    )
    parent_id_raw = payload.get("parentId")
    parent_id: str | None
    if parent_id_raw is None:
        parent_id = None
    elif not str(parent_id_raw).strip():
        parent_id = None
    else:
        parent_id = str(parent_id_raw).strip()
    remote_id = str(remote_id_raw).strip()
    return ConfluenceKnowledgePage(
        remote_id=remote_id,
        space_id=space_id,
        parent_id=parent_id,
        status="current",
        title=title_raw.strip(),
        created_at=created_at,
        version_number=version_number_raw,
        version_created_at=version_created_at,
        storage_value=None,
        web_url=page_url_builder(remote_id),
    )


def parse_confluence_knowledge_page_page(
    payload: Mapping[str, Any],
    *,
    requested_space_id: str,
    page_url_builder: Any,
) -> ConfluenceKnowledgePagePage:
    if not isinstance(payload, dict):
        raise ValueError("list response must be an object")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("results must be a list")
    seen_ids: set[str] = set()
    pages: list[ConfluenceKnowledgePage] = []
    for item in raw_results:
        page = _parse_inventory_page(
            item,
            requested_space_id=requested_space_id,
            page_url_builder=page_url_builder,
        )
        if page.remote_id in seen_ids:
            raise ValueError("duplicate page id on page")
        seen_ids.add(page.remote_id)
        pages.append(page)
    links = payload.get("_links")
    next_link: str | None = None
    if links is not None:
        if not isinstance(links, dict):
            raise ValueError("unexpected Confluence knowledge response")
        next_raw = links.get("next")
        if next_raw is not None:
            next_link = extract_confluence_knowledge_next_cursor(
                next_raw,
                space_id=requested_space_id,
            )
    if next_link is None:
        return ConfluenceKnowledgePagePage(pages=tuple(pages), next_cursor=None, is_last=True)
    return ConfluenceKnowledgePagePage(
        pages=tuple(pages),
        next_cursor=next_link,
        is_last=False,
    )


def parse_confluence_knowledge_page(
    payload: Mapping[str, Any],
    *,
    page_id: str,
    version_number: int,
    page_url_builder: Any,
) -> ConfluenceKnowledgePage:
    if not isinstance(payload, dict):
        raise ValueError("page payload must be an object")
    validated_page_id = validate_confluence_page_id(page_id)
    if version_number < 1:
        raise ValueError("version_number must be >= 1")
    remote_id_raw = payload.get("id")
    if remote_id_raw is None or str(remote_id_raw).strip() != validated_page_id:
        raise ValueError("page id does not match requested page")
    status_raw = payload.get("status")
    if status_raw != "current":
        raise ValueError("page status must be current")
    title_raw = payload.get("title")
    if not isinstance(title_raw, str) or not title_raw.strip():
        raise ValueError("page title is required")
    space_id_raw = payload.get("spaceId")
    if space_id_raw is None or not str(space_id_raw).strip():
        raise ValueError("page spaceId is required")
    created_at = _parse_timestamp(payload.get("createdAt"), field_name="createdAt")
    version_obj = payload.get("version")
    if not isinstance(version_obj, dict):
        raise ValueError("page version is required")
    version_number_raw = version_obj.get("number")
    if not isinstance(version_number_raw, int) or version_number_raw != version_number:
        raise ValueError("page version number does not match requested version")
    version_created_at = _parse_timestamp(
        version_obj.get("createdAt"),
        field_name="version.createdAt",
    )
    parent_id_raw = payload.get("parentId")
    parent_id: str | None
    if parent_id_raw is None:
        parent_id = None
    elif not str(parent_id_raw).strip():
        parent_id = None
    else:
        parent_id = str(parent_id_raw).strip()
    body_obj = payload.get("body")
    if not isinstance(body_obj, dict):
        raise ValueError("page body is required")
    storage_obj = body_obj.get("storage")
    if not isinstance(storage_obj, dict):
        raise ValueError("page body.storage is required")
    storage_value_raw = storage_obj.get("value")
    if not isinstance(storage_value_raw, str):
        raise ValueError("page body.storage.value must be a string")
    return ConfluenceKnowledgePage(
        remote_id=validated_page_id,
        space_id=str(space_id_raw).strip(),
        parent_id=parent_id,
        status="current",
        title=title_raw.strip(),
        created_at=created_at,
        version_number=version_number_raw,
        version_created_at=version_created_at,
        storage_value=storage_value_raw,
        web_url=page_url_builder(validated_page_id),
    )
