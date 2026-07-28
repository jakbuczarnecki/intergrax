# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Confluence pages knowledge source adapter (CONFLUENCE-KNOWLEDGE-ADAPTER-1)."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import json
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    ConfluenceWikiKnowledgeIntegration,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.knowledge_read import (
    CONFLUENCE_PAGES_CURSOR_VERSION,
    CONFLUENCE_PAGES_SOURCE_KIND,
    CONFLUENCE_SPACE_SCOPE_TYPE,
    ConfluenceKnowledgePage,
    ConfluenceKnowledgePagePage,
    validate_confluence_space_id,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeAdapterCapabilities,
    KnowledgeChange,
    KnowledgeChangeKind,
    KnowledgeContent,
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgeItemIdentity,
    KnowledgeItemProvenance,
    KnowledgeItemRevision,
    KnowledgePage,
    KnowledgePermissions,
    KnowledgeScopeInfo,
    KnowledgeSourceRef,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry

_CONFLUENCE_REMOTE_ID_RE = re.compile(r"^[1-9][0-9]*$")
_RICH_TEXT_MIME = "application/vnd.atlassian.confluence.storage+xml"


class _ConfluencePagesReconciliationCursor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["confluence.pages.cursor.v1"]
    space_id: str
    provider_cursor: str | None = Field(default=None, repr=False)
    complete: bool

    @field_validator("space_id")
    @classmethod
    def _validate_space_id(cls, value: str) -> str:
        return validate_confluence_space_id(value)

    @field_validator("provider_cursor")
    @classmethod
    def _validate_provider_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("provider_cursor must not be empty")
        return cleaned

    @model_validator(mode="after")
    def _cursor_rules(self) -> _ConfluencePagesReconciliationCursor:
        if not self.complete and not self.provider_cursor:
            raise ValueError("provider_cursor is required when complete is False")
        if self.complete and self.provider_cursor is not None:
            raise ValueError("provider_cursor must be None when complete is True")
        return self


class ConfluencePagesKnowledgeAdapter:
    """Thin mapping from Confluence wiki integration to vendor-neutral knowledge models."""

    @property
    def provider_id(self) -> str:
        return "confluence"

    @property
    def integration_kind(self) -> IntegrationCategory:
        return IntegrationCategory.WIKI_KNOWLEDGE

    @property
    def source_kind(self) -> str:
        return CONFLUENCE_PAGES_SOURCE_KIND

    @property
    def capabilities(self) -> KnowledgeAdapterCapabilities:
        return KnowledgeAdapterCapabilities(
            full_inventory=True,
            incremental_changes=False,
            content_fetch=True,
            binary_content=False,
            rich_text_content=True,
            structured_content=False,
            permissions=False,
            tombstones=False,
            remote_versions=True,
            reconciliation=True,
        )

    async def inspect_scope(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
    ) -> KnowledgeScopeInfo:
        self._require_confluence_integration(integration=integration, source=source)
        self._validate_source(source)
        return KnowledgeScopeInfo(
            source=source,
            capabilities=self.capabilities,
            safe_display_name=source.scope.safe_display_name,
        )

    async def read_page(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        cursor: KnowledgeCursor | None,
        limit: int,
    ) -> KnowledgePage:
        confluence_integration = self._require_confluence_integration(
            integration=integration,
            source=source,
        )
        space_id = self._validate_source(source)
        decoded = self._decode_cursor(cursor, space_id=space_id)
        if decoded is not None and decoded.complete:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Confluence reconciliation cursor is complete; restart reconciliation",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )

        provider_cursor = None if decoded is None else decoded.provider_cursor
        page = await asyncio.to_thread(
            confluence_integration.list_knowledge_pages,
            space_id=space_id,
            cursor=provider_cursor,
            limit=limit,
        )
        try:
            validated_next_cursor = self._validate_page_for_source(
                page,
                space_id=space_id,
            )
        except (ValueError, TypeError, ValidationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Confluence knowledge provider response is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        changes = tuple(self._page_to_change(item) for item in page.pages)
        if not page.is_last:
            checkpoint = self._encode_cursor(
                _ConfluencePagesReconciliationCursor(
                    schema_version=CONFLUENCE_PAGES_CURSOR_VERSION,
                    space_id=space_id,
                    provider_cursor=validated_next_cursor,
                    complete=False,
                )
            )
            return KnowledgePage(
                changes=changes,
                next_cursor=checkpoint,
                proposed_checkpoint=checkpoint,
                has_more=True,
            )

        final_checkpoint = self._encode_cursor(
            _ConfluencePagesReconciliationCursor(
                schema_version=CONFLUENCE_PAGES_CURSOR_VERSION,
                space_id=space_id,
                provider_cursor=None,
                complete=True,
            )
        )
        return KnowledgePage(
            changes=changes,
            next_cursor=None,
            proposed_checkpoint=final_checkpoint,
            has_more=False,
        )

    async def fetch_content(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgeContent:
        confluence_integration = self._require_confluence_integration(
            integration=integration,
            source=source,
        )
        space_id = self._validate_source(source)
        self._validate_item(item, source=source)
        version_number = self._parse_version_number(item.revision.version)
        page_id = str(item.identity.remote_id).strip()
        page = await asyncio.to_thread(
            confluence_integration.get_knowledge_page,
            page_id=page_id,
            version_number=version_number,
        )
        try:
            self._validate_fetched_page_identity(
                page,
                item=item,
                space_id=space_id,
            )
        except ValueError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Confluence page response identity does not match requested item",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        storage_value = page.storage_value if page.storage_value is not None else ""
        canonical_storage = f"<h1>{html.escape(page.title)}</h1>{storage_value}"
        content_hash = hashlib.sha256(canonical_storage.encode("utf-8")).hexdigest()
        return KnowledgeContent(
            mode=KnowledgeContentMode.RICH_TEXT,
            rich_text=canonical_storage,
            mime_type=_RICH_TEXT_MIME,
            encoding="utf-8",
            content_hash=content_hash,
        )

    async def fetch_permissions(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
        item: KnowledgeItemDescriptor,
    ) -> KnowledgePermissions:
        self._require_confluence_integration(integration=integration, source=source)
        self._validate_source(source)
        self._validate_item(item, source=source)
        raise VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
            safe_message="Confluence page permission projection is not implemented",
            provider_id=self.provider_id,
            source_kind=self.source_kind,
            retryable=False,
        )

    def _require_confluence_integration(
        self,
        *,
        integration: object,
        source: KnowledgeSourceRef,
    ) -> ConfluenceWikiKnowledgeIntegration:
        if not isinstance(integration, ConfluenceWikiKnowledgeIntegration):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Confluence knowledge adapter requires Confluence wiki integration",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        return integration

    def _validate_source(self, source: KnowledgeSourceRef) -> str:
        if (
            source.provider_id != self.provider_id
            or source.integration_kind != self.integration_kind
            or source.source_kind != self.source_kind
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Knowledge source identity is not supported by the Confluence pages adapter",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        scope = source.scope
        if scope.remote_scope_type != CONFLUENCE_SPACE_SCOPE_TYPE:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Knowledge source scope type is not supported",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if scope.parameters:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Knowledge source scope parameters are not supported",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        try:
            return validate_confluence_space_id(scope.remote_scope_id)
        except ValueError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Knowledge source scope identifier is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            ) from None

    def _validate_item(
        self,
        item: KnowledgeItemDescriptor,
        *,
        source: KnowledgeSourceRef,
    ) -> None:
        provenance = item.provenance
        if (
            provenance.provider_id != source.provider_id
            or provenance.source_kind != source.source_kind
        ):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Item provenance does not match the requested source",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if item.content_mode is not KnowledgeContentMode.RICH_TEXT:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Confluence page content mode must be rich text",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if item.item_type != "confluence_page":
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Confluence knowledge item type is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        if not item.content_available:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Confluence knowledge item content is not available",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        remote_id = item.identity.remote_id
        if not _CONFLUENCE_REMOTE_ID_RE.fullmatch(str(remote_id).strip()):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Confluence page remote id is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )
        self._parse_version_number(item.revision.version)
        parent_remote_id = item.identity.parent_remote_id
        if parent_remote_id is not None and not str(parent_remote_id).strip():
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Confluence page parent remote id is invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

    def _parse_version_number(self, version: str) -> int:
        cleaned = str(version).strip()
        if not _CONFLUENCE_REMOTE_ID_RE.fullmatch(cleaned):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_SCOPE,
                safe_message="Confluence page revision version is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return int(cleaned)

    def _validate_page_for_source(
        self,
        page: object,
        *,
        space_id: str,
    ) -> str | None:
        if not isinstance(page, ConfluenceKnowledgePagePage):
            raise ValueError("page must be a ConfluenceKnowledgePagePage")
        if not isinstance(page.is_last, bool):
            raise ValueError("is_last must be a boolean")
        validated_next_cursor: str | None
        if page.is_last:
            if page.next_cursor is not None:
                raise ValueError("next_cursor must be None when is_last is True")
            validated_next_cursor = None
        else:
            if not isinstance(page.next_cursor, str):
                raise ValueError("next_cursor must be a string when is_last is False")
            validated_next_cursor = page.next_cursor.strip()
            if not validated_next_cursor:
                raise ValueError("next_cursor must not be empty when is_last is False")
        if not isinstance(page.pages, tuple):
            raise ValueError("pages must be a tuple")
        seen_remote_ids: set[str] = set()
        for item in page.pages:
            if not isinstance(item, ConfluenceKnowledgePage):
                raise ValueError("page must be a ConfluenceKnowledgePage")
            if item.remote_id in seen_remote_ids:
                raise ValueError("duplicate page id on page")
            seen_remote_ids.add(item.remote_id)
            self._validate_inventory_page_for_source(item, space_id=space_id)
        return validated_next_cursor

    def _validate_inventory_page_for_source(
        self,
        page: ConfluenceKnowledgePage,
        *,
        space_id: str,
    ) -> None:
        if not isinstance(page.remote_id, str):
            raise ValueError("remote_id must be a string")
        remote_id = page.remote_id.strip()
        if not _CONFLUENCE_REMOTE_ID_RE.fullmatch(remote_id):
            raise ValueError("remote_id must be a positive numeric Confluence page ID")
        validated_space_id = validate_confluence_space_id(space_id)
        if page.space_id != validated_space_id:
            raise ValueError("page space id does not match source")
        if page.status != "current":
            raise ValueError("page status must be current")
        if not isinstance(page.created_at, datetime):
            raise ValueError("created_at must be a datetime")
        if page.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        if page.created_at.utcoffset() is None:
            raise ValueError("created_at must have a defined UTC offset")
        if not isinstance(page.version_created_at, datetime):
            raise ValueError("version_created_at must be a datetime")
        if page.version_created_at.tzinfo is None:
            raise ValueError("version_created_at must be timezone-aware")
        if page.version_created_at.utcoffset() is None:
            raise ValueError("version_created_at must have a defined UTC offset")

    def _validate_fetched_page_identity(
        self,
        page: ConfluenceKnowledgePage,
        *,
        item: KnowledgeItemDescriptor,
        space_id: str,
    ) -> None:
        if page.remote_id != str(item.identity.remote_id).strip():
            raise ValueError("Confluence page response identity does not match requested item")
        if page.space_id != space_id:
            raise ValueError("Confluence page response identity does not match requested item")
        if str(page.version_number) != str(item.revision.version).strip():
            raise ValueError("Confluence page response identity does not match requested item")
        parent_remote_id = item.identity.parent_remote_id
        if parent_remote_id is not None and page.parent_id != str(parent_remote_id).strip():
            raise ValueError("Confluence page response identity does not match requested item")
        if parent_remote_id is None and page.parent_id is not None:
            raise ValueError("Confluence page response identity does not match requested item")

    def _page_to_change(self, page: ConfluenceKnowledgePage) -> KnowledgeChange:
        descriptor = KnowledgeItemDescriptor(
            identity=KnowledgeItemIdentity(
                remote_id=page.remote_id,
                logical_key=None,
                parent_remote_id=page.parent_id,
            ),
            revision=KnowledgeItemRevision(
                version=str(page.version_number),
                updated_at=page.version_created_at,
            ),
            title=page.title,
            item_type="confluence_page",
            content_mode=KnowledgeContentMode.RICH_TEXT,
            content_available=True,
            provenance=KnowledgeItemProvenance(
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                remote_id=page.remote_id,
                web_url=page.web_url,
                safe_locator=page.remote_id,
            ),
            metadata={
                "space_id": page.space_id,
                "status": page.status,
                "version_number": page.version_number,
            },
        )
        return KnowledgeChange(
            kind=KnowledgeChangeKind.UPSERT,
            remote_id=page.remote_id,
            descriptor=descriptor,
        )

    def _encode_cursor(self, cursor: _ConfluencePagesReconciliationCursor) -> KnowledgeCursor:
        payload = cursor.model_dump()
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
        return KnowledgeCursor(value=encoded, version=CONFLUENCE_PAGES_CURSOR_VERSION)

    def _decode_cursor(
        self,
        cursor: KnowledgeCursor | None,
        *,
        space_id: str,
    ) -> _ConfluencePagesReconciliationCursor | None:
        if cursor is None:
            return None
        if cursor.version != CONFLUENCE_PAGES_CURSOR_VERSION:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Confluence reconciliation cursor version is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        try:
            padding = "=" * (-len(cursor.value) % 4)
            raw = base64.urlsafe_b64decode(cursor.value + padding)
            data = json.loads(raw.decode("utf-8"))
            decoded = _ConfluencePagesReconciliationCursor.model_validate(data)
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Confluence reconciliation cursor is invalid",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            ) from None
        if decoded.space_id != space_id:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_CURSOR,
                safe_message="Confluence reconciliation cursor scope does not match source",
                provider_id=self.provider_id,
                source_kind=self.source_kind,
                retryable=False,
            )
        return decoded


def register_confluence_pages_knowledge_adapter(
    registry: KnowledgeAdapterRegistry,
) -> ConfluencePagesKnowledgeAdapter:
    adapter = ConfluencePagesKnowledgeAdapter()
    registry.register(adapter)
    return adapter
