# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for ConfluencePagesKnowledgeAdapter."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.wiki_knowledge import WikiPageRecord, WikiSearchResult
from intergrax.integrations.providers.wiki_knowledge.confluence.adapter import _ConfluenceWikiKnowledge
from intergrax.integrations.providers.wiki_knowledge.confluence.client import ConfluenceRestClient
from intergrax.integrations.providers.wiki_knowledge.confluence.config import ConfluenceIntegrationConfig
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    ConfluenceWikiKnowledgeIntegration,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.knowledge_read import (
    CONFLUENCE_PAGES_CURSOR_VERSION,
    CONFLUENCE_PAGES_SOURCE_KIND,
    CONFLUENCE_SPACE_SCOPE_TYPE,
    ConfluenceKnowledgePage,
    ConfluenceKnowledgePagePage,
)
from intergrax.runtime.vendor_knowledge.adapters.confluence_pages import (
    ConfluencePagesKnowledgeAdapter,
    register_confluence_pages_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeContentMode,
    KnowledgeCursor,
    KnowledgeItemDescriptor,
    KnowledgeItemIdentity,
    KnowledgeItemProvenance,
    KnowledgeItemRevision,
    KnowledgeSourceRef,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def _config() -> ConfluenceIntegrationConfig:
    return ConfluenceIntegrationConfig(
        base_url="https://example.atlassian.net/wiki",
        email="bot@example.com",
        api_token="secret",
    )


def _source(
    *,
    remote_scope_id: str = "10000",
    remote_scope_type: str = CONFLUENCE_SPACE_SCOPE_TYPE,
    parameters: dict[str, Any] | None = None,
) -> KnowledgeSourceRef:
    return KnowledgeSourceRef(
        tenant_id="tenant-1",
        provider_id="confluence",
        integration_kind=IntegrationCategory.WIKI_KNOWLEDGE,
        source_kind=CONFLUENCE_PAGES_SOURCE_KIND,
        scope=KnowledgeSourceScope(
            remote_scope_id=remote_scope_id,
            remote_scope_type=remote_scope_type,
            safe_display_name="Engineering Space",
            parameters=parameters or {},
        ),
    )


def _page(
    *,
    remote_id: str = "20001",
    space_id: str = "10000",
    version_number: int = 3,
    parent_id: str | None = None,
    title: str = "Runbook",
    storage_value: str | None = None,
) -> ConfluenceKnowledgePage:
    return ConfluenceKnowledgePage(
        remote_id=remote_id,
        space_id=space_id,
        parent_id=parent_id,
        status="current",
        title=title,
        created_at=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        version_number=version_number,
        version_created_at=datetime(2024, 1, 2, 11, 0, tzinfo=timezone.utc),
        storage_value=storage_value,
        web_url=f"https://example.atlassian.net/wiki/pages/viewpage.action?pageId={remote_id}",
    )


def _integration() -> ConfluenceWikiKnowledgeIntegration:
    rest = ConfluenceRestClient(_config(), http_client=MagicMock())
    return ConfluenceWikiKnowledgeIntegration.from_client(_ConfluenceWikiKnowledge(rest))


class _FakeKnowledgeWiki:
    def __init__(
        self,
        *,
        pages: list[ConfluenceKnowledgePagePage] | None = None,
        page: ConfluenceKnowledgePage | None = None,
    ) -> None:
        self._pages = list(pages or [])
        self._page = page or _page(storage_value="<p>Body</p>")
        self.list_calls: list[dict[str, Any]] = []

    def get_page(self, page_id: str) -> WikiPageRecord:
        return WikiPageRecord(id=page_id, title="ok", space_key="OPS", body="", url="")

    def search_pages(self, query: str, *, limit: int = 25) -> WikiSearchResult:
        return WikiSearchResult(pages=[], total=0)

    def list_knowledge_pages(
        self,
        *,
        space_id: str,
        cursor: str | None,
        limit: int,
    ) -> ConfluenceKnowledgePagePage:
        self.list_calls.append({"space_id": space_id, "cursor": cursor, "limit": limit})
        if not self._pages:
            return ConfluenceKnowledgePagePage(pages=(_page(),), is_last=True)
        return self._pages.pop(0)

    def get_knowledge_page(
        self,
        *,
        page_id: str,
        version_number: int,
    ) -> ConfluenceKnowledgePage:
        return self._page


def _integration_with_wiki(wiki: _FakeKnowledgeWiki) -> ConfluenceWikiKnowledgeIntegration:
    return ConfluenceWikiKnowledgeIntegration.from_client(wiki)


def _encode_cursor(payload: dict[str, Any]) -> KnowledgeCursor:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return KnowledgeCursor(value=encoded, version=CONFLUENCE_PAGES_CURSOR_VERSION)


async def test_adapter_identity() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    assert adapter.provider_id == "confluence"
    assert adapter.integration_kind is IntegrationCategory.WIKI_KNOWLEDGE
    assert adapter.source_kind == CONFLUENCE_PAGES_SOURCE_KIND


async def test_capabilities_exact_set() -> None:
    caps = ConfluencePagesKnowledgeAdapter().capabilities
    assert caps.full_inventory is True
    assert caps.incremental_changes is False
    assert caps.content_fetch is True
    assert caps.binary_content is False
    assert caps.rich_text_content is True
    assert caps.structured_content is False
    assert caps.permissions is False
    assert caps.tombstones is False
    assert caps.remote_versions is True
    assert caps.reconciliation is True


async def test_valid_space_scope_inspect() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    info = await adapter.inspect_scope(integration=_integration(), source=_source())
    assert info.safe_display_name == "Engineering Space"


async def test_invalid_scope_type_rejected() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(
            integration=_integration(),
            source=_source(remote_scope_type="space_key"),
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_non_empty_parameters_rejected() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(),
            source=_source(parameters={"cql": "bad"}),
            cursor=None,
            limit=10,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE


async def test_space_key_instead_of_id_rejected() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(),
            source=_source(remote_scope_id="ENG"),
            cursor=None,
            limit=10,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_SCOPE
    assert "ENG" not in str(exc_info.value)


async def test_wrong_integration_type_rejected() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.inspect_scope(integration=object(), source=_source())
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_first_page_read() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    page = await adapter.read_page(
        integration=_integration_with_wiki(_FakeKnowledgeWiki()),
        source=_source(),
        cursor=None,
        limit=10,
    )
    assert len(page.changes) == 1
    assert page.has_more is False


async def test_multipage_continuation() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    wiki = _FakeKnowledgeWiki(
        pages=[
            ConfluenceKnowledgePagePage(
                pages=(_page(remote_id="20001"),),
                next_cursor="page-2",
                is_last=False,
            ),
            ConfluenceKnowledgePagePage(
                pages=(_page(remote_id="20002", version_number=7),),
                is_last=True,
            ),
        ]
    )
    integration = _integration_with_wiki(wiki)
    first = await adapter.read_page(integration=integration, source=_source(), cursor=None, limit=10)
    assert first.has_more is True
    second = await adapter.read_page(
        integration=integration,
        source=_source(),
        cursor=first.next_cursor,
        limit=10,
    )
    assert second.has_more is False
    assert wiki.list_calls[1]["cursor"] == "page-2"


async def test_intermediate_page_checkpoint_equals_next_cursor() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    page = await adapter.read_page(
        integration=_integration_with_wiki(
            _FakeKnowledgeWiki(
                pages=[
                    ConfluenceKnowledgePagePage(
                        pages=(_page(),),
                        next_cursor="page-2",
                        is_last=False,
                    )
                ]
            )
        ),
        source=_source(),
        cursor=None,
        limit=10,
    )
    assert page.next_cursor == page.proposed_checkpoint


async def test_final_complete_checkpoint() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    page = await adapter.read_page(
        integration=_integration_with_wiki(_FakeKnowledgeWiki()),
        source=_source(),
        cursor=None,
        limit=10,
    )
    assert page.next_cursor is None
    assert page.proposed_checkpoint is not None
    raw = base64.urlsafe_b64decode(page.proposed_checkpoint.value + "==")
    payload = json.loads(raw.decode("utf-8"))
    assert payload["complete"] is True
    assert payload["provider_cursor"] is None


async def test_complete_cursor_read_rejected() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    cursor = _encode_cursor(
        {
            "schema_version": CONFLUENCE_PAGES_CURSOR_VERSION,
            "space_id": "10000",
            "provider_cursor": None,
            "complete": True,
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(),
            source=_source(),
            cursor=cursor,
            limit=10,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR
    assert cursor.value not in str(exc_info.value)


async def test_cursor_other_space_rejected() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    cursor = _encode_cursor(
        {
            "schema_version": CONFLUENCE_PAGES_CURSOR_VERSION,
            "space_id": "99999",
            "provider_cursor": "page-2",
            "complete": False,
        }
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(),
            source=_source(),
            cursor=cursor,
            limit=10,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_CURSOR


async def test_malformed_cursor_does_not_leak_value() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    secret_cursor = KnowledgeCursor(value="not-valid-base64!!!", version=CONFLUENCE_PAGES_CURSOR_VERSION)
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration(),
            source=_source(),
            cursor=secret_cursor,
            limit=10,
        )
    assert secret_cursor.value not in str(exc_info.value)


async def test_cross_space_page_rejected() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    wiki = _FakeKnowledgeWiki(
        pages=[
            ConfluenceKnowledgePagePage(
                pages=(_page(space_id="99999"),),
                is_last=True,
            )
        ]
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration_with_wiki(wiki),
            source=_source(),
            cursor=None,
            limit=10,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_non_numeric_page_id_rejected_via_model_construct() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    wiki = _FakeKnowledgeWiki(
        pages=[
            ConfluenceKnowledgePagePage(
                pages=(
                    ConfluenceKnowledgePage.model_construct(
                        remote_id="abc",
                        space_id="10000",
                        status="current",
                        title="Title",
                        created_at=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                        version_number=1,
                        version_created_at=datetime(2024, 1, 2, 11, 0, tzinfo=timezone.utc),
                        web_url="https://example/pages/abc",
                    ),
                ),
                is_last=True,
            )
        ]
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration_with_wiki(wiki),
            source=_source(),
            cursor=None,
            limit=10,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_naive_timestamp_rejected_via_model_construct() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    wiki = _FakeKnowledgeWiki(
        pages=[
            ConfluenceKnowledgePagePage(
                pages=(
                    ConfluenceKnowledgePage.model_construct(
                        remote_id="20001",
                        space_id="10000",
                        status="current",
                        title="Title",
                        created_at=datetime(2024, 1, 1, 10, 0),
                        version_number=1,
                        version_created_at=datetime(2024, 1, 2, 11, 0, tzinfo=timezone.utc),
                        web_url="https://example/pages/20001",
                    ),
                ),
                is_last=True,
            )
        ]
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.read_page(
            integration=_integration_with_wiki(wiki),
            source=_source(),
            cursor=None,
            limit=10,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_descriptor_mapping() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    page = await adapter.read_page(
        integration=_integration_with_wiki(_FakeKnowledgeWiki()),
        source=_source(),
        cursor=None,
        limit=10,
    )
    descriptor = page.changes[0].descriptor
    assert descriptor is not None
    assert descriptor.identity.remote_id == "20001"
    assert descriptor.identity.logical_key is None
    assert descriptor.item_type == "confluence_page"
    assert descriptor.content_mode is KnowledgeContentMode.RICH_TEXT
    assert descriptor.revision.version == "3"
    assert descriptor.metadata is not None
    assert descriptor.metadata["space_id"] == "10000"
    assert descriptor.metadata["version_number"] == 3


async def test_fetch_content_exact_version() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    fetched = _page(storage_value="<p>Exact</p>")  # type: ignore[call-arg]
    integration = _integration_with_wiki(_FakeKnowledgeWiki(page=fetched))
    item = KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id="20001", parent_remote_id=None),
        revision=KnowledgeItemRevision(version="3"),
        title="Runbook",
        item_type="confluence_page",
        content_mode=KnowledgeContentMode.RICH_TEXT,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id="confluence",
            source_kind=CONFLUENCE_PAGES_SOURCE_KIND,
            remote_id="20001",
        ),
    )
    content = await adapter.fetch_content(integration=integration, source=_source(), item=item)
    assert content.mode is KnowledgeContentMode.RICH_TEXT
    assert "<p>Exact</p>" in content.rich_text


async def test_fetch_content_uses_asyncio_to_thread() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    integration = _integration_with_wiki(_FakeKnowledgeWiki(page=_page(storage_value="")))  # type: ignore[call-arg]
    item = KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id="20001"),
        revision=KnowledgeItemRevision(version="3"),
        title="Runbook",
        item_type="confluence_page",
        content_mode=KnowledgeContentMode.RICH_TEXT,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id="confluence",
            source_kind=CONFLUENCE_PAGES_SOURCE_KIND,
            remote_id="20001",
        ),
    )
    with patch(
        "intergrax.runtime.vendor_knowledge.adapters.confluence_pages.asyncio.to_thread",
        side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs),
    ) as to_thread:
        await adapter.fetch_content(integration=integration, source=_source(), item=item)
    to_thread.assert_called_once()


async def test_fetched_identity_mismatch_rejected() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    integration = _integration_with_wiki(
        _FakeKnowledgeWiki(page=_page(remote_id="29999", storage_value=""))  # type: ignore[call-arg]
    )
    item = KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id="20001"),
        revision=KnowledgeItemRevision(version="3"),
        title="Runbook",
        item_type="confluence_page",
        content_mode=KnowledgeContentMode.RICH_TEXT,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id="confluence",
            source_kind=CONFLUENCE_PAGES_SOURCE_KIND,
            remote_id="20001",
        ),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(integration=integration, source=_source(), item=item)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_fetched_version_mismatch_rejected() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    integration = _integration_with_wiki(
        _FakeKnowledgeWiki(page=_page(version_number=9, storage_value=""))  # type: ignore[call-arg]
    )
    item = KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id="20001"),
        revision=KnowledgeItemRevision(version="3"),
        title="Runbook",
        item_type="confluence_page",
        content_mode=KnowledgeContentMode.RICH_TEXT,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id="confluence",
            source_kind=CONFLUENCE_PAGES_SOURCE_KIND,
            remote_id="20001",
        ),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(integration=integration, source=_source(), item=item)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_fetched_space_mismatch_rejected() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    integration = _integration_with_wiki(
        _FakeKnowledgeWiki(page=_page(space_id="99999", storage_value=""))  # type: ignore[call-arg]
    )
    item = KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id="20001"),
        revision=KnowledgeItemRevision(version="3"),
        title="Runbook",
        item_type="confluence_page",
        content_mode=KnowledgeContentMode.RICH_TEXT,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id="confluence",
            source_kind=CONFLUENCE_PAGES_SOURCE_KIND,
            remote_id="20001",
        ),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_content(integration=integration, source=_source(), item=item)
    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE


async def test_rich_text_shape_and_title_escaping() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    fetched = _page(title="A & B <script>", storage_value="<p>Body</p>")  # type: ignore[call-arg]
    integration = _integration_with_wiki(_FakeKnowledgeWiki(page=fetched))
    item = KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id="20001"),
        revision=KnowledgeItemRevision(version="3"),
        title="A & B <script>",
        item_type="confluence_page",
        content_mode=KnowledgeContentMode.RICH_TEXT,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id="confluence",
            source_kind=CONFLUENCE_PAGES_SOURCE_KIND,
            remote_id="20001",
        ),
    )
    content = await adapter.fetch_content(integration=integration, source=_source(), item=item)
    assert content.mime_type == "application/vnd.atlassian.confluence.storage+xml"
    assert content.encoding == "utf-8"
    assert content.rich_text.startswith("<h1>A &amp; B &lt;script&gt;</h1>")
    assert re.fullmatch(r"[0-9a-f]{64}", content.content_hash or "")


async def test_deterministic_hash() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    fetched = _page(title="Title", storage_value="<p>x</p>")  # type: ignore[call-arg]
    integration = _integration_with_wiki(_FakeKnowledgeWiki(page=fetched))
    item = KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id="20001"),
        revision=KnowledgeItemRevision(version="3"),
        title="Title",
        item_type="confluence_page",
        content_mode=KnowledgeContentMode.RICH_TEXT,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id="confluence",
            source_kind=CONFLUENCE_PAGES_SOURCE_KIND,
            remote_id="20001",
        ),
    )
    first = await adapter.fetch_content(integration=integration, source=_source(), item=item)
    second = await adapter.fetch_content(integration=integration, source=_source(), item=item)
    assert first.content_hash == second.content_hash
    expected = hashlib.sha256(first.rich_text.encode("utf-8")).hexdigest()
    assert first.content_hash == expected


async def test_empty_body_still_gives_title_content() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    integration = _integration_with_wiki(_FakeKnowledgeWiki(page=_page(storage_value="")))  # type: ignore[call-arg]
    item = KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id="20001"),
        revision=KnowledgeItemRevision(version="3"),
        title="Runbook",
        item_type="confluence_page",
        content_mode=KnowledgeContentMode.RICH_TEXT,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id="confluence",
            source_kind=CONFLUENCE_PAGES_SOURCE_KIND,
            remote_id="20001",
        ),
    )
    content = await adapter.fetch_content(integration=integration, source=_source(), item=item)
    assert content.rich_text == "<h1>Runbook</h1>"


async def test_permissions_unsupported() -> None:
    adapter = ConfluencePagesKnowledgeAdapter()
    item = KnowledgeItemDescriptor(
        identity=KnowledgeItemIdentity(remote_id="20001"),
        revision=KnowledgeItemRevision(version="3"),
        title="Runbook",
        item_type="confluence_page",
        content_mode=KnowledgeContentMode.RICH_TEXT,
        content_available=True,
        provenance=KnowledgeItemProvenance(
            provider_id="confluence",
            source_kind=CONFLUENCE_PAGES_SOURCE_KIND,
            remote_id="20001",
        ),
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        await adapter.fetch_permissions(
            integration=_integration(),
            source=_source(),
            item=item,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY


async def test_registration_helper() -> None:
    registry = KnowledgeAdapterRegistry()
    adapter = register_confluence_pages_knowledge_adapter(registry)
    assert isinstance(adapter, ConfluencePagesKnowledgeAdapter)
    resolved = registry.resolve(source=_source())
    assert resolved is adapter


async def test_duplicate_registration_raises() -> None:
    registry = KnowledgeAdapterRegistry()
    register_confluence_pages_knowledge_adapter(registry)
    with pytest.raises(ValueError, match="already registered"):
        register_confluence_pages_knowledge_adapter(registry)
