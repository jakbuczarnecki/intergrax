# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""End-to-end Confluence knowledge adapter proof through facade and coordinator."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.wiki_knowledge.confluence.adapter import _ConfluenceWikiKnowledge
from intergrax.integrations.providers.wiki_knowledge.confluence.client import ConfluenceRestClient
from intergrax.integrations.providers.wiki_knowledge.confluence.config import ConfluenceIntegrationConfig
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    ConfluenceWikiKnowledgeIntegration,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.knowledge_read import (
    CONFLUENCE_PAGES_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.adapters.confluence_pages import (
    register_confluence_pages_knowledge_adapter,
)
from intergrax.runtime.vendor_knowledge.bindings import KnowledgeSourceBinding, KnowledgeSourceBindingStatus
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
from intergrax.runtime.vendor_knowledge.models import KnowledgeContentMode, KnowledgeSourceScope
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry
from intergrax.runtime.vendor_knowledge.sync_coordinator import VendorKnowledgeSyncCoordinator
from intergrax.runtime.vendor_knowledge.sync_document_store import (
    DocumentStoreKnowledgeRemoteItemStateRepository,
    DocumentStoreKnowledgeSourceLeaseRepository,
    DocumentStoreKnowledgeSyncCheckpointRepository,
)
from intergrax.runtime.vendor_knowledge.sync_models import KnowledgeSyncRunStatus
from tests.unit.runtime.vendor_knowledge._sync_fakes import (
    IdempotentRecordingSink,
    RecordingBindingService,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_API_TOKEN = "top-secret-token"
_EMAIL = "bot@example.com"
_SPACE_ID = "10000"
_PAGE_TOKEN = "page-2"


def _page_payload(
    *,
    page_id: str,
    version_number: int,
    space_id: str = _SPACE_ID,
    storage_value: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": page_id,
        "status": "current",
        "title": f"Page {page_id}",
        "spaceId": space_id,
        "createdAt": "2024-01-01T10:00:00.000Z",
        "version": {
            "number": version_number,
            "createdAt": "2024-01-02T11:00:00.000Z",
        },
    }
    if storage_value is not None:
        payload["body"] = {"storage": {"value": storage_value}}
    return payload


@dataclass
class _FakeHttpResponse:
    status_code: int = 200
    _payload: dict[str, Any] = field(default_factory=dict)
    text: str = ""

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _ConfluenceHttpFake:
    def __init__(self) -> None:
        self.get_calls: list[dict[str, Any]] = []

    def get(self, url: str, *, params: dict[str, Any] | None = None) -> _FakeHttpResponse:
        self.get_calls.append({"url": url, "params": params})
        parsed = urlparse(url)
        path = parsed.path
        request_params = params or {}
        if path.endswith(f"/spaces/{_SPACE_ID}/pages"):
            cursor = request_params.get("cursor")
            if cursor is None:
                return _FakeHttpResponse(
                    _payload={
                        "results": [_page_payload(page_id="20001", version_number=3)],
                        "_links": {
                            "next": f"/wiki/api/v2/spaces/{_SPACE_ID}/pages?cursor={_PAGE_TOKEN}"
                        },
                    }
                )
            if cursor == _PAGE_TOKEN:
                return _FakeHttpResponse(
                    _payload={
                        "results": [_page_payload(page_id="20002", version_number=7)],
                    }
                )
            raise AssertionError(f"unexpected cursor: {cursor}")
        if path.endswith("/pages/20001"):
            assert params == {"body-format": "storage", "version": 3}
            return _FakeHttpResponse(
                _payload=_page_payload(
                    page_id="20001",
                    version_number=3,
                    storage_value="<p>Page one</p>",
                )
            )
        if path.endswith("/pages/20002"):
            assert params == {"body-format": "storage", "version": 7}
            return _FakeHttpResponse(
                _payload=_page_payload(
                    page_id="20002",
                    version_number=7,
                    storage_value="<p>Page two</p>",
                )
            )
        return _FakeHttpResponse(status_code=404)


@dataclass
class _ConfluenceResolver:
    integration: ConfluenceWikiKnowledgeIntegration

    def resolve(self, *, source) -> ConfluenceWikiKnowledgeIntegration:
        return self.integration


def _public_blob(value: object) -> str:
    return json.dumps(value, default=str)


def _assert_no_secrets(blob: str) -> None:
    forbidden = (
        _API_TOKEN,
        _EMAIL,
        _PAGE_TOKEN,
        "Authorization",
        "raw-body",
    )
    for item in forbidden:
        assert item not in blob
    assert "credential_ref" not in blob


def _assert_envelope_safe(envelope) -> None:
    blob = _public_blob(
        {
            "descriptor": envelope.descriptor.model_dump(mode="json") if envelope.descriptor else None,
            "content": envelope.content.model_dump(mode="json") if envelope.content else None,
        }
    )
    _assert_no_secrets(blob)
    assert "connection_ref" not in blob


@pytest.mark.asyncio
async def test_confluence_facade_coordinator_reconciliation_proof() -> None:
    config = ConfluenceIntegrationConfig(
        base_url="https://example.atlassian.net/wiki",
        email=_EMAIL,
        api_token=_API_TOKEN,
    )
    http = _ConfluenceHttpFake()
    integration = ConfluenceWikiKnowledgeIntegration.from_client(
        _ConfluenceWikiKnowledge(ConfluenceRestClient(config, http_client=http))
    )
    registry = KnowledgeAdapterRegistry()
    register_confluence_pages_knowledge_adapter(registry)
    facade = VendorKnowledgeFacadeService(
        tenant_id="tenant-1",
        resolver=_ConfluenceResolver(integration=integration),
        adapter_registry=registry,
    )
    document_store = InMemoryDocumentStore()
    lease_repo = DocumentStoreKnowledgeSourceLeaseRepository(document_store)
    checkpoint_repo = DocumentStoreKnowledgeSyncCheckpointRepository(document_store)
    state_repo = DocumentStoreKnowledgeRemoteItemStateRepository(document_store)
    sink = IdempotentRecordingSink()
    binding = KnowledgeSourceBinding(
        binding_id="confluence-binding",
        tenant_id="tenant-1",
        provider_id="confluence",
        integration_kind=IntegrationCategory.WIKI_KNOWLEDGE,
        source_kind=CONFLUENCE_PAGES_SOURCE_KIND,
        connection_ref="conn-1",
        safe_display_name="Confluence Binding",
        scope=KnowledgeSourceScope(
            remote_scope_id=_SPACE_ID,
            remote_scope_type="confluence_space",
            safe_display_name="Engineering Space",
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=1,
    )
    coordinator = VendorKnowledgeSyncCoordinator(
        tenant_id="tenant-1",
        owner_id="owner-1",
        binding_service=RecordingBindingService(binding=binding),  # type: ignore[arg-type]
        facade=facade,
        lease_repository=lease_repo,
        checkpoint_repository=checkpoint_repo,
        item_state_repository=state_repo,
        sink=sink,
        lease_ttl_seconds=30,
    )

    first = await coordinator.reconcile_once(binding_id="confluence-binding", restart=True)
    second = await coordinator.reconcile_once(binding_id="confluence-binding", restart=False)

    assert first.status is KnowledgeSyncRunStatus.COMPLETED
    assert second.status is KnowledgeSyncRunStatus.COMPLETED
    assert first.has_more is True
    assert second.has_more is False
    assert len(sink.calls) == 2
    assert sink.calls[0].delivery_id != sink.calls[1].delivery_id

    for batch in sink.calls:
        assert batch.mode.value == "reconciliation"
        assert len(batch.envelopes) == 1
        envelope = batch.envelopes[0]
        assert envelope.change_kind.value == "upsert"
        assert envelope.content is not None
        assert envelope.content.mode is KnowledgeContentMode.RICH_TEXT
        assert envelope.content.rich_text is not None
        assert envelope.content.mime_type == "application/vnd.atlassian.confluence.storage+xml"
        assert envelope.content.rich_text.startswith("<h1>")
        assert "<p>Page " in envelope.content.rich_text
        descriptor = envelope.descriptor
        assert descriptor is not None
        assert descriptor.metadata is not None
        assert descriptor.metadata["space_id"] == _SPACE_ID
        assert descriptor.revision is not None
        page_id = descriptor.identity.remote_id
        if page_id == "20001":
            assert descriptor.revision.version == "3"
            assert "<p>Page one</p>" in envelope.content.rich_text
        elif page_id == "20002":
            assert descriptor.revision.version == "7"
            assert "<p>Page two</p>" in envelope.content.rich_text
        assert re.fullmatch(r"^[1-9][0-9]*$", descriptor.identity.remote_id)
        _assert_envelope_safe(envelope)

    remote_ids = set()
    for remote_id in ("20001", "20002"):
        state = state_repo.get(
            tenant_id="tenant-1",
            binding_id="confluence-binding",
            remote_id=remote_id,
        )
        assert state is not None
        remote_ids.add(state.remote_id)
    assert remote_ids == {"20001", "20002"}

    checkpoint = checkpoint_repo.get(tenant_id="tenant-1", binding_id="confluence-binding")
    assert checkpoint is not None
    assert checkpoint.cursor is not None
    assert checkpoint.cursor.version == "confluence.pages.cursor.v1"
    assert _PAGE_TOKEN not in _public_blob(checkpoint.model_dump(mode="json"))

    busy_lease = lease_repo.acquire(
        tenant_id="tenant-1",
        binding_id="confluence-binding",
        owner_id="owner-2",
        ttl_seconds=30,
    )
    assert busy_lease is not None

    list_calls = [call for call in http.get_calls if call["url"].endswith(f"/spaces/{_SPACE_ID}/pages")]
    assert len(list_calls) == 2
    assert "cursor" not in (list_calls[0]["params"] or {})
    assert list_calls[1]["params"]["cursor"] == _PAGE_TOKEN

    content_calls = [call for call in http.get_calls if "/pages/" in call["url"]]
    assert {call["url"].split("/pages/")[-1] for call in content_calls} == {"20001", "20002"}

    public_proof = _public_blob(
        {
            "first": first.model_dump(mode="json"),
            "second": second.model_dump(mode="json"),
            "envelopes": [
                {
                    "descriptor": batch.envelopes[0].descriptor.model_dump(mode="json"),
                    "content": batch.envelopes[0].content.model_dump(mode="json"),
                }
                for batch in sink.calls
            ],
        }
    )
    _assert_no_secrets(public_proof)


class _ConfluenceHttpFakeWrongSpace:
    def get(self, url: str, *, params: dict[str, Any] | None = None) -> _FakeHttpResponse:
        if url.endswith(f"/spaces/{_SPACE_ID}/pages"):
            return _FakeHttpResponse(
                _payload={
                    "results": [
                        _page_payload(
                            page_id="20099",
                            version_number=1,
                            space_id="99999",
                        )
                    ],
                }
            )
        raise AssertionError("unexpected url")


@pytest.mark.asyncio
async def test_confluence_coordinator_rejects_cross_space_first_page() -> None:
    config = ConfluenceIntegrationConfig(
        base_url="https://example.atlassian.net/wiki",
        email=_EMAIL,
        api_token=_API_TOKEN,
    )
    integration = ConfluenceWikiKnowledgeIntegration.from_client(
        _ConfluenceWikiKnowledge(
            ConfluenceRestClient(config, http_client=_ConfluenceHttpFakeWrongSpace())
        )
    )
    registry = KnowledgeAdapterRegistry()
    register_confluence_pages_knowledge_adapter(registry)
    facade = VendorKnowledgeFacadeService(
        tenant_id="tenant-1",
        resolver=_ConfluenceResolver(integration=integration),
        adapter_registry=registry,
    )
    document_store = InMemoryDocumentStore()
    sink = IdempotentRecordingSink()
    binding = KnowledgeSourceBinding(
        binding_id="confluence-binding",
        tenant_id="tenant-1",
        provider_id="confluence",
        integration_kind=IntegrationCategory.WIKI_KNOWLEDGE,
        source_kind=CONFLUENCE_PAGES_SOURCE_KIND,
        connection_ref="conn-1",
        safe_display_name="Confluence Binding",
        scope=KnowledgeSourceScope(
            remote_scope_id=_SPACE_ID,
            remote_scope_type="confluence_space",
            safe_display_name="Engineering Space",
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=1,
    )
    coordinator = VendorKnowledgeSyncCoordinator(
        tenant_id="tenant-1",
        owner_id="owner-1",
        binding_service=RecordingBindingService(binding=binding),  # type: ignore[arg-type]
        facade=facade,
        lease_repository=DocumentStoreKnowledgeSourceLeaseRepository(document_store),
        checkpoint_repository=DocumentStoreKnowledgeSyncCheckpointRepository(document_store),
        item_state_repository=DocumentStoreKnowledgeRemoteItemStateRepository(document_store),
        sink=sink,
        lease_ttl_seconds=30,
    )

    with pytest.raises(VendorKnowledgeError) as exc_info:
        await coordinator.reconcile_once(binding_id="confluence-binding", restart=True)

    assert exc_info.value.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert exc_info.value.retryable is False
    assert sink.calls == []
    checkpoint = DocumentStoreKnowledgeSyncCheckpointRepository(document_store).get(
        tenant_id="tenant-1",
        binding_id="confluence-binding",
    )
    assert checkpoint is None
    state = DocumentStoreKnowledgeRemoteItemStateRepository(document_store).get(
        tenant_id="tenant-1",
        binding_id="confluence-binding",
        remote_id="20099",
    )
    assert state is None
