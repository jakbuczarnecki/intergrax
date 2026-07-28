# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Confluence knowledge-read provider surface."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.contracts.wiki_knowledge import WikiPageRecord, WikiSearchResult
from intergrax.integrations.providers.wiki_knowledge.confluence.adapter import _ConfluenceWikiKnowledge
from intergrax.integrations.providers.wiki_knowledge.confluence.bundle import create_confluence_wiki_knowledge
from intergrax.integrations.providers.wiki_knowledge.confluence.client import ConfluenceRestClient
from intergrax.integrations.providers.wiki_knowledge.confluence.config import ConfluenceIntegrationConfig
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    ConfluenceWikiKnowledgeIntegration,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.knowledge_read import (
    ConfluenceKnowledgePage,
    ConfluenceKnowledgePagePage,
    extract_confluence_knowledge_next_cursor,
    parse_confluence_knowledge_page_page,
    validate_confluence_page_id,
    validate_confluence_space_id,
)

pytestmark = pytest.mark.unit

_API_TOKEN = "top-secret-token"
_EMAIL = "bot@example.com"
_SPACE_ID = "10000"
_PAGE_ID = "20001"
_CURSOR = "page-2"


def _config() -> ConfluenceIntegrationConfig:
    return ConfluenceIntegrationConfig(
        base_url="https://example.atlassian.net/wiki",
        email=_EMAIL,
        api_token=_API_TOKEN,
    )


def _page_payload(
    *,
    page_id: str = _PAGE_ID,
    space_id: str = _SPACE_ID,
    version_number: int = 3,
    title: str = "Runbook",
    parent_id: str | None = None,
    storage_value: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": page_id,
        "status": "current",
        "title": title,
        "spaceId": space_id,
        "createdAt": "2024-01-01T10:00:00.000Z",
        "version": {
            "number": version_number,
            "createdAt": "2024-01-02T11:00:00.000Z",
        },
    }
    if parent_id is not None:
        payload["parentId"] = parent_id
    if storage_value is not None:
        payload["body"] = {"storage": {"value": storage_value}}
    return payload


def _list_payload(
    *,
    pages: list[dict[str, Any]] | None = None,
    next_link: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"results": pages or [_page_payload()]}
    if next_link is not None:
        payload["_links"] = {"next": next_link}
    return payload


def _mock_http(*, get_payload: dict | None = None) -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = get_payload or {}
    response.raise_for_status.return_value = None
    client.get.return_value = response
    return client


class _CustomWikiKnowledge:
    def get_page(self, page_id: str) -> WikiPageRecord:
        return WikiPageRecord(id=page_id, title="ok", space_key="OPS", body="", url="")

    def search_pages(self, query: str, *, limit: int = 25) -> WikiSearchResult:
        return WikiSearchResult(pages=[], total=0)


def test_list_knowledge_pages_uses_v2_absolute_url() -> None:
    http = _mock_http(get_payload=_list_payload())
    client = ConfluenceRestClient(_config(), http_client=http)
    client.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=100)
    http.get.assert_called_once()
    url = http.get.call_args.args[0]
    assert url == "https://example.atlassian.net/wiki/api/v2/spaces/10000/pages"
    assert not url.startswith("/")


def test_list_knowledge_pages_uses_spaces_pages_endpoint() -> None:
    http = _mock_http(get_payload=_list_payload())
    client = ConfluenceRestClient(_config(), http_client=http)
    client.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=10)
    assert "/spaces/10000/pages" in http.get.call_args.args[0]


def test_list_knowledge_pages_status_current() -> None:
    http = _mock_http(get_payload=_list_payload())
    client = ConfluenceRestClient(_config(), http_client=http)
    client.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=10)
    params = http.get.call_args.kwargs["params"]
    assert params["status"] == "current"


def test_list_knowledge_pages_cursor_continuation() -> None:
    http = _mock_http(get_payload=_list_payload())
    client = ConfluenceRestClient(_config(), http_client=http)
    client.list_knowledge_pages(space_id=_SPACE_ID, cursor=_CURSOR, limit=10)
    params = http.get.call_args.kwargs["params"]
    assert params["cursor"] == _CURSOR


def test_list_knowledge_pages_first_page_without_cursor_param() -> None:
    http = _mock_http(get_payload=_list_payload())
    client = ConfluenceRestClient(_config(), http_client=http)
    client.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=10)
    params = http.get.call_args.kwargs["params"]
    assert "cursor" not in params


def test_limit_validation_rejects_out_of_range() -> None:
    http = _mock_http()
    client = ConfluenceRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="limit must be in range 1..250"):
        client.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=0)
    with pytest.raises(ValueError, match="limit must be in range 1..250"):
        client.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=251)
    http.get.assert_not_called()


def test_invalid_space_id_rejected_before_http() -> None:
    http = _mock_http()
    client = ConfluenceRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="invalid Confluence space id"):
        client.list_knowledge_pages(space_id="ENG", cursor=None, limit=10)
    http.get.assert_not_called()


def test_invalid_page_id_rejected_before_http() -> None:
    http = _mock_http()
    client = ConfluenceRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="invalid Confluence page id"):
        client.get_knowledge_page(page_id="bad", version_number=1)
    http.get.assert_not_called()


def test_duplicate_page_id_rejected() -> None:
    page = _page_payload()
    http = _mock_http(get_payload=_list_payload(pages=[page, page]))
    client = ConfluenceRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="unexpected Confluence knowledge response"):
        client.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=10)


def test_cross_space_response_rejected() -> None:
    http = _mock_http(
        get_payload=_list_payload(pages=[_page_payload(space_id="99999")])
    )
    client = ConfluenceRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="unexpected Confluence knowledge response"):
        client.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=10)


def test_timestamps_are_timezone_aware() -> None:
    http = _mock_http(get_payload=_list_payload())
    client = ConfluenceRestClient(_config(), http_client=http)
    page = client.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=10)
    assert page.pages[0].created_at.tzinfo is not None
    assert page.pages[0].version_created_at.tzinfo is not None


def test_links_next_cursor_extraction() -> None:
    next_link = f"/wiki/api/v2/spaces/{_SPACE_ID}/pages?cursor={_CURSOR}"
    http = _mock_http(get_payload=_list_payload(next_link=next_link))
    client = ConfluenceRestClient(_config(), http_client=http)
    page = client.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=10)
    assert page.is_last is False
    assert page.next_cursor == _CURSOR


def test_missing_cursor_in_next_link_rejected() -> None:
    next_link = f"/wiki/api/v2/spaces/{_SPACE_ID}/pages"
    http = _mock_http(get_payload=_list_payload(next_link=next_link))
    client = ConfluenceRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="unexpected Confluence knowledge response"):
        client.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=10)


def test_absolute_next_url_rejected() -> None:
    next_link = f"https://example.atlassian.net/wiki/api/v2/spaces/{_SPACE_ID}/pages?cursor={_CURSOR}"
    with pytest.raises(ValueError, match="unexpected Confluence knowledge response"):
        extract_confluence_knowledge_next_cursor(next_link, space_id=_SPACE_ID)


def test_wrong_next_path_rejected() -> None:
    next_link = f"/wiki/api/v2/spaces/99999/pages?cursor={_CURSOR}"
    with pytest.raises(ValueError, match="unexpected Confluence knowledge response"):
        extract_confluence_knowledge_next_cursor(next_link, space_id=_SPACE_ID)


def test_multiple_cursor_values_rejected() -> None:
    next_link = f"/wiki/api/v2/spaces/{_SPACE_ID}/pages?cursor={_CURSOR}&cursor=other"
    with pytest.raises(ValueError, match="unexpected Confluence knowledge response"):
        extract_confluence_knowledge_next_cursor(next_link, space_id=_SPACE_ID)


def test_get_knowledge_page_exact_version_fetch() -> None:
    http = _mock_http(get_payload=_page_payload(storage_value="<p>Body</p>"))
    client = ConfluenceRestClient(_config(), http_client=http)
    page = client.get_knowledge_page(page_id=_PAGE_ID, version_number=3)
    assert page.version_number == 3
    assert page.storage_value == "<p>Body</p>"
    url = http.get.call_args.args[0]
    assert url.endswith(f"/pages/{_PAGE_ID}")
    params = http.get.call_args.kwargs["params"]
    assert params["body-format"] == "storage"
    assert params["version"] == 3


def test_empty_storage_body_allowed() -> None:
    http = _mock_http(get_payload=_page_payload(storage_value=""))
    client = ConfluenceRestClient(_config(), http_client=http)
    page = client.get_knowledge_page(page_id=_PAGE_ID, version_number=3)
    assert page.storage_value == ""


def test_identity_version_mismatch_rejected() -> None:
    http = _mock_http(get_payload=_page_payload(version_number=5, storage_value=""))
    client = ConfluenceRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="unexpected Confluence knowledge response"):
        client.get_knowledge_page(page_id=_PAGE_ID, version_number=3)


@pytest.mark.parametrize("status_code,expected", [(429, IntegrationDependencyError), (503, IntegrationDependencyError)])
def test_rate_limit_and_5xx_map_to_dependency_error(status_code: int, expected: type[Exception]) -> None:
    http = _mock_http()
    http.get.return_value.status_code = status_code
    client = ConfluenceRestClient(_config(), http_client=http)
    with pytest.raises(expected):
        client.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=10)


@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_client_errors_map_to_configuration_error(status_code: int) -> None:
    http = _mock_http()
    http.get.return_value.status_code = status_code
    client = ConfluenceRestClient(_config(), http_client=http)
    with pytest.raises(IntegrationConfigurationError):
        client.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=10)


def test_list_space_404_is_configuration_error() -> None:
    http = _mock_http()
    http.get.return_value.status_code = 404
    client = ConfluenceRestClient(_config(), http_client=http)
    with pytest.raises(IntegrationConfigurationError):
        client.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=10)


def test_get_page_404_is_dependency_failure() -> None:
    http = _mock_http()
    http.get.return_value.status_code = 404
    client = ConfluenceRestClient(_config(), http_client=http)
    with pytest.raises(IntegrationDependencyError):
        client.get_knowledge_page(page_id=_PAGE_ID, version_number=3)


def test_transport_exception_maps_to_dependency_error() -> None:
    http = _mock_http()
    http.get.side_effect = RuntimeError("network down")
    client = ConfluenceRestClient(_config(), http_client=http)
    with pytest.raises(IntegrationDependencyError):
        client.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=10)


def test_json_parser_failure_is_safe() -> None:
    http = _mock_http()
    http.get.return_value.json.side_effect = ValueError("bad json")
    client = ConfluenceRestClient(_config(), http_client=http)
    with pytest.raises(ValueError, match="unexpected Confluence knowledge response"):
        client.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=10)


def test_errors_do_not_expose_secrets() -> None:
    http = _mock_http()
    http.get.return_value.status_code = 400
    http.get.return_value.text = "raw-body-with-secret"
    client = ConfluenceRestClient(_config(), http_client=http)
    with pytest.raises(IntegrationConfigurationError) as exc_info:
        client.list_knowledge_pages(space_id=_SPACE_ID, cursor=_CURSOR, limit=10)
    message = str(exc_info.value)
    assert _CURSOR not in message
    assert _SPACE_ID not in message
    assert "raw-body" not in message
    assert _API_TOKEN not in message
    assert _EMAIL not in message


def test_confluence_wiki_knowledge_delegates_knowledge_read() -> None:
    http = _mock_http(
        get_payload=_list_payload(),
    )
    rest = ConfluenceRestClient(_config(), http_client=http)
    wiki = _ConfluenceWikiKnowledge(rest)
    page = wiki.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=5)
    assert len(page.pages) == 1


def test_confluence_integration_delegates_knowledge_read() -> None:
    http = _mock_http(get_payload=_page_payload(storage_value="<p>x</p>"))
    rest = ConfluenceRestClient(_config(), http_client=http)
    integration = ConfluenceWikiKnowledgeIntegration.from_client(_ConfluenceWikiKnowledge(rest))
    page = integration.get_knowledge_page(page_id=_PAGE_ID, version_number=3)
    assert page.remote_id == _PAGE_ID


def test_custom_wiki_without_knowledge_read_fails_closed() -> None:
    integration = ConfluenceWikiKnowledgeIntegration.from_client(_CustomWikiKnowledge())
    assert integration.get_page("42").title == "ok"
    with pytest.raises(IntegrationConfigurationError, match="knowledge read capability"):
        integration.list_knowledge_pages(space_id=_SPACE_ID, cursor=None, limit=10)


def test_space_and_page_id_validators() -> None:
    assert validate_confluence_space_id("10000") == "10000"
    assert validate_confluence_page_id("20001") == "20001"
    with pytest.raises(ValueError):
        validate_confluence_space_id("ENG")
    with pytest.raises(ValueError):
        validate_confluence_page_id("0")


def test_existing_get_page_and_search_pages_still_work() -> None:
    http = _mock_http(
        get_payload={
            "id": "42",
            "title": "Runbook",
            "space": {"key": "OPS"},
            "version": {"number": 3},
            "body": {"storage": {"value": "<p>Restart service</p>"}},
        }
    )
    wiki = create_confluence_wiki_knowledge(**_config().model_dump(), http_client=http)
    page = wiki.get_page("42")
    assert page.id == "42"
    assert http.get.call_args.args[0] == "/content/42"


def test_strict_frozen_models() -> None:
    page = ConfluenceKnowledgePage(
        remote_id="20001",
        space_id="10000",
        parent_id=None,
        status="current",
        title="Title",
        created_at=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        version_number=1,
        version_created_at=datetime(2024, 1, 2, 11, 0, tzinfo=timezone.utc),
        storage_value=None,
        web_url="https://example.atlassian.net/wiki/pages/viewpage.action?pageId=20001",
    )
    with pytest.raises(Exception):
        page.title = "changed"  # type: ignore[misc]


def test_confluence_knowledge_page_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError):
        ConfluenceKnowledgePage(
            remote_id="20001",
            space_id="10000",
            status="current",
            title="Title",
            created_at=datetime(2024, 1, 1, 10, 0),
            version_number=1,
            version_created_at=datetime(2024, 1, 2, 11, 0, tzinfo=timezone.utc),
            web_url="https://example.atlassian.net/wiki/pages/viewpage.action?pageId=20001",
        )


def test_confluence_knowledge_page_normalizes_non_utc_aware_datetime() -> None:
    offset = timezone(timedelta(hours=-5))
    page = ConfluenceKnowledgePage(
        remote_id="20001",
        space_id="10000",
        status="current",
        title="Title",
        created_at=datetime(2024, 1, 1, 10, 0, tzinfo=offset),
        version_number=1,
        version_created_at=datetime(2024, 1, 2, 11, 0, tzinfo=offset),
        web_url="https://example.atlassian.net/wiki/pages/viewpage.action?pageId=20001",
    )
    assert page.created_at.tzinfo == timezone.utc


def test_parse_confluence_knowledge_page_page_is_last_without_next() -> None:
    payload = _list_payload()
    page = parse_confluence_knowledge_page_page(
        payload,
        requested_space_id=_SPACE_ID,
        page_url_builder=lambda page_id: f"https://example/pages/{page_id}",
    )
    assert page.is_last is True
    assert page.next_cursor is None


def test_confluence_knowledge_page_page_rejects_is_last_false_without_cursor() -> None:
    with pytest.raises(ValueError):
        ConfluenceKnowledgePagePage(pages=(), next_cursor=None, is_last=False)
