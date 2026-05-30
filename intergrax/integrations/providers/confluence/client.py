# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Confluence REST client — HTTP client injected from ``opens.py`` only."""

from __future__ import annotations

import re
from typing import Any, Mapping

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.wiki_knowledge import WikiPageRecord, WikiSearchResult
from intergrax.integrations.providers.confluence.config import ConfluenceIntegrationConfig


def _html_to_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    return " ".join(text.split())


def _space_key(payload: Mapping[str, Any]) -> str:
    space = payload.get("space")
    if isinstance(space, dict):
        key = space.get("key")
        if isinstance(key, str):
            return key
    return ""


def _page_body(payload: Mapping[str, Any]) -> str:
    body = payload.get("body")
    if not isinstance(body, dict):
        return ""
    storage = body.get("storage")
    if isinstance(storage, dict):
        value = storage.get("value")
        if isinstance(value, str):
            return _html_to_text(value)
    return ""


def _page_from_payload(config: ConfluenceIntegrationConfig, payload: Mapping[str, Any]) -> WikiPageRecord:
    page_id = str(payload.get("id") or "")
    version_obj = payload.get("version")
    version = version_obj.get("number") if isinstance(version_obj, dict) else None
    return WikiPageRecord(
        id=page_id,
        title=str(payload.get("title") or ""),
        space_key=_space_key(payload),
        body=_page_body(payload),
        url=config.page_url(page_id) if page_id else "",
        version=int(version) if isinstance(version, int) else None,
    )


def _escape_cql_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


class ConfluenceRestClient:
    """Minimal Confluence REST client — sync HTTP via injected client."""

    def __init__(
        self,
        config: ConfluenceIntegrationConfig,
        *,
        http_client: Any,
    ) -> None:
        if not config.base_url:
            raise IntegrationConfigurationError(
                "Confluence base_url is required (INTERGRAX_CONFLUENCE_BASE_URL)"
            )
        if not config.email or not config.api_token:
            raise IntegrationConfigurationError(
                "Confluence email and api_token are required "
                "(INTERGRAX_CONFLUENCE_EMAIL, INTERGRAX_CONFLUENCE_API_TOKEN)"
            )
        self._config = config
        self._http_client = http_client

    @property
    def config(self) -> ConfluenceIntegrationConfig:
        return self._config

    def get_page(self, page_id: str) -> WikiPageRecord:
        response = self._http_client.get(
            f"/content/{page_id}",
            params={"expand": "body.storage,space,version"},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise IntegrationConfigurationError("Unexpected Confluence get_page response")
        return _page_from_payload(self._config, payload)

    def search_pages(self, query: str, *, limit: int = 25) -> WikiSearchResult:
        escaped = _escape_cql_literal(query.strip())
        cql = f'type=page AND text ~ "{escaped}"' if escaped else "type=page"
        response = self._http_client.get(
            "/content/search",
            params={
                "cql": cql,
                "limit": max(1, int(limit)),
                "expand": "body.storage,space,version",
            },
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise IntegrationConfigurationError("Unexpected Confluence search response")
        raw_results = data.get("results")
        pages = [
            _page_from_payload(self._config, item)
            for item in raw_results
            if isinstance(item, dict)
        ]
        total_raw = data.get("totalSize", data.get("size", len(pages)))
        total = int(total_raw) if isinstance(total_raw, int) else len(pages)
        return WikiSearchResult(pages=pages, total=total)
