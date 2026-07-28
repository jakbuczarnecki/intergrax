# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Confluence REST client — HTTP client injected from ``opens.py`` only."""

from __future__ import annotations

import re
from typing import Any, Mapping

from intergrax.integrations.contracts.base import IntegrationConfigurationError, IntegrationDependencyError
from intergrax.integrations.contracts.wiki_knowledge import WikiPageRecord, WikiSearchResult
from intergrax.integrations.providers.wiki_knowledge.confluence.config import ConfluenceIntegrationConfig
from intergrax.integrations.providers.wiki_knowledge.confluence.knowledge_read import (
    ConfluenceKnowledgePage,
    ConfluenceKnowledgePagePage,
    parse_confluence_knowledge_page,
    parse_confluence_knowledge_page_page,
    validate_confluence_page_id,
    validate_confluence_space_id,
)


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


def _response_status_code(response: object) -> int | None:
    status_code = response.status_code  # type: ignore[attr-defined]
    return int(status_code) if isinstance(status_code, int) else None


def _raise_for_knowledge_response(
    response: object,
    *,
    operation: str,
    list_space: bool = False,
) -> None:
    status_code = _response_status_code(response)
    if status_code is None or status_code < 400:
        return
    if status_code == 429 or status_code >= 500:
        raise IntegrationDependencyError(f"Confluence {operation} dependency failure")
    if status_code in {400, 401, 403}:
        raise IntegrationConfigurationError(f"Confluence {operation} configuration failure")
    if list_space and status_code == 404:
        raise IntegrationConfigurationError(f"Confluence {operation} configuration failure")
    if operation == "get_knowledge_page" and status_code == 404:
        raise IntegrationDependencyError("Confluence page fetch dependency failure")
    raise IntegrationConfigurationError(f"Confluence {operation} configuration failure")


def _execute_knowledge_transport(operation: str, transport_fn: Any) -> object:
    try:
        return transport_fn()
    except (IntegrationConfigurationError, IntegrationDependencyError):
        raise
    except Exception:
        raise IntegrationDependencyError(
            "Confluence knowledge dependency is unavailable"
        ) from None


def _decode_knowledge_json(response: object) -> dict[str, Any]:
    try:
        json_method = response.json  # type: ignore[attr-defined]
        payload = json_method()
    except Exception:
        raise ValueError("unexpected Confluence knowledge response") from None
    if not isinstance(payload, dict):
        raise ValueError("unexpected Confluence knowledge response")
    return payload


def _validate_knowledge_page_scope(
    page: ConfluenceKnowledgePagePage,
    *,
    space_id: str,
) -> None:
    for item in page.pages:
        if item.space_id != space_id:
            raise ValueError("page spaceId does not match requested space")


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

    def list_knowledge_pages(
        self,
        *,
        space_id: str,
        cursor: str | None,
        limit: int,
    ) -> ConfluenceKnowledgePagePage:
        validated_space_id = validate_confluence_space_id(space_id)
        if limit < 1 or limit > 250:
            raise ValueError("limit must be in range 1..250")
        params: dict[str, str | int] = {
            "limit": int(limit),
            "status": "current",
        }
        if cursor is not None:
            if not isinstance(cursor, str):
                raise ValueError("cursor must be a string")
            token = cursor.strip()
            if not token:
                raise ValueError("cursor must be a non-empty string")
            params["cursor"] = token
        url = self._config.v2_api_url(f"/spaces/{validated_space_id}/pages")
        response = _execute_knowledge_transport(
            "list_knowledge_pages",
            lambda: self._http_client.get(url, params=params),
        )
        _raise_for_knowledge_response(
            response,
            operation="list_knowledge_pages",
            list_space=True,
        )
        payload = _decode_knowledge_json(response)
        try:
            page = parse_confluence_knowledge_page_page(
                payload,
                requested_space_id=validated_space_id,
                page_url_builder=self._config.page_url,
            )
        except (ValueError, TypeError):
            raise ValueError("unexpected Confluence knowledge response") from None
        _validate_knowledge_page_scope(page, space_id=validated_space_id)
        return page

    def get_knowledge_page(
        self,
        *,
        page_id: str,
        version_number: int,
    ) -> ConfluenceKnowledgePage:
        validated_page_id = validate_confluence_page_id(page_id)
        if version_number < 1:
            raise ValueError("version_number must be >= 1")
        url = self._config.v2_api_url(f"/pages/{validated_page_id}")
        response = _execute_knowledge_transport(
            "get_knowledge_page",
            lambda: self._http_client.get(
                url,
                params={
                    "body-format": "storage",
                    "version": int(version_number),
                },
            ),
        )
        _raise_for_knowledge_response(response, operation="get_knowledge_page")
        payload = _decode_knowledge_json(response)
        try:
            return parse_confluence_knowledge_page(
                payload,
                page_id=validated_page_id,
                version_number=version_number,
                page_url_builder=self._config.page_url,
            )
        except (ValueError, TypeError):
            raise ValueError("unexpected Confluence knowledge response") from None
