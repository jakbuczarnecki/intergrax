# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared Microsoft Graph knowledge-read transport, paging and delta foundation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.integrations.contracts.base import (
    IntegrationConfigurationError,
    IntegrationDependencyError,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.config import (
    Ms365GraphIntegrationConfig,
)

_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_MALFORMED_RESPONSE = "unexpected Microsoft Graph knowledge response"
_INVALID_CONTINUATION_URL = "invalid Microsoft Graph continuation URL"
_INVALID_INITIAL_PATH = "invalid Microsoft Graph knowledge request path"


class MsGraphKnowledgeContinuationKind(StrEnum):
    NEXT_PAGE = "next_page"
    DELTA = "delta"


class MsGraphKnowledgeContinuation(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    kind: MsGraphKnowledgeContinuationKind
    url: str = Field(repr=False)

    @field_validator("url", mode="before")
    @classmethod
    def _validate_url(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("continuation url must be a string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("continuation url must not be empty")
        parsed = urlparse(cleaned)
        if parsed.username or parsed.password:
            raise ValueError("continuation url must not contain credentials")
        return cleaned


@dataclass(frozen=True, slots=True)
class MsGraphKnowledgeCollectionPage:
    items: tuple[dict[str, Any], ...]
    continuation: MsGraphKnowledgeContinuation | None = field(default=None, repr=False)


class MsGraphKnowledgeSyncResetRequired(IntegrationDependencyError):
    """Delta token is no longer valid; full synchronization must restart."""

    def __init__(self) -> None:
        super().__init__("Microsoft Graph knowledge synchronization must restart")


def _default_port(scheme: str, explicit_port: int | None) -> int:
    if explicit_port is not None:
        return explicit_port
    return 443 if scheme == "https" else 80


def _parsed_graph_base(graph_base_url: str) -> tuple[str, str, int, str]:
    parsed = urlparse(graph_base_url.strip().rstrip("/"))
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(_INVALID_CONTINUATION_URL)
    base_path = parsed.path.rstrip("/") or "/"
    if base_path == "/":
        raise ValueError(_INVALID_CONTINUATION_URL)
    return (
        parsed.scheme,
        parsed.hostname,
        _default_port(parsed.scheme, parsed.port),
        base_path,
    )


def validate_msgraph_continuation_url(
    value: object,
    *,
    graph_base_url: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(_INVALID_CONTINUATION_URL)
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(_INVALID_CONTINUATION_URL)
    if any(ord(ch) < 32 for ch in cleaned):
        raise ValueError(_INVALID_CONTINUATION_URL)

    parsed = urlparse(cleaned)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(_INVALID_CONTINUATION_URL)
    if parsed.scheme != "https":
        raise ValueError(_INVALID_CONTINUATION_URL)
    if parsed.username or parsed.password:
        raise ValueError(_INVALID_CONTINUATION_URL)
    if parsed.fragment:
        raise ValueError(_INVALID_CONTINUATION_URL)
    if not parsed.hostname:
        raise ValueError(_INVALID_CONTINUATION_URL)
    if not parsed.path or parsed.path == "/":
        raise ValueError(_INVALID_CONTINUATION_URL)

    base_scheme, base_host, base_port, base_path = _parsed_graph_base(graph_base_url)
    if parsed.hostname != base_host:
        raise ValueError(_INVALID_CONTINUATION_URL)
    if _default_port(parsed.scheme, parsed.port) != base_port:
        raise ValueError(_INVALID_CONTINUATION_URL)

    url_path = parsed.path.rstrip("/") or "/"
    if url_path != base_path and not parsed.path.startswith(f"{base_path}/"):
        raise ValueError(_INVALID_CONTINUATION_URL)

    return cleaned


def parse_msgraph_collection_page(
    payload: object,
    *,
    graph_base_url: str,
    delta_mode: bool,
) -> MsGraphKnowledgeCollectionPage:
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_RESPONSE)

    raw_value = payload.get("value")
    if not isinstance(raw_value, list):
        raise ValueError(_MALFORMED_RESPONSE)

    items: list[dict[str, Any]] = []
    for item in raw_value:
        if not isinstance(item, dict):
            raise ValueError(_MALFORMED_RESPONSE)
        items.append(item)

    next_link = payload.get("@odata.nextLink")
    delta_link = payload.get("@odata.deltaLink")
    has_next = isinstance(next_link, str) and bool(next_link.strip())
    has_delta = isinstance(delta_link, str) and bool(delta_link.strip())

    if has_next and has_delta:
        raise ValueError(_MALFORMED_RESPONSE)

    if delta_mode:
        if not has_next and not has_delta:
            raise ValueError(_MALFORMED_RESPONSE)
    else:
        if has_delta:
            raise ValueError(_MALFORMED_RESPONSE)

    continuation: MsGraphKnowledgeContinuation | None = None
    if has_next:
        validated_url = validate_msgraph_continuation_url(next_link, graph_base_url=graph_base_url)
        continuation = MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=validated_url,
        )
    elif has_delta:
        validated_url = validate_msgraph_continuation_url(delta_link, graph_base_url=graph_base_url)
        continuation = MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=validated_url,
        )

    return MsGraphKnowledgeCollectionPage(items=tuple(items), continuation=continuation)


def _validate_initial_path(path: str) -> str:
    cleaned = path.strip()
    if not cleaned.startswith("/"):
        raise IntegrationConfigurationError(_INVALID_INITIAL_PATH)
    if "://" in cleaned or "@" in cleaned or "#" in cleaned:
        raise IntegrationConfigurationError(_INVALID_INITIAL_PATH)
    if ".." in cleaned.split("/"):
        raise IntegrationConfigurationError(_INVALID_INITIAL_PATH)
    return cleaned


def _response_status_code(response: object) -> int | None:
    status_code = response.status_code  # type: ignore[attr-defined]
    return int(status_code) if isinstance(status_code, int) else None


def _raise_for_knowledge_response(
    response: object,
    *,
    not_found_is_dependency: bool,
) -> None:
    status_code = _response_status_code(response)
    if status_code is None or status_code < 400:
        return
    if status_code in {408, 429} or status_code >= 500:
        raise IntegrationDependencyError("Microsoft Graph knowledge dependency failure")
    if status_code == 410:
        raise MsGraphKnowledgeSyncResetRequired()
    if status_code == 404:
        if not_found_is_dependency:
            raise IntegrationDependencyError("Microsoft Graph knowledge dependency failure")
        raise IntegrationConfigurationError("Microsoft Graph knowledge configuration failure")
    if status_code in {400, 401, 403}:
        raise IntegrationConfigurationError("Microsoft Graph knowledge configuration failure")
    raise IntegrationConfigurationError("Microsoft Graph knowledge configuration failure")


def _decode_knowledge_json(response: object) -> dict[str, Any]:
    try:
        json_method = response.json  # type: ignore[attr-defined]
        payload = json_method()
    except Exception:
        raise ValueError(_MALFORMED_RESPONSE) from None
    if not isinstance(payload, dict):
        raise ValueError(_MALFORMED_RESPONSE)
    return payload


def _execute_knowledge_transport(transport_fn: Any) -> object:
    try:
        return transport_fn()
    except (IntegrationConfigurationError, IntegrationDependencyError, MsGraphKnowledgeSyncResetRequired):
        raise
    except Exception:
        raise IntegrationDependencyError(
            "Microsoft Graph knowledge dependency is unavailable"
        ) from None


class MsGraphKnowledgeTransport:
    """Shared Graph knowledge read transport over an injected HTTP client."""

    def __init__(
        self,
        config: Ms365GraphIntegrationConfig,
        *,
        http_client: Any,
    ) -> None:
        self._config = config
        self._http_client = http_client

    def get_initial_json(
        self,
        *,
        path: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, str] | None = None,
        not_found_is_dependency: bool = False,
    ) -> dict[str, Any]:
        validated_path = _validate_initial_path(path)
        return self._request_json(
            lambda: self._http_client.get(
                validated_path,
                params=dict(params) if params is not None else None,
                headers=dict(headers) if headers is not None else None,
            ),
            not_found_is_dependency=not_found_is_dependency,
        )

    def get_continuation_json(
        self,
        *,
        continuation: MsGraphKnowledgeContinuation,
        headers: Mapping[str, str] | None = None,
        not_found_is_dependency: bool = False,
    ) -> dict[str, Any]:
        validated_url = validate_msgraph_continuation_url(
            continuation.url,
            graph_base_url=self._config.graph_base_url,
        )
        return self._request_json(
            lambda: self._http_client.get(
                validated_url,
                headers=dict(headers) if headers is not None else None,
            ),
            not_found_is_dependency=not_found_is_dependency,
        )

    def _request_json(
        self,
        transport_fn: Any,
        *,
        not_found_is_dependency: bool,
    ) -> dict[str, Any]:
        response = _execute_knowledge_transport(transport_fn)
        _raise_for_knowledge_response(
            response,
            not_found_is_dependency=not_found_is_dependency,
        )
        return _decode_knowledge_json(response)
