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
_INVALID_HTTP_RESPONSE = "Microsoft Graph knowledge dependency returned an invalid response"
_UNEXPECTED_REDIRECT = "Microsoft Graph knowledge dependency returned an unexpected redirect"
_INVALID_CONTINUATION = "invalid Microsoft Graph knowledge continuation"


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

    def __post_init__(self) -> None:
        if type(self.items) is not tuple:
            raise TypeError("items must be a tuple")
        for item in self.items:
            if not isinstance(item, dict):
                raise TypeError("each item must be a dict")
        if self.continuation is not None and not isinstance(
            self.continuation, MsGraphKnowledgeContinuation
        ):
            raise TypeError("continuation must be MsGraphKnowledgeContinuation or None")


class MsGraphKnowledgeSyncResetRequired(IntegrationDependencyError):
    """Delta token is no longer valid; full synchronization must restart."""

    def __init__(self) -> None:
        super().__init__("Microsoft Graph knowledge synchronization must restart")


def _default_port(scheme: str, explicit_port: int | None) -> int:
    if explicit_port is not None:
        return explicit_port
    return 443 if scheme == "https" else 80


def _parsed_graph_base(graph_base_url: object) -> tuple[str, str, int, str]:
    if not isinstance(graph_base_url, str):
        raise ValueError(_INVALID_CONTINUATION_URL)
    cleaned = graph_base_url.strip()
    if not cleaned:
        raise ValueError(_INVALID_CONTINUATION_URL)
    if any(ord(ch) < 32 for ch in cleaned):
        raise ValueError(_INVALID_CONTINUATION_URL)

    parsed = urlparse(cleaned.rstrip("/"))
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(_INVALID_CONTINUATION_URL)
    if parsed.username or parsed.password:
        raise ValueError(_INVALID_CONTINUATION_URL)
    if parsed.query:
        raise ValueError(_INVALID_CONTINUATION_URL)
    if parsed.fragment:
        raise ValueError(_INVALID_CONTINUATION_URL)
    if parsed.port is not None and not (1 <= parsed.port <= 65535):
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

    try:
        base_scheme, base_host, base_port, base_path = _parsed_graph_base(graph_base_url)
    except ValueError:
        raise ValueError(_INVALID_CONTINUATION_URL) from None

    if parsed.hostname != base_host:
        raise ValueError(_INVALID_CONTINUATION_URL)
    if _default_port(parsed.scheme, parsed.port) != base_port:
        raise ValueError(_INVALID_CONTINUATION_URL)

    url_path = parsed.path.rstrip("/") or "/"
    if url_path != base_path and not parsed.path.startswith(f"{base_path}/"):
        raise ValueError(_INVALID_CONTINUATION_URL)

    return cleaned


def _validate_odata_link(
    payload: dict[str, Any],
    key: str,
    *,
    graph_base_url: str,
) -> str | None:
    """Return validated URL when key is present; None when key is absent."""
    if key not in payload:
        return None
    raw_value = payload[key]
    if not isinstance(raw_value, str):
        raise ValueError(_MALFORMED_RESPONSE)
    trimmed = raw_value.strip()
    if not trimmed:
        raise ValueError(_MALFORMED_RESPONSE)
    try:
        return validate_msgraph_continuation_url(trimmed, graph_base_url=graph_base_url)
    except ValueError:
        raise ValueError(_MALFORMED_RESPONSE) from None


def parse_msgraph_collection_page(
    payload: object,
    *,
    graph_base_url: str,
    delta_mode: bool,
) -> MsGraphKnowledgeCollectionPage:
    if type(delta_mode) is not bool:
        raise ValueError(_MALFORMED_RESPONSE)

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

    has_next_key = "@odata.nextLink" in payload
    has_delta_key = "@odata.deltaLink" in payload

    if has_next_key and has_delta_key:
        raise ValueError(_MALFORMED_RESPONSE)

    next_link: str | None = None
    delta_link: str | None = None

    if has_next_key:
        next_link = _validate_odata_link(
            payload, "@odata.nextLink", graph_base_url=graph_base_url
        )
    if has_delta_key:
        delta_link = _validate_odata_link(
            payload, "@odata.deltaLink", graph_base_url=graph_base_url
        )

    if delta_mode:
        if not has_next_key and not has_delta_key:
            raise ValueError(_MALFORMED_RESPONSE)
        if has_next_key and has_delta_key:
            raise ValueError(_MALFORMED_RESPONSE)
    else:
        if has_delta_key:
            raise ValueError(_MALFORMED_RESPONSE)

    continuation: MsGraphKnowledgeContinuation | None = None
    if next_link is not None:
        continuation = MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
            url=next_link,
        )
    elif delta_link is not None:
        continuation = MsGraphKnowledgeContinuation(
            kind=MsGraphKnowledgeContinuationKind.DELTA,
            url=delta_link,
        )

    return MsGraphKnowledgeCollectionPage(items=tuple(items), continuation=continuation)


def _validate_initial_path(path: object) -> str:
    if not isinstance(path, str):
        raise IntegrationConfigurationError(_INVALID_INITIAL_PATH)
    cleaned = path.strip()
    if not cleaned:
        raise IntegrationConfigurationError(_INVALID_INITIAL_PATH)
    if any(ord(ch) < 32 for ch in cleaned):
        raise IntegrationConfigurationError(_INVALID_INITIAL_PATH)
    if cleaned == "/":
        raise IntegrationConfigurationError(_INVALID_INITIAL_PATH)
    if not cleaned.startswith("/"):
        raise IntegrationConfigurationError(_INVALID_INITIAL_PATH)
    if cleaned.startswith("//"):
        raise IntegrationConfigurationError(_INVALID_INITIAL_PATH)
    if "://" in cleaned:
        raise IntegrationConfigurationError(_INVALID_INITIAL_PATH)
    if "@" in cleaned:
        raise IntegrationConfigurationError(_INVALID_INITIAL_PATH)
    if "?" in cleaned:
        raise IntegrationConfigurationError(_INVALID_INITIAL_PATH)
    if "#" in cleaned:
        raise IntegrationConfigurationError(_INVALID_INITIAL_PATH)
    if "\\" in cleaned:
        raise IntegrationConfigurationError(_INVALID_INITIAL_PATH)
    segments = cleaned.split("/")
    if "." in segments or ".." in segments:
        raise IntegrationConfigurationError(_INVALID_INITIAL_PATH)
    return cleaned


def _response_status_code(response: object) -> int:
    try:
        status_code = response.status_code  # type: ignore[attr-defined]
    except AttributeError:
        raise IntegrationDependencyError(_INVALID_HTTP_RESPONSE) from None
    if type(status_code) is not int:
        raise IntegrationDependencyError(_INVALID_HTTP_RESPONSE) from None
    if status_code < 100 or status_code > 599:
        raise IntegrationDependencyError(_INVALID_HTTP_RESPONSE) from None
    return status_code


def _raise_for_knowledge_response(
    response: object,
    *,
    not_found_is_dependency: bool,
) -> None:
    status_code = _response_status_code(response)
    if 200 <= status_code <= 299:
        return
    if 300 <= status_code <= 399:
        raise IntegrationDependencyError(_UNEXPECTED_REDIRECT) from None
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
    except AttributeError:
        raise ValueError(_MALFORMED_RESPONSE) from None
    if not callable(json_method):
        raise ValueError(_MALFORMED_RESPONSE) from None
    try:
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
        if not isinstance(continuation, MsGraphKnowledgeContinuation):
            raise IntegrationConfigurationError(_INVALID_CONTINUATION)
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
