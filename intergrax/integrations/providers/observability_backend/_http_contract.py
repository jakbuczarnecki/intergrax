# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Minimal shared HTTP contracts for observability backend provider REST clients."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeAlias, TypeVar, runtime_checkable

ConfigT = TypeVar("ConfigT")

ProviderJsonObject: TypeAlias = dict[str, object]
ProviderJsonMapping: TypeAlias = Mapping[str, object]
ElasticsearchDocument: TypeAlias = dict[str, object]
ElasticsearchSearchBody: TypeAlias = dict[str, object]


@runtime_checkable
class ObservabilityHttpResponse(Protocol):
    """Minimal HTTP response surface used by observability provider clients."""

    @property
    def status_code(self) -> int:
        """HTTP status code."""
        ...

    def json(self) -> object:
        """Decode response body as JSON."""
        ...

    def raise_for_status(self) -> None:
        """Raise when the response indicates an HTTP error."""
        ...


@runtime_checkable
class ObservabilityHttpClient(Protocol):
    """Minimal sync HTTP client injected into observability provider REST clients."""

    def get(
        self,
        url: str,
        *,
        params: object | None = None,
    ) -> ObservabilityHttpResponse:
        """Issue an HTTP GET."""
        ...

    def post(
        self,
        url: str,
        *,
        json: object | None = None,
        params: object | None = None,
    ) -> ObservabilityHttpResponse:
        """Issue an HTTP POST."""
        ...

    def put(
        self,
        url: str,
        *,
        json: object | None = None,
    ) -> ObservabilityHttpResponse:
        """Issue an HTTP PUT."""
        ...

    def head(
        self,
        url: str,
        *,
        json: object | None = None,
    ) -> ObservabilityHttpResponse:
        """Issue an HTTP HEAD."""
        ...


ObservabilityHttpClientFactory: TypeAlias = Callable[[ConfigT], ObservabilityHttpClient]


class ObservabilityHttpxClientAdapter:
    """Bind a concrete httpx ``Client`` to :class:`ObservabilityHttpClient`."""

    def __init__(self, client: Any) -> None:
        # F: third-party httpx client stubs are not Protocol-compatible on parameter types.
        self._client = client

    def get(
        self,
        url: str,
        *,
        params: object | None = None,
    ) -> ObservabilityHttpResponse:
        return self._client.get(url, params=params)

    def post(
        self,
        url: str,
        *,
        json: object | None = None,
        params: object | None = None,
    ) -> ObservabilityHttpResponse:
        return self._client.post(url, json=json, params=params)

    def put(
        self,
        url: str,
        *,
        json: object | None = None,
    ) -> ObservabilityHttpResponse:
        return self._client.put(url, json=json)

    def head(
        self,
        url: str,
        *,
        json: object | None = None,
    ) -> ObservabilityHttpResponse:
        return self._client.head(url, json=json)
