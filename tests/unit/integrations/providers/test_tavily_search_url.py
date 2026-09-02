# © Artur Czarnecki. All rights reserved.

"""Regression: Tavily search requests must target /search without trailing-slash base URL."""

from __future__ import annotations

from typing import Any

import pytest

from intergrax.integrations.providers.search_provider.tavily.bundle import create_tavily_search_provider

pytestmark = pytest.mark.unit


class _CapturingHttp:
    def __init__(self, *, base_url: str, **kwargs: object) -> None:
        del kwargs
        self.base_url = base_url
        self.post_path: str | None = None

    def post(self, path: str, **kwargs: object) -> Any:
        del kwargs
        self.post_path = path
        return _FakeResponse()


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"results": []}


def test_tavily_factory_posts_to_search_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, _CapturingHttp] = {}

    def _fake_open(config: object, *, default_url: str) -> _CapturingHttp:
        del config
        client = _CapturingHttp(base_url=default_url)
        captured["client"] = client
        return client

    monkeypatch.setattr(
        "intergrax.integrations._shared.p3.factories._open_httpx_client",
        _fake_open,
    )
    provider = create_tavily_search_provider(api_key="tvly-test")
    provider.search("python", limit=1)

    client = captured["client"]
    assert client.base_url == "https://api.tavily.com"
    assert client.post_path == "/search"
    assert client.post_path != "/search/"
