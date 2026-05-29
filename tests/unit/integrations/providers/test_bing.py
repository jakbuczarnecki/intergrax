# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Bing integration provider (Phase M.4)."""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest

from intergrax.integrations._shared.conformance import assert_search_provider
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.bing.adapter import BingSearchProvider
from intergrax.integrations.providers.bing.bundle import (
    BingIntegrationBundle,
    create_bing_integration,
    create_bing_search_provider,
)
from intergrax.integrations.providers.bing.register import register_bing_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug
from intergrax.websearch.providers.bing_provider import BingWebProvider
from intergrax.websearch.schemas.query_spec import QuerySpec
from intergrax.websearch.schemas.search_hit import SearchHit

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


@pytest.fixture
def mock_web_provider() -> MagicMock:
    provider = MagicMock(spec=BingWebProvider)
    provider.name = "bing_web"
    provider.search.return_value = [
        SearchHit(
            provider="bing_web",
            query_issued="intergrax",
            rank=1,
            title="Intergrax",
            url="https://example.com/intergrax",
        )
    ]
    return provider


def test_create_bing_integration_bundle(mock_web_provider: MagicMock) -> None:
    bundle = create_bing_integration(provider=mock_web_provider, api_key="key")

    assert isinstance(bundle, BingIntegrationBundle)
    assert isinstance(bundle.search_provider, BingSearchProvider)
    assert bundle.web_search_provider is mock_web_provider
    assert bundle.config.api_key == "key"


def test_create_bing_search_provider_delegates_to_web_provider(
    mock_web_provider: MagicMock,
) -> None:
    search = create_bing_search_provider(provider=mock_web_provider)

    hits = search.search("intergrax", limit=5)

    assert len(hits) == 1
    assert hits[0].url == "https://example.com/intergrax"
    mock_web_provider.search.assert_called_once()
    spec: QuerySpec = mock_web_provider.search.call_args.args[0]
    assert spec.query == "intergrax"
    assert spec.top_k == 5


def test_register_and_resolve_via_profile(mock_web_provider: MagicMock) -> None:
    register_bing_integration()
    profile = IntegrationProfile(search_provider=IntegrationSlug.BING)

    provider = resolve(
        IntegrationCategory.SEARCH_PROVIDER,
        profile=profile,
        config={"provider": mock_web_provider},
    )

    assert_search_provider(provider)
    hits = provider.search("test")
    assert hits[0].title == "Intergrax"


def test_register_default_integrations_includes_bing(mock_web_provider: MagicMock) -> None:
    register_default_integrations()
    profile = IntegrationProfile(search_provider=IntegrationSlug.BING)

    provider = resolve(
        IntegrationCategory.SEARCH_PROVIDER,
        profile=profile,
        config={"provider": mock_web_provider},
    )

    assert isinstance(provider, BingSearchProvider)


def test_adapter_exposes_web_search_provider(mock_web_provider: MagicMock) -> None:
    adapter = BingSearchProvider(mock_web_provider)

    assert adapter.web_search_provider is mock_web_provider
    results: List[SearchHit] = list(adapter.search("q", limit=3))
    assert results[0].provider == "bing_web"
