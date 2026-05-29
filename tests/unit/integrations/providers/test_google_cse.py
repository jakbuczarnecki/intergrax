# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Google CSE integration provider (Phase M.4)."""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest

from intergrax.integrations._shared.conformance import assert_search_provider
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.google_cse.adapter import GoogleCSESearchProvider
from intergrax.integrations.providers.google_cse.bundle import (
    GoogleCSEIntegrationBundle,
    create_google_cse_integration,
    create_google_cse_search_provider,
)
from intergrax.integrations.providers.google_cse.register import register_google_cse_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug
from intergrax.websearch.providers.google_cse_provider import GoogleCSEProvider
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
    provider = MagicMock(spec=GoogleCSEProvider)
    provider.name = "google_cse"
    provider.search.return_value = [
        SearchHit(
            provider="google_cse",
            query_issued="intergrax",
            rank=1,
            title="Intergrax",
            url="https://example.com/intergrax",
        )
    ]
    return provider


def test_create_google_cse_integration_bundle(mock_web_provider: MagicMock) -> None:
    bundle = create_google_cse_integration(
        provider=mock_web_provider,
        api_key="key",
        cx="cx-id",
    )

    assert isinstance(bundle, GoogleCSEIntegrationBundle)
    assert isinstance(bundle.search_provider, GoogleCSESearchProvider)
    assert bundle.web_search_provider is mock_web_provider
    assert bundle.config.api_key == "key"
    assert bundle.config.cx == "cx-id"


def test_create_google_cse_search_provider_delegates_to_web_provider(
    mock_web_provider: MagicMock,
) -> None:
    search = create_google_cse_search_provider(provider=mock_web_provider)

    hits = search.search("intergrax", limit=5)

    assert len(hits) == 1
    assert hits[0].url == "https://example.com/intergrax"
    mock_web_provider.search.assert_called_once()
    spec: QuerySpec = mock_web_provider.search.call_args.args[0]
    assert spec.query == "intergrax"
    assert spec.top_k == 5


def test_register_and_resolve_via_profile(mock_web_provider: MagicMock) -> None:
    register_google_cse_integration()
    profile = IntegrationProfile(search_provider=IntegrationSlug.GOOGLE_CSE)

    provider = resolve(
        IntegrationCategory.SEARCH_PROVIDER,
        profile=profile,
        config={"provider": mock_web_provider},
    )

    assert_search_provider(provider)
    hits = provider.search("test")
    assert hits[0].title == "Intergrax"


def test_register_default_integrations_includes_google_cse(
    mock_web_provider: MagicMock,
) -> None:
    register_default_integrations()
    profile = IntegrationProfile(search_provider=IntegrationSlug.GOOGLE_CSE)

    provider = resolve(
        IntegrationCategory.SEARCH_PROVIDER,
        profile=profile,
        config={"provider": mock_web_provider},
    )

    assert isinstance(provider, GoogleCSESearchProvider)


def test_adapter_exposes_web_search_provider(mock_web_provider: MagicMock) -> None:
    adapter = GoogleCSESearchProvider(mock_web_provider)

    assert adapter.web_search_provider is mock_web_provider
    results: List[SearchHit] = list(adapter.search("q", limit=3))
    assert results[0].provider == "google_cse"
