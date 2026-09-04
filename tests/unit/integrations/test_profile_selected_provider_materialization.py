# © Artur Czarnecki. All rights reserved.

"""Regression tests for profile-driven integration materialization."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    IntegrationEntry,
    IntegrationStatus,
)
from intergrax.integrations.registry.catalog import clear_catalog, register_integration
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from local_workspace_application.host.environment_profile import (
    build_local_workspace_integration_profile,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated_catalog() -> None:
    clear_catalog()
    yield
    clear_catalog()


def _register_fake(
    slug: str,
    *,
    categories: tuple[IntegrationCategory, ...],
    factory: Any,
) -> None:
    register_integration(
        IntegrationEntry(
            slug=slug,
            categories=categories,
            factory=factory,
            status=IntegrationStatus.BETA,
        )
    )


def test_unselected_provider_factory_not_called() -> None:
    selected_factory = MagicMock(return_value={"slug": "selected_sql"})
    passive_factory = MagicMock(return_value={"slug": "passive_graph"})
    _register_fake(
        "selected_sql",
        categories=(IntegrationCategory.RELATIONAL_STORE,),
        factory=selected_factory,
    )
    _register_fake(
        "passive_graph",
        categories=(IntegrationCategory.GRAPH_STORE,),
        factory=passive_factory,
    )

    profile = IntegrationProfile.model_validate({"relational_store": "selected_sql"})
    resolved = resolve(IntegrationCategory.RELATIONAL_STORE, profile=profile)

    assert resolved == {"slug": "selected_sql"}
    selected_factory.assert_called_once()
    passive_factory.assert_not_called()


def test_selected_provider_materializes() -> None:
    factory = MagicMock(return_value={"slug": "selected_graph"})
    _register_fake(
        "selected_graph",
        categories=(IntegrationCategory.GRAPH_STORE,),
        factory=factory,
    )

    profile = IntegrationProfile.model_validate({"graph_store": "selected_graph"})
    resolved = resolve(IntegrationCategory.GRAPH_STORE, profile=profile)

    assert resolved == {"slug": "selected_graph"}
    factory.assert_called_once()


def test_selected_invalid_provider_fails_closed() -> None:
    def _raise_config(**_kwargs: object) -> dict[str, str]:
        raise IntegrationConfigurationError("invalid graph configuration")

    _register_fake(
        "broken_graph",
        categories=(IntegrationCategory.GRAPH_STORE,),
        factory=_raise_config,
    )
    profile = IntegrationProfile.model_validate({"graph_store": "broken_graph"})

    with pytest.raises(IntegrationConfigurationError, match="invalid graph configuration"):
        resolve(IntegrationCategory.GRAPH_STORE, profile=profile)


def test_catalog_addition_is_non_behavioral_for_unrelated_profile() -> None:
    selected_factory = MagicMock(return_value={"slug": "selected_sql"})
    passive_factory = MagicMock(return_value={"slug": "passive_graph"})
    _register_fake(
        "selected_sql",
        categories=(IntegrationCategory.RELATIONAL_STORE,),
        factory=selected_factory,
    )
    profile = IntegrationProfile.model_validate({"relational_store": "selected_sql"})
    first = resolve(IntegrationCategory.RELATIONAL_STORE, profile=profile)

    _register_fake(
        "passive_graph",
        categories=(IntegrationCategory.GRAPH_STORE,),
        factory=passive_factory,
    )
    second = resolve(IntegrationCategory.RELATIONAL_STORE, profile=profile)

    assert first == second == {"slug": "selected_sql"}
    selected_factory.assert_called()
    passive_factory.assert_not_called()


def test_lkw_integration_profile_has_no_graph_store_selection() -> None:
    profile = build_local_workspace_integration_profile()
    assert profile.slug_for_category(IntegrationCategory.GRAPH_STORE) is None
