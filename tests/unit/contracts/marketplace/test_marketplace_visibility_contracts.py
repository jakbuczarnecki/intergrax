# © Artur Czarnecki. All rights reserved.

"""ME-9 marketplace visibility contract validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.contracts.marketplace.query_context import MarketplaceQueryContext
from intergrax.contracts.marketplace.visibility import (
    MarketplaceVisibility,
    MarketplaceVisibilityScope,
)

pytestmark = pytest.mark.unit


def test_public_visibility_rejects_tenant_id() -> None:
    with pytest.raises(ValidationError, match="PUBLIC marketplace visibility"):
        MarketplaceVisibility(
            scope=MarketplaceVisibilityScope.PUBLIC,
            tenant_id="tenant-a",
        )


def test_public_visibility_rejects_organization_id() -> None:
    with pytest.raises(ValidationError, match="PUBLIC marketplace visibility"):
        MarketplaceVisibility(
            scope=MarketplaceVisibilityScope.PUBLIC,
            organization_id="org-a",
        )


def test_tenant_private_requires_tenant_id() -> None:
    with pytest.raises(ValidationError, match="TENANT_PRIVATE"):
        MarketplaceVisibility(scope=MarketplaceVisibilityScope.TENANT_PRIVATE)


def test_tenant_private_rejects_organization_id() -> None:
    with pytest.raises(ValidationError, match="organization_id"):
        MarketplaceVisibility(
            scope=MarketplaceVisibilityScope.TENANT_PRIVATE,
            tenant_id="tenant-a",
            organization_id="org-a",
        )


def test_tenant_private_accepts_explicit_tenant() -> None:
    visibility = MarketplaceVisibility(
        scope=MarketplaceVisibilityScope.TENANT_PRIVATE,
        tenant_id="tenant-a",
    )
    assert visibility.tenant_id == "tenant-a"


def test_organization_private_requires_organization_id() -> None:
    with pytest.raises(ValidationError, match="ORGANIZATION_PRIVATE"):
        MarketplaceVisibility(scope=MarketplaceVisibilityScope.ORGANIZATION_PRIVATE)


def test_organization_private_rejects_tenant_id() -> None:
    with pytest.raises(ValidationError, match="tenant_id and organization_id"):
        MarketplaceVisibility(
            scope=MarketplaceVisibilityScope.ORGANIZATION_PRIVATE,
            organization_id="org-a",
            tenant_id="tenant-a",
        )


def test_query_context_rejects_blank_tenant_id() -> None:
    with pytest.raises(ValidationError):
        MarketplaceQueryContext(tenant_id="   ")


def test_query_context_rejects_blank_organization_id() -> None:
    with pytest.raises(ValidationError):
        MarketplaceQueryContext(organization_id="   ")


def test_query_context_accepts_tenant_and_organization_together() -> None:
    context = MarketplaceQueryContext(tenant_id="tenant-a", organization_id="org-a")
    assert context.tenant_id == "tenant-a"
    assert context.organization_id == "org-a"
