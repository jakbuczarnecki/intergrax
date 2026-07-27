# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for IntegrationProfileVendorResolver."""

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationCategoryMismatchError,
    IntegrationDependencyError,
)
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.resolver import IntegrationProfileVendorResolver
from tests.unit.runtime.vendor_knowledge._fakes import FakeIntegration, make_source


class _SpyProfile:
    def __init__(self, *, result: object | None = None, error: Exception | None = None) -> None:
        self.calls: list[IntegrationCategory] = []
        self._result = result
        self._error = error

    def resolve(self, category: IntegrationCategory) -> object:
        self.calls.append(category)
        if self._error is not None:
            raise self._error
        return self._result


@pytest.mark.unit
def test_prebuilt_instance_returned_same_object() -> None:
    integration = FakeIntegration()
    profile = IntegrationProfile(
        issue_tracker=IntegrationBinding.from_instance(integration),
    )
    resolver = IntegrationProfileVendorResolver(profile=profile, tenant_id="tenant-1")

    result = resolver.resolve(source=make_source())

    assert result is integration


@pytest.mark.unit
def test_tenant_mismatch_fails_before_profile_resolution() -> None:
    spy = _SpyProfile(result=FakeIntegration())
    resolver = IntegrationProfileVendorResolver(profile=spy, tenant_id="tenant-1")  # type: ignore[arg-type]

    with pytest.raises(VendorKnowledgeError) as exc_info:
        resolver.resolve(source=make_source(tenant_id="other-tenant"))

    assert exc_info.value.code is VendorKnowledgeErrorCode.TENANT_MISMATCH
    assert spy.calls == []


@pytest.mark.unit
def test_connection_ref_fails_closed() -> None:
    spy = _SpyProfile(result=FakeIntegration())
    resolver = IntegrationProfileVendorResolver(profile=spy, tenant_id="tenant-1")  # type: ignore[arg-type]

    with pytest.raises(VendorKnowledgeError) as exc_info:
        resolver.resolve(source=make_source(connection_ref="conn-1"))

    assert exc_info.value.code is VendorKnowledgeErrorCode.CONFIGURATION_ERROR
    assert "connection-aware" in exc_info.value.safe_message.lower()
    assert spy.calls == []


@pytest.mark.unit
def test_provider_mismatch() -> None:
    integration = FakeIntegration(provider_id="other")
    profile = IntegrationProfile(
        issue_tracker=IntegrationBinding.from_instance(integration),
    )
    resolver = IntegrationProfileVendorResolver(profile=profile, tenant_id="tenant-1")

    with pytest.raises(VendorKnowledgeError) as exc_info:
        resolver.resolve(source=make_source(provider_id="example"))

    assert exc_info.value.code is VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND


@pytest.mark.unit
def test_integration_category_mismatch_from_identity() -> None:
    integration = FakeIntegration(
        integration_kind=IntegrationCategory.WIKI_KNOWLEDGE.value,
    )
    profile = IntegrationProfile(
        issue_tracker=IntegrationBinding.from_instance(integration),
    )
    resolver = IntegrationProfileVendorResolver(profile=profile, tenant_id="tenant-1")

    with pytest.raises(VendorKnowledgeError) as exc_info:
        resolver.resolve(source=make_source())

    assert exc_info.value.code is VendorKnowledgeErrorCode.INTEGRATION_CATEGORY_MISMATCH


@pytest.mark.unit
def test_missing_profile_integration() -> None:
    profile = IntegrationProfile()
    resolver = IntegrationProfileVendorResolver(profile=profile, tenant_id="tenant-1")

    with pytest.raises(VendorKnowledgeError) as exc_info:
        resolver.resolve(source=make_source())

    assert exc_info.value.code is VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND


@pytest.mark.unit
def test_platform_dependency_error_mapping() -> None:
    spy = _SpyProfile(
        error=IntegrationDependencyError(
            "backend down token=super-secret",
            integration_name="example",
        )
    )
    resolver = IntegrationProfileVendorResolver(profile=spy, tenant_id="tenant-1")  # type: ignore[arg-type]

    with pytest.raises(VendorKnowledgeError) as exc_info:
        resolver.resolve(source=make_source())

    error = exc_info.value
    assert error.code is VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE
    assert error.retryable is True
    assert "super-secret" not in error.safe_message
    assert "token=" not in error.safe_message


@pytest.mark.unit
def test_category_mismatch_error_mapping() -> None:
    spy = _SpyProfile(error=IntegrationCategoryMismatchError("example", "issue_tracker"))
    resolver = IntegrationProfileVendorResolver(profile=spy, tenant_id="tenant-1")  # type: ignore[arg-type]

    with pytest.raises(VendorKnowledgeError) as exc_info:
        resolver.resolve(source=make_source())

    assert exc_info.value.code is VendorKnowledgeErrorCode.INTEGRATION_CATEGORY_MISMATCH


@pytest.mark.unit
def test_unexpected_error_mapping_without_leaking_text() -> None:
    spy = _SpyProfile(error=RuntimeError("leak bearer abc.def and https://x?token=1"))
    resolver = IntegrationProfileVendorResolver(profile=spy, tenant_id="tenant-1")  # type: ignore[arg-type]

    with pytest.raises(VendorKnowledgeError) as exc_info:
        resolver.resolve(source=make_source())

    error = exc_info.value
    assert error.code is VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE
    assert error.retryable is False
    assert "bearer" not in error.safe_message.lower()
    assert "token=" not in error.safe_message
    assert "https://" not in error.safe_message


@pytest.mark.unit
def test_no_vendor_specific_conditional_behavior() -> None:
    alpha = FakeIntegration(provider_id="alpha", integration_id="alpha:issue_tracker")
    beta = FakeIntegration(provider_id="beta", integration_id="beta:issue_tracker")
    profile_alpha = IntegrationProfile(
        issue_tracker=IntegrationBinding.from_instance(alpha),
    )
    profile_beta = IntegrationProfile(
        issue_tracker=IntegrationBinding.from_instance(beta),
    )

    resolver_alpha = IntegrationProfileVendorResolver(
        profile=profile_alpha, tenant_id="tenant-1"
    )
    resolver_beta = IntegrationProfileVendorResolver(
        profile=profile_beta, tenant_id="tenant-1"
    )

    assert resolver_alpha.resolve(source=make_source(provider_id="alpha")) is alpha
    assert resolver_beta.resolve(source=make_source(provider_id="beta")) is beta
