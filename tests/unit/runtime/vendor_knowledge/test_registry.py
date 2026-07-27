# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for KnowledgeAdapterRegistry."""

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry
from tests.unit.runtime.vendor_knowledge._fakes import FakeAdapter, FakeIntegration, make_source


@pytest.mark.unit
def test_register_and_resolve_exact_key() -> None:
    registry = KnowledgeAdapterRegistry()
    adapter = FakeAdapter(source_kind="issues")
    registry.register(adapter)

    source = make_source(source_kind="issues")
    resolved = registry.resolve(source=source)

    assert resolved is adapter
    assert registry.registered_keys() == (
        ("example", IntegrationCategory.ISSUE_TRACKER, "issues"),
    )


@pytest.mark.unit
def test_two_source_kinds_same_provider_category() -> None:
    registry = KnowledgeAdapterRegistry()
    issues = FakeAdapter(source_kind="issues")
    comments = FakeAdapter(source_kind="comments")
    registry.register(issues)
    registry.register(comments)

    assert registry.resolve(source=make_source(source_kind="issues")) is issues
    assert registry.resolve(source=make_source(source_kind="comments")) is comments


@pytest.mark.unit
def test_duplicate_exact_key_rejected_without_overwrite() -> None:
    registry = KnowledgeAdapterRegistry()
    original = FakeAdapter(source_kind="issues")
    duplicate = FakeAdapter(source_kind="issues")
    registry.register(original)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(duplicate)

    assert registry.resolve(source=make_source(source_kind="issues")) is original


@pytest.mark.unit
def test_unknown_adapter_returns_adapter_not_found() -> None:
    registry = KnowledgeAdapterRegistry()
    source = make_source(source_kind="missing")

    with pytest.raises(VendorKnowledgeError) as exc_info:
        registry.resolve(source=source)

    error = exc_info.value
    assert error.code is VendorKnowledgeErrorCode.ADAPTER_NOT_FOUND
    assert error.retryable is False
    assert "scope" not in error.safe_message.lower()
    assert "connection" not in error.safe_message.lower()


@pytest.mark.unit
def test_no_global_registry_state_between_instances() -> None:
    first = KnowledgeAdapterRegistry()
    second = KnowledgeAdapterRegistry()
    first.register(FakeAdapter(source_kind="issues"))

    assert first.registered_keys()
    assert second.registered_keys() == ()

    with pytest.raises(VendorKnowledgeError) as exc_info:
        second.resolve(source=make_source(source_kind="issues"))
    assert exc_info.value.code is VendorKnowledgeErrorCode.ADAPTER_NOT_FOUND


@pytest.mark.unit
def test_registration_performs_no_integration_construction() -> None:
    registry = KnowledgeAdapterRegistry()
    integration = FakeIntegration(constructed=False)
    adapter = FakeAdapter()

    registry.register(adapter)

    assert integration.constructed is False
    assert adapter.inspect_calls == []
    assert adapter.read_calls == []
