# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for vendor-knowledge facade errors."""

from __future__ import annotations

import pytest

from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)


@pytest.mark.unit
def test_str_returns_safe_message_only() -> None:
    error = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.AUTHENTICATION,
        safe_message="Authentication failed for source",
        provider_id="example",
        source_kind="pages",
    )
    assert str(error) == "Authentication failed for source"
    assert "token" not in str(error).lower()


@pytest.mark.unit
def test_error_code_values() -> None:
    assert VendorKnowledgeErrorCode.INVALID_REQUEST.value == "invalid_request"
    assert VendorKnowledgeErrorCode.CONFIGURATION.value == "configuration"
    assert VendorKnowledgeErrorCode.UNKNOWN_SOURCE.value == "unknown_source"
    assert VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY.value == "unsupported_capability"
    assert VendorKnowledgeErrorCode.AUTHENTICATION.value == "authentication"
    assert VendorKnowledgeErrorCode.AUTHORIZATION.value == "authorization"
    assert VendorKnowledgeErrorCode.RATE_LIMIT.value == "rate_limit"
    assert VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE.value == "dependency_unavailable"
    assert VendorKnowledgeErrorCode.CURSOR_INVALID.value == "cursor_invalid"
    assert VendorKnowledgeErrorCode.ITEM_NOT_FOUND.value == "item_not_found"
    assert VendorKnowledgeErrorCode.ITEM_REVOKED.value == "item_revoked"
    assert VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE.value == "invalid_provider_response"


@pytest.mark.unit
def test_retryable_defaults() -> None:
    rate = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.RATE_LIMIT,
        safe_message="Rate limited",
    )
    unavailable = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
        safe_message="Dependency unavailable",
    )
    authn = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.AUTHENTICATION,
        safe_message="Auth failed",
    )
    authz = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.AUTHORIZATION,
        safe_message="Forbidden",
    )
    config = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.CONFIGURATION,
        safe_message="Bad config",
    )
    unsupported = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
        safe_message="Unsupported",
    )

    assert rate.retryable is True
    assert unavailable.retryable is True
    assert authn.retryable is False
    assert authz.retryable is False
    assert config.retryable is False
    assert unsupported.retryable is False


@pytest.mark.unit
def test_error_has_no_raw_response_body_attribute() -> None:
    error = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
        safe_message="Provider response invalid",
        provider_id="example",
        source_kind="pages",
    )
    assert not hasattr(error, "response_body")
    assert not hasattr(error, "raw_response")
    assert not hasattr(error, "body")


@pytest.mark.unit
def test_error_representation_has_no_token_or_secret() -> None:
    error = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.AUTHORIZATION,
        safe_message="Access denied",
        provider_id="example",
        source_kind="drive",
        retryable=False,
    )
    text = f"{error!r} {error!s} {error.__dict__!r}".lower()
    assert "token" not in text
    assert "secret" not in text
    assert "authorization: bearer" not in text
    assert "api_key" not in text
