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
        code=VendorKnowledgeErrorCode.AUTHENTICATION_FAILED,
        safe_message="Authentication failed for source",
        provider_id="example",
        source_kind="pages",
    )
    assert str(error) == "Authentication failed for source"
    assert "token" not in str(error).lower()


@pytest.mark.unit
def test_error_code_values() -> None:
    assert VendorKnowledgeErrorCode.CONFIGURATION_ERROR.value == "configuration_error"
    assert VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND.value == "integration_not_found"
    assert (
        VendorKnowledgeErrorCode.INTEGRATION_CATEGORY_MISMATCH.value
        == "integration_category_mismatch"
    )
    assert VendorKnowledgeErrorCode.ADAPTER_NOT_FOUND.value == "adapter_not_found"
    assert VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY.value == "unsupported_capability"
    assert VendorKnowledgeErrorCode.AUTHENTICATION_FAILED.value == "authentication_failed"
    assert VendorKnowledgeErrorCode.AUTHORIZATION_DENIED.value == "authorization_denied"
    assert VendorKnowledgeErrorCode.RATE_LIMITED.value == "rate_limited"
    assert VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE.value == "dependency_unavailable"
    assert VendorKnowledgeErrorCode.REMOTE_ITEM_NOT_FOUND.value == "remote_item_not_found"
    assert VendorKnowledgeErrorCode.REMOTE_ITEM_REVOKED.value == "remote_item_revoked"
    assert VendorKnowledgeErrorCode.INVALID_CURSOR.value == "invalid_cursor"
    assert VendorKnowledgeErrorCode.INVALID_SCOPE.value == "invalid_scope"
    assert VendorKnowledgeErrorCode.TENANT_MISMATCH.value == "tenant_mismatch"
    assert (
        VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE.value == "invalid_provider_response"
    )


@pytest.mark.unit
def test_retryable_defaults() -> None:
    rate = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.RATE_LIMITED,
        safe_message="Rate limited",
    )
    unavailable = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
        safe_message="Dependency unavailable",
    )
    authn = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.AUTHENTICATION_FAILED,
        safe_message="Auth failed",
    )
    authz = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.AUTHORIZATION_DENIED,
        safe_message="Forbidden",
    )
    config = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
        safe_message="Bad config",
    )
    unsupported = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.UNSUPPORTED_CAPABILITY,
        safe_message="Unsupported",
    )
    tenant = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.TENANT_MISMATCH,
        safe_message="Tenant mismatch",
    )
    invalid_scope = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.INVALID_SCOPE,
        safe_message="Invalid scope",
    )

    assert rate.retryable is True
    assert unavailable.retryable is True
    assert authn.retryable is False
    assert authz.retryable is False
    assert config.retryable is False
    assert unsupported.retryable is False
    assert tenant.retryable is False
    assert invalid_scope.retryable is False


@pytest.mark.unit
def test_rejects_unsafe_bearer_or_header_message() -> None:
    with pytest.raises(ValueError):
        VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.AUTHORIZATION_DENIED,
            safe_message="Authorization: Bearer abc.def.ghi",
        )
    with pytest.raises(ValueError):
        VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.AUTHENTICATION_FAILED,
            safe_message="api_key=super-secret",
        )


@pytest.mark.unit
def test_rejects_credential_bearing_url_message() -> None:
    with pytest.raises(ValueError):
        VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
            safe_message="Bad endpoint https://user:pass@example.test/item",
        )
    with pytest.raises(ValueError):
        VendorKnowledgeError(
            code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
            safe_message="See https://example.test/callback?access_token=abc",
        )


@pytest.mark.unit
def test_repr_does_not_include_message_body() -> None:
    message = "Authentication failed for source"
    error = VendorKnowledgeError(
        code=VendorKnowledgeErrorCode.AUTHENTICATION_FAILED,
        safe_message=message,
        provider_id="example",
        source_kind="pages",
    )
    rendered = repr(error)
    assert message not in rendered
    assert "safe_message" not in rendered
    assert "AUTHENTICATION_FAILED" in rendered or "authentication_failed" in rendered
    assert "example" in rendered
    assert "pages" in rendered


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
        code=VendorKnowledgeErrorCode.AUTHORIZATION_DENIED,
        safe_message="Access denied",
        provider_id="example",
        source_kind="drive",
        retryable=False,
    )
    text = f"{error!r} {error!s} {error.__dict__!r}".lower()
    assert "bearer " not in text
    assert "authorization:" not in text
    assert "api_key=" not in text
