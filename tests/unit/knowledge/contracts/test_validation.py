# © Artur Czarnecki. All rights reserved.

"""Knowledge secret-safe metadata and URL validation."""

from __future__ import annotations

import pytest

from intergrax.knowledge.contracts.validation import (
    KNOWLEDGE_SECRET_POLICY,
    assert_safe_mapping,
    freeze_knowledge_metadata,
    validate_json_value,
    validate_safe_url,
)
from intergrax.core.security import is_secret_like_key

pytestmark = pytest.mark.unit


def test_access_token_rejected() -> None:
    with pytest.raises(ValueError, match="secret-bearing key"):
        assert_safe_mapping({"access_token": "x"}, field_name="metadata")


def test_password_rejected() -> None:
    with pytest.raises(ValueError, match="secret-bearing key"):
        assert_safe_mapping({"password": "x"}, field_name="metadata")


def test_nested_secret_key_rejected() -> None:
    with pytest.raises(ValueError, match="secret-bearing key"):
        validate_json_value({"items": [{"api_key": "x"}]}, field_name="metadata")


def test_credential_ref_accepted() -> None:
    assert not is_secret_like_key("credential_ref", policy=KNOWLEDGE_SECRET_POLICY)
    result = assert_safe_mapping(
        {"credential_ref": "vault://item/1", "region": "eu"},
        field_name="metadata",
    )
    assert result["credential_ref"] == "vault://item/1"


def test_false_positive_control_for_knowledge_policy() -> None:
    assert not is_secret_like_key("tokenizer", policy=KNOWLEDGE_SECRET_POLICY)
    assert not is_secret_like_key("secretary", policy=KNOWLEDGE_SECRET_POLICY)
    assert is_secret_like_key("access_token", policy=KNOWLEDGE_SECRET_POLICY)
    assert is_secret_like_key("client_secret", policy=KNOWLEDGE_SECRET_POLICY)
    assert is_secret_like_key("api_key", policy=KNOWLEDGE_SECRET_POLICY)
    assert not is_secret_like_key("api-key", policy=KNOWLEDGE_SECRET_POLICY)


def test_url_embedded_username_password_rejected() -> None:
    with pytest.raises(ValueError, match="must not embed credentials"):
        validate_safe_url("https://user:pass@example.test/item", field_name="web_url")


def test_url_secret_query_key_rejected() -> None:
    with pytest.raises(ValueError, match="secret-bearing query parameter"):
        validate_safe_url("https://example.test/item?access_token=abc", field_name="web_url")


def test_safe_url_accepted() -> None:
    assert (
        validate_safe_url("https://example.test/item?page=1", field_name="web_url")
        == "https://example.test/item?page=1"
    )


def test_knowledge_freezing_unchanged() -> None:
    frozen = freeze_knowledge_metadata({"region": "eu", "nested": {"n": 1}})
    with pytest.raises(TypeError, match="immutable"):
        frozen["region"] = "us"  # type: ignore[index]
