# © Artur Czarnecki. All rights reserved.

"""Agent Distribution secret-safe config validation."""

from __future__ import annotations

import pytest

from intergrax.agent_distribution._config_validation import (
    validate_non_secret_distribution_config,
)

pytestmark = pytest.mark.unit


def test_rejects_secret_config_key() -> None:
    with pytest.raises(ValueError, match="secret_refs"):
        validate_non_secret_distribution_config({"api_key": "public"})


def test_rejects_nested_secret_key() -> None:
    with pytest.raises(ValueError, match="provider.api_key"):
        validate_non_secret_distribution_config({"provider": {"api_key": "public"}})


def test_rejects_nested_list_secret_key() -> None:
    with pytest.raises(ValueError, match=r"items\[0\].password"):
        validate_non_secret_distribution_config({"items": [{"password": "x"}]})


def test_rejects_sk_literal() -> None:
    with pytest.raises(ValueError, match="secret literal"):
        validate_non_secret_distribution_config({"note": "sk-live-not-a-ref"})


def test_rejects_bearer_literal() -> None:
    with pytest.raises(ValueError, match="secret literal"):
        validate_non_secret_distribution_config({"note": "Bearer abc.def"})


def test_rejects_jwt_like_literal() -> None:
    with pytest.raises(ValueError, match="secret literal"):
        validate_non_secret_distribution_config({"note": "eyJhbGciOiJIUzI1NiJ9.payload"})


def test_rejects_nested_mapping_list_secret_value() -> None:
    with pytest.raises(ValueError, match="secret literal"):
        validate_non_secret_distribution_config(
            {"provider": [{"endpoint": "sk-live-not-a-ref"}]}
        )


def test_accepts_safe_config() -> None:
    result = validate_non_secret_distribution_config(
        {"provider": {"region": "eu-west-1", "limits": {"rpm": 120}}}
    )
    assert result["provider"]["region"] == "eu-west-1"


def test_list_of_safe_scalars_remains_accepted() -> None:
    result = validate_non_secret_distribution_config({"labels": ["prod", "search"]})
    assert result["labels"] == ["prod", "search"]


def test_secret_looking_string_in_list_is_not_treated_as_mapping_value() -> None:
    result = validate_non_secret_distribution_config({"notes": ["sk-live-not-a-ref"]})
    assert result["notes"] == ["sk-live-not-a-ref"]
