# © Artur Czarnecki. All rights reserved.

"""Canonical secret-safe validation engine."""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError

import pytest

from intergrax.core.security import (
    FORBIDDEN_KEY,
    SECRET_LIKE_VALUE,
    SecretSafetyValidationError,
    SecretSafeValidationPolicy,
    is_secret_like_key,
    is_secret_like_value,
    validate_secret_safe_value,
)

pytestmark = pytest.mark.unit

_VALUE_PATTERN = re.compile(
    r"^(sk-|xox[baprs]-|Bearer\s|eyJ[A-Za-z0-9_-]+\.)",
    re.IGNORECASE,
)

_NAME_POLICY = SecretSafeValidationPolicy(
    forbidden_key_names=frozenset({"password", "api_key", "token", "credential"}),
    forbidden_key_suffixes=("_secret", "_token"),
    allowed_keys=frozenset({"credential_ref"}),
    split_key_segments=True,
    scan_string_values=False,
)

_FRAGMENT_POLICY = SecretSafeValidationPolicy(
    forbidden_key_fragments=frozenset({"token", "secret", "api_key", "credential"}),
    scan_string_values=False,
)

_VALUE_POLICY = SecretSafeValidationPolicy(
    forbidden_key_names=frozenset({"password"}),
    forbidden_value_patterns=(_VALUE_PATTERN,),
    scan_string_values=True,
    traverse_sequences=True,
)

_NO_VALUE_SCAN_POLICY = SecretSafeValidationPolicy(
    forbidden_key_names=frozenset({"password"}),
    forbidden_value_patterns=(_VALUE_PATTERN,),
    scan_string_values=False,
)


def test_exact_forbidden_key() -> None:
    assert is_secret_like_key("password", policy=_NAME_POLICY)
    assert is_secret_like_key("API_KEY", policy=_NAME_POLICY)


def test_suffix_forbidden_key() -> None:
    assert is_secret_like_key("client_secret", policy=_NAME_POLICY)
    assert is_secret_like_key("refresh_token", policy=_NAME_POLICY)


def test_segment_and_fragment_forbidden_key() -> None:
    assert is_secret_like_key("foo-token", policy=_NAME_POLICY)
    assert is_secret_like_key("tokenizer", policy=_FRAGMENT_POLICY)
    assert is_secret_like_key("secretary", policy=_FRAGMENT_POLICY)


def test_allowlisted_key_wins() -> None:
    assert not is_secret_like_key("credential_ref", policy=_NAME_POLICY)
    validate_secret_safe_value(
        {"credential_ref": "vault://item/1"},
        policy=_NAME_POLICY,
    )


def test_safe_key_accepted() -> None:
    assert not is_secret_like_key("region", policy=_NAME_POLICY)
    assert not is_secret_like_key("tokenizer", policy=_NAME_POLICY)
    assert not is_secret_like_key("secretary", policy=_NAME_POLICY)
    validate_secret_safe_value({"region": "eu-west-1"}, policy=_NAME_POLICY)


def test_secret_like_value_detected_when_policy_scans_values() -> None:
    assert is_secret_like_value("sk-live-example", policy=_VALUE_POLICY)
    with pytest.raises(SecretSafetyValidationError) as exc_info:
        validate_secret_safe_value({"note": "sk-live-example"}, policy=_VALUE_POLICY)
    assert exc_info.value.reason_code == SECRET_LIKE_VALUE
    assert "sk-live-example" not in str(exc_info.value)


def test_secret_like_value_accepted_when_policy_does_not_scan_values() -> None:
    validate_secret_safe_value(
        {"documentation_uri": "sk-live-example"},
        policy=_NO_VALUE_SCAN_POLICY,
    )


def test_nested_mapping_path() -> None:
    with pytest.raises(SecretSafetyValidationError) as exc_info:
        validate_secret_safe_value(
            {"provider": {"api_key": "public"}},
            policy=_NAME_POLICY,
        )
    assert exc_info.value.reason_code == FORBIDDEN_KEY
    assert exc_info.value.path == "provider.api_key"


def test_nested_list_path() -> None:
    with pytest.raises(SecretSafetyValidationError) as exc_info:
        validate_secret_safe_value(
            {"items": [{"password": "x"}]},
            policy=_NAME_POLICY,
        )
    assert exc_info.value.path == "items[0].password"


def test_safe_errors_do_not_contain_secret_literal() -> None:
    secret = "sk-super-secret-literal"
    with pytest.raises(SecretSafetyValidationError) as exc_info:
        validate_secret_safe_value({"token": secret}, policy=_VALUE_POLICY)
    assert secret not in str(exc_info.value)
    assert secret not in exc_info.value.path
    assert secret not in exc_info.value.context_label


def test_policy_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        _NAME_POLICY.scan_string_values = True  # type: ignore[misc]
