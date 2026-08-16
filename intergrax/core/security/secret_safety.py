# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical secret-safe metadata/config detection (not a secret manager)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from re import Pattern
from typing import NoReturn
from urllib.parse import parse_qsl, urlparse

FORBIDDEN_KEY = "FORBIDDEN_KEY"
SECRET_LIKE_VALUE = "SECRET_LIKE_VALUE"
CREDENTIAL_IN_URL = "CREDENTIAL_IN_URL"
SECRET_QUERY_PARAMETER = "SECRET_QUERY_PARAMETER"


class SecretSafetyValidationError(ValueError):
    """Neutral secret-safety violation. Message contains no secret literals."""

    def __init__(
        self,
        message: str,
        *,
        path: str,
        reason_code: str,
        context_label: str,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.reason_code = reason_code
        self.context_label = context_label


@dataclass(frozen=True, slots=True)
class SecretSafeValidationPolicy:
    """Immutable domain-owned matching policy for secret-safe validation."""

    forbidden_key_names: frozenset[str] = frozenset()
    forbidden_key_fragments: frozenset[str] = frozenset()
    forbidden_key_suffixes: tuple[str, ...] = ()
    allowed_keys: frozenset[str] = frozenset()
    forbidden_value_patterns: tuple[Pattern[str], ...] = ()
    normalize_hyphens: bool = True
    split_key_segments: bool = False
    scan_string_values: bool = False
    traverse_sequences: bool = True


def normalize_metadata_key(key: str, *, normalize_hyphens: bool = True) -> str:
    """Trim, lowercase, and optionally fold hyphens to underscores."""
    normalized = key.strip().lower()
    if normalize_hyphens:
        return normalized.replace("-", "_")
    return normalized


def is_secret_like_key(key: str, *, policy: SecretSafeValidationPolicy) -> bool:
    canonical = key.strip().lower()
    if not canonical:
        return False
    if canonical in policy.allowed_keys:
        return False
    if canonical in policy.forbidden_key_names:
        return True
    if any(canonical.endswith(suffix) for suffix in policy.forbidden_key_suffixes):
        return True
    folded = normalize_metadata_key(canonical, normalize_hyphens=policy.normalize_hyphens)
    if policy.forbidden_key_fragments and any(
        fragment in folded for fragment in policy.forbidden_key_fragments
    ):
        return True
    if policy.split_key_segments:
        segments = [segment for segment in folded.split("_") if segment]
        if any(segment in policy.forbidden_key_names for segment in segments):
            return True
    return False


def is_secret_like_value(value: str, *, policy: SecretSafeValidationPolicy) -> bool:
    if not policy.forbidden_value_patterns:
        return False
    stripped = value.strip()
    return any(pattern.match(stripped) is not None for pattern in policy.forbidden_value_patterns)


def _child_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _sequence_path(path: str, index: int) -> str:
    return f"{path}[{index}]"


def _raise(
    *,
    path: str,
    reason_code: str,
    context_label: str,
    message: str,
) -> NoReturn:
    raise SecretSafetyValidationError(
        message,
        path=path,
        reason_code=reason_code,
        context_label=context_label,
    )


def validate_secret_safe_value(
    value: object,
    *,
    policy: SecretSafeValidationPolicy,
    path: str = "",
    context_label: str = "value",
) -> None:
    """Reject secret-like keys (and optionally string values) in mappings/sequences."""
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                key = str(raw_key)
            else:
                key = raw_key
            label = _child_path(path, key)
            if is_secret_like_key(key, policy=policy):
                _raise(
                    path=label,
                    reason_code=FORBIDDEN_KEY,
                    context_label=context_label,
                    message=f"{context_label} contains a forbidden secret-like key at '{label}'",
                )
            if (
                policy.scan_string_values
                and isinstance(child, str)
                and is_secret_like_value(child, policy=policy)
            ):
                _raise(
                    path=label,
                    reason_code=SECRET_LIKE_VALUE,
                    context_label=context_label,
                    message=f"{context_label} contains a secret-like value at '{label}'",
                )
            validate_secret_safe_value(
                child,
                policy=policy,
                path=label,
                context_label=context_label,
            )
        return
    if not policy.traverse_sequences:
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            validate_secret_safe_value(
                child,
                policy=policy,
                path=_sequence_path(path, index),
                context_label=context_label,
            )


def validate_secret_safe_url(
    url: str,
    *,
    field_name: str,
    policy: SecretSafeValidationPolicy,
) -> str:
    """Reject embedded URL credentials and secret-like query parameter names."""
    cleaned = url.strip()
    parsed = urlparse(cleaned)
    if parsed.username is not None or parsed.password is not None:
        _raise(
            path=field_name,
            reason_code=CREDENTIAL_IN_URL,
            context_label=field_name,
            message=f"{field_name} must not embed credentials",
        )
    for raw_key, _raw_value in parse_qsl(parsed.query, keep_blank_values=True):
        if is_secret_like_key(raw_key, policy=policy):
            _raise(
                path=raw_key,
                reason_code=SECRET_QUERY_PARAMETER,
                context_label=field_name,
                message=f"{field_name} must not include a secret-bearing query parameter",
            )
    return cleaned
