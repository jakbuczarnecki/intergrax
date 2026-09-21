# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared JSON and credential-free validation primitives for knowledge contracts."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from intergrax.contracts.structured_json_value import (
    JsonObject,
    JsonPrimitive,
    JsonValue,
    validate_structured_json_value,
)
from intergrax.core.security import (
    CREDENTIAL_IN_URL,
    SECRET_QUERY_PARAMETER,
    SecretSafetyValidationError,
    SecretSafeValidationPolicy,
    is_secret_like_key,
    validate_secret_safe_url,
)

KNOWLEDGE_SECRET_POLICY = SecretSafeValidationPolicy(
    forbidden_key_names=frozenset(
        {
            "token",
            "access_token",
            "refresh_token",
            "password",
            "secret",
            "api_key",
            "authorization",
            "credential",
            "bearer",
        }
    ),
    forbidden_key_suffixes=(
        "_token",
        "_password",
        "_secret",
        "_api_key",
        "_authorization",
        "_credential",
        "_bearer",
    ),
    allowed_keys=frozenset({"credential_ref"}),
    split_key_segments=True,
    scan_string_values=False,
)

_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*$")

_MUTATION_ERROR = "knowledge metadata is immutable"


def is_url_like(value: str) -> bool:
    cleaned = value.strip()
    if "://" not in cleaned:
        return False
    scheme, _, rest = cleaned.partition("://")
    if not scheme or not rest:
        return False
    return _URL_SCHEME_RE.fullmatch(scheme) is not None


def require_non_empty_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def require_non_empty_trimmed_str(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def validate_safe_url(url: str, *, field_name: str) -> str:
    cleaned = url.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string when provided")
    try:
        return validate_secret_safe_url(
            cleaned,
            field_name=field_name,
            policy=KNOWLEDGE_SECRET_POLICY,
        )
    except SecretSafetyValidationError as exc:
        if exc.reason_code == CREDENTIAL_IN_URL:
            raise ValueError(f"{field_name} must not embed credentials") from exc
        if exc.reason_code == SECRET_QUERY_PARAMETER:
            raise ValueError(
                f"{field_name} must not include secret-bearing query parameter '{exc.path}'"
            ) from exc
        raise ValueError(str(exc)) from exc


def _enforce_knowledge_metadata_policies(
    value: JsonValue,
    *,
    field_name: str,
    path: str = "",
) -> JsonValue:
    if isinstance(value, dict):
        for key, child in value.items():
            if is_secret_like_key(key, policy=KNOWLEDGE_SECRET_POLICY):
                raise ValueError(
                    f"{field_name} must not contain secret-bearing key '{path + key}'"
                )
            _enforce_knowledge_metadata_policies(
                child,
                field_name=field_name,
                path=f"{path}{key}.",
            )
        return value
    if isinstance(value, list):
        for index, child in enumerate(value):
            _enforce_knowledge_metadata_policies(
                child,
                field_name=field_name,
                path=f"{path}[{index}].",
            )
        return value
    if isinstance(value, str) and is_url_like(value):
        label = path.rstrip(".") if path else field_name
        validate_safe_url(value, field_name=f"{field_name} value '{label}'")
    return value


def _reject_knowledge_legacy_tuple_containers(
    value: object,
    *,
    field_name: str,
    path: str = "",
) -> None:
    """Reject tuple containers anywhere (historical Knowledge admission before platform delegation)."""
    if isinstance(value, tuple):
        label = path.rstrip(".") if path else field_name
        raise ValueError(f"{field_name} must contain JSON-compatible values at '{label}'")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_knowledge_legacy_tuple_containers(
                child,
                field_name=field_name,
                path=f"{path}{key}.",
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_knowledge_legacy_tuple_containers(
                child,
                field_name=field_name,
                path=f"{path}[{index}].",
            )


def validate_json_value(value: object, *, field_name: str, path: str = "") -> JsonValue:
    """Validate JSON structure, then apply Knowledge secret and URL safety policies."""
    _reject_knowledge_legacy_tuple_containers(value, field_name=field_name, path=path)
    structured = validate_structured_json_value(value, field_name=field_name, path=path)
    return _enforce_knowledge_metadata_policies(structured, field_name=field_name, path=path)


def assert_safe_mapping(value: Mapping[str, Any], *, field_name: str) -> dict[str, JsonValue]:
    validated = validate_json_value(value, field_name=field_name)
    if not isinstance(validated, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return validated


def assert_knowledge_metadata(
    value: Mapping[str, Any],
    *,
    field_name: str,
    reserved_keys: frozenset[str],
) -> dict[str, JsonValue]:
    for key in value:
        if key in reserved_keys:
            raise ValueError(f"{field_name} must not contain reserved key '{key}'")
    return assert_safe_mapping(value, field_name=field_name)


def _json_value_to_plain(value: JsonValue) -> JsonValue:
    if isinstance(value, _FrozenJsonObject):
        return value.to_plain()
    if isinstance(value, _FrozenJsonArray):
        return value.to_plain()
    if isinstance(value, dict):
        return {key: _json_value_to_plain(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_value_to_plain(child) for child in value]
    return value


class _FrozenJsonArray(Sequence[JsonValue]):
    __slots__ = ("_items",)

    def __init__(self, items: list[JsonValue]) -> None:
        object.__setattr__(
            self,
            "_items",
            tuple(_freeze_json_value(item) for item in items),
        )

    def __getitem__(self, index: int | slice) -> JsonValue | _FrozenJsonArray:
        result = self._items[index]
        if isinstance(index, slice):
            return _FrozenJsonArray(list(result))
        return result

    def __iter__(self) -> Iterator[JsonValue]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FrozenJsonArray):
            return self._items == other._items
        if isinstance(other, list):
            return self.to_plain() == other
        return NotImplemented

    def __repr__(self) -> str:
        return repr(self.to_plain())

    def to_plain(self) -> list[JsonValue]:
        return [_json_value_to_plain(item) for item in self._items]

    def __setitem__(self, index: int | slice, value: object) -> None:
        raise TypeError(_MUTATION_ERROR)

    def __delitem__(self, index: int | slice) -> None:
        raise TypeError(_MUTATION_ERROR)

    def append(self, value: object) -> None:
        raise TypeError(_MUTATION_ERROR)

    def extend(self, values: object) -> None:
        raise TypeError(_MUTATION_ERROR)

    def insert(self, index: int, value: object) -> None:
        raise TypeError(_MUTATION_ERROR)

    def remove(self, value: object) -> None:
        raise TypeError(_MUTATION_ERROR)

    def pop(self, index: int = -1) -> None:
        raise TypeError(_MUTATION_ERROR)

    def clear(self) -> None:
        raise TypeError(_MUTATION_ERROR)

    def sort(self, *args: object, **kwargs: object) -> None:
        raise TypeError(_MUTATION_ERROR)

    def reverse(self) -> None:
        raise TypeError(_MUTATION_ERROR)

    def __iadd__(self, other: object) -> _FrozenJsonArray:
        raise TypeError(_MUTATION_ERROR)


class _FrozenJsonObject(Mapping[str, JsonValue]):
    __slots__ = ("_items",)

    def __init__(self, value: dict[str, JsonValue]) -> None:
        object.__setattr__(
            self,
            "_items",
            {key: _freeze_json_value(child) for key, child in value.items()},
        )

    def __getitem__(self, key: str) -> JsonValue:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FrozenJsonObject):
            return self._items == other._items
        if isinstance(other, Mapping):
            return self.to_plain() == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        return repr(self.to_plain())

    def keys(self) -> Iterator[str]:
        return iter(self._items.keys())

    def values(self) -> Iterator[JsonValue]:
        return iter(self._items.values())

    def items(self) -> Iterator[tuple[str, JsonValue]]:
        return iter(self._items.items())

    def get(self, key: str, default: JsonValue | None = None) -> JsonValue | None:
        return self._items.get(key, default)

    def to_plain(self) -> dict[str, JsonValue]:
        return {key: _json_value_to_plain(child) for key, child in self._items.items()}

    def __setitem__(self, key: str, value: object) -> None:
        raise TypeError(_MUTATION_ERROR)

    def __delitem__(self, key: str) -> None:
        raise TypeError(_MUTATION_ERROR)

    def update(self, *args: object, **kwargs: object) -> None:
        raise TypeError(_MUTATION_ERROR)

    def clear(self) -> None:
        raise TypeError(_MUTATION_ERROR)

    def pop(self, key: str, default: object = ...) -> None:
        raise TypeError(_MUTATION_ERROR)

    def popitem(self) -> None:
        raise TypeError(_MUTATION_ERROR)

    def setdefault(self, key: str, default: object = None) -> None:
        raise TypeError(_MUTATION_ERROR)

    def __ior__(self, other: object) -> _FrozenJsonObject:
        raise TypeError(_MUTATION_ERROR)


def _freeze_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return _FrozenJsonObject(value)
    if isinstance(value, list):
        return _FrozenJsonArray(value)
    return value


def knowledge_metadata_to_plain(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    if isinstance(value, _FrozenJsonObject):
        return value.to_plain()
    return {key: _json_value_to_plain(child) for key, child in value.items()}


def freeze_knowledge_metadata(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    if isinstance(value, _FrozenJsonObject):
        return value
    return _FrozenJsonObject(dict(value))
