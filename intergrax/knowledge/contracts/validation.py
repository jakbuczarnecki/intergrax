# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared JSON and credential-free validation primitives for knowledge contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, Sequence
from enum import Enum
from typing import Any
from urllib.parse import parse_qsl, urlparse

type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

_SECRET_KEY_NAMES: frozenset[str] = frozenset(
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
)

_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*$")

_SECRET_KEY_SUFFIXES: tuple[str, ...] = (
    "_token",
    "_password",
    "_secret",
    "_api_key",
    "_authorization",
    "_credential",
    "_bearer",
)

_ALLOWED_SECRET_LIKE_KEYS: frozenset[str] = frozenset({"credential_ref"})

_MUTATION_ERROR = "knowledge metadata is immutable"


def _normalize_key(key: str) -> str:
    return str(key).strip().lower()


def is_forbidden_secret_key(key: str) -> bool:
    normalized = _normalize_key(key)
    if not normalized:
        return False
    if normalized in _ALLOWED_SECRET_LIKE_KEYS:
        return False
    if normalized in _SECRET_KEY_NAMES:
        return True
    if any(normalized.endswith(suffix) for suffix in _SECRET_KEY_SUFFIXES):
        return True
    segments = [segment for segment in normalized.replace("-", "_").split("_") if segment]
    return any(segment in _SECRET_KEY_NAMES for segment in segments)


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
    parsed = urlparse(cleaned)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} must not embed credentials")
    for raw_key, _raw_value in parse_qsl(parsed.query, keep_blank_values=True):
        if is_forbidden_secret_key(raw_key):
            raise ValueError(
                f"{field_name} must not include secret-bearing query parameter '{raw_key}'"
            )
    return cleaned


def _validate_finite_float(value: float, *, field_name: str, path: str) -> float:
    if not math.isfinite(value):
        label = path.rstrip(".") if path else field_name
        raise ValueError(f"{field_name} must not contain non-finite float at '{label}'")
    return value


def validate_json_value(value: object, *, field_name: str, path: str = "") -> JsonValue:
    if isinstance(value, Enum):
        raise ValueError(f"{field_name} must contain JSON-compatible values at '{path.rstrip('.')}'")

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        if is_url_like(value):
            label = path.rstrip(".") if path else field_name
            validate_safe_url(value, field_name=f"{field_name} value '{label}'")
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return _validate_finite_float(value, field_name=field_name, path=path)
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                child_path = path or field_name
                raise ValueError(f"{field_name} keys must be strings at '{child_path}'")
            key = raw_key
            if is_forbidden_secret_key(key):
                raise ValueError(
                    f"{field_name} must not contain secret-bearing key '{path + key}'"
                )
            result[key] = validate_json_value(
                child,
                field_name=field_name,
                path=f"{path}{key}.",
            )
        return result
    if isinstance(value, list):
        return [
            validate_json_value(
                child,
                field_name=field_name,
                path=f"{path}[{index}].",
            )
            for index, child in enumerate(value)
        ]

    child_path = path.rstrip(".") if path else field_name
    raise ValueError(f"{field_name} must contain JSON-compatible values at '{child_path}'")


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


def freeze_knowledge_metadata(value: dict[str, JsonValue]) -> _FrozenJsonObject:
    return _FrozenJsonObject(value)
