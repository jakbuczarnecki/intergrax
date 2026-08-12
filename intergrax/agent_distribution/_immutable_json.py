# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Recursively immutable JSON-like values for Agent Distribution contracts."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from enum import Enum
from typing import Any

type JsonPrimitive = str | int | float | bool | None
type DistributionJsonValue = JsonPrimitive | list[DistributionJsonValue] | dict[str, DistributionJsonValue]
type DistributionJsonObject = dict[str, DistributionJsonValue]

_MUTATION_ERROR = "agent distribution config is immutable"


def _validate_finite_float(value: float, *, field_name: str, path: str) -> float:
    if not math.isfinite(value):
        label = path.rstrip(".") if path else field_name
        raise ValueError(f"{field_name} must not contain non-finite float at '{label}'")
    return value


def validate_distribution_json_value(
    value: object,
    *,
    field_name: str,
    path: str = "",
) -> DistributionJsonValue:
    """Validate and normalize one bounded JSON-compatible distribution value."""
    if isinstance(value, Enum):
        raise ValueError(
            f"{field_name} must contain JSON-compatible values at '{path.rstrip('.')}'"
        )

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        return _validate_finite_float(value, field_name=field_name, path=path)
    if isinstance(value, Mapping):
        result: dict[str, DistributionJsonValue] = {}
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                child_path = path or field_name
                raise ValueError(f"{field_name} keys must be strings at '{child_path}'")
            result[raw_key] = validate_distribution_json_value(
                child,
                field_name=field_name,
                path=f"{path}{raw_key}.",
            )
        return result
    if isinstance(value, list):
        return [
            validate_distribution_json_value(
                child,
                field_name=field_name,
                path=f"{path}[{index}].",
            )
            for index, child in enumerate(value)
        ]

    child_path = path.rstrip(".") if path else field_name
    raise ValueError(f"{field_name} must contain JSON-compatible values at '{child_path}'")


def assert_distribution_json_object(
    value: Mapping[str, Any],
    *,
    field_name: str,
) -> DistributionJsonObject:
    validated = validate_distribution_json_value(value, field_name=field_name)
    if not isinstance(validated, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return validated


def _json_value_to_plain(value: DistributionJsonValue) -> DistributionJsonValue:
    if isinstance(value, _FrozenJsonObject):
        return value.to_plain()
    if isinstance(value, _FrozenJsonArray):
        return value.to_plain()
    if isinstance(value, dict):
        return {key: _json_value_to_plain(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_value_to_plain(child) for child in value]
    return value


class _FrozenJsonArray(Sequence[DistributionJsonValue]):
    __slots__ = ("_items",)

    def __init__(self, items: list[DistributionJsonValue]) -> None:
        object.__setattr__(
            self,
            "_items",
            tuple(_freeze_json_value(item) for item in items),
        )

    def __getitem__(self, index: int | slice) -> DistributionJsonValue | _FrozenJsonArray:
        result = self._items[index]
        if isinstance(index, slice):
            return _FrozenJsonArray(list(result))
        return result

    def __iter__(self) -> Iterator[DistributionJsonValue]:
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

    def to_plain(self) -> list[DistributionJsonValue]:
        return [_json_value_to_plain(item) for item in self._items]

    def __setitem__(self, index: int | slice, value: object) -> None:
        raise TypeError(_MUTATION_ERROR)

    def __delitem__(self, index: int | slice) -> None:
        raise TypeError(_MUTATION_ERROR)

    def append(self, value: object) -> None:
        raise TypeError(_MUTATION_ERROR)


class _FrozenJsonObject(Mapping[str, DistributionJsonValue]):
    __slots__ = ("_items",)

    def __init__(self, value: dict[str, DistributionJsonValue]) -> None:
        object.__setattr__(
            self,
            "_items",
            {key: _freeze_json_value(child) for key, child in value.items()},
        )

    def __getitem__(self, key: str) -> DistributionJsonValue:
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

    def values(self) -> Iterator[DistributionJsonValue]:
        return iter(self._items.values())

    def items(self) -> Iterator[tuple[str, DistributionJsonValue]]:
        return iter(self._items.items())

    def get(self, key: str, default: DistributionJsonValue | None = None) -> DistributionJsonValue | None:
        return self._items.get(key, default)

    def to_plain(self) -> dict[str, DistributionJsonValue]:
        return {key: _json_value_to_plain(child) for key, child in self._items.items()}

    def __setitem__(self, key: str, value: object) -> None:
        raise TypeError(_MUTATION_ERROR)

    def __delitem__(self, key: str) -> None:
        raise TypeError(_MUTATION_ERROR)

    def update(self, *args: object, **kwargs: object) -> None:
        raise TypeError(_MUTATION_ERROR)


def _freeze_json_value(value: DistributionJsonValue) -> DistributionJsonValue:
    if isinstance(value, dict):
        return _FrozenJsonObject(value)
    if isinstance(value, list):
        return _FrozenJsonArray(value)
    return value


def distribution_json_to_plain(
    value: Mapping[str, DistributionJsonValue],
) -> dict[str, DistributionJsonValue]:
    if isinstance(value, _FrozenJsonObject):
        return value.to_plain()
    return {key: _json_value_to_plain(child) for key, child in value.items()}


def freeze_distribution_json_object(
    value: Mapping[str, DistributionJsonValue],
) -> Mapping[str, DistributionJsonValue]:
    if isinstance(value, _FrozenJsonObject):
        return value
    return _FrozenJsonObject(dict(value))
