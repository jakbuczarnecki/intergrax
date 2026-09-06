"""Strict JSON field decoding for frozen data pack contracts."""

from __future__ import annotations

from pathlib import PurePosixPath

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackFormatError,
)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def require_mapping(payload: object, *, field_name: str) -> dict[str, JsonValue]:
    if not isinstance(payload, dict):
        raise VpiDataPackFormatError(f"{field_name} must be a JSON object")
    return payload


def require_known_keys(
    payload: dict[str, JsonValue],
    *,
    allowed: frozenset[str],
    context: str,
) -> None:
    unknown = sorted(set(payload.keys()) - allowed)
    if unknown:
        raise VpiDataPackFormatError(f"unexpected fields in {context}: {', '.join(unknown)}")


def require_str(payload: dict[str, JsonValue], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise VpiDataPackFormatError(f"{key} must be a string")
    if not value.strip():
        raise VpiDataPackFormatError(f"{key} must be a non-empty string")
    return value


def require_relative_path(payload: dict[str, JsonValue], key: str) -> str:
    value = require_str(payload, key)
    path = PurePosixPath(value)
    if path.is_absolute():
        raise VpiDataPackFormatError(f"{key} must be a relative path")
    if path.drive:
        raise VpiDataPackFormatError(f"{key} must be a relative path")
    if ".." in path.parts:
        raise VpiDataPackFormatError(f"{key} must not contain parent traversal")
    if not path.parts or any(part in {"", "."} for part in path.parts):
        raise VpiDataPackFormatError(f"{key} must be a non-empty relative path")
    return value


def require_optional_str(payload: dict[str, JsonValue], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise VpiDataPackFormatError(f"{key} must be a string or null")
    if not value.strip():
        raise VpiDataPackFormatError(f"{key} must be a non-empty string when present")
    return value


def require_int(payload: dict[str, JsonValue], key: str, *, minimum: int = 0) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise VpiDataPackFormatError(f"{key} must be an integer")
    if value < minimum:
        raise VpiDataPackFormatError(f"{key} must be >= {minimum}")
    return value


def require_sha256_hex(payload: dict[str, JsonValue], key: str) -> str:
    value = require_str(payload, key)
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise VpiDataPackFormatError(f"{key} must be a 64-character lowercase sha256 hex digest")
    return normalized


def require_str_list(payload: dict[str, JsonValue], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise VpiDataPackFormatError(f"{key} must be a list")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise VpiDataPackFormatError(f"{key}[{index}] must be a string")
        items.append(item)
    return tuple(items)
