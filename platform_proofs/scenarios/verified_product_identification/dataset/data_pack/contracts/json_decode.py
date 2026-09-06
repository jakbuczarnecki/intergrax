"""Strict JSON field decoding for frozen data pack contracts."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackFormatError,
)


def require_mapping(payload: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise VpiDataPackFormatError(f"{field_name} must be a JSON object")
    return payload


def require_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise VpiDataPackFormatError(f"{key} must be a string")
    if not value.strip():
        raise VpiDataPackFormatError(f"{key} must be a non-empty string")
    return value


def require_optional_str(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise VpiDataPackFormatError(f"{key} must be a string or null")
    if not value.strip():
        raise VpiDataPackFormatError(f"{key} must be a non-empty string when present")
    return value


def require_int(payload: dict[str, object], key: str, *, minimum: int = 0) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise VpiDataPackFormatError(f"{key} must be an integer")
    if value < minimum:
        raise VpiDataPackFormatError(f"{key} must be >= {minimum}")
    return value


def require_str_list(payload: dict[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise VpiDataPackFormatError(f"{key} must be a list")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise VpiDataPackFormatError(f"{key}[{index}] must be a string")
        items.append(item)
    return tuple(items)
