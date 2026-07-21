# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Internal JSON-safety and stable-identifier helpers for hosting contracts."""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import JsonValue
from pydantic.types import SecretBytes, SecretStr

from intergrax.utils import attribute_access

_BOUNDED_IDENTIFIER_MAX_LENGTH = 256
_INSTANCE_ID_MAX_LENGTH = 128
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9._:-]*$")


def validate_bounded_identifier(
    value: str,
    *,
    field_name: str,
    max_length: int = _BOUNDED_IDENTIFIER_MAX_LENGTH,
    pattern: re.Pattern[str] | None = _SAFE_IDENTIFIER_RE,
) -> str:
    """Validate a bounded diagnostic-safe identifier string."""
    identifier = value.strip()
    if not identifier:
        raise ValueError(f"{field_name} must not be empty")
    if len(identifier) > max_length:
        raise ValueError(f"{field_name} must be at most {max_length} characters")
    if any(character.isspace() or ord(character) < 32 for character in identifier):
        raise ValueError(f"{field_name} must not contain whitespace or control characters")
    if pattern is not None and not pattern.match(identifier):
        raise ValueError(f"{field_name} has invalid characters")
    return identifier


def validate_instance_id(value: str) -> str:
    """Validate a hosted application instance identifier."""
    instance_id = value.strip()
    if not instance_id:
        raise ValueError("instance_id must not be empty")
    if len(instance_id) > _INSTANCE_ID_MAX_LENGTH:
        raise ValueError(f"instance_id must be at most {_INSTANCE_ID_MAX_LENGTH} characters")
    if any(character.isspace() or ord(character) < 32 for character in instance_id):
        raise ValueError("instance_id must not contain whitespace or control characters")
    return instance_id


def validate_positive_bounded_seconds(
    value: float,
    *,
    field_name: str,
    max_seconds: float = 86_400.0,
) -> float:
    """Validate a positive bounded timeout/interval in seconds."""
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    if value > max_seconds:
        raise ValueError(f"{field_name} must be at most {max_seconds} seconds")
    return value


def validate_non_negative_bounded_seconds(
    value: float,
    *,
    field_name: str,
    max_seconds: float = 86_400.0,
) -> float:
    """Validate a non-negative bounded timeout in seconds."""
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    if value > max_seconds:
        raise ValueError(f"{field_name} must be at most {max_seconds} seconds")
    return value


def validate_bounded_priority(value: int, *, field_name: str = "priority") -> int:
    """Validate a bounded hook/subscription priority."""
    if value < -1_000 or value > 1_000:
        raise ValueError(f"{field_name} must be between -1000 and 1000")
    return value


def validate_json_value(value: object) -> JsonValue:
    """Validate and deep-copy a JSON-safe public value."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("public JSON values must contain only finite numbers")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, bytes | bytearray | memoryview):
        raise ValueError("public JSON values must not contain bytes")
    if isinstance(value, SecretStr | SecretBytes):
        raise ValueError("public JSON values must not contain secret wrappers")
    if isinstance(value, list):
        return [validate_json_value(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("public JSON object keys must be strings")
            normalized[key] = validate_json_value(item)
        return normalized
    raise ValueError("public JSON values must contain only JSON-safe data")


def normalize_public_json_mapping(metadata: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Validate and deep-copy a string-keyed public JSON mapping."""
    return {key: validate_json_value(value) for key, value in metadata.items()}


def deep_copy_public_json(value: JsonValue) -> JsonValue:
    """Return a validated deep copy of public JSON data."""
    return validate_json_value(value)


def canonical_public_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Encode public JSON deterministically for digests and comparisons."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def public_json_digest(payload: Mapping[str, Any]) -> str:
    """Return a deterministic sha256 digest for canonical public JSON."""
    digest = hashlib.sha256(canonical_public_json_bytes(payload)).hexdigest()
    return f"sha256:{digest}"


def derive_stable_callable_id(
    callback: Callable[..., object],
    *,
    field_name: str = "handler_id",
) -> str:
    """Derive a stable callable identifier from a module-level callable."""
    if isinstance(callback, functools.partial):
        raise ValueError(
            f"{field_name} is not reliably stable; provide an explicit stable identifier"
        )
    module = attribute_access.optional(callback, "__module__", None)
    qualname = attribute_access.optional(callback, "__qualname__", None)
    if not module or not qualname:
        raise ValueError(
            f"{field_name} is not reliably stable; provide an explicit stable identifier"
        )
    if qualname == "<lambda>" or "<locals>" in qualname:
        raise ValueError(
            f"{field_name} is not reliably stable; provide an explicit stable identifier"
        )
    return validate_bounded_identifier(f"{module}.{qualname}", field_name=field_name)


def derive_stable_type_id(
    value: object,
    *,
    field_name: str = "component_type_id",
) -> str:
    """Derive a stable type identifier from a class or runtime-checkable component."""
    candidate = value if isinstance(value, type) else type(value)
    module = attribute_access.optional(candidate, "__module__", None)
    qualname = attribute_access.optional(candidate, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname:
        raise ValueError(
            f"{field_name} is not reliably stable; provide an explicit stable identifier"
        )
    return validate_bounded_identifier(f"{module}.{qualname}", field_name=field_name)


def stable_service_type_id(service_type: type[object]) -> str:
    """Return a deterministic service type identifier for diagnostics."""
    module = attribute_access.optional(service_type, "__module__", "unknown")
    qualname = attribute_access.optional(service_type, "__qualname__", service_type.__name__)
    return f"{module}.{qualname}"
