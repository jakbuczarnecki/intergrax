# © Artur Czarnecki. All rights reserved.

"""Runtime-internal adapters between typed middleware payloads and legacy encryption helpers."""

from __future__ import annotations

from intergrax.contracts.middleware_hook_semantics import (
    DataProtectionHookPayload,
    DataProtectionRestrictedValue,
)


def data_protection_payload_to_encryption_dict(
    payload: DataProtectionHookPayload,
) -> dict[str, object]:
    """Isolate dict-shaped encryption evaluation inside Tier-1 (not middleware ABI)."""
    result: dict[str, object] = {}
    if payload.data_classification is not None:
        result["data_classification"] = payload.data_classification.value
    if payload.classification is not None:
        result["classification"] = payload.classification
    if payload.namespace is not None:
        result["namespace"] = payload.namespace
    if payload.key is not None:
        result["key"] = payload.key
    if payload.write_policy is not None:
        result["write_policy"] = payload.write_policy
    if payload.value is not None:
        result["value"] = _restricted_value_to_dict(payload.value)
    return result


def data_protection_from_memory_write_state(
    runtime_state: dict[str, object],
) -> DataProtectionHookPayload:
    """Build typed protection payload from internal memory-write hook state."""
    memory_write = runtime_state.get("memory_write")
    if isinstance(memory_write, dict):
        return _protection_from_mapping(memory_write)
    return _protection_from_mapping(runtime_state)


def _protection_from_mapping(raw: dict[str, object]) -> DataProtectionHookPayload:
    value_raw = raw.get("value")
    value: DataProtectionRestrictedValue | None = None
    if isinstance(value_raw, dict):
        value = DataProtectionRestrictedValue(
            data_classification=_coerce_classification(value_raw.get("data_classification")),
            classification=_as_optional_str(value_raw.get("classification")),
            secret=_as_optional_str(value_raw.get("secret")),
            payload=_as_optional_str(value_raw.get("payload")),
            content=_as_optional_str(value_raw.get("content")),
            data=_as_optional_str(value_raw.get("data")),
        )
    return DataProtectionHookPayload(
        data_classification=_coerce_classification(
            raw.get("data_classification") or raw.get("classification"),
        ),
        classification=_as_optional_str(raw.get("classification")),
        namespace=_as_optional_str(raw.get("namespace")),
        key=_as_optional_str(raw.get("key")),
        write_policy=_as_optional_str(raw.get("write_policy")),
        value=value,
    )


def _restricted_value_to_dict(value: DataProtectionRestrictedValue) -> dict[str, object]:
    out: dict[str, object] = {}
    if value.data_classification is not None:
        out["data_classification"] = value.data_classification.value
    if value.classification is not None:
        out["classification"] = value.classification
    if value.secret is not None:
        out["secret"] = value.secret
    if value.payload is not None:
        out["payload"] = value.payload
    if value.content is not None:
        out["content"] = value.content
    if value.data is not None:
        out["data"] = value.data
    return out


def _as_optional_str(raw: object) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _coerce_classification(raw: object) -> DataClassification | None:
    from intergrax.contracts.data_classification import DataClassification

    if raw is None:
        return None
    try:
        return DataClassification(str(raw).lower())
    except ValueError:
        return None
