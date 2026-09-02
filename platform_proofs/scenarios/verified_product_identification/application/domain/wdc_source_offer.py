"""Typed WDC source offer — provider-neutral source payload boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.domain.json_value import (
    JsonObject,
    JsonValue,
)


def _normalize_json_object(value: dict[object, object]) -> JsonObject:
    normalized: JsonObject = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            msg = "JSON object keys must be strings"
            raise ValueError(msg)
        normalized[raw_key] = _normalize_json_value(raw_value)
    return normalized


def _normalize_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        return _normalize_json_object(value)
    msg = f"unsupported JSON value type: {type(value).__name__}"
    raise ValueError(msg)


def parse_wdc_record_json(record_json: str) -> JsonObject:
    """Parse one lossless ``record_json`` row into a bounded JSON object."""
    parsed = json.loads(record_json)
    if not isinstance(parsed, dict):
        msg = "top-level JSON value must be an object"
        raise ValueError(msg)
    return _normalize_json_object(parsed)


@dataclass(frozen=True, slots=True)
class WdcIdentifierEntry:
    """One typed identifier entry from a WDC ``identifiers`` list item."""

    source_key: str
    source_value: str


@dataclass(frozen=True, slots=True)
class WdcKeyValuePair:
    """One key/value attribute from WDC ``keyValuePairs``."""

    source_key: str
    source_value: str
    raw_value: JsonValue


@dataclass(frozen=True, slots=True)
class WdcSourceOffer:
    """Minimal typed WDC source offer used for deterministic search derivation."""

    offer_id: str
    cluster_id: int | None
    category: str | None
    identifiers: tuple[WdcIdentifierEntry, ...]
    title: str | None
    description: str | None
    brand: str | None
    price: str | None
    key_value_pairs: tuple[WdcKeyValuePair, ...]
    spec_table_content: str | None


def _optional_non_empty_string(value: JsonValue) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _optional_int(value: JsonValue) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _offer_id_from_record(record: JsonObject) -> str:
    raw_id = record.get("id")
    if isinstance(raw_id, bool):
        msg = "WDC source offer id must not be a boolean"
        raise ValueError(msg)
    if isinstance(raw_id, int):
        return str(raw_id)
    if isinstance(raw_id, str) and raw_id.strip():
        return raw_id.strip()
    msg = "WDC source offer requires a non-empty id"
    raise ValueError(msg)


def _parse_identifiers(record: JsonObject) -> tuple[WdcIdentifierEntry, ...]:
    raw_identifiers = record.get("identifiers")
    if raw_identifiers is None:
        return ()
    if not isinstance(raw_identifiers, list):
        return ()

    entries: list[WdcIdentifierEntry] = []
    for item in raw_identifiers:
        if not isinstance(item, dict):
            continue
        for source_key, raw_value in item.items():
            if not isinstance(source_key, str) or not source_key.strip():
                continue
            source_value = _identifier_value_to_string(raw_value)
            if source_value is None:
                continue
            entries.append(
                WdcIdentifierEntry(
                    source_key=source_key,
                    source_value=source_value,
                )
            )
    return tuple(entries)


def _identifier_value_to_string(value: JsonValue) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    return None


def _parse_key_value_pairs(record: JsonObject) -> tuple[WdcKeyValuePair, ...]:
    raw_pairs = record.get("keyValuePairs")
    if raw_pairs is None or not isinstance(raw_pairs, dict) or not raw_pairs:
        return ()

    pairs: list[WdcKeyValuePair] = []
    for source_key, raw_value in raw_pairs.items():
        if not isinstance(source_key, str) or not source_key.strip():
            continue
        source_value = _attribute_value_to_string(raw_value)
        if source_value is None:
            continue
        pairs.append(
            WdcKeyValuePair(
                source_key=source_key,
                source_value=source_value,
                raw_value=raw_value,
            )
        )
    return tuple(sorted(pairs, key=lambda pair: pair.source_key.casefold()))


def _attribute_value_to_string(value: JsonValue) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    return None


def parse_wdc_source_offer(record: JsonObject) -> WdcSourceOffer:
    """Parse one bounded WDC JSON object into a typed source offer."""
    return WdcSourceOffer(
        offer_id=_offer_id_from_record(record),
        cluster_id=_optional_int(record.get("cluster_id")),
        category=_optional_non_empty_string(record.get("category")),
        identifiers=_parse_identifiers(record),
        title=_optional_non_empty_string(record.get("title")),
        description=_optional_non_empty_string(record.get("description")),
        brand=_optional_non_empty_string(record.get("brand")),
        price=_optional_non_empty_string(record.get("price")),
        key_value_pairs=_parse_key_value_pairs(record),
        spec_table_content=_optional_non_empty_string(record.get("specTableContent")),
    )


def parse_wdc_source_offer_json(record_json: str) -> WdcSourceOffer:
    """Parse one ``record_json`` payload into a typed WDC source offer."""
    return parse_wdc_source_offer(parse_wdc_record_json(record_json))
