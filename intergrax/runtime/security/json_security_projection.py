# © Artur Czarnecki. All rights reserved.

"""Deterministic JSON → textual projection for token scanning (security-local; not middleware ABI)."""

from __future__ import annotations

from intergrax.contracts.structured_json_value import JsonObject, JsonValue


def json_value_to_security_scan_fragment(value: JsonValue) -> str:
    """Stable, recursive text for substring token scans; does not mutate the source value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        parts = [json_value_to_security_scan_fragment(item) for item in value]
        return " ".join(parts)
    parts = [
        f"{key}={json_value_to_security_scan_fragment(child)}"
        for key, child in sorted(value.items())
    ]
    return " ".join(parts)


def json_object_to_security_scan_text(arguments: JsonObject) -> str:
    """Flatten a tool-argument object into one deterministic scan blob."""
    if not arguments:
        return ""
    parts = [
        f"{key}={json_value_to_security_scan_fragment(child)}"
        for key, child in sorted(arguments.items())
    ]
    return " ".join(parts)


def json_object_to_string_argument_map(arguments: JsonObject) -> dict[str, str]:
    """Per-key scan fragments for ``ToolInvocationRequest`` token policy (top-level keys preserved)."""
    return {
        key: json_value_to_security_scan_fragment(child)
        for key, child in sorted(arguments.items())
    }
