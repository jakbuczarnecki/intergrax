# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Ollama JSON Schema compatibility projection for constrained generation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

# Ollama grammar compilation fails when string maxLength is >= 2000.
OLLAMA_UNSUPPORTED_MAX_LENGTH_THRESHOLD = 2000

_NON_VALIDATION_METADATA_KEYS = frozenset(
    {"title", "description", "examples", "deprecated", "readOnly", "writeOnly"}
)


class RecursiveSchemaReferenceError(ValueError):
    """Raised when a JSON Schema contains unsupported recursive $ref chains."""


def prepare_ollama_generation_schema(output_model: type) -> dict[str, object]:
    """Build a provider-compatible generation schema from a Pydantic output model."""
    if not hasattr(output_model, "model_json_schema"):
        raise TypeError(f"{output_model!r} does not provide model_json_schema()")
    canonical_schema = output_model.model_json_schema()  # type: ignore[attr-defined]
    return project_json_schema_for_ollama(canonical_schema)


def project_json_schema_for_ollama(schema: Mapping[str, object]) -> dict[str, object]:
    """Return a deep-copied schema safe for Ollama grammar compilation."""
    projected = copy.deepcopy(dict(schema))
    _remove_unsupported_string_max_length(projected)
    return projected


def remove_non_validation_metadata(schema: Mapping[str, object]) -> dict[str, object]:
    """Remove JSON Schema keys that do not affect structural validation."""
    projected = copy.deepcopy(dict(schema))
    _remove_keys_recursive(projected, _NON_VALIDATION_METADATA_KEYS)
    return projected


def remove_defaults(schema: Mapping[str, object]) -> dict[str, object]:
    """Remove default values from a JSON Schema copy."""
    projected = copy.deepcopy(dict(schema))
    _remove_keys_recursive(projected, frozenset({"default"}))
    return projected


def remove_discriminator_metadata(schema: Mapping[str, object]) -> dict[str, object]:
    """Remove discriminator metadata while preserving branch constraints."""
    projected = copy.deepcopy(dict(schema))
    _remove_keys_recursive(projected, frozenset({"discriminator"}))
    return projected


def inline_refs(schema: Mapping[str, object]) -> dict[str, object]:
    """Resolve $defs/$ref into a fully inlined schema copy."""
    projected = copy.deepcopy(dict(schema))
    defs = projected.pop("$defs", {})
    if not isinstance(defs, dict):
        raise ValueError("$defs must be a mapping when present")
    return _resolve_refs(projected, defs, frozenset())


def convert_oneof_to_anyof(schema: Mapping[str, object]) -> dict[str, object]:
    """Convert oneOf branches to anyOf in a schema copy."""
    projected = copy.deepcopy(dict(schema))
    _convert_oneof_recursive(projected)
    return projected


def _remove_unsupported_string_max_length(
    node: Any,
    *,
    threshold: int = OLLAMA_UNSUPPORTED_MAX_LENGTH_THRESHOLD,
) -> None:
    if isinstance(node, dict):
        max_length = node.get("maxLength")
        if isinstance(max_length, int) and max_length >= threshold:
            del node["maxLength"]
        for value in node.values():
            _remove_unsupported_string_max_length(value, threshold=threshold)
    elif isinstance(node, list):
        for item in node:
            _remove_unsupported_string_max_length(item, threshold=threshold)


def _remove_keys_recursive(node: Any, keys: frozenset[str]) -> None:
    if isinstance(node, dict):
        for key in list(node.keys()):
            if key in keys:
                del node[key]
            else:
                _remove_keys_recursive(node[key], keys)
    elif isinstance(node, list):
        for item in node:
            _remove_keys_recursive(item, keys)


def _resolve_refs(node: Any, defs: dict[str, Any], resolving: frozenset[str]) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            if not ref.startswith("#/$defs/"):
                raise ValueError(f"unsupported $ref target: {ref}")
            def_name = ref[len("#/$defs/") :]
            if def_name in resolving:
                raise RecursiveSchemaReferenceError(
                    f"unsupported recursive schema reference: {def_name}"
                )
            if def_name not in defs:
                raise ValueError(f"unknown schema definition: {def_name}")
            return _resolve_refs(
                copy.deepcopy(defs[def_name]),
                defs,
                resolving | {def_name},
            )
        return {key: _resolve_refs(value, defs, resolving) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_refs(item, defs, resolving) for item in node]
    return node


def _convert_oneof_recursive(node: Any) -> None:
    if isinstance(node, dict):
        if "oneOf" in node:
            node["anyOf"] = node.pop("oneOf")
        for value in node.values():
            _convert_oneof_recursive(value)
    elif isinstance(node, list):
        for item in node:
            _convert_oneof_recursive(item)
