# © Artur Czarnecki. All rights reserved.

"""Canonical strict tool argument validation at the adapter boundary (DS-E2E-15J-L0)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from intergrax.llm_adapters.contracts.strict_tool_arguments import (
    CanonicalFunctionToolDefinition,
    coerce_canonical_tool_definitions,
)
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall


class StrictToolContractValidationError(RuntimeError):
    """Tool call arguments failed canonical schema validation."""


def _type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _validate_type(instance: object, expected: str, *, path: str) -> None:
    if expected == "number" and isinstance(instance, (int, float)) and not isinstance(
        instance, bool
    ):
        return
    if expected == "integer" and isinstance(instance, int) and not isinstance(instance, bool):
        return
    if _type_name(instance) != expected:
        raise StrictToolContractValidationError(
            f"{path}: expected type {expected}, got {_type_name(instance)}"
        )


def validate_json_against_canonical_schema(
    instance: object,
    schema: Mapping[str, object],
    *,
    path: str = "$",
) -> None:
    """Validate a JSON value against a canonical JSON Schema subset (fail-closed)."""
    if not isinstance(schema, Mapping):
        raise StrictToolContractValidationError(f"{path}: schema must be an object")

    const = schema.get("const")
    if const is not None and instance != const:
        raise StrictToolContractValidationError(
            f"{path}: expected const {const!r}, got {instance!r}"
        )

    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        if instance not in enum:
            raise StrictToolContractValidationError(
                f"{path}: value {instance!r} not in enum {enum!r}"
            )

    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        matched = False
        last_error: StrictToolContractValidationError | None = None
        for branch_index, branch in enumerate(one_of):
            if not isinstance(branch, Mapping):
                continue
            try:
                validate_json_against_canonical_schema(
                    instance,
                    branch,
                    path=f"{path}.oneOf[{branch_index}]",
                )
                matched = True
                break
            except StrictToolContractValidationError as exc:
                last_error = exc
        if not matched:
            detail = str(last_error) if last_error is not None else "no branch matched"
            raise StrictToolContractValidationError(
                f"{path}: oneOf validation failed ({detail})"
            )
        return

    schema_type = schema.get("type")
    if schema_type is None:
        return

    if isinstance(schema_type, list):
        if not any(_type_name(instance) == item for item in schema_type):
            raise StrictToolContractValidationError(
                f"{path}: expected one of types {schema_type}, got {_type_name(instance)}"
            )
        if "object" in schema_type and isinstance(instance, dict):
            schema_type = "object"
        elif "array" in schema_type and isinstance(instance, list):
            schema_type = "array"
        else:
            schema_type = schema_type[0]

    if not isinstance(schema_type, str):
        raise StrictToolContractValidationError(f"{path}: invalid schema type")

    _validate_type(instance, schema_type, path=path)

    if schema_type == "object":
        if not isinstance(instance, dict):
            raise StrictToolContractValidationError(f"{path}: expected object")
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            required = schema.get("required")
            if isinstance(required, list):
                for key in required:
                    if not isinstance(key, str):
                        continue
                    if key not in instance:
                        raise StrictToolContractValidationError(
                            f"{path}: missing required property {key!r}"
                        )
            additional_properties = schema.get("additionalProperties", True)
            allowed = set(properties.keys()) if isinstance(properties, Mapping) else set()
            for key, value in instance.items():
                if key in properties and isinstance(properties[key], Mapping):
                    validate_json_against_canonical_schema(
                        value,
                        properties[key],
                        path=f"{path}.{key}",
                    )
                elif additional_properties is False and key not in allowed:
                    raise StrictToolContractValidationError(
                        f"{path}: additional property {key!r} is not allowed"
                    )
        return

    if schema_type == "array":
        if not isinstance(instance, list):
            raise StrictToolContractValidationError(f"{path}: expected array")
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(instance) < min_items:
            raise StrictToolContractValidationError(
                f"{path}: expected at least {min_items} items, got {len(instance)}"
            )
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(instance):
                validate_json_against_canonical_schema(
                    item,
                    items,
                    path=f"{path}[{index}]",
                )


def _tool_name_from_definition(definition: CanonicalFunctionToolDefinition) -> str:
    function = definition.wire_schema.get("function")
    if not isinstance(function, Mapping):
        raise StrictToolContractValidationError("canonical tool missing function object")
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        raise StrictToolContractValidationError("canonical tool missing function.name")
    return name


def _parameters_from_definition(
    definition: CanonicalFunctionToolDefinition,
) -> Mapping[str, object]:
    function = definition.wire_schema.get("function")
    if not isinstance(function, Mapping):
        raise StrictToolContractValidationError("canonical tool missing function object")
    parameters = function.get("parameters")
    if not isinstance(parameters, Mapping):
        raise StrictToolContractValidationError("canonical tool missing function.parameters")
    return parameters


def validate_tool_call_against_definition(
    call: LLMToolCall,
    definition: CanonicalFunctionToolDefinition,
) -> None:
    """Validate one provider tool call against a canonical function tool definition."""
    expected_name = _tool_name_from_definition(definition)
    if call.name != expected_name:
        raise StrictToolContractValidationError(
            f"tool call name {call.name!r} does not match canonical tool {expected_name!r}"
        )
    try:
        payload = json.loads(call.arguments_json or "{}")
    except json.JSONDecodeError as exc:
        raise StrictToolContractValidationError(
            f"tool call {call.name!r} arguments JSON is malformed"
        ) from exc
    validate_json_against_canonical_schema(
        payload,
        _parameters_from_definition(definition),
        path=f"$.{call.name}",
    )


def _definitions_by_canonical_name(
    definitions: Sequence[CanonicalFunctionToolDefinition],
) -> dict[str, CanonicalFunctionToolDefinition]:
    by_name: dict[str, CanonicalFunctionToolDefinition] = {}
    for index, definition in enumerate(definitions):
        name = _tool_name_from_definition(definition)
        if name in by_name:
            first_index = next(
                idx
                for idx, candidate in enumerate(definitions)
                if _tool_name_from_definition(candidate) == name
            )
            raise StrictToolContractValidationError(
                f"duplicate canonical tool name {name!r} at tools[{index}] "
                f"(first declared at tools[{first_index}])"
            )
        by_name[name] = definition
    return by_name


def validate_tool_calls_against_canonical_definitions(
    tool_calls: Sequence[LLMToolCall],
    tool_definitions: Sequence[CanonicalFunctionToolDefinition | Mapping[str, object]],
) -> None:
    """Validate emitted tool calls against canonical bindings (fail-closed)."""
    definitions = coerce_canonical_tool_definitions(tool_definitions)
    definitions_by_name = _definitions_by_canonical_name(definitions)
    for index, call in enumerate(tool_calls):
        definition = definitions_by_name.get(call.name)
        if definition is None:
            raise StrictToolContractValidationError(
                f"tool_calls[{index}] name {call.name!r} is not in canonical tool definitions"
            )
        validate_tool_call_against_definition(call, definition)
