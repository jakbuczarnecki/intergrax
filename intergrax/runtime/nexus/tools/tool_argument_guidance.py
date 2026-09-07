# © Artur Czarnecki. All rights reserved.

"""Provider-neutral per-tool argument guidance derived from canonical business schemas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class ToolArgumentGuidanceProjectionError(ValueError):
    """Canonical business schema cannot be projected into model-facing argument guidance."""


@dataclass(frozen=True, slots=True)
class ArgumentFieldConstraint:
    enum_values: tuple[str, ...] | None = None
    min_length: int | None = None
    max_length: int | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    pattern: str | None = None


@dataclass(frozen=True, slots=True)
class ArgumentFieldGuidance:
    path: str
    type_label: str
    required: bool
    description: str | None = None
    constraints: ArgumentFieldConstraint | None = None
    nested_fields: tuple[ArgumentFieldGuidance, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolArgumentGuidance:
    tool_id: str
    fields: tuple[ArgumentFieldGuidance, ...]
    additional_properties: bool


def build_tool_argument_guidance(
    tool_id: str,
    parameters: Mapping[str, object],
) -> ToolArgumentGuidance:
    """Derive typed argument guidance from one canonical business parameters schema."""
    if parameters.get("type") != "object":
        raise ToolArgumentGuidanceProjectionError(
            f"{tool_id}: parameters root must be an object schema"
        )
    properties = parameters.get("properties")
    if properties is None:
        properties = {}
    if not isinstance(properties, Mapping):
        raise ToolArgumentGuidanceProjectionError(
            f"{tool_id}: parameters.properties must be an object"
        )
    required_raw = parameters.get("required")
    required_names = (
        frozenset(str(name) for name in required_raw)
        if isinstance(required_raw, list)
        else frozenset()
    )
    additional_properties = parameters.get("additionalProperties", True)
    if not isinstance(additional_properties, bool):
        raise ToolArgumentGuidanceProjectionError(
            f"{tool_id}: additionalProperties must be boolean when present"
        )
    defs = _schema_defs(parameters)
    fields = _project_object_fields(
        properties=properties,
        required_names=required_names,
        defs=defs,
        path_prefix="",
    )
    return ToolArgumentGuidance(
        tool_id=tool_id,
        fields=fields,
        additional_properties=additional_properties,
    )


def build_atomic_planner_argument_guidance(
    *,
    tool_ids: Sequence[str],
    parameters_by_tool_id: Mapping[str, Mapping[str, object]],
) -> tuple[ToolArgumentGuidance, ...]:
    """Build per-admitted-tool guidance in deterministic tool_id order."""
    guidance: list[ToolArgumentGuidance] = []
    for tool_id in sorted(tool_ids):
        parameters = parameters_by_tool_id.get(tool_id)
        if parameters is None:
            raise ToolArgumentGuidanceProjectionError(
                f"missing canonical parameters for admitted tool id: {tool_id}"
            )
        guidance.append(build_tool_argument_guidance(tool_id, parameters))
    return tuple(guidance)


def render_tool_argument_guidance(guidance: ToolArgumentGuidance) -> str:
    """Render one tool contract as compact provider-neutral text."""
    lines: list[str] = [f"{guidance.tool_id}:"]
    required_fields = tuple(field for field in guidance.fields if field.required)
    optional_fields = tuple(field for field in guidance.fields if not field.required)
    if required_fields:
        lines.append("  required:")
        for field in required_fields:
            lines.append(f"    {_render_field_line(field)}")
    if optional_fields:
        lines.append("  optional:")
        for field in optional_fields:
            lines.append(f"    {_render_field_line(field)}")
    if not guidance.additional_properties:
        lines.append("  additional fields: forbidden")
    return "\n".join(lines)


def render_atomic_planner_argument_guidance(
    guidance_entries: Sequence[ToolArgumentGuidance],
) -> str:
    """Render correlated tool_id → argument contract guidance for atomic planner transport."""
    if not guidance_entries:
        raise ToolArgumentGuidanceProjectionError(
            "atomic planner argument guidance requires at least one admitted tool"
        )
    ordered = tuple(sorted(guidance_entries, key=lambda entry: entry.tool_id))
    sections = [render_tool_argument_guidance(entry) for entry in ordered]
    return (
        "Encode a JSON object matching the selected tool argument contract:\n\n"
        + "\n\n".join(sections)
    )


def guidance_text_byte_size(text: str) -> int:
    """Deterministic serialized guidance size diagnostic."""
    return len(text.encode("utf-8"))


def _schema_defs(schema: Mapping[str, object]) -> Mapping[str, object]:
    defs = schema.get("$defs")
    if defs is None:
        defs = schema.get("definitions")
    if defs is None:
        return {}
    if not isinstance(defs, Mapping):
        raise ToolArgumentGuidanceProjectionError("schema $defs must be an object")
    return defs


def _resolve_ref(
    schema: Mapping[str, object],
    *,
    defs: Mapping[str, object],
) -> Mapping[str, object]:
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        return schema
    def_name = ref.rsplit("/", 1)[-1]
    resolved = defs.get(def_name)
    if not isinstance(resolved, Mapping):
        raise ToolArgumentGuidanceProjectionError(f"unresolved schema $ref: {ref}")
    return resolved


def _project_object_fields(
    *,
    properties: Mapping[str, object],
    required_names: frozenset[str],
    defs: Mapping[str, object],
    path_prefix: str,
) -> tuple[ArgumentFieldGuidance, ...]:
    fields: list[ArgumentFieldGuidance] = []
    for name in sorted(properties.keys()):
        raw_property = properties[name]
        if not isinstance(raw_property, Mapping):
            raise ToolArgumentGuidanceProjectionError(
                f"property {name} must be an object schema"
            )
        resolved = _resolve_ref(raw_property, defs=defs)
        path = f"{path_prefix}.{name}" if path_prefix else name
        fields.append(
            _project_field_guidance(
                path=path,
                schema=resolved,
                required=name in required_names,
                defs=defs,
            )
        )
    return tuple(fields)


def _project_field_guidance(
    *,
    path: str,
    schema: Mapping[str, object],
    required: bool,
    defs: Mapping[str, object],
) -> ArgumentFieldGuidance:
    if "anyOf" in schema:
        return _project_any_of_field(
            path=path,
            schema=schema,
            required=required,
            defs=defs,
        )
    if "oneOf" in schema:
        raise ToolArgumentGuidanceProjectionError(
            f"{path}: oneOf field schemas are not supported for argument guidance"
        )
    if "allOf" in schema:
        raise ToolArgumentGuidanceProjectionError(
            f"{path}: allOf field schemas are not supported for argument guidance"
        )

    schema_type = schema.get("type")
    description = _optional_description(schema.get("description"))
    constraints = _extract_constraints(schema)

    if schema_type == "object" or (
        schema_type is None and isinstance(schema.get("properties"), Mapping)
    ):
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            properties = {}
        nested_required_raw = schema.get("required")
        nested_required = (
            frozenset(str(name) for name in nested_required_raw)
            if isinstance(nested_required_raw, list)
            else frozenset()
        )
        nested_additional = schema.get("additionalProperties", True)
        if not isinstance(nested_additional, bool):
            raise ToolArgumentGuidanceProjectionError(
                f"{path}: nested additionalProperties must be boolean when present"
            )
        nested_fields = _project_object_fields(
            properties=properties,
            required_names=nested_required,
            defs=defs,
            path_prefix=path,
        )
        type_label = "object"
        if nested_additional is False and not nested_fields:
            type_label = "object"
        elif nested_fields:
            type_label = "object"
        return ArgumentFieldGuidance(
            path=path,
            type_label=type_label,
            required=required,
            description=description,
            constraints=constraints,
            nested_fields=nested_fields,
        )

    if schema_type == "array":
        items = schema.get("items")
        if not isinstance(items, Mapping):
            raise ToolArgumentGuidanceProjectionError(
                f"{path}: array items schema must be an object"
            )
        item_resolved = _resolve_ref(items, defs=defs)
        item_type = _primitive_or_container_type_label(item_resolved, defs=defs)
        nested_fields: tuple[ArgumentFieldGuidance, ...] = ()
        if item_type == "object":
            item_properties = item_resolved.get("properties")
            if not isinstance(item_properties, Mapping):
                item_properties = {}
            item_required_raw = item_resolved.get("required")
            item_required = (
                frozenset(str(name) for name in item_required_raw)
                if isinstance(item_required_raw, list)
                else frozenset()
            )
            nested_fields = _project_object_fields(
                properties=item_properties,
                required_names=item_required,
                defs=defs,
                path_prefix=f"{path}[]",
            )
        return ArgumentFieldGuidance(
            path=path,
            type_label=f"array<{item_type}>",
            required=required,
            description=description,
            constraints=constraints,
            nested_fields=nested_fields,
        )

    type_label = _primitive_or_container_type_label(schema, defs=defs)
    return ArgumentFieldGuidance(
        path=path,
        type_label=type_label,
        required=required,
        description=description,
        constraints=constraints,
    )


def _project_any_of_field(
    *,
    path: str,
    schema: Mapping[str, object],
    required: bool,
    defs: Mapping[str, object],
) -> ArgumentFieldGuidance:
    branches = schema.get("anyOf")
    if not isinstance(branches, list) or not branches:
        raise ToolArgumentGuidanceProjectionError(f"{path}: anyOf must be a non-empty list")
    non_null: list[Mapping[str, object]] = []
    for branch in branches:
        if not isinstance(branch, Mapping):
            raise ToolArgumentGuidanceProjectionError(f"{path}: anyOf branch must be an object")
        if branch.get("type") == "null":
            continue
        non_null.append(_resolve_ref(branch, defs=defs))
    if len(non_null) != 1:
        raise ToolArgumentGuidanceProjectionError(
            f"{path}: only nullable anyOf schemas are supported for argument guidance"
        )
    inner = dict(non_null[0])
    if description := _optional_description(schema.get("description")):
        inner.setdefault("description", description)
    return _project_field_guidance(
        path=path,
        schema=inner,
        required=required,
        defs=defs,
    )


def _primitive_or_container_type_label(
    schema: Mapping[str, object],
    *,
    defs: Mapping[str, object],
) -> str:
    schema = _resolve_ref(schema, defs=defs)
    schema_type = schema.get("type")
    if schema_type == "integer":
        return "integer"
    if schema_type == "number":
        return "number"
    if schema_type == "boolean":
        return "boolean"
    if schema_type == "string":
        return "string"
    if schema_type == "object" or isinstance(schema.get("properties"), Mapping):
        return "object"
    if schema_type == "array":
        items = schema.get("items")
        if isinstance(items, Mapping):
            item_label = _primitive_or_container_type_label(
                _resolve_ref(items, defs=defs),
                defs=defs,
            )
            return f"array<{item_label}>"
        return "array"
    raise ToolArgumentGuidanceProjectionError(
        f"unsupported or missing schema type: {schema_type!r}"
    )


def _extract_constraints(schema: Mapping[str, object]) -> ArgumentFieldConstraint | None:
    enum_raw = schema.get("enum")
    enum_values: tuple[str, ...] | None = None
    if isinstance(enum_raw, list):
        enum_values = tuple(str(value) for value in enum_raw)

    min_length = _optional_int(schema.get("minLength"))
    max_length = _optional_int(schema.get("maxLength"))
    minimum = _optional_number(schema.get("minimum"))
    maximum = _optional_number(schema.get("maximum"))
    pattern = schema.get("pattern")
    pattern_value = pattern if isinstance(pattern, str) and pattern else None

    if (
        enum_values is None
        and min_length is None
        and max_length is None
        and minimum is None
        and maximum is None
        and pattern_value is None
    ):
        return None
    return ArgumentFieldConstraint(
        enum_values=enum_values,
        min_length=min_length,
        max_length=max_length,
        minimum=minimum,
        maximum=maximum,
        pattern=pattern_value,
    )


def _render_field_line(field: ArgumentFieldGuidance) -> str:
    parts = [f"{field.path}: {field.type_label}"]
    if field.constraints is not None:
        if field.constraints.enum_values is not None:
            allowed = ", ".join(field.constraints.enum_values)
            parts.append(f"allowed: [{allowed}]")
        if field.constraints.min_length is not None:
            parts.append(f"minLength: {field.constraints.min_length}")
        if field.constraints.max_length is not None:
            parts.append(f"maxLength: {field.constraints.max_length}")
        if field.constraints.minimum is not None:
            parts.append(f"min: {field.constraints.minimum}")
        if field.constraints.maximum is not None:
            parts.append(f"max: {field.constraints.maximum}")
        if field.constraints.pattern is not None:
            parts.append(f"pattern: {field.constraints.pattern}")
    line = ", ".join(parts)
    if field.description:
        line = f"{line} — {field.description}"
    if field.nested_fields:
        nested_lines = [_render_field_line(nested) for nested in field.nested_fields]
        line = f"{line}\n" + "\n".join(f"      {nested_line}" for nested_line in nested_lines)
    return line


def _optional_description(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _optional_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None
