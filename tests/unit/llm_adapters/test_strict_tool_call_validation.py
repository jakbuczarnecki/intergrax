# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J-L0 — strict tool call validation matrix."""

from __future__ import annotations

import json

import pytest

from intergrax.llm_adapters._shared.strict_tool_enforcement import (
    enforce_strict_tool_call_conformance,
    resolve_canonical_tool_definitions,
)
from intergrax.llm_adapters.contracts.strict_tool_call_validation import (
    StrictToolContractValidationError,
    validate_tool_calls_against_canonical_definitions,
)
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.llm_adapters.providers.native_ollama_adapter import NativeOllamaAdapter
from intergrax.llm_adapters.providers.ollama_capabilities import (
    OllamaModelCapabilityResolver,
)

pytestmark = pytest.mark.unit


def _resolver(capabilities: list[str]) -> OllamaModelCapabilityResolver:
    from types import SimpleNamespace

    return OllamaModelCapabilityResolver(
        show_model=lambda _model: SimpleNamespace(capabilities=capabilities),
    )


def _schema() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "record_incident",
            "description": "Record incident details",
            "parameters": {
                "type": "object",
                "properties": {
                    "incident_id": {"type": "string"},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "region": {
                                "type": "object",
                                "properties": {"code": {"type": "string"}},
                                "required": ["code"],
                                "additionalProperties": False,
                            }
                        },
                        "required": ["region"],
                        "additionalProperties": False,
                    },
                    "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "count": {"type": "integer"},
                    "note": {"type": "string"},
                },
                "required": ["incident_id", "severity", "filters"],
                "additionalProperties": False,
            },
        },
    }


def _valid_args() -> dict[str, object]:
    return {
        "incident_id": "INC-1",
        "severity": "high",
        "filters": {"region": {"code": "eu-west"}},
        "tags": ["ops"],
        "count": 3,
    }


def _call(arguments: dict[str, object], *, name: str = "record_incident", call_id: str = "tc-1") -> LLMToolCall:
    return LLMToolCall(
        id=call_id,
        name=name,
        arguments_json=json.dumps(arguments, separators=(",", ":")),
    )


def test_valid_required_args() -> None:
    validate_tool_calls_against_canonical_definitions([_call(_valid_args())], [_schema()])


def test_missing_required_arg() -> None:
    args = _valid_args()
    del args["incident_id"]
    with pytest.raises(StrictToolContractValidationError, match="missing required property 'incident_id'"):
        validate_tool_calls_against_canonical_definitions([_call(args)], [_schema()])


def test_wrong_primitive_type_rejects_string_integer() -> None:
    args = _valid_args()
    args["count"] = "5"
    with pytest.raises(StrictToolContractValidationError, match="expected type integer"):
        validate_tool_calls_against_canonical_definitions([_call(args)], [_schema()])


def test_wrong_nested_type() -> None:
    args = _valid_args()
    args["filters"] = {"region": {"code": 42}}
    with pytest.raises(StrictToolContractValidationError, match="expected type string"):
        validate_tool_calls_against_canonical_definitions([_call(args)], [_schema()])


def test_enum_violation() -> None:
    args = _valid_args()
    args["severity"] = "urgent"
    with pytest.raises(StrictToolContractValidationError, match="not in enum"):
        validate_tool_calls_against_canonical_definitions([_call(args)], [_schema()])


def test_additional_property() -> None:
    args = _valid_args()
    args["unexpected"] = True
    with pytest.raises(StrictToolContractValidationError, match="additional property 'unexpected'"):
        validate_tool_calls_against_canonical_definitions([_call(args)], [_schema()])


def test_invalid_array_item() -> None:
    args = _valid_args()
    args["tags"] = [1]
    with pytest.raises(StrictToolContractValidationError, match="expected type string"):
        validate_tool_calls_against_canonical_definitions([_call(args)], [_schema()])


def test_null_violation() -> None:
    args = _valid_args()
    args["note"] = None
    with pytest.raises(StrictToolContractValidationError, match="expected type string"):
        validate_tool_calls_against_canonical_definitions([_call(args)], [_schema()])


def test_unknown_tool() -> None:
    with pytest.raises(StrictToolContractValidationError, match="not in canonical tool definitions"):
        validate_tool_calls_against_canonical_definitions(
            [_call(_valid_args(), name="other_tool")],
            [_schema()],
        )


def test_malformed_json() -> None:
    call = LLMToolCall(id="tc-bad", name="record_incident", arguments_json="{bad")
    with pytest.raises(StrictToolContractValidationError, match="malformed"):
        validate_tool_calls_against_canonical_definitions([call], [_schema()])


def test_empty_valid_args_when_schema_allows() -> None:
    schema = {
        "type": "function",
        "function": {
            "name": "noop",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }
    validate_tool_calls_against_canonical_definitions([_call({}, name="noop")], [schema])


def test_multiple_valid_calls() -> None:
    schema = {
        "type": "function",
        "function": {
            "name": "noop",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }
    calls = [
        _call({"value": "a"}, name="noop", call_id="tc-1"),
        _call({"value": "b"}, name="noop", call_id="tc-2"),
    ]
    validate_tool_calls_against_canonical_definitions(calls, [schema])


def test_one_invalid_among_multiple_calls_fails_entire_response() -> None:
    schema = {
        "type": "function",
        "function": {
            "name": "noop",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }
    calls = [
        _call({"value": "ok"}, name="noop", call_id="tc-1"),
        _call({}, name="noop", call_id="tc-2"),
    ]
    with pytest.raises(StrictToolContractValidationError, match="missing required property 'value'"):
        validate_tool_calls_against_canonical_definitions(calls, [schema])


def test_duplicate_tool_definitions_fail_before_dispatch() -> None:
    schema = _schema()
    with pytest.raises(StrictToolContractValidationError, match="duplicate canonical tool name"):
        resolve_canonical_tool_definitions([schema, schema])


def test_no_coercion_string_integer() -> None:
    args = _valid_args()
    args["count"] = "5"
    with pytest.raises(StrictToolContractValidationError):
        validate_tool_calls_against_canonical_definitions([_call(args)], [_schema()])


def test_native_ollama_declares_strict_capability_when_tools_supported() -> None:
    adapter = NativeOllamaAdapter(
        client=object(),
        model="qwen2.5:14b",
        capability_resolver=_resolver(["tools", "completion"]),
    )
    assert adapter.supports_strict_tool_argument_conformance() is True


def test_native_ollama_strict_capability_false_without_tools() -> None:
    adapter = NativeOllamaAdapter(
        client=object(),
        model="llama3.1:latest",
        capability_resolver=_resolver(["completion"]),
    )
    assert adapter.supports_strict_tool_argument_conformance() is False


def test_enforce_rejects_invalid_provider_payload() -> None:
    with pytest.raises(StrictToolContractValidationError):
        enforce_strict_tool_call_conformance(
            [_call({"incident_id": "x", "severity": "high", "filters": {"region": {}}})],
            [_schema()],
        )
