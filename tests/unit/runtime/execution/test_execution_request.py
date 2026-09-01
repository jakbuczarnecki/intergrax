# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path

import pytest

from intergrax.runtime.execution import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.inference_profile import (
    InferenceProfileId,
    validate_inference_profile_id,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_PUBLIC_FIELD_NAMES = frozenset(
    {
        "strategy",
        "mode",
        "executor",
        "agent",
        "use_nexus",
        "nexus",
        "react",
        "planner",
        "graph_executor",
        "metadata",
        "options",
        "config",
    }
)


@dataclass(frozen=True, slots=True)
class PromptInput:
    text: str


@dataclass(frozen=True, slots=True)
class SummaryOutput:
    summary: str


@dataclass(frozen=True, slots=True)
class OrchestrationWork:
    topology_id: str


def test_request_stores_strongly_typed_non_dict_input() -> None:
    payload = PromptInput(text="summarize this")

    request = ExecutionRequest[PromptInput, SummaryOutput](input=payload)

    assert request.input is payload
    assert request.input.text == "summarize this"
    assert not isinstance(request.input, dict)


def test_execution_request_is_immutable_value_object() -> None:
    original_input = PromptInput(text="immutable")
    original_capabilities = frozenset({ExecutionCapability.TOOLS})
    request = ExecutionRequest[PromptInput, SummaryOutput](
        input=original_input,
        capabilities=original_capabilities,
    )

    assert is_dataclass(request)
    assert request.input is original_input
    assert request.input.text == "immutable"
    assert request.capabilities == original_capabilities

    updated_capabilities = frozenset({ExecutionCapability.STREAMING})
    replacement = replace(request, capabilities=updated_capabilities)

    assert replacement is not request
    assert request.input is original_input
    assert request.capabilities == original_capabilities
    assert replacement.input is original_input
    assert replacement.capabilities == updated_capabilities

    source = Path("intergrax/runtime/execution/request.py").read_text(encoding="utf-8")
    assert "@dataclass(frozen=True, slots=True)" in source
    assert "class ExecutionRequest" in source


def test_output_type_declaration_is_preserved_exactly() -> None:
    request = ExecutionRequest[PromptInput, SummaryOutput](
        input=PromptInput(text="typed"),
        output_type=SummaryOutput,
    )

    assert request.output_type is SummaryOutput


def test_capabilities_default_to_empty_frozenset() -> None:
    request = ExecutionRequest[PromptInput, SummaryOutput](
        input=PromptInput(text="default-caps"),
    )

    assert isinstance(request.capabilities, frozenset)
    assert request.capabilities == frozenset()


def test_capabilities_explicit_frozenset_is_preserved() -> None:
    request = ExecutionRequest[PromptInput, SummaryOutput](
        input=PromptInput(text="caps"),
        capabilities=frozenset({ExecutionCapability.TOOLS, ExecutionCapability.STREAMING}),
    )

    assert isinstance(request.capabilities, frozenset)
    assert request.capabilities == frozenset(
        {ExecutionCapability.TOOLS, ExecutionCapability.STREAMING}
    )


def test_direct_inference_requirements_without_strategy_controls() -> None:
    request = ExecutionRequest[PromptInput, SummaryOutput](
        input=PromptInput(text="direct inference"),
    )

    assert request.capabilities == frozenset()
    assert request.output_type is None


def test_tool_requiring_work_without_tool_strategy_fields() -> None:
    request = ExecutionRequest[PromptInput, SummaryOutput](
        input=PromptInput(text="lookup and answer"),
        capabilities=frozenset({ExecutionCapability.TOOLS}),
    )

    assert ExecutionCapability.TOOLS in request.capabilities
    public_fields = {field.name for field in fields(ExecutionRequest)}
    assert "tool_selection" not in public_fields
    assert "tool_invocation" not in public_fields
    assert "react" not in public_fields


def test_orchestration_requirement_without_nexus_naming() -> None:
    request = ExecutionRequest[OrchestrationWork, SummaryOutput](
        input=OrchestrationWork(topology_id="multi-unit-topology"),
        capabilities=frozenset({ExecutionCapability.ORCHESTRATION}),
    )

    assert ExecutionCapability.ORCHESTRATION in request.capabilities
    public_fields = {field.name for field in fields(ExecutionRequest)}
    assert "nexus" not in public_fields
    assert "use_nexus" not in public_fields


def test_no_strategy_mode_or_executor_public_fields() -> None:
    public_fields = {field.name for field in fields(ExecutionRequest)}

    assert public_fields.isdisjoint(_FORBIDDEN_PUBLIC_FIELD_NAMES)


def test_no_metadata_options_or_config_escape_hatch_fields() -> None:
    public_fields = {field.name for field in fields(ExecutionRequest)}

    assert "metadata" not in public_fields
    assert "options" not in public_fields
    assert "config" not in public_fields
    assert public_fields == frozenset(
        {"input", "output_type", "capabilities", "inference_profile_id"},
    )


def test_package_root_exports_execution_request_symbols() -> None:
    from intergrax.runtime.execution import ExecutionCapability as ExportedCapability
    from intergrax.runtime.execution import ExecutionRequest as ExportedRequest

    assert ExportedCapability is ExecutionCapability
    assert ExportedRequest is ExecutionRequest


@pytest.mark.parametrize(
    ("profile_id", "expected"),
    [
        (None, None),
        (InferenceProfileId("primary"), InferenceProfileId("primary")),
    ],
)
def test_inference_profile_id_accepts_none_and_valid_profile(
    profile_id: InferenceProfileId | None,
    expected: InferenceProfileId | None,
) -> None:
    request = ExecutionRequest[PromptInput, SummaryOutput](
        input=PromptInput(text="profile"),
        inference_profile_id=profile_id,
    )

    assert request.inference_profile_id == expected


@pytest.mark.parametrize(
    "invalid_profile_id",
    [
        InferenceProfileId(""),
        InferenceProfileId("   "),
        InferenceProfileId(" primary "),
    ],
)
def test_inference_profile_id_rejects_invalid_direct_constructor_values(
    invalid_profile_id: InferenceProfileId,
) -> None:
    with pytest.raises(ValueError):
        ExecutionRequest[PromptInput, SummaryOutput](
            input=PromptInput(text="invalid profile"),
            inference_profile_id=invalid_profile_id,
        )


def test_inference_profile_id_rejects_non_string_runtime_type() -> None:
    with pytest.raises(TypeError):
        validate_inference_profile_id(42)
