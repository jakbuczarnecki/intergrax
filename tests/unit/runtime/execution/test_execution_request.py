# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, fields
from dataclasses import FrozenInstanceError

import pytest

from intergrax.runtime.execution import ExecutionCapability, ExecutionRequest

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


def test_request_is_immutable() -> None:
    request = ExecutionRequest[PromptInput, SummaryOutput](
        input=PromptInput(text="immutable"),
        capabilities=frozenset({ExecutionCapability.TOOLS}),
    )

    with pytest.raises(FrozenInstanceError):
        request.input = PromptInput(text="mutated")  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        request.capabilities = frozenset({ExecutionCapability.STREAMING})  # type: ignore[misc]


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
    assert public_fields == frozenset({"input", "output_type", "capabilities"})


def test_package_root_exports_execution_request_symbols() -> None:
    from intergrax.runtime.execution import ExecutionCapability as ExportedCapability
    from intergrax.runtime.execution import ExecutionRequest as ExportedRequest

    assert ExportedCapability is ExecutionCapability
    assert ExportedRequest is ExecutionRequest
