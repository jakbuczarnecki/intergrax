# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, fields

import pytest

from intergrax.runtime.execution import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.strategy import ExecutionStrategy, StrategyResolver

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
class Output:
    value: str


@dataclass(frozen=True, slots=True)
class OrchestrationWork:
    topology_id: str


@dataclass(frozen=True, slots=True)
class ToolWork:
    tool_name: str


def _request(
    *,
    input_payload: PromptInput | OrchestrationWork | ToolWork,
    output_type: type[Output] | None = None,
    capabilities: frozenset[ExecutionCapability] | None = None,
) -> ExecutionRequest:
    caps = frozenset() if capabilities is None else capabilities
    return ExecutionRequest(input=input_payload, output_type=output_type, capabilities=caps)


@pytest.fixture
def resolver() -> StrategyResolver:
    return StrategyResolver()


def test_empty_capabilities_resolve_to_inference(resolver: StrategyResolver) -> None:
    request = _request(input_payload=PromptInput(text="plain"))

    assert resolver.resolve(request) is ExecutionStrategy.INFERENCE


def test_streaming_only_resolves_to_inference(resolver: StrategyResolver) -> None:
    request = _request(
        input_payload=PromptInput(text="stream"),
        capabilities=frozenset({ExecutionCapability.STREAMING}),
    )

    assert resolver.resolve(request) is ExecutionStrategy.INFERENCE


def test_tools_only_resolves_to_agentic(resolver: StrategyResolver) -> None:
    request = _request(
        input_payload=PromptInput(text="tools"),
        capabilities=frozenset({ExecutionCapability.TOOLS}),
    )

    assert resolver.resolve(request) is ExecutionStrategy.AGENTIC


def test_tools_and_streaming_resolve_to_agentic(resolver: StrategyResolver) -> None:
    request = _request(
        input_payload=PromptInput(text="tools+stream"),
        capabilities=frozenset({ExecutionCapability.TOOLS, ExecutionCapability.STREAMING}),
    )

    assert resolver.resolve(request) is ExecutionStrategy.AGENTIC


def test_orchestration_only_resolves_to_orchestration(resolver: StrategyResolver) -> None:
    request = _request(
        input_payload=OrchestrationWork(topology_id="root"),
        capabilities=frozenset({ExecutionCapability.ORCHESTRATION}),
    )

    assert resolver.resolve(request) is ExecutionStrategy.ORCHESTRATION


def test_orchestration_and_streaming_resolve_to_orchestration(
    resolver: StrategyResolver,
) -> None:
    request = _request(
        input_payload=OrchestrationWork(topology_id="root"),
        capabilities=frozenset(
            {ExecutionCapability.ORCHESTRATION, ExecutionCapability.STREAMING}
        ),
    )

    assert resolver.resolve(request) is ExecutionStrategy.ORCHESTRATION


def test_tools_and_orchestration_resolve_to_orchestration(resolver: StrategyResolver) -> None:
    request = _request(
        input_payload=OrchestrationWork(topology_id="multi"),
        capabilities=frozenset(
            {ExecutionCapability.TOOLS, ExecutionCapability.ORCHESTRATION}
        ),
    )

    assert resolver.resolve(request) is ExecutionStrategy.ORCHESTRATION


def test_all_capabilities_resolve_to_orchestration(resolver: StrategyResolver) -> None:
    request = _request(
        input_payload=OrchestrationWork(topology_id="all"),
        capabilities=frozenset(
            {
                ExecutionCapability.TOOLS,
                ExecutionCapability.ORCHESTRATION,
                ExecutionCapability.STREAMING,
            }
        ),
    )

    assert resolver.resolve(request) is ExecutionStrategy.ORCHESTRATION


def test_output_type_without_capabilities_resolves_to_inference(
    resolver: StrategyResolver,
) -> None:
    request = _request(
        input_payload=PromptInput(text="typed"),
        output_type=Output,
    )

    assert resolver.resolve(request) is ExecutionStrategy.INFERENCE


def test_output_type_with_tools_resolves_to_agentic(resolver: StrategyResolver) -> None:
    request = _request(
        input_payload=PromptInput(text="typed tools"),
        output_type=Output,
        capabilities=frozenset({ExecutionCapability.TOOLS}),
    )

    assert resolver.resolve(request) is ExecutionStrategy.AGENTIC


def test_input_shape_does_not_affect_strategy(resolver: StrategyResolver) -> None:
    caps = frozenset({ExecutionCapability.TOOLS})
    prompt_request = _request(input_payload=PromptInput(text="prompt"), capabilities=caps)
    tool_request = _request(input_payload=ToolWork(tool_name="lookup"), capabilities=caps)
    orchestration_request = _request(
        input_payload=OrchestrationWork(topology_id="child"),
        capabilities=caps,
    )

    assert resolver.resolve(prompt_request) is ExecutionStrategy.AGENTIC
    assert resolver.resolve(tool_request) is ExecutionStrategy.AGENTIC
    assert resolver.resolve(orchestration_request) is ExecutionStrategy.AGENTIC


def test_same_request_resolved_repeatedly_returns_same_strategy(
    resolver: StrategyResolver,
) -> None:
    request = _request(
        input_payload=PromptInput(text="repeat"),
        capabilities=frozenset({ExecutionCapability.ORCHESTRATION}),
    )

    first = resolver.resolve(request)
    second = resolver.resolve(request)

    assert first is second is ExecutionStrategy.ORCHESTRATION


def test_distinct_resolver_instances_produce_same_result() -> None:
    request = _request(
        input_payload=PromptInput(text="shared"),
        capabilities=frozenset({ExecutionCapability.TOOLS, ExecutionCapability.STREAMING}),
    )

    assert StrategyResolver().resolve(request) is StrategyResolver().resolve(request)
    assert StrategyResolver().resolve(request) is ExecutionStrategy.AGENTIC


def test_resolver_does_not_mutate_request(resolver: StrategyResolver) -> None:
    request = _request(
        input_payload=PromptInput(text="immutable"),
        output_type=Output,
        capabilities=frozenset({ExecutionCapability.TOOLS}),
    )
    snapshot = (
        request.input,
        request.output_type,
        request.capabilities,
    )

    resolver.resolve(request)

    assert (request.input, request.output_type, request.capabilities) == snapshot


def test_execution_request_has_no_strategy_mode_or_executor_fields() -> None:
    public_fields = {field.name for field in fields(ExecutionRequest)}

    assert public_fields.isdisjoint(_FORBIDDEN_PUBLIC_FIELD_NAMES)
    assert public_fields == frozenset({"input", "output_type", "capabilities"})


def test_package_root_does_not_export_strategy_symbols() -> None:
    import intergrax.runtime.execution as execution_package

    assert "StrategyResolver" not in execution_package.__all__
    assert "ExecutionStrategy" not in execution_package.__all__
