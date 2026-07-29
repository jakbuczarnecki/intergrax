# © Artur Czarnecki. All rights reserved.

"""TOKEN-10B: deterministic tool-planning schema and prepared schema forwarding."""

from __future__ import annotations

import copy
import json
from typing import Any, Sequence

import pytest
from pydantic import BaseModel, ConfigDict

from intergrax.llm.messages import ChatMessage, compute_model_facing_messages_hash
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.tools.tool_planning_service import (
    ToolPlanningService,
    build_tool_planning_schema,
)
from intergrax.runtime.token_optimization.prompt_assembly import build_cache_stable_tool_envelope
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.registry import ToolRegistry

pytestmark = pytest.mark.unit


class _AlphaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str = "alpha"


class _BetaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str = "beta"


class _GammaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str = "gamma"


def _noop_handler(_input: BaseModel) -> dict[str, str]:
    return {"ok": "true"}


def _registry_register_order_b() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolContract(
            tool_id="zeta.tool",
            name="Zeta",
            description="SYNTH-ZETA-DESC",
            input_schema=_GammaInput,
            output_schema=dict,
            error_mapping={},
            side_effects=False,
            risk_level=ToolRiskLevel.LOW,
            category="test",
            tags=(),
        ),
        _noop_handler,
    )
    registry.register(
        ToolContract(
            tool_id="alpha.tool",
            name="Alpha",
            description="SYNTH-ALPHA-DESC",
            input_schema=_AlphaInput,
            output_schema=dict,
            error_mapping={},
            side_effects=False,
            risk_level=ToolRiskLevel.LOW,
            category="test",
            tags=(),
        ),
        _noop_handler,
    )
    registry.register(
        ToolContract(
            tool_id="beta.tool",
            name="Beta",
            description="SYNTH-BETA-DESC",
            input_schema=_BetaInput,
            output_schema=dict,
            error_mapping={},
            side_effects=False,
            risk_level=ToolRiskLevel.LOW,
            category="test",
            tags=(),
        ),
        _noop_handler,
    )
    return registry


def _registry_register_order_a() -> ToolRegistry:
    registry = ToolRegistry()
    for tool_id, schema, description in (
        ("beta.tool", _BetaInput, "SYNTH-BETA-DESC"),
        ("alpha.tool", _AlphaInput, "SYNTH-ALPHA-DESC"),
        ("zeta.tool", _GammaInput, "SYNTH-ZETA-DESC"),
    ):
        registry.register(
            ToolContract(
                tool_id=tool_id,
                name=tool_id,
                description=description,
                input_schema=schema,
                output_schema=dict,
                error_mapping={},
                side_effects=False,
                risk_level=ToolRiskLevel.LOW,
                category="test",
                tags=(),
            ),
            _noop_handler,
        )
    return registry


def test_different_registration_orders_produce_same_schema_order() -> None:
    schema_a = build_tool_planning_schema(_registry_register_order_a())
    schema_b = build_tool_planning_schema(_registry_register_order_b())
    names_a = [entry["function"]["name"] for entry in schema_a]
    names_b = [entry["function"]["name"] for entry in schema_b]
    assert names_a == ["alpha.tool", "beta.tool", "zeta.tool"]
    assert names_a == names_b


def test_different_registration_orders_produce_same_envelope_hash() -> None:
    envelope_a = build_cache_stable_tool_envelope(
        build_tool_planning_schema(_registry_register_order_a())
    )
    envelope_b = build_cache_stable_tool_envelope(
        build_tool_planning_schema(_registry_register_order_b())
    )
    assert envelope_a.envelope_hash == envelope_b.envelope_hash


def test_identical_schema_produces_identical_hash() -> None:
    schema = build_tool_planning_schema(_registry_register_order_a())
    first = build_cache_stable_tool_envelope(schema)
    second = build_cache_stable_tool_envelope(schema)
    assert first.envelope_hash == second.envelope_hash


def test_changed_tool_description_changes_hash() -> None:
    registry = ToolRegistry()
    contract = ToolContract(
        tool_id="alpha.tool",
        name="Alpha",
        description="SYNTH-DESC-ONE",
        input_schema=_AlphaInput,
        output_schema=dict,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
        category="test",
        tags=(),
    )
    registry.register(contract, _noop_handler)
    first = build_cache_stable_tool_envelope(build_tool_planning_schema(registry))

    mutated = ToolContract(
        tool_id="alpha.tool",
        name="Alpha",
        description="SYNTH-DESC-TWO",
        input_schema=_AlphaInput,
        output_schema=dict,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
        category="test",
        tags=(),
    )
    registry_b = ToolRegistry()
    registry_b.register(mutated, _noop_handler)
    second = build_cache_stable_tool_envelope(build_tool_planning_schema(registry_b))
    assert first.envelope_hash != second.envelope_hash


def test_changed_input_schema_changes_hash() -> None:
    registry = ToolRegistry()

    class _SchemaOne(BaseModel):
        model_config = ConfigDict(extra="forbid")
        field_a: str

    class _SchemaTwo(BaseModel):
        model_config = ConfigDict(extra="forbid")
        field_b: str

    registry.register(
        ToolContract(
            tool_id="alpha.tool",
            name="Alpha",
            description="SYNTH-DESC",
            input_schema=_SchemaOne,
            output_schema=dict,
            error_mapping={},
            side_effects=False,
            risk_level=ToolRiskLevel.LOW,
            category="test",
            tags=(),
        ),
        _noop_handler,
    )
    first = build_cache_stable_tool_envelope(build_tool_planning_schema(registry))

    registry_b = ToolRegistry()
    registry_b.register(
        ToolContract(
            tool_id="alpha.tool",
            name="Alpha",
            description="SYNTH-DESC",
            input_schema=_SchemaTwo,
            output_schema=dict,
            error_mapping={},
            side_effects=False,
            risk_level=ToolRiskLevel.LOW,
            category="test",
            tags=(),
        ),
        _noop_handler,
    )
    second = build_cache_stable_tool_envelope(build_tool_planning_schema(registry_b))
    assert first.envelope_hash != second.envelope_hash


def test_added_or_removed_tool_changes_hash() -> None:
    registry = _registry_register_order_a()
    full = build_cache_stable_tool_envelope(build_tool_planning_schema(registry))
    subset = build_cache_stable_tool_envelope(
        build_tool_planning_schema(registry, allowed_tool_ids=("alpha.tool",))
    )
    assert full.envelope_hash != subset.envelope_hash


def test_duplicate_tool_ids_rejected_in_envelope() -> None:
    duplicate_schema = [
        {"type": "function", "function": {"name": "dup.tool", "description": "a", "parameters": {}}},
        {"type": "function", "function": {"name": "dup.tool", "description": "b", "parameters": {}}},
    ]
    with pytest.raises(ValueError, match="duplicate tool name"):
        build_cache_stable_tool_envelope(duplicate_schema)


def test_canonical_tool_contract_not_mutated() -> None:
    registry = _registry_register_order_a()
    original = registry.get("alpha.tool").contract.description
    _ = build_tool_planning_schema(registry)
    assert registry.get("alpha.tool").contract.description == original


class _CapturingAdapter(LLMAdapter):
    provider = "fake-capture"
    model = "fake-capture"

    def __init__(self) -> None:
        super().__init__()
        self.received_schema: list[dict[str, Any]] | None = None
        self.generate_with_tools_calls = 0

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def supports_tools(self) -> bool:
        return True

    def supports_structured_output(self) -> bool:
        return False

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        return build_adapter_response(content="unused")

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        self.generate_with_tools_calls += 1
        self.received_schema = tools_schema
        return build_adapter_response(content="")


def _single_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolContract(
            tool_id="alpha.tool",
            name="Alpha",
            description="SYNTH-DESC",
            input_schema=_AlphaInput,
            output_schema=dict,
            error_mapping={},
            side_effects=False,
            risk_level=ToolRiskLevel.LOW,
            category="test",
            tags=(),
        ),
        _noop_handler,
    )
    return registry


def test_prepared_schema_with_unexpected_tool_rejected() -> None:
    registry = _single_tool_registry()
    schema = build_tool_planning_schema(registry)
    bad = list(schema)
    bad.append(
        {
            "type": "function",
            "function": {
                "name": "unexpected.tool",
                "description": "SYNTH-UNEXPECTED",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    )
    adapter = _CapturingAdapter()
    planner = ToolPlanningService(adapter, registry)
    with pytest.raises(ValueError, match="unexpected tools"):
        planner.plan_native_round(
            [ChatMessage(role="user", content="SYNTH-PLAN")],
            allowed_tool_ids=("alpha.tool",),
            prepared_tools_schema=bad,
        )


def test_prepared_schema_missing_expected_tool_rejected() -> None:
    registry = _single_tool_registry()
    adapter = _CapturingAdapter()
    planner = ToolPlanningService(adapter, registry)
    with pytest.raises(ValueError, match="missing expected tools"):
        planner.plan_native_round(
            [ChatMessage(role="user", content="SYNTH-PLAN")],
            allowed_tool_ids=("alpha.tool",),
            prepared_tools_schema=[],
        )


def test_exact_fingerprinted_schema_passed_to_adapter() -> None:
    registry = _single_tool_registry()
    schema = build_tool_planning_schema(registry)
    envelope = build_cache_stable_tool_envelope(schema)
    adapter = _CapturingAdapter()
    planner = ToolPlanningService(adapter, registry)
    planner.plan_native_round(
        [ChatMessage(role="user", content="SYNTH-PLAN")],
        allowed_tool_ids=("alpha.tool",),
        prepared_tools_schema=list(envelope.tools_schema),
    )
    assert adapter.received_schema is not None
    assert json.dumps(adapter.received_schema, sort_keys=True) == json.dumps(
        list(envelope.tools_schema),
        sort_keys=True,
    )


def test_nested_schema_is_deep_copied() -> None:
    registry = _single_tool_registry()
    schema = build_tool_planning_schema(registry)
    nested = schema[0]["function"]["parameters"]["properties"]
    assert isinstance(nested, dict)
    adapter = _CapturingAdapter()
    planner = ToolPlanningService(adapter, registry)
    planner.plan_native_round(
        [ChatMessage(role="user", content="SYNTH-PLAN")],
        allowed_tool_ids=("alpha.tool",),
        prepared_tools_schema=schema,
    )
    assert adapter.received_schema is not None
    adapter.received_schema[0]["function"]["description"] = "SYNTH-MUTATED"
    assert schema[0]["function"]["description"] == "SYNTH-DESC"


def test_matching_prepared_schema_hash_passes() -> None:
    registry = _single_tool_registry()
    schema = build_tool_planning_schema(registry)
    envelope = build_cache_stable_tool_envelope(schema)
    adapter = _CapturingAdapter()
    planner = ToolPlanningService(adapter, registry)
    planner.plan_native_round(
        [ChatMessage(role="user", content="SYNTH-PLAN")],
        allowed_tool_ids=("alpha.tool",),
        prepared_tools_schema=list(envelope.tools_schema),
        prepared_tools_schema_hash=envelope.envelope_hash,
    )
    assert adapter.generate_with_tools_calls == 1


def test_mismatching_prepared_schema_hash_fails_before_adapter_call() -> None:
    registry = _single_tool_registry()
    schema = build_tool_planning_schema(registry)
    envelope = build_cache_stable_tool_envelope(schema)
    adapter = _CapturingAdapter()
    planner = ToolPlanningService(adapter, registry)
    with pytest.raises(ValueError, match="prepared tools schema hash mismatch"):
        planner.plan_native_round(
            [ChatMessage(role="user", content="SYNTH-PLAN")],
            allowed_tool_ids=("alpha.tool",),
            prepared_tools_schema=list(envelope.tools_schema),
            prepared_tools_schema_hash="0" * 64,
        )
    assert adapter.generate_with_tools_calls == 0


def test_matching_prepared_messages_hash_passes() -> None:
    registry = _single_tool_registry()
    schema = build_tool_planning_schema(registry)
    envelope = build_cache_stable_tool_envelope(schema)
    messages = [ChatMessage(role="user", content="SYNTH-PLAN")]
    messages_hash = compute_model_facing_messages_hash(messages)
    adapter = _CapturingAdapter()
    planner = ToolPlanningService(adapter, registry)
    planner.plan_native_round(
        messages,
        allowed_tool_ids=("alpha.tool",),
        prepared_tools_schema=list(envelope.tools_schema),
        prepared_messages_hash=messages_hash,
    )
    assert adapter.generate_with_tools_calls == 1


def test_mismatching_prepared_messages_hash_fails_before_adapter_call() -> None:
    registry = _single_tool_registry()
    schema = build_tool_planning_schema(registry)
    envelope = build_cache_stable_tool_envelope(schema)
    adapter = _CapturingAdapter()
    planner = ToolPlanningService(adapter, registry)
    with pytest.raises(ValueError, match="prepared messages hash mismatch"):
        planner.plan_native_round(
            [ChatMessage(role="user", content="SYNTH-PLAN")],
            allowed_tool_ids=("alpha.tool",),
            prepared_tools_schema=list(envelope.tools_schema),
            prepared_messages_hash="0" * 64,
        )
    assert adapter.generate_with_tools_calls == 0


def test_message_hash_calculated_after_pruning() -> None:
    registry = _single_tool_registry()
    schema = build_tool_planning_schema(registry)
    envelope = build_cache_stable_tool_envelope(schema)
    messages = [
        ChatMessage(role="system", content="SYNTH-SYSTEM"),
        ChatMessage(role="assistant", content="SYNTH-ASSISTANT", tool_calls=[]),
        ChatMessage(role="tool", content="SYNTH-TOOL", tool_call_id="call-1"),
        ChatMessage(role="user", content="SYNTH-USER"),
    ]
    from intergrax.runtime.nexus.tools.tool_planning_service import _prune_messages_for_openai

    pruned = _prune_messages_for_openai(list(messages))
    pruned_hash = compute_model_facing_messages_hash(pruned)
    adapter = _CapturingAdapter()
    planner = ToolPlanningService(adapter, registry)
    planner.plan_native_round(
        messages,
        allowed_tool_ids=("alpha.tool",),
        prepared_tools_schema=list(envelope.tools_schema),
        prepared_messages_hash=pruned_hash,
    )
    assert adapter.generate_with_tools_calls == 1


def test_adapter_receives_value_identical_schema() -> None:
    registry = _single_tool_registry()
    schema = build_tool_planning_schema(registry)
    envelope = build_cache_stable_tool_envelope(schema)
    adapter = _CapturingAdapter()
    planner = ToolPlanningService(adapter, registry)
    planner.plan_native_round(
        [ChatMessage(role="user", content="SYNTH-PLAN")],
        allowed_tool_ids=("alpha.tool",),
        prepared_tools_schema=list(envelope.tools_schema),
        prepared_tools_schema_hash=envelope.envelope_hash,
    )
    assert adapter.received_schema is not None
    assert adapter.received_schema == list(envelope.tools_schema)


def test_adapter_not_called_after_integrity_failure() -> None:
    registry = _single_tool_registry()
    schema = build_tool_planning_schema(registry)
    envelope = build_cache_stable_tool_envelope(schema)
    adapter = _CapturingAdapter()
    planner = ToolPlanningService(adapter, registry)
    with pytest.raises(ValueError, match="prepared tools schema hash mismatch"):
        planner.plan_native_round(
            [ChatMessage(role="user", content="SYNTH-PLAN")],
            allowed_tool_ids=("alpha.tool",),
            prepared_tools_schema=list(envelope.tools_schema),
            prepared_tools_schema_hash="deadbeef" * 8,
        )
    assert adapter.generate_with_tools_calls == 0
