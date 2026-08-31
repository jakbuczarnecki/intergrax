from __future__ import annotations

import json

import pytest

from platform_proofs.scenarios.indirect_prompt_injection.application.order_provider_models import (
    OrderProviderNote,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.tools import (
    SCENARIO_TOOL_IDS,
    TOOL_ORDER_GET,
    TOOL_ORDER_GET_NOTES,
    TOOL_ORDER_UPDATE_SHIPPING_ADDRESS,
    OrderIdInput,
    register_scenario_tools,
)
from platform_proofs.scenarios.indirect_prompt_injection.fixtures.orders import (
    FORBIDDEN_LEAK_MARKERS,
    build_attack_fixture,
    AttackVariantId,
)
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from tests.unit.platform_proofs.scenarios.indirect_prompt_injection.in_process_order_provider import (
    InProcessOrderProviderClient,
)

pytestmark = pytest.mark.unit


def test_tool_contracts_registered_with_expected_side_effects() -> None:
    registry = ToolRegistry()
    client = InProcessOrderProviderClient()
    client.reset(notes=[OrderProviderNote(note_id="n1", content="hello")])
    register_scenario_tools(registry, provider_client=client)

    get_contract = registry.get(TOOL_ORDER_GET).contract
    notes_contract = registry.get(TOOL_ORDER_GET_NOTES).contract
    write_contract = registry.get(TOOL_ORDER_UPDATE_SHIPPING_ADDRESS).contract

    assert get_contract.side_effects is False
    assert notes_contract.side_effects is False
    assert write_contract.side_effects is True
    assert SCENARIO_TOOL_IDS == (
        TOOL_ORDER_GET,
        TOOL_ORDER_GET_NOTES,
        TOOL_ORDER_UPDATE_SHIPPING_ADDRESS,
    )


def test_provider_adapter_reads_and_writes() -> None:
    client = InProcessOrderProviderClient()
    client.reset(notes=[OrderProviderNote(note_id="n1", content="status check")])
    order = client.get_order("48291")
    notes = client.get_notes("48291")
    assert order.order_id == "48291"
    assert notes.notes

    client.update_shipping_address("48291", "456 Oak Street")
    assert client.mutation_state().write_count == 1


def test_fixture_notes_do_not_leak_attack_markers() -> None:
    fixture = build_attack_fixture(AttackVariantId.ATTACK_AUTHORITY_LAUNDERING)
    blob = json.dumps([note.model_dump(mode="json") for note in fixture.provider_notes]).lower()
    for marker in FORBIDDEN_LEAK_MARKERS:
        assert marker.lower() not in blob


def test_read_tools_execute_via_registry_handler() -> None:
    registry = ToolRegistry()
    client = InProcessOrderProviderClient()
    client.reset(notes=[OrderProviderNote(note_id="n1", content="note")])
    register_scenario_tools(registry, provider_client=client)
    handler = registry.get(TOOL_ORDER_GET).handler
    result = handler.execute(
        ToolExecutionRequest(
            run_id="run-1",
            tool_id=TOOL_ORDER_GET,
            step_id="1",
            input=OrderIdInput(order_id="48291"),
        )
    )
    assert result.order_id == "48291"
