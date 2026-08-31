"""Scenario tool declarations — canonical ToolContract registrations."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler

from platform_proofs.scenarios.indirect_prompt_injection.application.order_provider_client import (
    OrderProviderClient,
)

TOOL_ORDER_GET = "order.get"
TOOL_ORDER_GET_NOTES = "order.get_notes"
TOOL_ORDER_UPDATE_SHIPPING_ADDRESS = "order.update_shipping_address"

SCENARIO_TOOL_IDS: tuple[str, ...] = (
    TOOL_ORDER_GET,
    TOOL_ORDER_GET_NOTES,
    TOOL_ORDER_UPDATE_SHIPPING_ADDRESS,
)

READ_TOOL_IDS: tuple[str, ...] = (TOOL_ORDER_GET, TOOL_ORDER_GET_NOTES)
WRITE_TOOL_IDS: tuple[str, ...] = (TOOL_ORDER_UPDATE_SHIPPING_ADDRESS,)


class OrderIdInput(BaseModel):
    order_id: str = Field(min_length=1)


class OrderGetOutput(BaseModel):
    order_id: str
    status: str
    shipping_address: str
    fulfillment_status: str


class OrderNotesOutput(BaseModel):
    order_id: str
    notes: list[dict[str, object]]


class UpdateShippingAddressInput(BaseModel):
    order_id: str = Field(min_length=1)
    new_shipping_address: str = Field(min_length=1)


class UpdateShippingAddressOutput(BaseModel):
    order_id: str
    status: str
    shipping_address: str
    fulfillment_status: str
    confirmation: str


def _order_tool_contract(
    tool_id: str,
    input_model: type[BaseModel],
    output_model: type[BaseModel],
    *,
    description: str,
    side_effects: bool,
) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description=description,
        input_schema=input_model,
        output_schema=output_model,
        error_mapping={},
        side_effects=side_effects,
        risk_level=ToolRiskLevel.MEDIUM if side_effects else ToolRiskLevel.LOW,
    )


class _OrderGetHandler(ToolHandler[OrderIdInput, OrderGetOutput]):
    def __init__(self, client: OrderProviderClient) -> None:
        self._client = client

    def execute(self, request: ToolExecutionRequest[OrderIdInput]) -> OrderGetOutput:
        payload = self._client.get_order(request.input.order_id)
        return OrderGetOutput(
            order_id=str(payload["order_id"]),
            status=str(payload["status"]),
            shipping_address=str(payload["shipping_address"]),
            fulfillment_status=str(payload["fulfillment_status"]),
        )


class _OrderGetNotesHandler(ToolHandler[OrderIdInput, OrderNotesOutput]):
    def __init__(self, client: OrderProviderClient) -> None:
        self._client = client

    def execute(self, request: ToolExecutionRequest[OrderIdInput]) -> OrderNotesOutput:
        payload = self._client.get_notes(request.input.order_id)
        notes_raw = payload.get("notes", [])
        notes = [dict(item) for item in notes_raw if isinstance(item, dict)]
        return OrderNotesOutput(order_id=str(payload["order_id"]), notes=notes)


class _UpdateShippingAddressHandler(
    ToolHandler[UpdateShippingAddressInput, UpdateShippingAddressOutput]
):
    def __init__(self, client: OrderProviderClient) -> None:
        self._client = client

    def execute(
        self, request: ToolExecutionRequest[UpdateShippingAddressInput]
    ) -> UpdateShippingAddressOutput:
        payload = self._client.update_shipping_address(
            request.input.order_id,
            request.input.new_shipping_address,
        )
        return UpdateShippingAddressOutput(
            order_id=str(payload["order_id"]),
            status=str(payload["status"]),
            shipping_address=str(payload["shipping_address"]),
            fulfillment_status=str(payload["fulfillment_status"]),
            confirmation=str(payload.get("confirmation", "shipping_address_updated")),
        )


def register_scenario_tools(
    registry: ToolRegistry,
    *,
    provider_client: OrderProviderClient,
) -> None:
    registry.register(
        _order_tool_contract(
            TOOL_ORDER_GET,
            OrderIdInput,
            OrderGetOutput,
            description="Read current order facts including status and shipping address.",
            side_effects=False,
        ),
        _OrderGetHandler(provider_client),
    )
    registry.register(
        _order_tool_contract(
            TOOL_ORDER_GET_NOTES,
            OrderIdInput,
            OrderNotesOutput,
            description="Read support notes attached to an order.",
            side_effects=False,
        ),
        _OrderGetNotesHandler(provider_client),
    )
    registry.register(
        _order_tool_contract(
            TOOL_ORDER_UPDATE_SHIPPING_ADDRESS,
            UpdateShippingAddressInput,
            UpdateShippingAddressOutput,
            description="Update the shipping address for an order.",
            side_effects=True,
        ),
        _UpdateShippingAddressHandler(provider_client),
    )
