"""Typed HTTP contracts for the controlled Order Service."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class OrderProviderNote(BaseModel):
    model_config = ConfigDict(extra="allow")

    note_id: str
    content: str
    author: str = ""


class OrderProviderOrder(BaseModel):
    order_id: str
    status: str
    shipping_address: str
    fulfillment_status: str


class OrderProviderNotesResponse(BaseModel):
    order_id: str
    notes: list[OrderProviderNote]


class OrderProviderMutation(BaseModel):
    method: str
    path: str
    payload: dict[str, object]


class ProviderMutationState(BaseModel):
    write_count: int
    mutations: tuple[OrderProviderMutation, ...]


class OrderProviderUpdateResponse(OrderProviderOrder):
    confirmation: str = "shipping_address_updated"


class OrderProviderResetRequest(BaseModel):
    notes: list[OrderProviderNote] = Field(default_factory=list)
