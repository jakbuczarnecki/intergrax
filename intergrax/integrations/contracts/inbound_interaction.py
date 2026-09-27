# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vendor-neutral inbound interaction models for integration contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InboundInteraction(BaseModel):
    """Normalized inbound envelope before runtime task materialization."""

    channel: str
    tenant_id: str
    user_id: str
    message: str = ""
    capability: str | None = None
    session_id: str | None = None
    interaction_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_payload: dict[str, Any] = Field(default_factory=dict)


__all__ = ["InboundInteraction"]
