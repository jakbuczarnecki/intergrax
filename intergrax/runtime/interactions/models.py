# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vendor-neutral inbound interaction models (§18, Phase H.2)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class InboundInteraction(BaseModel):
    """
    Normalized inbound envelope before ``Task`` materialization.

    Parsers produce this; adapters map it to ``Task`` — keeping vendor logic
    out of NexusLoop.
    """

    channel: str
    tenant_id: str
    user_id: str
    message: str = ""
    capability: Optional[str] = None
    session_id: Optional[str] = None
    interaction_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_payload: Dict[str, Any] = Field(default_factory=dict)


class InteractionIntakeResponse(BaseModel):
    """HTTP response for inbound Slack / Teams / lab interaction webhooks."""

    task_id: str
    tenant_id: str
    user_id: str
    capability: Optional[str] = None
    message: str = ""
    interaction_channel: str = ""
    executed: bool = False
    state: Optional[str] = None
    answer: Optional[str] = None
    run_id: Optional[str] = None
    resume_token: Optional[str] = None
    checkpoint_id: Optional[str] = None
