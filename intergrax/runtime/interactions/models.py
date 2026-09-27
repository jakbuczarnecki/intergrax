# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Vendor-neutral inbound interaction models (§18, Phase H.2)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from intergrax.integrations.contracts.inbound_interaction import InboundInteraction


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


__all__ = ["InboundInteraction", "InteractionIntakeResponse"]
