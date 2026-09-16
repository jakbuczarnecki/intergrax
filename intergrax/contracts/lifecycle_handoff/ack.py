# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral domain handoff acknowledgment — not lifecycle terminal state (ME-RB4-C1)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class DomainLifecycleHandoffDisposition(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEFERRED = "deferred"


class DomainLifecycleHandoffAck(BaseModel):
    """Domain authority acknowledgment at the handoff boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition: DomainLifecycleHandoffDisposition
    domain_reference: str | None = None
    reason_detail: str = ""


__all__ = [
    "DomainLifecycleHandoffAck",
    "DomainLifecycleHandoffDisposition",
]
