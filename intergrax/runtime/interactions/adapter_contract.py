# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Inbound interaction adapter contract (§18, Phase H.2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, runtime_checkable

from intergrax.runtime.interactions.models import InboundInteraction

if TYPE_CHECKING:
    from intergrax.runtime.task.task import Task


@runtime_checkable
class InteractionPayloadParser(Protocol):
    """Maps a raw vendor payload to a canonical ``InboundInteraction``."""

    def parse(
        self,
        payload: Dict[str, Any],
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> InboundInteraction: ...


class InteractionAdapter(ABC):
    """
    Inbound interaction surface — external events → normalized ``Task``.

    Implementations: Slack slash command, Teams activity, lab JSON, CLI, etc.
    Nexus depends on this contract, not on vendor SDKs.
    """

    @property
    @abstractmethod
    def channel(self) -> str:
        """Logical channel id (``slash_command``, ``lab``, ``teams``, …)."""

    @abstractmethod
    def can_handle(self, payload: Dict[str, Any]) -> bool:
        """Return True when this adapter can parse ``payload``."""

    @abstractmethod
    def to_inbound(self, payload: Dict[str, Any], *, tenant_id: str, user_id: str) -> InboundInteraction:
        """Parse vendor payload into canonical inbound envelope."""

    def to_task(
        self,
        payload: Dict[str, Any],
        *,
        tenant_id: str,
        user_id: Optional[str] = None,
    ) -> Task:
        """Materialize a Nexus ``Task`` from ``payload``."""
        resolved_user = user_id or str(payload.get("user_id") or "anonymous")
        inbound = self.to_inbound(payload, tenant_id=tenant_id, user_id=resolved_user)
        return inbound_to_task(inbound)


def inbound_to_task(inbound: InboundInteraction) -> Task:
    """Shared Task builder — reusable across all interaction adapters."""
    from intergrax.runtime.interactions.metadata_keys import (
        INTERACTION_CHANNEL_KEY,
        INTERACTION_RAW_ID_KEY,
    )
    from intergrax.runtime.task.task import Task, TaskContext

    metadata = dict(inbound.metadata)
    if inbound.interaction_id:
        metadata.setdefault(INTERACTION_RAW_ID_KEY, inbound.interaction_id)
    metadata.setdefault(INTERACTION_CHANNEL_KEY, inbound.channel)

    return Task(
        tenant_id=inbound.tenant_id,
        user_id=inbound.user_id,
        session_id=inbound.session_id,
        message=inbound.message,
        context=TaskContext(
            capability=inbound.capability,
            metadata={
                k: v
                for k, v in inbound.raw_payload.items()
                if k not in {"tenant_id", "user_id", "message", "capability", "text", "command"}
            },
        ),
        metadata=metadata,
    )
