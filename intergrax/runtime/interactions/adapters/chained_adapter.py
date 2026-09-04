# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Try multiple interaction adapters without vendor coupling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from intergrax.runtime.interactions.adapter_contract import InteractionAdapter

if TYPE_CHECKING:
    from intergrax.runtime.task.task import Task


class ChainedInteractionAdapter(InteractionAdapter):
    """First matching adapter wins — used by factory ``auto`` mode."""

    def __init__(self, adapters: Iterable[InteractionAdapter]) -> None:
        self._adapters: List[InteractionAdapter] = list(adapters)

    @property
    def channel(self) -> str:
        return "auto"

    @property
    def adapters(self) -> List[InteractionAdapter]:
        return list(self._adapters)

    def can_handle(self, payload: Dict[str, Any]) -> bool:
        return any(adapter.can_handle(payload) for adapter in self._adapters)

    def to_inbound(self, payload: Dict[str, Any], *, tenant_id: str, user_id: str):
        for adapter in self._adapters:
            if adapter.can_handle(payload):
                return adapter.to_inbound(payload, tenant_id=tenant_id, user_id=user_id)
        raise ValueError("No interaction adapter matched payload")

    def to_task(
        self,
        payload: Dict[str, Any],
        *,
        tenant_id: str,
        user_id: Optional[str] = None,
    ) -> Task:
        for adapter in self._adapters:
            if adapter.can_handle(payload):
                return adapter.to_task(payload, tenant_id=tenant_id, user_id=user_id)
        raise ValueError("No interaction adapter matched payload")
