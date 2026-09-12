# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing action provider SPI — translate proposed action to external intent (SELF-HEALING R1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.external_operations.intent import ExternalOperationIntent
from intergrax.contracts.external_operations.provider import ProviderPayloadBounds
from intergrax.contracts.self_healing.decision import SelfHealingDecision, SelfHealingProposedAction


@runtime_checkable
class SelfHealingActionProvider(Protocol):
    """
    Maps an admitted decision action to ``ExternalOperationIntent``.

    Must not perform governance, admission, execution, diagnostics, or lifecycle authority.
    """

    @property
    def provider_id(self) -> str:
        ...

    @property
    def supported_action_types(self) -> frozenset[str]:
        ...

    @property
    def payload_bounds(self) -> ProviderPayloadBounds:
        ...

    def translate(
        self,
        decision: SelfHealingDecision,
        action: SelfHealingProposedAction,
        *,
        tenant_id: str,
        task_id: str,
        requested_by: str,
    ) -> ExternalOperationIntent:
        """Pure translation — no I/O."""
        ...


__all__ = ["SelfHealingActionProvider"]
