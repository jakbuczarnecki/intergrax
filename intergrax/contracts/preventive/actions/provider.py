# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Preventive action provider SPI — translate proposal to external intent only (PREVENTIVE R7)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.external_operations.intent import ExternalOperationIntent
from intergrax.contracts.external_operations.provider import ProviderPayloadBounds
from intergrax.contracts.preventive.actions.proposal import PreventiveActionProposal


@runtime_checkable
class PreventiveActionProvider(Protocol):
    """
    Maps an admitted proposal to ``ExternalOperationIntent``.

    Must not perform governance, admission, execution, diagnostics, or lifecycle authority.
    """

    @property
    def provider_id(self) -> str:
        ...

    @property
    def supported_action_types(self) -> frozenset[str]:
        """Qualified or short action type ids this provider handles."""

    @property
    def payload_bounds(self) -> ProviderPayloadBounds:
        ...

    def translate(
        self,
        proposal: PreventiveActionProposal,
        *,
        task_id: str,
        requested_by: str,
    ) -> ExternalOperationIntent:
        """Pure translation — no I/O."""
        ...


__all__ = ["PreventiveActionProvider"]
