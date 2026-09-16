# © Artur Czarnecki. All rights reserved.

"""Thin bridge: marketplace Tool handoff payload → reference Tool host lifecycle."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.contracts.lifecycle_handoff.ack import DomainLifecycleHandoffAck
from intergrax.contracts.tools.marketplace_lifecycle_handoff import (
    ToolLifecycleHandoffPayload,
)
from testing_support.reference_tool_host_lifecycle_service import (
    ReferenceToolHostLifecycleService,
    ToolActivationRequest,
)


class ToolMarketplaceAcquisitionBridge:
    """Maps ``ToolLifecycleHandoffPayload`` to host Tool lifecycle ``activate``."""

    def __init__(
        self,
        lifecycle: ReferenceToolHostLifecycleService,
        request_factory: Callable[[ToolLifecycleHandoffPayload], ToolActivationRequest],
    ) -> None:
        self._lifecycle = lifecycle
        self._request_factory = request_factory

    def submit_marketplace_lifecycle_handoff(
        self,
        payload: ToolLifecycleHandoffPayload,
        *,
        request_id: str,
        correlation_id: str | None,
    ) -> DomainLifecycleHandoffAck:
        del request_id, correlation_id
        activation_request = self._request_factory(payload)
        return self._lifecycle.activate(activation_request)


__all__ = ["ToolMarketplaceAcquisitionBridge"]
