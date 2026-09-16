# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Thin bridge from marketplace tool handoff payload to Tool domain acquisition port.

Controlled boundary: may import Tool domain public contracts only.
"""

from __future__ import annotations

from collections.abc import Callable

from intergrax.contracts.lifecycle_handoff.ack import (
    DomainLifecycleHandoffAck,
    DomainLifecycleHandoffDisposition,
)
from intergrax.contracts.tools.marketplace_lifecycle_handoff import (
    ToolLifecycleHandoffPayload,
)
from intergrax.tools.dynamic_acquisition import (
    DynamicToolAcquisitionConflictError,
    DynamicToolAcquisitionPort,
    DynamicToolAcquisitionRequest,
    DynamicToolAcquisitionResolutionError,
    DynamicToolAcquisitionActivationError,
)


class ToolMarketplaceAcquisitionBridge:
    """Maps ``ToolLifecycleHandoffPayload`` to ``DynamicToolAcquisitionPort.acquire``."""

    def __init__(
        self,
        acquisition: DynamicToolAcquisitionPort,
        request_factory: Callable[
            [ToolLifecycleHandoffPayload],
            DynamicToolAcquisitionRequest,
        ],
    ) -> None:
        self._acquisition = acquisition
        self._request_factory = request_factory

    def submit_marketplace_lifecycle_handoff(
        self,
        payload: ToolLifecycleHandoffPayload,
        *,
        request_id: str,
        correlation_id: str | None,
    ) -> DomainLifecycleHandoffAck:
        del correlation_id
        acquisition_request = self._request_factory(payload)
        if acquisition_request.capability_identity_key != payload.capability_identity_key:
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail="capability_identity_key mismatch",
            )
        try:
            result = self._acquisition.acquire(acquisition_request)
        except DynamicToolAcquisitionResolutionError as exc:
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail=str(exc),
            )
        except DynamicToolAcquisitionConflictError as exc:
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail=str(exc),
            )
        except DynamicToolAcquisitionActivationError as exc:
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail=str(exc),
            )
        return DomainLifecycleHandoffAck(
            disposition=DomainLifecycleHandoffDisposition.ACCEPTED,
            domain_reference=result.domain_reference,
            reason_detail=f"handoff accepted for marketplace request {request_id}",
        )


__all__ = ["ToolMarketplaceAcquisitionBridge"]
