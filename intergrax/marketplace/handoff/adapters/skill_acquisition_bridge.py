# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Thin bridge from marketplace skill handoff payload to Skill domain acquisition port.

Controlled boundary: may import Skill domain public contracts only.
"""

from __future__ import annotations

from collections.abc import Callable

from intergrax.contracts.lifecycle_handoff.ack import (
    DomainLifecycleHandoffAck,
    DomainLifecycleHandoffDisposition,
)
from intergrax.contracts.skills.marketplace_lifecycle_handoff import (
    SkillLifecycleHandoffPayload,
)
from intergrax.skills.dynamic_acquisition import (
    DynamicSkillAcquisitionBindingError,
    DynamicSkillAcquisitionConflictError,
    DynamicSkillAcquisitionPort,
    DynamicSkillAcquisitionRequest,
    DynamicSkillAcquisitionResolutionError,
)


class SkillMarketplaceAcquisitionBridge:
    """Maps ``SkillLifecycleHandoffPayload`` to ``DynamicSkillAcquisitionPort.acquire``."""

    def __init__(
        self,
        acquisition: DynamicSkillAcquisitionPort,
        request_factory: Callable[
            [SkillLifecycleHandoffPayload],
            DynamicSkillAcquisitionRequest,
        ],
    ) -> None:
        self._acquisition = acquisition
        self._request_factory = request_factory

    def submit_marketplace_lifecycle_handoff(
        self,
        payload: SkillLifecycleHandoffPayload,
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
        except DynamicSkillAcquisitionResolutionError as exc:
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail=str(exc),
            )
        except DynamicSkillAcquisitionConflictError as exc:
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail=str(exc),
            )
        except DynamicSkillAcquisitionBindingError as exc:
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.REJECTED,
                reason_detail=str(exc),
            )
        return DomainLifecycleHandoffAck(
            disposition=DomainLifecycleHandoffDisposition.ACCEPTED,
            domain_reference=result.domain_reference,
            reason_detail=f"handoff accepted for marketplace request {request_id}",
        )


__all__ = ["SkillMarketplaceAcquisitionBridge"]
