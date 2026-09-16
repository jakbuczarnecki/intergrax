# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Thin bridge from marketplace agent handoff payload to AC-4 acquisition port.

Controlled boundary: may import Agent Distribution public contracts only.
"""

from __future__ import annotations

from collections.abc import Callable

from intergrax.agent_distribution.dynamic_acquisition import (
    DynamicAgentAcquisitionPort,
    DynamicAgentAcquisitionRequest,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.marketplace.domain_lifecycle_ports import (
    DomainLifecycleHandoffAck,
    DomainLifecycleHandoffDisposition,
)
from intergrax.contracts.marketplace.lifecycle_handoff_payloads import (
    AgentLifecycleHandoffPayload,
)


class AgentDistributionAcquisitionBridge:
    """
    Maps marketplace handoff payload to ``DynamicAgentAcquisitionPort.acquire``.

    Acquisition request construction remains caller-supplied — marketplace does not
    invent install/bind/activate fields.
    """

    def __init__(
        self,
        acquisition: DynamicAgentAcquisitionPort,
        request_factory: Callable[[AgentLifecycleHandoffPayload], DynamicAgentAcquisitionRequest],
        *,
        principal: RequestIdentity,
    ) -> None:
        self._acquisition = acquisition
        self._request_factory = request_factory
        self._principal = principal

    def submit_marketplace_lifecycle_handoff(
        self,
        payload: AgentLifecycleHandoffPayload,
        *,
        request_id: str,
        correlation_id: str | None,
    ) -> DomainLifecycleHandoffAck:
        del correlation_id
        acquisition_request = self._request_factory(payload)
        result = self._acquisition.acquire(
            acquisition_request,
            principal=self._principal,
        )
        return DomainLifecycleHandoffAck(
            disposition=DomainLifecycleHandoffDisposition.ACCEPTED,
            domain_reference=result.runtime_revision_id,
            reason_detail=f"handoff accepted for marketplace request {request_id}",
        )


__all__ = ["AgentDistributionAcquisitionBridge"]
