# © Artur Czarnecki. All rights reserved.

"""Neutral host orchestration wiring bundle passed into Execution Engine composition."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.decision_wiring import ApplicationDecisionWiring
from intergrax.applications._shared.guardrail_wiring import ApplicationGuardrailWiring
from intergrax.applications._shared.security_wiring import ApplicationSecurityWiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


@dataclass(frozen=True, slots=True)
class HostOrchestrationApplicationWiringBundle:
    """Resolved Tier-3 wiring applied inside Execution Engine after backend materialization."""

    environment: ApplicationEnvironmentProfile
    security_wiring: ApplicationSecurityWiring | None = None
    guardrail_wiring: ApplicationGuardrailWiring | None = None
    decision_wiring: ApplicationDecisionWiring | None = None


__all__ = ["HostOrchestrationApplicationWiringBundle"]
