# © Artur Czarnecki. All rights reserved.

"""Tier-3 implementation of the neutral host-orchestration application wiring port."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.decision_wiring import (
    ApplicationDecisionWiring,
    apply_application_decision_wiring,
)
from intergrax.applications._shared.guardrail_assembly_resolver import (
    assert_guardrail_assembly_valid,
)
from intergrax.applications._shared.guardrail_wiring import (
    ApplicationGuardrailWiring,
    apply_application_guardrail_wiring,
    wire_application_guardrail,
)
from intergrax.applications._shared.observability_wiring import (
    wire_observability_event_subscriptions,
)
from intergrax.applications._shared.reliability_wiring import (
    apply_reliability_governance_wiring,
)
from intergrax.applications._shared.security_assembly_resolver import (
    assert_security_assembly_valid,
)
from intergrax.applications._shared.security_wiring import (
    ApplicationSecurityWiring,
    apply_application_security_wiring,
    wire_application_security,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.host_orchestration_application_wiring_target import (
    HostOrchestrationApplicationWiringTarget,
    HostOrchestrationAssemblyInspectionTarget,
)
from intergrax.contracts.host_orchestration_wiring_capabilities import (
    HostOrchestrationRuntimeEventPort,
)


@dataclass(frozen=True, slots=True)
class ApplicationHostOrchestrationWiring:
    """Resolved Tier-3 wiring applied by Execution Engine after backend materialization."""

    environment: ApplicationEnvironmentProfile
    security_wiring: ApplicationSecurityWiring | None = None
    guardrail_wiring: ApplicationGuardrailWiring | None = None
    decision_wiring: ApplicationDecisionWiring | None = None

    def apply(self, target: HostOrchestrationApplicationWiringTarget) -> None:
        """Attach security, guardrail and decision wiring to the orchestration host target."""
        env = self.environment
        if self.security_wiring is not None:
            apply_application_security_wiring(target, self.security_wiring, env=env)
        if self.guardrail_wiring is not None:
            apply_application_guardrail_wiring(target, self.guardrail_wiring, env)
        if self.decision_wiring is not None:
            apply_application_decision_wiring(
                target,
                self.decision_wiring,
                environment=env,
            )

    def assert_assembly(self, target: HostOrchestrationAssemblyInspectionTarget) -> None:
        """Validate security and guardrail assembly against the materialized host."""
        if self.security_wiring is None or self.guardrail_wiring is None:
            raise ValueError(
                "host orchestration assembly validation requires security and guardrail wiring",
            )
        assert_security_assembly_valid(
            self.security_wiring,
            self.environment,
            orchestration_host=target,
        )
        assert_guardrail_assembly_valid(
            self.guardrail_wiring,
            self.environment,
            orchestration_host=target,
        )

    def apply_runtime_wiring(
        self,
        target: HostOrchestrationApplicationWiringTarget,
        *,
        event_bus: HostOrchestrationRuntimeEventPort,
    ) -> None:
        """Subscribe observability and attach reliability governance on the host target."""
        wire_observability_event_subscriptions(event_bus, self.environment.observability_profile)
        apply_reliability_governance_wiring(target, self.environment)


def build_application_host_orchestration_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    security_wiring: ApplicationSecurityWiring | None = None,
    guardrail_wiring: ApplicationGuardrailWiring | None = None,
    decision_wiring: ApplicationDecisionWiring | None = None,
) -> ApplicationHostOrchestrationWiring:
    """Resolve application wiring for the host (security and guardrail default from ``env``)."""
    return ApplicationHostOrchestrationWiring(
        environment=env,
        security_wiring=security_wiring or wire_application_security(env),
        guardrail_wiring=guardrail_wiring or wire_application_guardrail(env),
        decision_wiring=decision_wiring,
    )


__all__ = [
    "ApplicationHostOrchestrationWiring",
    "build_application_host_orchestration_wiring",
]
