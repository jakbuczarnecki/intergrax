# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Apply Tier-3 host wiring inside Execution Engine owner zone."""

from __future__ import annotations

from intergrax.applications._shared.decision_wiring import (
    ApplicationDecisionWiring,
    apply_application_decision_wiring,
)
from intergrax.applications._shared.guardrail_wiring import (
    ApplicationGuardrailWiring,
    apply_application_guardrail_wiring,
)
from intergrax.applications._shared.security_wiring import (
    ApplicationSecurityWiring,
    apply_application_security_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.execution.host_orchestration_wiring_bundle import (
    HostOrchestrationApplicationWiringBundle,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop


def apply_host_orchestration_application_wiring_bundle(
    backend: NexusLoop,
    bundle: HostOrchestrationApplicationWiringBundle,
) -> None:
    """Apply resolved application wiring to the private orchestration backend."""
    env = bundle.environment
    if bundle.security_wiring is not None:
        apply_application_security_wiring(backend, bundle.security_wiring, env=env)
    if bundle.guardrail_wiring is not None:
        apply_application_guardrail_wiring(backend, bundle.guardrail_wiring, env)
    if bundle.decision_wiring is not None:
        apply_application_decision_wiring(
            backend,
            bundle.decision_wiring,
            environment=env,
        )


__all__ = ["apply_host_orchestration_application_wiring_bundle"]
