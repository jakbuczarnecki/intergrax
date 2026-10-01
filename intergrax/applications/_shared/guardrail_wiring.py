# © Artur Czarnecki. All rights reserved.

"""Tier-3 guardrail wiring (M-P12-WIRE.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.application_guardrail_middleware import LlmGuardrailMiddleware
from intergrax.applications._shared.guardrail_runtime_bridge import (
    GuardrailWiringOptions,
    resolve_guardrail_backend,
    resolve_guardrail_wiring_options,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.host_orchestration_application_wiring_target import (
    HostOrchestrationApplicationWiringTarget,
)
from intergrax.integrations.contracts.llm_guardrail import LlmGuardrailBackend


@dataclass(frozen=True, slots=True)
class ApplicationGuardrailWiring:
    options: GuardrailWiringOptions
    backend: LlmGuardrailBackend | None


def wire_application_guardrail(env: ApplicationEnvironmentProfile) -> ApplicationGuardrailWiring:
    return ApplicationGuardrailWiring(
        options=resolve_guardrail_wiring_options(env),
        backend=resolve_guardrail_backend(env),
    )


def _attach_middleware(
    target: HostOrchestrationApplicationWiringTarget,
    middleware: LlmGuardrailMiddleware,
) -> None:
    target.middleware.attach_runtime_middleware_if_absent(middleware)


def apply_application_guardrail_wiring(
    target: HostOrchestrationApplicationWiringTarget,
    wiring: ApplicationGuardrailWiring,
    env: ApplicationEnvironmentProfile,
) -> ApplicationGuardrailWiring:
    """Attach vendor guardrail middleware when profile binding is present."""
    if not wiring.options.enabled or wiring.backend is None:
        return wiring
    if not env.guardrail_profile.enabled:
        return wiring
    _attach_middleware(
        target,
        LlmGuardrailMiddleware(
            wiring.backend,
            env.guardrail_profile,
            event_bus=target.event_bus,
        ),
    )
    return wiring
