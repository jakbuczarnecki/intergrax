# © Artur Czarnecki. All rights reserved.

"""Tier-3 guardrail wiring (M-P12-WIRE.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.guardrail_runtime_bridge import (
    GuardrailWiringOptions,
    resolve_guardrail_backend,
    resolve_guardrail_wiring_options,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations.contracts.llm_guardrail import LlmGuardrailBackend
from intergrax.runtime.nexus.nexus_loop import NexusLoop


@dataclass(frozen=True, slots=True)
class ApplicationGuardrailWiring:
    options: GuardrailWiringOptions
    backend: LlmGuardrailBackend | None


def wire_application_guardrail(env: ApplicationEnvironmentProfile) -> ApplicationGuardrailWiring:
    return ApplicationGuardrailWiring(
        options=resolve_guardrail_wiring_options(env),
        backend=resolve_guardrail_backend(env),
    )


def apply_application_guardrail_wiring(
    nexus: NexusLoop,
    wiring: ApplicationGuardrailWiring,
) -> ApplicationGuardrailWiring:
    """Return wiring for host middleware; NexusLoop stores backend via host closure."""
    _ = nexus
    return wiring
