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
from intergrax.integrations.contracts.llm_guardrail import LlmGuardrailBackend
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
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


def _attach_middleware(nexus: NexusLoop, middleware: LlmGuardrailMiddleware) -> None:
    pipeline = nexus._middleware  # noqa: SLF001
    if isinstance(pipeline, MiddlewarePipeline):
        pipeline._middleware = sorted(  # noqa: SLF001
            [middleware, *pipeline._middleware],
            key=lambda item: item.priority,
        )


def apply_application_guardrail_wiring(
    nexus: NexusLoop,
    wiring: ApplicationGuardrailWiring,
    env: ApplicationEnvironmentProfile,
) -> ApplicationGuardrailWiring:
    """Attach vendor guardrail middleware when profile binding is present."""
    if not wiring.options.enabled or wiring.backend is None:
        return wiring
    if not env.guardrail_profile.enabled:
        return wiring
    _attach_middleware(
        nexus,
        LlmGuardrailMiddleware(wiring.backend, env.guardrail_profile),
    )
    return wiring
