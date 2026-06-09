# © Artur Czarnecki. All rights reserved.

"""Guardrail assembly validation for Tier-3 hosts (M-P12-WIRE.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intergrax.applications._shared.guardrail_wiring import ApplicationGuardrailWiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.nexus_loop import NexusLoop


@dataclass(frozen=True, slots=True)
class GuardrailAssemblyValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()


class GuardrailAssemblyError(ValueError):
    def __init__(self, errors: Sequence[str]) -> None:
        self.errors: tuple[str, ...] = tuple(errors)
        super().__init__("; ".join(self.errors))


def validate_guardrail_wiring(
    wiring: ApplicationGuardrailWiring,
    env: ApplicationEnvironmentProfile,
) -> GuardrailAssemblyValidationResult:
    errors: list[str] = []
    binding = env.integration_profile.llm_guardrail
    if wiring.options.enabled and binding is None:
        errors.append("llm_guardrail binding missing while guardrail wiring enabled")
    if wiring.options.enabled and wiring.backend is None:
        errors.append("guardrail backend failed to resolve from integration profile")
    if wiring.options.enabled and wiring.backend is not None:
        primary = wiring.options.backend_slug or ""
        if primary and primary not in wiring.backend.slug.split("+"):
            errors.append("guardrail backend slug mismatch")
    if env.guardrail_profile.enabled and not wiring.options.enabled:
        errors.append("guardrail_profile.enabled requires integration_profile.llm_guardrail")
    return GuardrailAssemblyValidationResult(valid=not errors, errors=tuple(errors))


def validate_guardrail_nexus_wiring(
    wiring: ApplicationGuardrailWiring,
    nexus: NexusLoop,
) -> GuardrailAssemblyValidationResult:
    if not wiring.options.enabled:
        return GuardrailAssemblyValidationResult(valid=True)
    pipeline = nexus._middleware  # noqa: SLF001
    if not isinstance(pipeline, MiddlewarePipeline):
        return GuardrailAssemblyValidationResult(
            valid=False,
            errors=("NexusLoop middleware pipeline missing",),
        )
    names = {middleware.name for middleware in pipeline._middleware}  # noqa: SLF001
    if "LlmGuardrailMiddleware" not in names:
        return GuardrailAssemblyValidationResult(
            valid=False,
            errors=("missing LlmGuardrailMiddleware on NexusLoop",),
        )
    return GuardrailAssemblyValidationResult(valid=True)


def assert_guardrail_assembly_valid(
    wiring: ApplicationGuardrailWiring,
    env: ApplicationEnvironmentProfile,
    *,
    nexus: NexusLoop | None = None,
) -> None:
    errors = list(validate_guardrail_wiring(wiring, env).errors)
    if nexus is not None and wiring.options.enabled:
        errors.extend(validate_guardrail_nexus_wiring(wiring, nexus).errors)
    if errors:
        raise GuardrailAssemblyError(errors)
