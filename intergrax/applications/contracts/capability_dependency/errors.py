# © Artur Czarnecki. All rights reserved.

"""Capability dependency validation failures (P1.3)."""

from __future__ import annotations

from intergrax.applications.contracts.capability_dependency.validation import (
    CapabilityDependencyValidationResult,
)


class CapabilityDependencyValidationError(RuntimeError):
    """Dependency validation failed before execution may proceed."""

    def __init__(self, result: CapabilityDependencyValidationResult) -> None:
        self.result = result
        lines = [
            (
                f"- {failure.owner.canonical_key} requires "
                f"{failure.dependency.canonical_key} "
                f"({failure.status.value}): {failure.reason}"
            )
            for failure in result.required_failures
        ]
        message = "Capability dependency validation failed:\n" + "\n".join(lines)
        super().__init__(message)


class RequiredCapabilityDependencyUnavailableError(CapabilityDependencyValidationError):
    """Required dependency is unavailable or unknown — fail closed."""


class CapabilityDependencyDeclarationConflictError(CapabilityDependencyValidationError):
    """Conflicting dependency declarations cannot be merged deterministically."""
