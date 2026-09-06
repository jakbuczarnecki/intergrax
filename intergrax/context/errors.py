# © Artur Czarnecki. All rights reserved.

"""Typed Context provider lifecycle errors (P1.9)."""

from __future__ import annotations

from intergrax.context.contracts import ContextFragmentSource, ContextProviderDescriptor


class ContextProviderLifecycleError(RuntimeError):
    """Base error for provider lifecycle and provenance semantics."""


class ContextProviderRegistrationError(ContextProviderLifecycleError, ValueError):
    """Registry rejected provider registration."""


class ContextProviderContractViolationError(ContextProviderLifecycleError, ValueError):
    """Provider returned fragments violating declared contract."""

    def __init__(
        self,
        *,
        descriptor: ContextProviderDescriptor,
        reason_code: str,
        detail: str = "",
    ) -> None:
        self.descriptor = descriptor
        self.reason_code = reason_code
        self.detail = detail
        message = (
            f"provider contract violation for {descriptor.provider_id}@"
            f"{descriptor.provider_version}: {reason_code}"
        )
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)


class RequiredContextSourceUnavailableError(ContextProviderLifecycleError, ValueError):
    """Mandatory context source could not be satisfied before model call."""

    def __init__(
        self,
        *,
        source: ContextFragmentSource,
        reason_code: str,
        provider_id: str = "",
    ) -> None:
        self.source = source
        self.reason_code = reason_code
        self.provider_id = provider_id
        detail = f" ({provider_id})" if provider_id else ""
        super().__init__(
            f"required context source unavailable: {source.value} "
            f"[{reason_code}]{detail}",
        )
