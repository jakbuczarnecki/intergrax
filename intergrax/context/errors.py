# © Artur Czarnecki. All rights reserved.

"""Typed Context provider lifecycle errors (P1.9)."""

from __future__ import annotations

from intergrax.context.contracts import ContextFragmentSource, ContextProviderDescriptor
from intergrax.contracts.execution_identity import ExecutionId


class ContextProviderLifecycleError(RuntimeError):
    """Base error for provider lifecycle and provenance semantics."""


class ContextProviderRegistrationError(ContextProviderLifecycleError, ValueError):
    """Registry rejected provider registration."""


class ContextProviderBindingError(ContextProviderLifecycleError):
    """Execution provider binding is missing or inconsistent."""


class ContextProviderBindingUnavailableError(ContextProviderBindingError):
    """Pinned provider semantics are unavailable for an execution."""

    def __init__(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId | str,
        provider_id: str,
        reason_code: str = "provider.binding_unavailable",
    ) -> None:
        self.tenant_id = tenant_id
        self.execution_id = str(execution_id)
        self.provider_id = provider_id
        self.reason_code = reason_code
        super().__init__(
            f"context provider binding unavailable for execution {self.execution_id}: "
            f"{provider_id} ({reason_code})",
        )


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
