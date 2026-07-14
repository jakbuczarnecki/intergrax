# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosting engine error hierarchy (APP-HOST-W2/W3)."""

from __future__ import annotations


class HostedApplicationEngineError(Exception):
    """Base error for hosted application engine operations."""


class HostedApplicationConfigurationError(HostedApplicationEngineError):
    """Raised when hosting configuration or factory signatures are invalid."""


class HostedApplicationDefinitionError(HostedApplicationEngineError):
    """Raised when profile composition or definition resolution fails."""


class HostedApplicationLifecycleTransitionError(HostedApplicationEngineError):
    """Raised when a lifecycle transition is not allowed."""


class HostedApplicationStartupError(HostedApplicationEngineError):
    """Raised when hosted application startup fails fatally."""


class HostedApplicationShutdownError(HostedApplicationEngineError):
    """Raised when hosted application shutdown fails fatally."""


class HostedApplicationHookError(HostedApplicationEngineError):
    """Raised when a blocking hook fails."""


class HostedApplicationComponentError(HostedApplicationEngineError):
    """Raised when a required component operation fails fatally."""


class HostedApplicationRuntimeError(HostedApplicationEngineError):
    """Raised when opaque application runtime operations fail."""


class HostedApplicationDiagnosticError(HostedApplicationEngineError):
    """Raised when hosting diagnostics or failure records are invalid."""


class HostedApplicationInstanceGuardError(HostedApplicationEngineError):
    """Raised when instance guard operations fail."""


class HostedApplicationInstanceConflictError(HostedApplicationInstanceGuardError):
    """Raised when another active instance owns the configured scope."""

    def __init__(self, message: str, snapshot: object | None = None) -> None:
        super().__init__(message)
        self.snapshot = snapshot


class HostedApplicationInstanceOwnershipError(HostedApplicationInstanceGuardError):
    """Raised when instance lease ownership verification fails."""


class HostedApplicationControlError(HostedApplicationEngineError):
    """Raised when control coordinator operations are invalid."""


class HostedApplicationShutdownTimeoutError(HostedApplicationShutdownError):
    """Raised when bounded shutdown phases exceed their deadline."""


class HostedApplicationSignalError(HostedApplicationEngineError):
    """Raised when signal adapter installation or handling fails."""


class HostedApplicationRestartPolicyError(HostedApplicationEngineError):
    """Raised when restart policy evaluation or configuration is invalid."""


class HostedApplicationSupervisorError(HostedApplicationEngineError):
    """Raised when supervisor orchestration fails."""
