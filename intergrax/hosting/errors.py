# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosting engine error hierarchy (APP-HOST-W2/W3)."""

from __future__ import annotations

from enum import StrEnum

from intergrax.hosting.process_bootstrap import HostedProcessBootstrapPhase


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


class HostedApplicationSupervisorFailureReason(StrEnum):
    """Deterministic bounded failure reasons for supervisor pre-engine failures."""

    ENGINE_FACTORY_FAILED = "engine_factory_failed"
    ENGINE_FACTORY_INVALID_RESULT = "engine_factory_invalid_result"
    ENGINE_INSTANCE_ID_MISMATCH = "engine_instance_id_mismatch"
    ENGINE_PROFILE_DIGEST_MISMATCH = "engine_profile_digest_mismatch"
    ENGINE_DEFINITION_DIGEST_MISMATCH = "engine_definition_digest_mismatch"
    ENGINE_APPLICATION_ID_MISMATCH = "engine_application_id_mismatch"

    @property
    def phase(self) -> HostedProcessBootstrapPhase:
        if self in (
            HostedApplicationSupervisorFailureReason.ENGINE_FACTORY_FAILED,
            HostedApplicationSupervisorFailureReason.ENGINE_FACTORY_INVALID_RESULT,
        ):
            return HostedProcessBootstrapPhase.ENGINE_CONSTRUCTION
        return HostedProcessBootstrapPhase.ENGINE_CONTRACT_VALIDATION


class HostedApplicationSupervisorError(HostedApplicationEngineError):
    """Raised when supervisor orchestration fails."""

    def __init__(
        self,
        message: str,
        *,
        reason: HostedApplicationSupervisorFailureReason,
        phase: HostedProcessBootstrapPhase,
    ) -> None:
        super().__init__(message)
        self._reason = reason
        self._phase = phase

    @property
    def reason(self) -> HostedApplicationSupervisorFailureReason:
        return self._reason

    @property
    def phase(self) -> HostedProcessBootstrapPhase:
        return self._phase
