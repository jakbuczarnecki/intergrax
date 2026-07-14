# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosting engine error hierarchy (APP-HOST-W2)."""

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
