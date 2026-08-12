# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed domain conflicts for Agent Distribution services (AP-4)."""

from __future__ import annotations


class AgentDistributionError(Exception):
    """Base error for Agent Distribution domain services."""


class AgentDistributionNotFoundError(AgentDistributionError):
    """Requested durable record does not exist."""


class InstallationLifecycleError(AgentDistributionError):
    """Illegal installation lifecycle transition."""


class InstallationSlotConflict(AgentDistributionError):
    """Concurrent or stale installation slot mutation."""


class BindingRevisionConflict(AgentDistributionError):
    """Optimistic binding revision mismatch."""


class BindingLifecycleError(AgentDistributionError):
    """Illegal binding lifecycle transition."""


class RuntimeRevisionConflict(AgentDistributionError):
    """Concurrent or stale runtime revision mutation."""


class RuntimeRevisionLifecycleError(AgentDistributionError):
    """Illegal runtime revision lifecycle transition."""
