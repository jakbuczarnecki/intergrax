# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool domain errors."""

from __future__ import annotations


class ToolDomainError(Exception):
    """Base error for Tool domain orchestration."""


class DynamicToolAcquisitionResolutionError(ToolDomainError):
    """Exact tool package resolution failed or selected identity mismatch."""


class DynamicToolAcquisitionConflictError(ToolDomainError):
    """Idempotent acquisition replay conflict for the same operation id."""


class DynamicToolAcquisitionActivationError(ToolDomainError):
    """Resolved package could not be activated for host profile."""


__all__ = [
    "DynamicToolAcquisitionActivationError",
    "DynamicToolAcquisitionConflictError",
    "DynamicToolAcquisitionResolutionError",
    "ToolDomainError",
]
