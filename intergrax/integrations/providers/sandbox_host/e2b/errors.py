# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""E2B sandbox host integration errors."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationConfigurationError


class E2bSandboxHostError(IntegrationConfigurationError):
    """E2B sandbox host operation failed."""


class E2bSandboxSecurityError(E2bSandboxHostError):
    """Security configuration or attestation failed for an E2B sandbox session."""
