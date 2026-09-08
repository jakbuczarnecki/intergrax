# © Artur Czarnecki. All rights reserved.

"""E2B credential discovery for physical qualification — environment only."""

from __future__ import annotations

from enum import Enum

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.sandbox_host.e2b.config import E2bSandboxHostConfig


class E2bCredentialStatus(str, Enum):
    """Credential availability without exposing secret material."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


def resolve_e2b_credentials() -> E2bCredentialStatus:
    """Resolve E2B API credentials from the process environment.

    Priority: ``INTERGRAX_E2B_API_KEY`` > ``E2B_API_KEY``. Never logs or returns
    secret values.
    """
    try:
        E2bSandboxHostConfig.from_env().resolved_api_key()
    except IntegrationConfigurationError:
        return E2bCredentialStatus.UNAVAILABLE
    except Exception:
        return E2bCredentialStatus.UNAVAILABLE
    return E2bCredentialStatus.AVAILABLE
