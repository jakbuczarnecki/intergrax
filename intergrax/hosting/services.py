# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Compatibility re-export — canonical owner is ``hosting.contracts.service_registry``."""

from intergrax.hosting.contracts.service_registry import (
    HostedApplicationServiceRegistry,
    HostedApplicationServiceRegistryCompatibilityError,
    HostedApplicationServiceRegistryDuplicateError,
    HostedApplicationServiceRegistryError,
    HostedApplicationServiceRegistryMissingError,
    HostedApplicationServiceRegistryStateError,
)

__all__ = [
    "HostedApplicationServiceRegistry",
    "HostedApplicationServiceRegistryCompatibilityError",
    "HostedApplicationServiceRegistryDuplicateError",
    "HostedApplicationServiceRegistryError",
    "HostedApplicationServiceRegistryMissingError",
    "HostedApplicationServiceRegistryStateError",
]
