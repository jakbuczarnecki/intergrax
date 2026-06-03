# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-0 Integration Library — catalog, contracts, and registry (§7.1, Phase M)."""

from intergrax.integrations.contracts.base import (
    HealthStatus,
    IntegrationCategory,
    IntegrationEntry,
    IntegrationError,
    IntegrationMetadata,
    IntegrationStatus,
    UnknownIntegrationError,
    UnknownIntegrationCategoryError,
)
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.core.slug import SlugInput, coerce_slug
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.factory import (
    build_profile_from_env,
    build_profile_from_mapping,
    resolve,
    resolve_from_profile,
)
from intergrax.integrations.registry.profile import IntegrationProfile

__all__ = [
    "HealthStatus",
    "IntegrationCategory",
    "IntegrationEntry",
    "IntegrationError",
    "IntegrationManifest",
    "IntegrationMetadata",
    "IntegrationProfile",
    "IntegrationStatus",
    "SlugInput",
    "UnknownIntegrationCategoryError",
    "UnknownIntegrationError",
    "build_profile_from_env",
    "build_profile_from_mapping",
    "coerce_slug",
    "register_default_integrations",
    "resolve",
    "resolve_from_profile",
]
