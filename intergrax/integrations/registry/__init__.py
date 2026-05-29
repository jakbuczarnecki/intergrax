# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.registry.catalog import (
    catalog_snapshot,
    clear_catalog,
    get_entry,
    iter_entries,
    list_slugs,
    metadata_for_slug,
    register_integration,
    unregister_integration,
)
from intergrax.integrations.registry.factory import (
    build_profile_from_env,
    build_profile_from_mapping,
    resolve,
    resolve_from_profile,
    resolve_slug,
)
from intergrax.integrations.registry.profile import IntegrationProfile, default_lab_profile
from intergrax.integrations.registry.slugs import (
    CLOUD_PLATFORM_DEFAULTS,
    IntegrationSlug,
    SlugInput,
    coerce_slug,
)

__all__ = [
    "IntegrationProfile",
    "IntegrationSlug",
    "SlugInput",
    "build_profile_from_env",
    "build_profile_from_mapping",
    "catalog_snapshot",
    "clear_catalog",
    "coerce_slug",
    "default_lab_profile",
    "get_entry",
    "iter_entries",
    "list_slugs",
    "metadata_for_slug",
    "register_integration",
    "resolve",
    "resolve_from_profile",
    "resolve_slug",
    "unregister_integration",
]
