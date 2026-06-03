# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.registry.bootstrap import (
    register_default_integrations,
    reset_default_integrations_state,
)
from intergrax.integrations.core import (
    IntegrationBinding,
    IntegrationManifest,
    IntegrationPlugin,
    IntegrationRef,
)
from intergrax.integrations.core.defaults import CLOUD_PLATFORM_DEFAULTS
from intergrax.integrations.core.slug import SlugInput, coerce_slug, slug_value
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
from intergrax.integrations.registry.plugin_register import (
    register_from_manifest,
    register_integration_plugin,
)
from intergrax.integrations.registry.factory import (
    build_profile_from_env,
    build_profile_from_mapping,
    resolve,
    resolve_from_profile,
    resolve_slug,
)
from intergrax.integrations.registry.profile import IntegrationProfile, default_lab_profile

__all__ = [
    "CLOUD_PLATFORM_DEFAULTS",
    "IntegrationBinding",
    "IntegrationManifest",
    "IntegrationPlugin",
    "IntegrationProfile",
    "IntegrationRef",
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
    "register_default_integrations",
    "register_from_manifest",
    "register_integration",
    "register_integration_plugin",
    "resolve",
    "resolve_from_profile",
    "resolve_slug",
    "reset_default_integrations_state",
    "slug_value",
    "unregister_integration",
]
