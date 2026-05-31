# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Concrete integration providers — registered via ``registry.catalog`` (Phase M.4+).

Layout::

    providers/
        <category>/          # IntegrationCategory value (relational_store, …)
            <slug>/          # vendor package (s3, postgresql, …)
                bundle.py
                USAGE.md
                …

See ``layout.py`` for slug → category mapping.
"""

from intergrax.integrations.providers.layout import SLUG_CATEGORY, provider_import_path, provider_package_path

__all__ = ["SLUG_CATEGORY", "provider_import_path", "provider_package_path"]
