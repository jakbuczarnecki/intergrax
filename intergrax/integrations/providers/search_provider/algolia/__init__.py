# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.search_provider.algolia.bundle import create_algolia_search_provider
from intergrax.integrations.providers.search_provider.algolia.register import register_algolia_integration

__all__ = ["create_algolia_search_provider", "register_algolia_integration"]
