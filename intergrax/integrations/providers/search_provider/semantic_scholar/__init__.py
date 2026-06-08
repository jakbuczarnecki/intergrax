# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.search_provider.semantic_scholar.bundle import create_semantic_scholar_search_provider
from intergrax.integrations.providers.search_provider.semantic_scholar.register import register_semantic_scholar_integration

__all__ = ["create_semantic_scholar_search_provider", "register_semantic_scholar_integration"]
