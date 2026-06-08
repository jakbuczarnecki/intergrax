# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.search_provider.arxiv.bundle import create_arxiv_search_provider
from intergrax.integrations.providers.search_provider.arxiv.register import register_arxiv_integration

__all__ = ["create_arxiv_search_provider", "register_arxiv_integration"]
