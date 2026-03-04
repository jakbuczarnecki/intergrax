# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.rag.document_loaders.registry.document_handler_registry import (
    DocumentHandlerRegistry,
)


def build_default_registry() -> DocumentHandlerRegistry:
    """
    Build a document handler registry preconfigured with default handlers.

    The registry is intentionally returned without importing specific
    handlers here. Concrete handlers should be registered by the caller
    or by optional bootstrap modules.

    Returns
    -------
    DocumentHandlerRegistry
        Initialized registry instance.
    """

    registry = DocumentHandlerRegistry()

    # Default handlers will be registered by optional modules.
    # This function intentionally avoids importing handler implementations
    # to prevent circular dependencies and heavy bootstrap imports.

    return registry