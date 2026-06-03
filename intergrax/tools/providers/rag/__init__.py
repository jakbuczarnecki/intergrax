# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""RAG tool bundle — register via ``intergrax.tools.registry.shipped_plugins`` or ``register_rag_tool_bundle``."""

from intergrax.tools.providers.rag.service import RAG_TOOL_ID

__all__ = ["RAG_TOOL_ID"]
